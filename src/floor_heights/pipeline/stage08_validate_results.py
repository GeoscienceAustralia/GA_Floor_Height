#!/usr/bin/env python
"""Stage-08: Validate FFH estimation results against Frontier SI ground truth.

This stage compares the estimated floor heights from stage07 with Frontier SI
LIDAR-derived ground truth values to assess the accuracy of the pipeline.

Metrics calculated:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- ME (Mean Error / Bias)
- Correlation coefficient
- Success rates for each FFH method
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from floor_heights.config import CONFIG
from floor_heights.db.schemas import (
    BatchWriter,
    Stage08ValidationRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import read_table
from floor_heights.utils.progress import PipelineProgress

STAGE_NAME = "stage08_validation"
STAGE_DESCRIPTION = "Validate FFH results against Frontier SI ground truth"

FIGURE_DPI = 300
FIGURE_FORMAT = "png"


def get_frontier_si_ground_truth() -> pd.DataFrame:
    """Load Frontier SI ground truth validation data from validation_labels parquet file."""

    validation_file = CONFIG.output_root.parent / "data" / "processed" / "validation_labels.parquet"

    if not validation_file.exists():
        logger.error(f"Validation labels file not found: {validation_file}")
        raise FileNotFoundError(f"Please run the data loader to create {validation_file}")

    df = pd.read_parquet(validation_file)

    df = df[df["dataset"] == "FrontierSI Validation"].copy()

    df = df[df["frontiersi_floor_height_m"].notna()].copy()

    df = df.rename(columns={"frontiersi_floor_height_m": "floor_height_m"})
    df["floor_height_m"] = df["floor_height_m"].astype(float)

    logger.info(f"Loaded {len(df)} Frontier SI ground truth records from {validation_file}")
    return df


def get_ffh_results() -> pd.DataFrame:
    """Load FFH estimation results from stage07."""
    df = read_table(
        "stage07_floor_heights",
        columns=["id", "building_id", "gnaf_id", "region_name", "ffh1", "ffh2", "ffh3", "method"],
    )

    logger.info(f"Loaded {len(df)} FFH estimation results from stage07")
    return df


def merge_results_with_ground_truth(ffh_results: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Merge FFH results with ground truth based on building_id and gnaf_id."""
    logger.debug(f"FFH results shape before merge: {ffh_results.shape}")
    logger.debug(f"Ground truth shape before merge: {ground_truth.shape}")

    ffh_duplicates = ffh_results.groupby(["building_id", "gnaf_id"]).size()
    if (ffh_duplicates > 1).any():
        logger.warning(f"Found {(ffh_duplicates > 1).sum()} duplicate building_id+gnaf_id combinations in FFH results")
        ffh_results = ffh_results.drop_duplicates(subset=["building_id", "gnaf_id"], keep="first")

    gt_duplicates = ground_truth.groupby(["building_id", "gnaf_id"]).size()
    if (gt_duplicates > 1).any():
        logger.warning(f"Found {(gt_duplicates > 1).sum()} duplicate building_id+gnaf_id combinations in ground truth")
        ground_truth = ground_truth.drop_duplicates(subset=["building_id", "gnaf_id"], keep="first")

    merged = ffh_results.merge(
        ground_truth[["building_id", "gnaf_id", "floor_height_m"]],
        on=["building_id", "gnaf_id"],
        how="inner",
        suffixes=("", "_gt"),
    )

    merged = merged.rename(columns={"floor_height_m": "ground_truth_ffh"})

    logger.info(f"Matched {len(merged)} buildings with Frontier SI ground truth")

    unmatched_gt = len(ground_truth) - len(merged)
    if unmatched_gt > 0:
        logger.info(f"{unmatched_gt} Frontier SI ground truth records had no matching FFH results")

    unmatched_ffh = len(ffh_results) - len(merged)
    if unmatched_ffh > 0:
        logger.info(f"{unmatched_ffh} FFH results had no matching Frontier SI ground truth")

    return merged


def calculate_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Calculate validation metrics."""
    mask = ~np.isnan(predicted) & ~np.isnan(ground_truth)
    predicted = predicted[mask]
    ground_truth = ground_truth[mask]

    if len(predicted) == 0:
        return {
            "n_samples": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "me": np.nan,
            "correlation": np.nan,
            "std": np.nan,
            "p95": np.nan,
        }

    rmse = np.sqrt(mean_squared_error(ground_truth, predicted))
    mae = mean_absolute_error(ground_truth, predicted)
    me = np.mean(predicted - ground_truth)
    errors = predicted - ground_truth

    correlation = np.corrcoef(ground_truth, predicted)[0, 1] if len(predicted) > 1 else np.nan

    return {
        "n_samples": len(predicted),
        "rmse": rmse,
        "mae": mae,
        "me": me,
        "correlation": correlation,
        "std": np.std(errors),
        "p95": np.percentile(np.abs(errors), 95) if len(errors) > 0 else np.nan,
    }


def plot_validation_results(merged_df: pd.DataFrame, output_dir: Path) -> None:
    """Create validation plots for each FFH method."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ffh_methods = ["ffh1", "ffh2", "ffh3"]
    method_names = {
        "ffh1": "FFH1 (Floor to Ground Feature)",
        "ffh2": "FFH2 (Floor to Nearest Ground)",
        "ffh3": "FFH3 (Floor to DTM)",
    }

    regions = merged_df["region_name"].unique()
    logger.info(f"Creating validation plots for regions: {', '.join(regions)}")

    _create_region_plots(merged_df, output_dir, "all_regions", ffh_methods, method_names)

    for region in regions:
        region_df = merged_df[merged_df["region_name"] == region]
        if len(region_df) > 0:
            _create_region_plots(region_df, output_dir, region, ffh_methods, method_names)


def _create_region_plots(
    data_df: pd.DataFrame, output_dir: Path, region_name: str, ffh_methods: list, method_names: dict
) -> None:
    """Create validation plots for a specific region or all regions."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"FFH Validation Results - {region_name.replace('_', ' ').title()}", fontsize=16)

    for _idx, (method, ax) in enumerate(zip(ffh_methods, axes, strict=False)):
        valid_mask = data_df[method].notna()
        valid_df = data_df[valid_mask]

        if len(valid_df) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(method_names[method])
            continue

        predicted = np.array(valid_df[method].values)
        ground_truth = np.array(valid_df["ground_truth_ffh"].values)

        metrics = calculate_metrics(predicted, ground_truth)

        ax.scatter(ground_truth, predicted, alpha=0.6, s=30)

        min_val = min(ground_truth.min(), predicted.min())
        max_val = max(ground_truth.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal")

        metrics_text = (
            f"n = {metrics['n_samples']}\n"
            f"RMSE = {metrics['rmse']:.3f}m\n"
            f"MAE = {metrics['mae']:.3f}m\n"
            f"ME = {metrics['me']:+.3f}m\n"
            f"r = {metrics['correlation']:.3f}"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

        ax.set_xlabel("Ground Truth FFH (m)")
        ax.set_ylabel("Predicted FFH (m)")
        ax.set_title(method_names[method])
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    fig_path = output_dir / f"validation_scatter_{region_name}.{FIGURE_FORMAT}"
    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {region_name} validation plot to {fig_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"FFH Error Distributions - {region_name.replace('_', ' ').title()}", fontsize=16)

    for _idx, (method, ax) in enumerate(zip(ffh_methods, axes, strict=False)):
        valid_mask = data_df[method].notna()
        valid_df = data_df[valid_mask]

        if len(valid_df) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(method_names[method])
            continue

        errors = np.array(valid_df[method].values) - np.array(valid_df["ground_truth_ffh"].values)

        ax.hist(errors, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
        ax.axvline(np.mean(errors), color="green", linestyle="-", linewidth=2, label=f"Mean = {np.mean(errors):.3f}m")

        ax.set_xlabel("Error (Predicted - Ground Truth) [m]")
        ax.set_ylabel("Count")
        ax.set_title(method_names[method])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = output_dir / f"error_distribution_{region_name}.{FIGURE_FORMAT}"
    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {region_name} error distribution plot to {fig_path}")


def generate_validation_report(merged_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a detailed validation report."""
    report_path = output_dir / "validation_report.txt"

    with report_path.open("w") as f:
        f.write("FFH Validation Report - Frontier SI Ground Truth\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total buildings with Frontier SI ground truth: {len(merged_df)}\n")
        f.write(f"Buildings with FFH1: {merged_df['ffh1'].notna().sum()}\n")
        f.write(f"Buildings with FFH2: {merged_df['ffh2'].notna().sum()}\n")
        f.write(f"Buildings with FFH3: {merged_df['ffh3'].notna().sum()}\n\n")

        f.write("Regional breakdown:\n")
        for region in merged_df["region_name"].unique():
            region_count = len(merged_df[merged_df["region_name"] == region])
            f.write(f"  {region}: {region_count} buildings\n")
        f.write("\n")

        ffh_methods = ["ffh1", "ffh2", "ffh3"]
        for method in ffh_methods:
            f.write(f"\n{method.upper()} Validation Metrics\n")
            f.write("-" * 30 + "\n")

            valid_mask = merged_df[method].notna()
            if valid_mask.sum() == 0:
                f.write("No valid predictions\n")
                continue

            valid_df = merged_df[valid_mask]
            metrics = calculate_metrics(
                np.array(valid_df[method].values), np.array(valid_df["ground_truth_ffh"].values)
            )

            f.write(f"Number of samples: {metrics['n_samples']}\n")
            f.write(f"RMSE: {metrics['rmse']:.3f} m\n")
            f.write(f"MAE: {metrics['mae']:.3f} m\n")
            f.write(f"Mean Error (bias): {metrics['me']:+.3f} m\n")
            f.write(f"Correlation coefficient: {metrics['correlation']:.3f}\n")
            f.write(f"Error std dev: {metrics['std']:.3f} m\n")
            f.write(f"Error 95th percentile: {metrics['p95']:.3f} m\n")

            f.write("\nBy region:\n")
            for region in valid_df["region_name"].unique():
                region_mask = valid_df["region_name"] == region
                if region_mask.sum() > 0:
                    region_metrics = calculate_metrics(
                        np.array(valid_df.loc[region_mask, method].values),
                        np.array(valid_df.loc[region_mask, "ground_truth_ffh"].values),
                    )
                    f.write(
                        f"  {region}: n={region_metrics['n_samples']}, "
                        f"RMSE={region_metrics['rmse']:.3f}m, "
                        f"MAE={region_metrics['mae']:.3f}m\n"
                    )

    logger.info(f"Saved validation report to {report_path}")


def save_validation_results(merged_df: pd.DataFrame, output_dir: Path) -> None:
    """Save merged validation results to file."""
    output_path = output_dir / "validation_results.parquet"
    merged_df.to_parquet(output_path)
    logger.info(f"Saved validation results to {output_path}")

    csv_path = output_dir / "validation_results.csv"
    merged_df.to_csv(csv_path, index=False)
    logger.info(f"Saved validation results to {csv_path}")


def process_validation_batch(merged_df: pd.DataFrame, writer: BatchWriter, prog: PipelineProgress) -> None:
    """Process validation results and write to database."""
    ffh_methods = ["ffh1", "ffh2", "ffh3"]

    for _, row in merged_df.iterrows():
        for method in ffh_methods:
            if pd.notna(row[method]):
                error = row[method] - row["ground_truth_ffh"]

                record = Stage08ValidationRecord(
                    id=str(row["id"]),
                    building_id=row["building_id"],
                    region_name=row["region_name"],
                    gnaf_id=row.get("gnaf_id", ""),
                    ffh_method=method,
                    predicted_ffh=float(row[method]),
                    ground_truth_ffh=float(row["ground_truth_ffh"]),
                    error=float(error),
                    absolute_error=float(abs(error)),
                    squared_error=float(error**2),
                )
                writer.add(record)

        prog.update("suc", 1)


def run_validation() -> None:
    """Run validation against Frontier SI ground truth."""
    logger.info("Starting Stage 08: Validation against Frontier SI ground truth")

    try:
        ground_truth = get_frontier_si_ground_truth()
        ffh_results = get_ffh_results()

        if len(ffh_results) == 0:
            logger.warning("No FFH results found. Run stage07 first.")
            return

        merged_df = merge_results_with_ground_truth(ffh_results, ground_truth)

        if len(merged_df) == 0:
            logger.warning("No matching buildings found between FFH results and Frontier SI ground truth")
            return

        output_dir = CONFIG.output_root / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_validation_results(merged_df, output_dir)
        generate_validation_report(merged_df, output_dir)
        save_validation_results(merged_df, output_dir)

        with (
            PipelineProgress("Saving validation results", len(merged_df), show_elapsed=True) as prog,
            BatchWriter(STAGE_NAME, batch_size=1000, progress_tracker=prog) as writer,
        ):
            process_validation_batch(merged_df, writer, prog)

        logger.info("\nValidation Summary (All Regions):")
        for method in ["ffh1", "ffh2", "ffh3"]:
            valid_mask = merged_df[method].notna()
            if valid_mask.sum() > 0:
                metrics = calculate_metrics(
                    np.array(merged_df.loc[valid_mask, method].values),
                    np.array(merged_df.loc[valid_mask, "ground_truth_ffh"].values),
                )
                logger.info(
                    f"  {method.upper()}: n={metrics['n_samples']}, "
                    f"RMSE={metrics['rmse']:.3f}m, MAE={metrics['mae']:.3f}m, "
                    f"ME={metrics['me']:+.3f}m, r={metrics['correlation']:.3f}"
                )

        logger.info("\nValidation Summary by Region:")
        for region in merged_df["region_name"].unique():
            region_df = merged_df[merged_df["region_name"] == region]
            logger.info(f"\n  {region}:")
            for method in ["ffh1", "ffh2", "ffh3"]:
                valid_mask = region_df[method].notna()
                if valid_mask.sum() > 0:
                    metrics = calculate_metrics(
                        np.array(region_df.loc[valid_mask, method].values),
                        np.array(region_df.loc[valid_mask, "ground_truth_ffh"].values),
                    )
                    logger.info(
                        f"    {method.upper()}: n={metrics['n_samples']}, "
                        f"RMSE={metrics['rmse']:.3f}m, MAE={metrics['mae']:.3f}m"
                    )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


def run_stage(
    regions: list[str] | None = None,
    skip_existing: bool = False,
) -> None:
    """Run stage08 validation.

    Args:
        regions: Not used (validation uses all available data)
        skip_existing: If True, skip if validation already exists
    """
    initialize_all_stage_tables()

    if skip_existing:
        existing = read_table(STAGE_NAME)
        if not existing.empty:
            logger.info(f"Found {len(existing)} existing validation records. Skipping.")
            return

    try:
        run_validation()
        logger.info("Stage-08 complete")
    except Exception as e:
        logger.error(f"Stage-08 failed: {e}")
        raise


if __name__ == "__main__":
    run_stage()
