#!/usr/bin/env python
"""Stage-07: Estimate First Floor Heights from detected features and LiDAR projections.

This stage combines:
- Object detection results from stage04a/04b (doors, windows, stairs, foundations)
- LiDAR point cloud projections from stage05 (elevation, depth, classification rasters)
- Ground elevations from stage06 (gap-filling ground levels)

To estimate First Floor Heights (FFH) using multiple methods:
1. FFH1: Floor feature to ground feature (when both detected)
2. FFH2: Floor feature to nearest ground area from LiDAR
3. FFH3: Floor feature to ground elevation from DTM
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from floor_heights.config import CONFIG
from floor_heights.db.schemas import (
    BatchWriter,
    Stage07FloorHeightRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import (
    read_table,
    validate_file_exists_and_valid,
)
from floor_heights.utils.point_cloud_processings import (
    calculate_gapfill_depth,
    compute_feature_properties,
    estimate_FFH,
    get_closest_ground_to_feature,
    select_best_feature,
)
from floor_heights.utils.progress import PipelineProgress

FRONTDOOR_STANDARDS = {
    "width_m": CONFIG.ffh_estimation.frontdoor_standards.width_m,
    "height_m": CONFIG.ffh_estimation.frontdoor_standards.height_m,
    "area_m2": CONFIG.ffh_estimation.frontdoor_standards.area_m2,
    "ratio": CONFIG.ffh_estimation.frontdoor_standards.ratio,
}
FEATURE_WEIGHTS = {
    "area_m2": CONFIG.ffh_estimation.feature_weights.area_m2,
    "ratio": CONFIG.ffh_estimation.feature_weights.ratio,
    "confidence": CONFIG.ffh_estimation.feature_weights.confidence,
    "x_location": CONFIG.ffh_estimation.feature_weights.x_location,
    "y_location": CONFIG.ffh_estimation.feature_weights.y_location,
}
TARGET_CLASSES = CONFIG.ffh_estimation.target_classes

MIN_FFH = CONFIG.ffh_estimation.min_ffh
MAX_FFH = CONFIG.ffh_estimation.max_ffh

NODATA_DEPTH = CONFIG.ffh_estimation.nodata_depth
NODATA_ELEVATION = CONFIG.ffh_estimation.nodata_elevation


STAGE_NAME = "stage07_floor_heights"
STAGE_DESCRIPTION = "First Floor Height estimation from features and LiDAR"


def load_raster_arrays(
    region: str, row_id: int, building_id: str, gnaf_id: str, panorama_id: str, projection_mode: str = "standard"
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load elevation, classification, and depth raster arrays for a building."""
    try:
        gnaf_id = gnaf_id if gnaf_id and pd.notna(gnaf_id) else "NO_GNAF"

        proj_suffix = "_ca" if projection_mode == "classification_aware" else ""
        projection_dir = CONFIG.region_folder(region) / f"projections{proj_suffix}"

        building_dir = projection_dir / f"{row_id}_{building_id}_{gnaf_id}"
        if not building_dir.exists():
            logger.debug(f"Building directory not found: {building_dir}")
            return None, None, None

        base_pid = panorama_id[:-4] if panorama_id.lower().endswith(".jpg") else panorama_id
        elevation_files = list(building_dir.glob(f"*_{base_pid}_*_elevation_resampled.tif"))
        classification_files = list(building_dir.glob(f"*_{base_pid}_*_classification_resampled.tif"))
        depth_files = list(building_dir.glob(f"*_{base_pid}_*_depth_resampled.tif"))

        if not (elevation_files and classification_files and depth_files):
            elevation_files = list(building_dir.glob("*_elevation_resampled.tif"))
            classification_files = list(building_dir.glob("*_classification_resampled.tif"))
            depth_files = list(building_dir.glob("*_depth_resampled.tif"))

        if not (elevation_files and classification_files and depth_files):
            logger.debug(f"Raster files not found in {building_dir}")
            return None, None, None

        if not validate_file_exists_and_valid(elevation_files[0], file_type="image", min_size_bytes=1000):
            logger.debug(f"Invalid elevation file: {elevation_files[0]}")
            return None, None, None
        if not validate_file_exists_and_valid(classification_files[0], file_type="image", min_size_bytes=1000):
            logger.debug(f"Invalid classification file: {classification_files[0]}")
            return None, None, None
        if not validate_file_exists_and_valid(depth_files[0], file_type="image", min_size_bytes=1000):
            logger.debug(f"Invalid depth file: {depth_files[0]}")
            return None, None, None

        elevation_arr = np.array(Image.open(elevation_files[0]))
        classification_arr = np.array(Image.open(classification_files[0]))
        depth_arr = np.array(Image.open(depth_files[0]))

        return elevation_arr, classification_arr, depth_arr

    except Exception as e:
        logger.debug(f"Failed to load rasters for {building_id}: {e}")
        return None, None, None


def _process_single_building(args_tuple) -> tuple[str, dict[str, Any] | None]:
    """Process FFH estimation for a single building.

    Note: This function runs in a separate process and should only read from
    the database, not write to it, to avoid lock conflicts.
    """
    (region, building_row, detections_df, ground_elevations_df, projection_mode, processed_set) = args_tuple

    row_id = building_row["id"]
    building_id = building_row["building_id"]
    gnaf_id = building_row.get("gnaf_id")
    if not gnaf_id or pd.isna(gnaf_id):
        gnaf_id = "NO_GNAF"
    panorama_id = building_row["pano_id"]

    unique_id = f"{row_id}_{building_id}_{gnaf_id}"

    if unique_id in processed_set:
        return ("skip", None)

    try:
        detection_mask = (detections_df["building_id"] == building_id) & (detections_df["pano_id"] == panorama_id)
        building_detections = detections_df[detection_mask].copy()

        if building_detections.empty:
            return ("fail", None)
        ground_elev_mask = ground_elevations_df["unique_id"] == unique_id
        if not ground_elev_mask.any():
            return ("fail", None)

        ground_elevation_dtm = ground_elevations_df.loc[ground_elev_mask, "ground_elevation_m"].iloc[0]

        elevation_arr, classification_arr, depth_arr = load_raster_arrays(
            region, row_id, building_id, gnaf_id, panorama_id, projection_mode
        )

        if elevation_arr is None:
            return ("fail", None)
        gapfill_depth = calculate_gapfill_depth(depth_arr, classification_arr, nodata_depth=NODATA_DEPTH)
        building_detections["class"] = building_detections["class_name"].str.title()

        img_height, img_width = elevation_arr.shape
        building_detections["x1"] = building_detections["bbox_x1"].clip(0, img_width - 1)
        building_detections["x2"] = building_detections["bbox_x2"].clip(0, img_width - 1)
        building_detections["y1"] = building_detections["bbox_y1"].clip(0, img_height - 1)
        building_detections["y2"] = building_detections["bbox_y2"].clip(0, img_height - 1)

        for idx in building_detections.index:
            row = building_detections.loc[idx]
            props = compute_feature_properties(
                row=row,
                elevation_arr=elevation_arr,
                depth_arr=depth_arr,
                gapfill_depth=gapfill_depth,
                nodata=NODATA_ELEVATION,
            )
            building_detections.loc[idx, "top_elevation"] = float(props[0]) if props[0] is not None else None
            building_detections.loc[idx, "bottom_elevation"] = float(props[1]) if props[1] is not None else None
            building_detections.loc[idx, "width_m"] = float(props[2]) if props[2] is not None else None
            building_detections.loc[idx, "height_m"] = float(props[3]) if props[3] is not None else None
            building_detections.loc[idx, "area_m2"] = float(props[4]) if props[4] is not None else None
            building_detections.loc[idx, "ratio"] = float(props[5]) if props[5] is not None else None

        selected_features = select_best_feature(
            building_detections,
            weights=FEATURE_WEIGHTS,
            classes=TARGET_CLASSES,
            img_width=img_width,
            img_height=img_height,
            frontdoor_standards=FRONTDOOR_STANDARDS,
        ).reset_index(drop=True)

        if selected_features.empty:
            return ("skip", None)

        for idx in selected_features.index:
            row = selected_features.loc[idx]
            nearest_ground = get_closest_ground_to_feature(
                row=row,
                classification_arr=classification_arr,
                elevation_arr=elevation_arr,
                min_area=CONFIG.ffh_estimation.min_ground_area_pixels,
            )
            selected_features.loc[idx, "nearest_ground_elev"] = nearest_ground

        try:
            ffh1, ffh2, ffh3 = estimate_FFH(
                selected_features, ground_elevation_gapfill=ground_elevation_dtm, min_ffh=MIN_FFH, max_ffh=MAX_FFH
            )
        except TypeError as e:
            logger.debug(f"FFH calculation error for {building_id}: {e}")
            ffh1, ffh2, ffh3 = None, None, None

        if "Front Door" in selected_features["class"].values:
            selected_features[selected_features["class"] == "Front Door"]["bottom_elevation"].iloc[0]
            selected_features[selected_features["class"] == "Front Door"]["nearest_ground_elev"].iloc[0]
        elif not selected_features.empty:
            for feat_class in ["Stairs", "Foundation"]:
                if feat_class in selected_features["class"].values:
                    selected_features[selected_features["class"] == feat_class]["top_elevation"].iloc[0]
                    selected_features[selected_features["class"] == feat_class]["nearest_ground_elev"].iloc[0]
                    break

        for feat_class in ["Garage Door", "Stairs", "Foundation"]:
            if feat_class in selected_features["class"].values:
                selected_features[selected_features["class"] == feat_class]["bottom_elevation"].iloc[0]
                break

        result_data = {"ffh1": ffh1, "ffh2": ffh2, "ffh3": ffh3, "method": "feature_detection"}

        return ("success", result_data)

    except Exception as e:
        import traceback

        error_detail = f"{e!s} | {traceback.format_exc().splitlines()[-1]}"
        logger.error(f"Failed to process building {building_id}: {error_detail}")
        return ("fail", None)


def process_region(
    region: str,
    projection_mode: str = "standard",
    workers: int | None = None,
    skip_existing: bool = True,
    sample: int | None = None,
) -> None:
    """Process FFH estimation for all buildings in a region.

    Args:
        region: Region name to process
        workers: Number of parallel workers (default: CPU count)
        skip_existing: Skip already processed buildings
        sample: Process only first N buildings (for testing)
    """
    logger.info(f"Starting Stage 07: FFH estimation for {region}")

    best_views_df = read_table(
        "stage04b_best_views",
        region=region.lower(),
        filters={"selection_type": "best"},
        columns=["id", "building_id", "gnaf_id", "pano_id"],
    )

    if best_views_df.empty:
        logger.warning(f"No best views found for {region}. Run stage04b first.")
        return

    logger.info(f"Found {len(best_views_df)} best views")

    detections_df = read_table("stage04a_detections", region=region.lower())

    if detections_df.empty:
        logger.warning(f"No object detections found for {region}. Run stage04a first.")
        return

    logger.info(f"Found {len(detections_df)} object detections")

    ground_elevations_df = read_table(
        "stage06_ground_elevations",
        region=region.lower(),
        columns=["id", "building_id", "gnaf_id", "ground_elevation_m"],
    )

    if ground_elevations_df.empty:
        logger.warning(f"No ground elevations found for {region}. Run stage06 first.")
        return

    ground_elevations_df["gnaf_id"] = ground_elevations_df["gnaf_id"].fillna("NO_GNAF")
    ground_elevations_df["unique_id"] = (
        ground_elevations_df["id"].astype(str)
        + "_"
        + ground_elevations_df["building_id"]
        + "_"
        + ground_elevations_df["gnaf_id"]
    )

    logger.info(f"Found {len(ground_elevations_df)} ground elevations")

    proj_suffix = "_ca" if projection_mode == "classification_aware" else ""
    projection_dir = CONFIG.region_folder(region) / f"projections{proj_suffix}"
    if not projection_dir.exists():
        logger.error(
            f"No point cloud projections found for {region} in {projection_mode} mode. Run stage05 with --projection-mode {projection_mode} first."
        )
        return

    processed_set = set()
    if skip_existing:
        existing = read_table(
            STAGE_NAME, region=region.lower(), columns=["id", "building_id", "gnaf_id", "ffh1", "ffh2", "ffh3"]
        )

        if not existing.empty:
            existing = existing[existing["ffh1"].notna() | existing["ffh2"].notna() | existing["ffh3"].notna()]
        if not existing.empty:
            existing["gnaf_id"] = existing["gnaf_id"].fillna("NO_GNAF")

            processed_set = set(existing["id"].astype(str) + "_" + existing["building_id"] + "_" + existing["gnaf_id"])
            logger.info(f"Found {len(processed_set)} already processed buildings")

    if sample:
        best_views_df = best_views_df.head(sample)
        logger.info(f"Sampling first {len(best_views_df)} buildings")

    if workers is None or workers <= 0:
        from multiprocessing import cpu_count

        workers = cpu_count()

    logger.info(f"Using {workers} parallel workers")

    counters = {"ffh1": 0, "ffh2": 0, "ffh3": 0, "no_ffh": 0}

    with (
        PipelineProgress(
            f"Estimating FFH for {region}", len(best_views_df), show_elapsed=True, custom_fields=counters
        ) as prog,
        BatchWriter(STAGE_NAME, batch_size=5000, progress_tracker=prog) as writer,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        args_list = [
            (region, row.to_dict(), detections_df, ground_elevations_df, projection_mode, processed_set)
            for _, row in best_views_df.iterrows()
        ]

        futures = {pool.submit(_process_single_building, args): (i, args[1]) for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            try:
                idx, row_data = futures[future]
                status, result_data = future.result()

                if status == "success" and result_data:
                    if (
                        result_data.get("ffh1") is not None
                        or result_data.get("ffh2") is not None
                        or result_data.get("ffh3") is not None
                    ):
                        writer.add(
                            Stage07FloorHeightRecord(
                                id=str(row_data["id"]),
                                building_id=row_data["building_id"],
                                region_name=region.lower(),
                                gnaf_id=row_data.get("gnaf_id", ""),
                                **result_data,
                            )
                        )

                        if result_data.get("ffh1") is not None:
                            counters["ffh1"] += 1
                        if result_data.get("ffh2") is not None:
                            counters["ffh2"] += 1
                        if result_data.get("ffh3") is not None:
                            counters["ffh3"] += 1

                        prog.fields["ffh1"] = counters["ffh1"]
                        prog.fields["ffh2"] = counters["ffh2"]
                        prog.fields["ffh3"] = counters["ffh3"]

                        if prog.task_id is not None:
                            prog.progress.update(
                                prog.task_id,
                                ffh1=counters["ffh1"],
                                ffh2=counters["ffh2"],
                                ffh3=counters["ffh3"],
                            )

                        prog.update("suc", 1)
                    else:
                        counters["no_ffh"] += 1
                        prog.fields["no_ffh"] = counters["no_ffh"]
                        if prog.task_id is not None:
                            prog.progress.update(prog.task_id, no_ffh=counters["no_ffh"])

                elif status == "skip":
                    prog.update("skp", 1)
                else:
                    prog.update("fail", 1)

            except Exception as e:
                logger.error(f"Worker failed: {e}")
                prog.advance()
                prog.update("fail", 1)

    summary_stats = read_table(STAGE_NAME, region=region.lower())

    if not summary_stats.empty:
        logger.info(f"\nFFH estimation statistics for {region}:")
        logger.info(f"  Buildings processed: {len(summary_stats):,}")

        for ffh_col, col_name in [("FFH1", "ffh1"), ("FFH2", "ffh2"), ("FFH3", "ffh3")]:
            valid_values = summary_stats[col_name].dropna()
            if len(valid_values) > 0:
                logger.info(f"\n  {ffh_col} statistics:")
                logger.info(f"    Count: {len(valid_values)}")
                logger.info(f"    Mean: {valid_values.mean():.3f}m")
                logger.info(f"    Median: {valid_values.median():.3f}m")
                logger.info(f"    Std: {valid_values.std():.3f}m")
                logger.info(f"    Range: [{valid_values.min():.3f}, {valid_values.max():.3f}]m")

    logger.success(f"{region}: {prog.get_summary()}")


def run_stage(
    regions: list[str] | None = None,
    projection_mode: str = "standard",
    workers: int | None = None,
    skip_existing: bool = True,
    sample: int | None = None,
) -> None:
    """Run stage 07 for specified regions.

    Args:
        regions: List of regions to process (default: all from config)
        projection_mode: Type of projection ('standard', 'classification_aware')
        workers: Number of parallel workers
        skip_existing: Skip already processed buildings
        sample: Process only first N buildings per region
    """
    initialize_all_stage_tables()

    if not regions:
        regions = CONFIG.region_names

    for region in regions:
        try:
            process_region(region, projection_mode, workers, skip_existing, sample)
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            break
        except Exception as e:
            logger.error(f"Failed to process {region}: {e}")
            continue


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        region: str | None = typer.Option(None, "--region", "-r"),
        projection_mode: str = typer.Option(
            "standard", "--projection-mode", "-p", help="Projection mode: 'standard' or 'classification_aware'"
        ),
        workers: int = typer.Option(-1, "--workers", "-w"),
        skip_existing: bool = typer.Option(True, "--skip-existing"),
        sample: int | None = typer.Option(None, "--sample"),
    ):
        if region:
            process_region(region, projection_mode, workers, skip_existing, sample)
        else:
            run_stage(None, projection_mode, workers, skip_existing, sample)

    app()
