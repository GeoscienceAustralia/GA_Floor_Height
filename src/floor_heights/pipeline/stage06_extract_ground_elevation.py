#!/usr/bin/env python

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pdal
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from floor_heights.config import CONFIG
from floor_heights.db.schemas import (
    BatchWriter,
    Stage06GroundElevationRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import (
    get_processed_stage_ids,
    read_table,
    validate_file_exists_and_valid,
)
from floor_heights.utils.progress import PipelineProgress

DTM_RESOLUTION = CONFIG.ground_elevation.dtm_resolution
GENERATE_DTM = CONFIG.ground_elevation.generate_dtm

STAGE_NAME = "stage06_ground_elevations"
STAGE_DESCRIPTION = "Ground elevation extraction from LiDAR"


def get_lidar_path(row_id: int) -> Path | None:
    clips = read_table("stage01_clips", filters={"id": row_id}, columns=["clip_path"])

    if clips.empty:
        return None

    clip_path = clips.iloc[0]["clip_path"]
    lidar_path = CONFIG.output_root / clip_path

    return lidar_path if validate_file_exists_and_valid(lidar_path, file_type="las", min_size_bytes=100) else None


def get_crs_from_las(las_file_path: Path) -> str:
    pipeline_info = {"pipeline": [{"type": "readers.las", "filename": str(las_file_path)}]}
    info_pipeline = pdal.Pipeline(json.dumps(pipeline_info))
    info_pipeline.execute()
    metadata = info_pipeline.metadata
    return metadata["metadata"]["readers.las"]["srs"]["horizontal"]


def process_extract_ground_elevations(
    las_file_path: Path, resolution: float, crs: str, output_tiff: Path | None = None
) -> tuple[dict[str, Any], Path | None]:
    pipeline_steps = [
        {"type": "readers.las", "filename": str(las_file_path)},
        {
            "type": "filters.range",
            "limits": f"Classification[{CONFIG.ground_elevation.ground_classification}:{CONFIG.ground_elevation.ground_classification}]",
        },
        {"type": "filters.outlier"},
        {
            "type": "filters.csf",
            "ignore": f"Classification[{CONFIG.ground_elevation.noise_classification}:{CONFIG.ground_elevation.noise_classification}]",
            "resolution": CONFIG.ground_elevation.csf_resolution,
            "hdiff": CONFIG.ground_elevation.csf_hdiff,
            "smooth": CONFIG.ground_elevation.csf_smooth,
        },
        {
            "type": "filters.range",
            "limits": f"Classification[{CONFIG.ground_elevation.ground_classification}:{CONFIG.ground_elevation.ground_classification}]",
        },
    ]

    pipeline_json = {"pipeline": pipeline_steps}
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()

    if len(pipeline.arrays) == 0 or len(pipeline.arrays[0]) == 0:
        return {
            "lidar_elev_mean": None,
            "lidar_elev_med": None,
            "lidar_elev_min": None,
            "lidar_elev_max": None,
            "lidar_elev_std": None,
            "lidar_elev_25pct": None,
            "lidar_elev_75pct": None,
            "ground_point_count": 0,
        }, None

    ground_points = pipeline.arrays[0]
    elevations = ground_points["Z"]

    actual_dtm_path = None

    if output_tiff is not None and len(elevations) >= CONFIG.ground_elevation.min_points_for_dtm:
        x_coords = ground_points["X"]
        y_coords = ground_points["Y"]
        x_extent = np.max(x_coords) - np.min(x_coords)
        y_extent = np.max(y_coords) - np.min(y_coords)

        if (
            x_extent >= CONFIG.ground_elevation.min_extent_for_dtm
            and y_extent >= CONFIG.ground_elevation.min_extent_for_dtm
        ):
            try:
                pipeline_steps.append(
                    {
                        "type": "writers.gdal",
                        "filename": str(output_tiff),
                        "dimension": "Z",
                        "output_type": "idw",
                        "resolution": resolution,
                        "gdaldriver": "GTiff",
                        "data_type": "float32",
                        "nodata": CONFIG.ground_elevation.dtm_nodata,
                        "override_srs": crs,
                    }
                )
                pipeline_json = {"pipeline": pipeline_steps}
                pipeline = pdal.Pipeline(json.dumps(pipeline_json))
                pipeline.execute()
                actual_dtm_path = output_tiff
            except Exception:
                pass

    if len(elevations) == 0:
        return {
            "lidar_elev_mean": None,
            "lidar_elev_med": None,
            "lidar_elev_min": None,
            "lidar_elev_max": None,
            "lidar_elev_std": None,
            "lidar_elev_25pct": None,
            "lidar_elev_75pct": None,
            "ground_point_count": 0,
        }, actual_dtm_path

    stats = {
        "lidar_elev_mean": float(np.mean(elevations)),
        "lidar_elev_med": float(np.median(elevations)),
        "lidar_elev_min": float(np.min(elevations)),
        "lidar_elev_max": float(np.max(elevations)),
        "lidar_elev_std": float(np.std(elevations)),
        "lidar_elev_25pct": float(np.percentile(elevations, 25)),
        "lidar_elev_75pct": float(np.percentile(elevations, 75)),
        "ground_point_count": len(elevations),
    }

    return stats, actual_dtm_path


def get_dtm_output_path(region: str, row_id: int, building_id: str, gnaf_id: str) -> Path:
    gnaf_id = gnaf_id if gnaf_id and pd.notna(gnaf_id) else "NO_GNAF"
    dtm_dir = CONFIG.region_folder(region) / "lidar" / "dtm"
    dtm_dir.mkdir(parents=True, exist_ok=True)

    dtm_filename = f"{row_id}_{building_id}_{gnaf_id}_DTM.tif"
    return dtm_dir / dtm_filename


def _process_single_building(args_tuple) -> tuple[str, dict[str, Any] | None]:
    region, row_data, generate_dtm, dtm_resolution, clip_path = args_tuple

    row = row_data

    gnaf_id = row["gnaf_id"] if pd.notna(row.get("gnaf_id")) else "NO_GNAF"

    if not clip_path:
        logger.debug(f"No clip path for id {row['id']}")
        return ("fail", None)

    lidar_path = CONFIG.output_root / clip_path

    if not validate_file_exists_and_valid(lidar_path, file_type="las", min_size_bytes=100):
        logger.debug(f"LiDAR file not found or invalid for id {row['id']}: {lidar_path}")
        return ("fail", None)

    try:
        crs = get_crs_from_las(lidar_path)

        dtm_path = None
        if generate_dtm:
            dtm_path = get_dtm_output_path(region, row["id"], row["building_id"], gnaf_id)

        stats, actual_dtm_path = process_extract_ground_elevations(
            las_file_path=lidar_path, resolution=dtm_resolution, crs=crs, output_tiff=dtm_path
        )

        if dtm_path and not actual_dtm_path:
            pass

        if stats["ground_point_count"] == 0:
            return ("fail", None)

        result_data = {
            "ground_elevation_m": stats["lidar_elev_med"],
            "ground_points_count": stats["ground_point_count"],
            "confidence_score": min(1.0, stats["ground_point_count"] / 100.0),
            "method": "lidar_ground_points",
        }

        return ("success", result_data)

    except Exception as e:
        logger.error(f"Failed to process id {row['id']}: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return ("fail", None)


def process_region(
    region: str,
    workers: int | None = None,
    skip_existing: bool = True,
    generate_dtm: bool = GENERATE_DTM,
    dtm_resolution: float = DTM_RESOLUTION,
    sample: int | None = None,
) -> None:
    logger.info(f"Starting Stage 06: Ground elevation extraction for {region}")
    logger.info(f"Generate DTM: {generate_dtm}")
    if generate_dtm:
        logger.info(f"DTM resolution: {dtm_resolution}m")

    processed_ids = get_processed_stage_ids(STAGE_NAME, region) if skip_existing else set()
    logger.info(f"Found {len(processed_ids)} already processed buildings")

    clipped = read_table("stage01_clips", region=region.lower(), columns=["id", "building_id", "gnaf_id"])

    if clipped.empty:
        logger.warning(f"No clipped buildings found for {region}")
        return

    if processed_ids:
        clipped = clipped[~clipped["id"].isin(processed_ids)]
        logger.info(f"Processing {len(clipped)} remaining buildings")

    if sample:
        clipped = clipped.head(sample)
        logger.info(f"Sampling first {len(clipped)} buildings")

    if clipped.empty:
        logger.info("No buildings to process")
        return

    if workers is None or workers <= 0:
        from multiprocessing import cpu_count

        workers = cpu_count()

    logger.info(f"Using {workers} parallel workers")

    clips_data = read_table("stage01_clips", region=region.lower(), columns=["id", "clip_path"])
    clip_path_map = {row["id"]: row["clip_path"] for _, row in clips_data.iterrows()}

    with (
        PipelineProgress(
            f"Extracting ground elevations for {region}",
            len(clipped),
            show_elapsed=True,
        ) as prog,
        BatchWriter(STAGE_NAME, batch_size=5000, progress_tracker=prog) as writer,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        args_list = [
            (
                region,
                {"id": row["id"], "building_id": row["building_id"], "gnaf_id": row.get("gnaf_id", "NO_GNAF")},
                generate_dtm,
                dtm_resolution,
                clip_path_map.get(row["id"]),
            )
            for _, row in clipped.iterrows()
        ]

        futures = {pool.submit(_process_single_building, args): (i, args[1]) for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            try:
                idx, row_data = futures[future]
                status, result_data = future.result()

                if status == "success" and result_data:
                    writer.add(
                        Stage06GroundElevationRecord(
                            id=str(row_data["id"]),
                            building_id=row_data["building_id"],
                            region_name=region.lower(),
                            gnaf_id=row_data.get("gnaf_id", ""),
                            **result_data,
                        )
                    )
                    prog.update("suc", 1)
                elif status == "skip":
                    prog.update("skp", 1)
                else:
                    prog.update("fail", 1)

            except Exception as e:
                logger.error(f"Worker failed: {e}")
                prog.update("fail", 1)

    summary_stats = read_table(STAGE_NAME, region=region.lower(), columns=["ground_elevation_m", "ground_points_count"])

    if not summary_stats.empty:
        elevations = summary_stats["ground_elevation_m"].dropna()
        if len(elevations) > 0:
            logger.info(f"\nGround elevation statistics for {region}:")
            logger.info(f"  Buildings processed: {len(summary_stats):,}")
            logger.info(f"  Mean elevation: {elevations.mean():.3f}m")
            logger.info(f"  Elevation range: {elevations.min():.3f}m - {elevations.max():.3f}m")
            logger.info(f"  Total ground points: {summary_stats['ground_points_count'].sum():,}")

    logger.success(f"{region}: {prog.get_summary()}")


def run_stage(
    regions: list[str] | None = None,
    workers: int | None = None,
    skip_existing: bool = True,
    generate_dtm: bool = GENERATE_DTM,
    dtm_resolution: float = DTM_RESOLUTION,
    sample: int | None = None,
) -> None:
    """Run stage06 with the given parameters.

    Args:
        regions: List of regions to process (None for all)
        workers: Number of workers (-1 for CPU count)
        skip_existing: Skip already processed buildings
        generate_dtm: Generate DTM output files
        dtm_resolution: Resolution for DTM generation
        sample: Process only first N buildings
    """
    initialize_all_stage_tables()

    if not regions:
        regions = CONFIG.region_names

    for region in regions:
        try:
            process_region(region, workers, skip_existing, generate_dtm, dtm_resolution, sample)
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
        workers: int = typer.Option(-1, "--workers", "-w"),
        skip_existing: bool = typer.Option(True, "--skip-existing"),
        generate_dtm: bool = typer.Option(True, "--generate-dtm"),
        dtm_resolution: float = typer.Option(DTM_RESOLUTION, "--dtm-resolution"),
        sample: int | None = typer.Option(None, "--sample"),
    ):
        if region:
            process_region(region, workers, skip_existing, generate_dtm, dtm_resolution, sample)
        else:
            run_stage(None, workers, skip_existing, generate_dtm, dtm_resolution, sample)

    app()
