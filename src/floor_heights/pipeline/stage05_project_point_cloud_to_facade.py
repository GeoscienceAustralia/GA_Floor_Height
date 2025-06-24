#!/usr/bin/env python

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyproj
from loguru import logger
from skimage.io import imsave

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from floor_heights.config import CONFIG, REGION_CRS
from floor_heights.db.schemas import (
    BatchWriter,
    Stage05ProjectionRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import (
    get_processed_stage_ids,
    read_table,
    validate_file_exists_and_valid,
)
from floor_heights.utils.point_cloud_processings import (
    fill_small_nans,
    project_las_to_equirectangular,
    project_las_to_equirectangular_classification_aware,
    resize_preserve_nans,
)
from floor_heights.utils.progress import processing_progress

CLIP_UPPER_PROP = CONFIG.constants.clip_upper_prop
CLIP_LOWER_PROP = CONFIG.constants.clip_lower_prop
WIDTH_PANORAMA = CONFIG.projection.panorama_width
HEIGHT_PANORAMA = CONFIG.projection.panorama_height
DOWNSCALE_FACTOR = CONFIG.projection.downscale_factor
MAX_HOLE_SIZE = CONFIG.projection.max_hole_size
NODATA_FLOAT = CONFIG.projection.nodata_float
NODATA_INT = CONFIG.projection.nodata_int

STAGE_NAME = "stage05_projections"
STAGE_DESCRIPTION = "Point cloud to facade projections"


def get_lidar_path(row_id: int) -> Path | None:
    clips = read_table("stage01_clips", filters={"id": row_id}, columns=["clip_path"])

    if clips.empty:
        return None

    clip_path = clips.iloc[0]["clip_path"]
    lidar_path = CONFIG.output_root / clip_path

    return lidar_path if validate_file_exists_and_valid(lidar_path, file_type="las", min_size_bytes=100) else None


def get_clip_bounds(row_id: int, pano_id: str, edge_idx: int, view_type: str) -> dict[str, float] | None:
    clips = read_table(
        "stage03_clips",
        filters={"id": row_id, "pano_id": pano_id, "edge_idx": edge_idx, "view_type": view_type},
        columns=["clip_left", "clip_right", "clip_top", "clip_bottom"],
    )

    if clips.empty:
        return None

    return clips.iloc[0].to_dict()


def get_panorama_metadata(pano_id: str) -> dict[str, float] | None:
    df = read_table(
        "panoramas",
        filters={"imgID": pano_id},
        columns=["Latitude_deg", "Longitude_deg", "Heading_deg", "Pitch_deg", "Roll_deg", "LTP_z_m"],
    )

    if df.empty:
        return None

    row = df.iloc[0]
    return {
        "lat": row["Latitude_deg"],
        "lon": row["Longitude_deg"],
        "heading": row["Heading_deg"],
        "pitch": row["Pitch_deg"],
        "roll": row["Roll_deg"],
        "elevation": row["LTP_z_m"],
    }


def calculate_effective_heading(pano_heading: float) -> float:
    effective_heading = -pano_heading
    while effective_heading < 0:
        effective_heading += 360
    while effective_heading >= 360:
        effective_heading -= 360

    logger.debug(
        f"Panorama heading: {pano_heading:.1f}°, effective heading: {effective_heading:.1f}° "
        f"(using -heading to match stage03 coordinate system)"
    )

    return effective_heading


def get_output_paths(
    region: str,
    row_id: int,
    building_id: str,
    gnaf_id: str,
    pano_id: str,
    edge_idx: int,
    view_type: str,
    projection_mode: str = "standard",
) -> dict[str, Any]:
    base_pid = pano_id[:-4] if pano_id.lower().endswith(".jpg") else pano_id
    gnaf_id = gnaf_id if gnaf_id and pd.notna(gnaf_id) else "NO_GNAF"

    proj_suffix = "_ca" if projection_mode == "classification_aware" else ""
    output_dir = CONFIG.region_folder(region) / f"projections{proj_suffix}" / f"{row_id}_{building_id}_{gnaf_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{base_pid}_edge{edge_idx}_{view_type}"

    relative_dir = Path(region.capitalize()) / f"projections{proj_suffix}" / f"{row_id}_{building_id}_{gnaf_id}"

    return {
        "rgb": output_dir / f"{base_name}_rgb.tif",
        "elevation": output_dir / f"{base_name}_elevation_resampled.tif",
        "intensity": output_dir / f"{base_name}_intensity_resampled.tif",
        "classification": output_dir / f"{base_name}_classification_resampled.tif",
        "depth": output_dir / f"{base_name}_depth_resampled.tif",
        "relative_dir": relative_dir,
        "base_name": base_name,
    }


def _process_single_building(args_tuple) -> tuple[str, dict[str, Any] | None]:
    region, row_data, selection_mode, projection_mode = args_tuple

    row = row_data

    gnaf_id = row["gnaf_id"] if pd.notna(row.get("gnaf_id")) else "NO_GNAF"

    lidar_path = get_lidar_path(row["id"])
    if not lidar_path or not validate_file_exists_and_valid(lidar_path, file_type="las", min_size_bytes=100):
        logger.debug(f"LiDAR file not found or invalid for id {row['id']}")
        return ("skip", None)

    pano_metadata = get_panorama_metadata(row["pano_id"])
    if not pano_metadata:
        logger.debug(f"No panorama metadata found for {row['pano_id']}")
        return ("skip", None)

    clip_bounds = get_clip_bounds(row["id"], row["pano_id"], row["edge_idx"], row["view_type"])
    if not clip_bounds:
        logger.debug(f"No clipping bounds found for id {row['id']}")
        return ("skip", None)

    output_paths = get_output_paths(
        region,
        row["id"],
        row["building_id"],
        gnaf_id,
        row["pano_id"],
        row["edge_idx"],
        row["view_type"],
        projection_mode,
    )

    try:
        transformer = pyproj.Transformer.from_crs(
            CONFIG.crs.geographic, f"EPSG:{REGION_CRS[region.lower()]}", always_xy=True
        )
        x_proj, y_proj = transformer.transform(pano_metadata["lon"], pano_metadata["lat"])
        camera_pos_proj = [x_proj, y_proj, pano_metadata["elevation"]]

        effective_heading = calculate_effective_heading(pano_metadata["heading"])

        camera_angles = [effective_heading, -pano_metadata["pitch"], -pano_metadata["roll"]]

        width = int(WIDTH_PANORAMA / DOWNSCALE_FACTOR)
        height = int(HEIGHT_PANORAMA / DOWNSCALE_FACTOR)

        if projection_mode == "classification_aware":
            rgb, z, depth, classification, intensity = project_las_to_equirectangular_classification_aware(
                input_las=str(lidar_path),
                camera_pos=camera_pos_proj,
                camera_angles=camera_angles,
                width=width,
                height=height,
            )
        else:
            rgb, z, depth, classification, intensity = project_las_to_equirectangular(
                input_las=str(lidar_path),
                camera_pos=camera_pos_proj,
                camera_angles=camera_angles,
                width=width,
                height=height,
            )

        z_filled = fill_small_nans(z, max_hole_size=MAX_HOLE_SIZE, nodata_value=NODATA_FLOAT)
        intensity_filled = fill_small_nans(intensity, max_hole_size=MAX_HOLE_SIZE, nodata_value=NODATA_INT)
        depth_filled = fill_small_nans(depth, max_hole_size=MAX_HOLE_SIZE, nodata_value=NODATA_FLOAT)

        z_resampled = resize_preserve_nans(
            z_filled,
            HEIGHT_PANORAMA,
            WIDTH_PANORAMA,
            order=CONFIG.projection.elevation_resample_order,
            nodata_value=NODATA_FLOAT,
        )
        intensity_resampled = resize_preserve_nans(
            intensity_filled,
            HEIGHT_PANORAMA,
            WIDTH_PANORAMA,
            order=CONFIG.projection.intensity_resample_order,
            nodata_value=NODATA_INT,
        )
        depth_resampled = resize_preserve_nans(
            depth_filled,
            HEIGHT_PANORAMA,
            WIDTH_PANORAMA,
            order=CONFIG.projection.depth_resample_order,
            nodata_value=NODATA_FLOAT,
        )
        classification_resampled = resize_preserve_nans(
            classification,
            HEIGHT_PANORAMA,
            WIDTH_PANORAMA,
            order=CONFIG.projection.classification_resample_order,
            nodata_value=NODATA_INT,
        )

        house_loc_left = int(clip_bounds["clip_left"])
        house_loc_right = int(clip_bounds["clip_right"])
        upper_row = int(clip_bounds["clip_top"])
        lower_row = int(clip_bounds["clip_bottom"])

        z_processed = z_resampled[upper_row:lower_row, house_loc_left:house_loc_right]
        intensity_processed = intensity_resampled[upper_row:lower_row, house_loc_left:house_loc_right]
        classification_processed = classification_resampled[upper_row:lower_row, house_loc_left:house_loc_right]
        depth_processed = depth_resampled[upper_row:lower_row, house_loc_left:house_loc_right]

        min_row = round(upper_row / DOWNSCALE_FACTOR)
        max_row = round(lower_row / DOWNSCALE_FACTOR)
        min_col = round(house_loc_left / DOWNSCALE_FACTOR)
        max_col = round(house_loc_right / DOWNSCALE_FACTOR)
        rgb_clipped = rgb[min_row:max_row, min_col:max_col, :]

        imsave(str(output_paths["rgb"]), rgb_clipped)
        imsave(str(output_paths["classification"]), classification_processed.astype(np.uint8))
        imsave(str(output_paths["depth"]), depth_processed.astype(np.float32))
        imsave(str(output_paths["elevation"]), z_processed.astype(np.float32))
        imsave(str(output_paths["intensity"]), intensity_processed.astype(np.uint8))

        valid_pixels = np.sum(z_processed != NODATA_FLOAT)
        total_pixels = z_processed.size
        coverage_percent = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

        z_valid = z_processed[z_processed != NODATA_FLOAT]
        if z_valid.size > 0:
            float(np.min(z_valid))
            float(np.max(z_valid))
        else:
            pass

        depth_valid = depth_processed[depth_processed != NODATA_FLOAT]
        if depth_valid.size > 0:
            float(np.min(depth_valid))
            float(np.max(depth_valid))
        else:
            pass

        projection_data = {
            "projection_path": str(
                output_paths["relative_dir"] / f"{output_paths['base_name']}_elevation_resampled.tif"
            ),
            "point_count": int(valid_pixels),
            "coverage_percent": coverage_percent,
        }

        return ("success", projection_data)

    except Exception as e:
        logger.error(f"Failed to process id {row['id']}: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return ("fail", None)


def process_region(
    region: str,
    selection_mode: str = "best",
    projection_mode: str = "standard",
    workers: int | None = None,
    skip_existing: bool = True,
    sample: int | None = None,
) -> None:
    logger.info(f"Starting Stage 05: Point cloud projection for {region}")
    logger.info(f"Selection mode: {selection_mode}")
    logger.info(f"Projection mode: {projection_mode}")

    processed_ids = get_processed_stage_ids(STAGE_NAME, region) if skip_existing else set()
    logger.info(f"Found {len(processed_ids)} already processed buildings")

    best_views = read_table(
        "stage04b_best_views", region=region.lower(), filters={"selection_type": selection_mode, "status": "success"}
    )

    if best_views.empty:
        logger.warning(f"No {selection_mode} views found for {region}")
        return

    if processed_ids:
        best_views = best_views[~best_views["id"].isin(processed_ids)]
        logger.info(f"Processing {len(best_views)} remaining buildings")

    if sample:
        best_views = best_views.head(sample)
        logger.info(f"Sampling first {len(best_views)} buildings")

    if best_views.empty:
        logger.info("No buildings to process")
        return

    if workers is None or workers <= 0:
        from multiprocessing import cpu_count

        workers = cpu_count()

    logger.info(f"Using {workers} parallel workers")

    with (
        processing_progress(f"Projecting {region}", len(best_views)) as prog,
        BatchWriter(STAGE_NAME, batch_size=5000, progress_tracker=prog) as writer,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        args_list = [(region, row.to_dict(), selection_mode, projection_mode) for _, row in best_views.iterrows()]

        futures = {pool.submit(_process_single_building, args): (i, args[1]) for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            try:
                idx, row_data = futures[future]
                status, projection_data = future.result()

                if status == "success" and projection_data:
                    writer.add(
                        Stage05ProjectionRecord(
                            id=row_data["id"],
                            building_id=row_data["building_id"],
                            region_name=region.lower(),
                            gnaf_id=row_data.get("gnaf_id", ""),
                            pano_id=row_data["pano_id"],
                            edge_idx=row_data["edge_idx"],
                            view_type=row_data["view_type"],
                            projection_path=projection_data["projection_path"],
                            point_count=projection_data["point_count"],
                            coverage_percent=projection_data["coverage_percent"],
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

    logger.success(f"{region}: {prog.get_summary()}")


def run_stage(
    regions: list[str] | None = None,
    selection_mode: str = "best",
    projection_mode: str = "standard",
    workers: int | None = None,
    skip_existing: bool = True,
    sample: int | None = None,
) -> None:
    """Run stage05 with the given parameters.

    Args:
        regions: List of regions to process (None for all)
        selection_mode: View selection mode
        projection_mode: Projection mode ('standard' or 'classification_aware')
        workers: Number of workers (-1 for CPU count)
        skip_existing: Skip already processed buildings
        sample: Process only first N buildings
    """
    initialize_all_stage_tables()

    if not regions:
        regions = CONFIG.region_names

    for region in regions:
        try:
            process_region(region, selection_mode, projection_mode, workers, skip_existing, sample)
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
        selection_type: str = typer.Option("best", "--selection-type", "-s"),
        projection_mode: str = typer.Option(
            "standard", "--projection-mode", "-p", help="Projection mode: 'standard' or 'classification_aware'"
        ),
        workers: int = typer.Option(-1, "--workers", "-w"),
        skip_existing: bool = typer.Option(True, "--skip-existing"),
        sample: int | None = typer.Option(None, "--sample"),
    ):
        if region:
            process_region(region, selection_type, projection_mode, workers, skip_existing, sample)
        else:
            run_stage(None, selection_type, projection_mode, workers, skip_existing, sample)

    app()
