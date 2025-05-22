#!/usr/bin/env python
"""Stage-06: Project point clouds to facade rasters."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TextColumn,
)
from skimage.io import imsave
from sqlalchemy import (
    Index,
    MetaData,
    Table,
    create_engine,
    select,
    inspect,
)
from floor_heights.utils.point_cloud_processings import (
    project_las_to_equirectangular,
    fill_small_nans,
    resize_preserve_nans,
)

DB = os.getenv("DB_CONNECTION_STRING")
if not DB:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")

cfg_path = Path(__file__).resolve().parents[3] / "config" / "common.yaml"
common_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
OUTPUT_ROOT = Path(common_cfg.get("output_root", "output"))
REGIONS = common_cfg.get("regions", [])

UPPER_CROP = 0.25
LOWER_CROP = 0.6
WIDTH_PANORAMA = 11000
HEIGHT_PANORAMA = 5500
DOWNSCALE_FACTOR = 8

WORKERS = max(1, (cpu_count() or 1))

engine = create_engine(DB, future=True, pool_pre_ping=True)
meta = MetaData()

tbl_regions = Table("regions", meta, autoload_with=engine)
tbl_clipped_views = Table("panorama_clipped_views", meta, autoload_with=engine)
tbl_lidar_clouds = Table("lidar_clipped_clouds", meta, autoload_with=engine)
tbl_panoramas = Table("panoramas", meta, autoload_with=engine)


def rfolder(region: str) -> Path:
    """Get the region's output folder."""
    return OUTPUT_ROOT / region.capitalize()


def get_file_paths(region: str, building_id: int) -> dict:
    """Generate all the necessary file paths for a building."""
    lidar_dir = rfolder(region) / "lidar" / "clipped"
    lidar_file = lidar_dir / f"{building_id}.copc.laz"

    output_dir = rfolder(region) / "lidar" / "projected"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "lidar_file": lidar_file,
        "output_dir": output_dir,
        "rgb_out": output_dir / f"{building_id}_rgb.tif",
        "elevation_out": output_dir / f"{building_id}_elevation_resampled.tif",
        "intensity_out": output_dir / f"{building_id}_intensity_resampled.tif",
        "classification_out": output_dir
        / f"{building_id}_classification_resampled.tif",
        "depth_out": output_dir / f"{building_id}_depth_resampled.tif",
    }

    return paths


def check_requirements(row: pd.Series, file_paths: dict) -> Optional[str]:
    """Check if all requirements are met to process this building."""
    if not file_paths["lidar_file"].exists():
        return "lidar_missing"

    all_exist = all(
        file_paths[key].exists()
        for key in [
            "rgb_out",
            "elevation_out",
            "intensity_out",
            "classification_out",
            "depth_out",
        ]
    )
    if all_exist:
        return "already_exists"

    return None


def process_building(
    row: pd.Series, file_paths: dict, transformer: Any, region: str
) -> Dict:
    """Process a single building's LiDAR data to create facade rasters."""
    try:
        lat, lon, elev = row.latitude_deg, row.longitude_deg, row.ltp_z_m
        x_proj, y_proj = transformer.transform(lon, lat)  # lon, lat order
        camera_pos_proj = [x_proj, y_proj, elev]
        camera_angles = [row.heading_deg, -row.pitch_deg, -row.roll_deg]

        width = int(WIDTH_PANORAMA / DOWNSCALE_FACTOR)
        height = int(HEIGHT_PANORAMA / DOWNSCALE_FACTOR)

        las_file_path = str(file_paths["lidar_file"])
        rgb, z, depth, classification, intensity = project_las_to_equirectangular(
            input_las=las_file_path,
            camera_pos=camera_pos_proj,
            camera_angles=camera_angles,
            width=width,
            height=height,
        )

        z_arr_filled = fill_small_nans(z, max_hole_size=10, nodata_value=9999)
        intensity_filled = fill_small_nans(
            intensity, max_hole_size=10, nodata_value=255
        )
        depth_filled = fill_small_nans(depth, max_hole_size=10, nodata_value=9999)

        z_filled_resampled = resize_preserve_nans(
            z_arr_filled, HEIGHT_PANORAMA, WIDTH_PANORAMA, order=1, nodata_value=9999
        )
        intensity_filled_resampled = resize_preserve_nans(
            intensity_filled, HEIGHT_PANORAMA, WIDTH_PANORAMA, order=1, nodata_value=255
        )
        depth_filled_resampled = resize_preserve_nans(
            depth_filled, HEIGHT_PANORAMA, WIDTH_PANORAMA, order=1, nodata_value=9999
        )
        classification_resampled = resize_preserve_nans(
            classification, HEIGHT_PANORAMA, WIDTH_PANORAMA, order=0, nodata_value=255
        )

        house_loc_left, house_loc_right = int(row.clip_left), int(row.clip_right)

        z_processed = z_filled_resampled[
            int(round(UPPER_CROP * HEIGHT_PANORAMA)) : int(
                round(LOWER_CROP * HEIGHT_PANORAMA)
            ),
            house_loc_left:house_loc_right,
        ]
        intensity_processed = intensity_filled_resampled[
            int(round(UPPER_CROP * HEIGHT_PANORAMA)) : int(
                round(LOWER_CROP * HEIGHT_PANORAMA)
            ),
            house_loc_left:house_loc_right,
        ]
        classification_processed = classification_resampled[
            int(round(UPPER_CROP * HEIGHT_PANORAMA)) : int(
                round(LOWER_CROP * HEIGHT_PANORAMA)
            ),
            house_loc_left:house_loc_right,
        ]
        depth_processed = depth_filled_resampled[
            int(round(UPPER_CROP * HEIGHT_PANORAMA)) : int(
                round(LOWER_CROP * HEIGHT_PANORAMA)
            ),
            house_loc_left:house_loc_right,
        ]

        min_row, max_row = (
            int(round(UPPER_CROP * height)),
            int(round(LOWER_CROP * height)),
        )
        min_col, max_col = (
            int(round(house_loc_left / DOWNSCALE_FACTOR)),
            int(round(house_loc_right / DOWNSCALE_FACTOR)),
        )
        rgb_arr_clipped = rgb[min_row:max_row, min_col:max_col, :]

        imsave(file_paths["rgb_out"], rgb_arr_clipped)
        imsave(file_paths["classification_out"], classification_processed)
        imsave(file_paths["depth_out"], depth_processed)
        imsave(file_paths["elevation_out"], z_processed)
        imsave(file_paths["intensity_out"], intensity_processed)

        return {
            "status": "success",
            "building_id": row.building_id,
            "pano_id": row.pano_id,
            "region": region,
        }

    except Exception as e:
        logger.error(f"Error processing building {row.building_id}: {e}")
        return {"status": "error", "error": str(e), "building_id": row.building_id}


def get_region_building_data(region: str) -> pd.DataFrame:
    """Fetch data for buildings with panorama views and LiDAR data."""
    stmt = select(
        tbl_clipped_views.c.building_id,
        tbl_clipped_views.c.pano_id,
        tbl_clipped_views.c.clip_left,
        tbl_clipped_views.c.clip_right,
        tbl_panoramas.c.latitude_deg,
        tbl_panoramas.c.longitude_deg,
        tbl_panoramas.c.ltp_z_m,
        tbl_panoramas.c.heading_deg,
        tbl_panoramas.c.pitch_deg,
        tbl_panoramas.c.roll_deg,
    ).where(
        tbl_clipped_views.c.region == region,
        tbl_clipped_views.c.pano_id == tbl_panoramas.c.imgid,
    )

    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    return df


def process_region(region: str) -> None:
    """Process all buildings in a region."""
    logger.info(f"Processing region: {region}")

    with engine.connect() as conn:
        region_id, epsg = conn.execute(
            select(tbl_regions.c.id, tbl_regions.c.target_epsg).where(
                tbl_regions.c.name == region
            )
        ).one()

    import pyproj

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True,
    )

    df = get_region_building_data(region)
    if df.empty:
        logger.warning(f"No buildings with panorama views found for region {region}")
        return

    logger.info(f"Found {len(df)} building/panorama pairs for processing in {region}")

    counters = {"success": 0, "already_exists": 0, "lidar_missing": 0, "error": 0}
    processed_records = []

    with (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(compact=True),
            TextColumn("✓{task.fields[suc]}"),
            TextColumn("➟{task.fields[skp]}"),
            TextColumn("⊘{task.fields[miss]}"),
            TextColumn("✗{task.fields[err]}"),
            transient=True,
        ) as progress,
        ThreadPoolExecutor(max_workers=WORKERS) as executor,
    ):
        task_id = progress.add_task(
            f"Projecting {region}", total=len(df), suc=0, skp=0, miss=0, err=0
        )

        futures = {}
        for _, row in df.iterrows():
            file_paths = get_file_paths(region, row.building_id)

            skip_reason = check_requirements(row, file_paths)
            if skip_reason == "already_exists":
                counters["already_exists"] += 1
                progress.update(task_id, advance=1, skp=counters["already_exists"])
                continue
            elif skip_reason == "lidar_missing":
                counters["lidar_missing"] += 1
                progress.update(task_id, advance=1, miss=counters["lidar_missing"])
                continue

            future = executor.submit(
                process_building, row, file_paths, transformer, region
            )
            futures[future] = row.building_id

        for future in as_completed(futures):
            result = future.result()
            status = result["status"]

            if status == "success":
                counters["success"] += 1
                processed_records.append(result)
            else:
                counters["error"] += 1

            progress.update(
                task_id,
                advance=1,
                suc=counters["success"],
                skp=counters["already_exists"],
                miss=counters["lidar_missing"],
                err=counters["error"],
            )

    write_results_to_db(processed_records, region)

    logger.success(
        f"{region}: success={counters['success']}  skip={counters['already_exists']}  "
        f"lidar_missing={counters['lidar_missing']}  errors={counters['error']}"
    )


def write_results_to_db(processed_records: List[Dict], region: str) -> None:
    """Write processing results to database."""
    if not processed_records:
        return

    df = pd.DataFrame(processed_records)

    with engine.begin() as conn:
        df.to_sql("stage06_projected_rasters", conn, if_exists="append", index=False)

    tbl = Table(
        "stage06_projected_rasters",
        MetaData(),
        autoload_with=engine,
        extend_existing=True,
    )

    Index("stage06_projected_rasters_building_id_idx", tbl.c.building_id).create(
        bind=engine, checkfirst=True
    )
    Index("stage06_projected_rasters_region_idx", tbl.c.region).create(
        bind=engine, checkfirst=True
    )

    logger.info(
        f"{region}: wrote {len(processed_records):,} results → stage06_projected_rasters"
    )


def main() -> None:
    """Main function to process LiDAR point clouds to facade rasters."""
    parser = argparse.ArgumentParser(
        description="Project point clouds to facade rasters"
    )
    parser.add_argument(
        "--region",
        choices=REGIONS,
        help="Process only this region (or all regions if not specified)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["trace", "debug", "info", "warning", "error", "critical"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    regions_to_process = [args.region] if args.region else REGIONS

    try:
        for region in regions_to_process:
            with engine.begin() as conn:
                inspector = inspect(engine)
                if inspector.has_table("stage06_projected_rasters"):
                    tbl = Table(
                        "stage06_projected_rasters", MetaData(), autoload_with=engine
                    )
                    conn.execute(tbl.delete().where(tbl.c.region == region))
    except Exception as e:
        logger.warning(f"Could not clear existing records: {e}")

    for region in regions_to_process:
        try:
            process_region(region)
        except Exception as e:
            logger.error(f"Error processing region {region}: {e}")
            logger.exception("Traceback:")

    logger.info("Stage-06 complete")


if __name__ == "__main__":
    main()
