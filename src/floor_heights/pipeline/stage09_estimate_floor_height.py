"""Stage-09: Floor height estimation using detected features and LiDAR data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlalchemy import (
    Index,
    MetaData,
    Table,
    create_engine,
    inspect,
    select,
)

from floor_heights.utils.point_cloud_processings import (
    compute_feature_properties,
    estimate_FFH,
    select_best_feature,
)

DB = os.getenv("DB_CONNECTION_STRING")
if not DB:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")

cfg_path = Path(__file__).resolve().parents[3] / "config" / "common.yaml"
common_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
OUTPUT_ROOT = Path(common_cfg.get("output_root", "output"))
REGIONS = common_cfg.get("regions", [])

engine = create_engine(DB, future=True, pool_pre_ping=True)
meta = MetaData()

FRONTDOOR_STANDARDS = {
    "width_m": 0.82,
    "height_m": 2.04,
    "area_m2": 1.67,
    "ratio": 0.40,
}

WEIGHTS = {"area_m2": 1, "ratio": 1, "confidence": 1, "x_location": 1, "y_location": 1}

CLASSES = ["Foundation", "Front Door", "Garage Door", "Stairs", "Window"]


def fetch_building_data(region_name: str) -> pd.DataFrame:
    regions_table = Table("regions", meta, autoload_with=engine)
    buildings_table = Table("building_points_processed", meta, autoload_with=engine)
    detections_table = Table("object_detections", meta, autoload_with=engine)
    elevations_table = Table("ground_elevation_stats", meta, autoload_with=engine)

    with engine.connect() as conn:
        region_id = conn.execute(
            select(regions_table.c.id).where(regions_table.c.name == region_name)
        ).scalar_one()

        stmt = (
            select(
                buildings_table.c.footprint_id.label("building_id"),
                elevations_table.c.lidar_elev_25pct,
                elevations_table.c.lidar_elev_min,
                elevations_table.c.lidar_elev_mean,
            )
            .select_from(
                buildings_table.join(
                    elevations_table,
                    buildings_table.c.footprint_id == elevations_table.c.building_id,
                ).join(
                    detections_table,
                    buildings_table.c.footprint_id == detections_table.c.building_id,
                )
            )
            .where(buildings_table.c.region_id == region_id)
            .distinct()
        )

        return pd.read_sql(stmt, conn)


def fetch_detections_for_building(building_id: int) -> pd.DataFrame:
    detections_table = Table("object_detections", meta, autoload_with=engine)

    stmt = select(
        detections_table.c.class_name.label("class"),
        detections_table.c.confidence,
        detections_table.c.bbox_x1,
        detections_table.c.bbox_y1,
        detections_table.c.bbox_x2,
        detections_table.c.bbox_y2,
        detections_table.c.view_metadata,
    ).where(detections_table.c.building_id == building_id)

    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def load_raster_data(building_id: int, region_name: str) -> Optional[Tuple]:
    """TODO: Implement raster loading from Stage 06 output.
    This requires:
    1. Locate elevation, depth, and classification rasters for building from Stage 06
    2. Load raster arrays using PIL or rasterio
    3. Return arrays for feature property computation
    """
    return None


def compute_ground_elevation(building_data: pd.Series) -> Optional[float]:
    elevation_fields = ["lidar_elev_25pct", "lidar_elev_min", "lidar_elev_mean"]

    return next(
        (
            float(building_data[field])
            for field in elevation_fields
            if pd.notna(building_data.get(field))
        ),
        None,
    )


def get_best_feature_type(selected_features: pd.DataFrame) -> Optional[str]:
    if selected_features.empty:
        return None

    floor_priority = ["Front Door", "Stairs", "Foundation"]
    return next(
        (
            feature_class
            for feature_class in floor_priority
            if feature_class in selected_features["class"].values
        ),
        None,
    )


def select_final_ffh(
    ffh1: Optional[float], ffh2: Optional[float], ffh3: Optional[float]
) -> Tuple[Optional[float], str]:
    ffh_methods = [
        (ffh1, "feature_to_feature"),
        (ffh2, "feature_to_nearest_ground"),
        (ffh3, "feature_to_dtm"),
    ]

    for ffh_value, method_name in ffh_methods:
        if ffh_value is not None:
            return ffh_value, method_name

    return None, "none"


def process_building_ffh(
    args: Tuple[int, pd.Series, str],
) -> Tuple[int, Optional[Dict]]:
    building_id, building_data, region_name = args

    detections_df = fetch_detections_for_building(building_id)
    if detections_df.empty:
        return building_id, None

    ground_elevation = compute_ground_elevation(building_data)
    if ground_elevation is None:
        return building_id, None

    raster_data = load_raster_data(building_id, region_name)

    if raster_data is not None:
        elevation_arr, depth_arr, classification_arr = raster_data

        for idx in range(len(detections_df)):
            props = compute_feature_properties(
                row=detections_df.iloc[idx],
                elevation_arr=elevation_arr,
                depth_arr=depth_arr,
                gapfill_depth=9999,
            )

            if props and len(props) == 6:
                top_elev, bottom_elev, width, height, area, ratio = props
                detections_df.at[idx, "top_elevation"] = float(top_elev)
                detections_df.at[idx, "bottom_elevation"] = float(bottom_elev)
                detections_df.at[idx, "width_m"] = float(width)
                detections_df.at[idx, "height_m"] = float(height)
                detections_df.at[idx, "area_m2"] = float(area)
                detections_df.at[idx, "ratio"] = float(ratio)

    img_width, img_height = 2048, 1024

    selected_features = select_best_feature(
        detections_df,
        weights=WEIGHTS,
        classes=CLASSES,
        img_width=img_width,
        img_height=img_height,
        frontdoor_standards=FRONTDOOR_STANDARDS,
    )

    if selected_features.empty:
        return building_id, None

    ffh1, ffh2, ffh3 = estimate_FFH(
        selected_features, ground_elevation, min_ffh=0.0, max_ffh=1.5
    )

    final_ffh, estimation_method = select_final_ffh(ffh1, ffh2, ffh3)

    if final_ffh is None:
        return building_id, None

    best_feature_type = get_best_feature_type(selected_features)

    return building_id, {
        "building_id": building_id,
        "estimated_ffh": float(final_ffh),
        "confidence_score": float(selected_features["confidence"].max()),
        "estimation_method": estimation_method,
        "ground_elevation": float(ground_elevation),
        "best_feature_type": best_feature_type,
        "estimation_metadata": json.dumps(
            {
                "ffh1": ffh1,
                "ffh2": ffh2,
                "ffh3": ffh3,
                "selected_features_count": len(selected_features),
                "total_detections": len(detections_df),
            }
        ),
    }


def clear_existing_estimates(table_name: str, region_id: int) -> None:
    if not inspect(engine).has_table(table_name):
        return

    with engine.begin() as conn:
        estimates_table = Table(table_name, MetaData(), autoload_with=engine)
        conn.execute(
            estimates_table.delete().where(estimates_table.c.region_id == region_id)
        )


def create_estimate_indexes(table_name: str) -> None:
    estimates_table = Table(table_name, MetaData(), autoload_with=engine)
    Index(f"{table_name}_building_id_idx", estimates_table.c.building_id).create(
        bind=engine, checkfirst=True
    )
    Index(f"{table_name}_region_id_idx", estimates_table.c.region_id).create(
        bind=engine, checkfirst=True
    )


def get_region_id(region_name: str) -> int:
    regions_table = Table("regions", meta, autoload_with=engine)
    with engine.connect() as conn:
        return int(
            conn.execute(
                select(regions_table.c.id).where(regions_table.c.name == region_name)
            ).scalar_one()
        )


def write_ffh_estimates(results: List[Dict], region_name: str) -> None:
    if not results:
        return

    table_name = "floor_height_estimates"
    table_exists = inspect(engine).has_table(table_name)
    region_id = get_region_id(region_name)

    clear_existing_estimates(table_name, region_id)

    df = pd.DataFrame(results).assign(
        region_id=region_id, created_at=pd.Timestamp.now(tz="UTC")
    )

    df.to_sql(
        table_name,
        con=engine,
        if_exists="replace" if not table_exists else "append",
        index=False,
    )

    create_estimate_indexes(table_name)


def execute_ffh_estimation(
    args_list: List[Tuple], workers: int, region_name: str
) -> Tuple[List[Dict], int, int]:
    results = []
    success_count = fail_count = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(
            f"Estimating {region_name} (S: 0, F: 0)", total=len(args_list)
        )

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_building_ffh, args) for args in args_list
            ]

            for future in futures:
                building_id, estimate = future.result()
                progress.update(task_id, advance=1)

                if estimate:
                    results.append(estimate)
                    success_count += 1
                else:
                    fail_count += 1

                progress.update(
                    task_id,
                    description=f"Estimating {region_name} (S: {success_count}, F: {fail_count})",
                )

    return results, success_count, fail_count


def process_region(region_name: str, workers: int) -> None:
    logger.info(f"Processing FFH estimation for region: {region_name}")

    buildings_df = fetch_building_data(region_name)
    if buildings_df.empty:
        logger.warning(
            f"No buildings with detection and elevation data found for {region_name}"
        )
        return

    args_list = [
        (row["building_id"], row, region_name) for _, row in buildings_df.iterrows()
    ]

    results, success_count, fail_count = execute_ffh_estimation(
        args_list, workers, region_name
    )

    write_ffh_estimates(results, region_name)
    logger.success(f"{region_name}: {success_count} FFH estimates, {fail_count} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate floor heights using detected features and LiDAR data"
    )
    parser.add_argument("--region", choices=REGIONS, help="Process single region")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count(), 10),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    regions = [args.region] if args.region else REGIONS
    for region in regions:
        process_region(region, args.workers)


if __name__ == "__main__":
    main()
