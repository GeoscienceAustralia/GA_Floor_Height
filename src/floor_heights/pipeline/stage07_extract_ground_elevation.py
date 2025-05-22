"""Stage-07: Ground elevation extraction from clipped LiDAR using PDAL pipelines."""

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
import pdal
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
    process_extract_ground_elevations,
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


def fetch_buildings_with_las(region_name: str) -> List[Tuple[int, int, Path]]:
    regions_table = Table("regions", meta, autoload_with=engine)
    buildings_table = Table("building_points_processed", meta, autoload_with=engine)

    with engine.connect() as conn:
        region_id = conn.execute(
            select(regions_table.c.id).where(regions_table.c.name == region_name)
        ).scalar_one()

        buildings = conn.execute(
            select(buildings_table.c.footprint_id, buildings_table.c.region_id).where(
                buildings_table.c.region_id == region_id
            )
        ).fetchall()

    las_dir = OUTPUT_ROOT / region_name / "lidar" / "clipped"
    return [
        (building_id, region_id, las_file)
        for building_id, region_id in buildings
        if (las_file := las_dir / f"building_{building_id}.las").exists()
    ]


def extract_crs_from_las(las_file: Path) -> str:
    pipeline_info = {"pipeline": [{"type": "readers.las", "filename": str(las_file)}]}
    info_pipeline = pdal.Pipeline(json.dumps(pipeline_info))
    info_pipeline.execute()
    return str(info_pipeline.metadata["metadata"]["readers.las"]["srs"]["horizontal"])


def process_building_elevation(
    args: Tuple[int, int, Path, Path, float],
) -> Tuple[int, Optional[Dict]]:
    building_id, region_id, las_file, dtm_dir, resolution = args
    dtm_file = dtm_dir / f"building_{building_id}_dtm.tif"

    crs_str = extract_crs_from_las(las_file)
    stats = process_extract_ground_elevations(
        las_file_path=str(las_file),
        resolution=resolution,
        crs=crs_str,
        output_tiff=str(dtm_file),
    )

    return building_id, {
        "building_id": building_id,
        "region_id": region_id,
        "dtm_file_path": str(dtm_file),
        "processing_metadata": json.dumps({"crs": crs_str, "resolution": resolution}),
        **stats,
    }


def clear_existing_stats(table_name: str, region_id: int) -> None:
    if not inspect(engine).has_table(table_name):
        return

    with engine.begin() as conn:
        stats_table = Table(table_name, MetaData(), autoload_with=engine)
        conn.execute(stats_table.delete().where(stats_table.c.region_id == region_id))


def create_indexes(table_name: str) -> None:
    stats_table = Table(table_name, MetaData(), autoload_with=engine)
    Index(f"{table_name}_building_id_idx", stats_table.c.building_id).create(
        bind=engine, checkfirst=True
    )
    Index(f"{table_name}_region_id_idx", stats_table.c.region_id).create(
        bind=engine, checkfirst=True
    )


def write_elevation_stats(results: List[Dict], region_id: int) -> None:
    if not results:
        return

    table_name = "ground_elevation_stats"
    table_exists = inspect(engine).has_table(table_name)

    clear_existing_stats(table_name, region_id)

    df = pd.DataFrame(results).assign(created_at=pd.Timestamp.now(tz="UTC"))
    df.to_sql(
        table_name,
        con=engine,
        if_exists="replace" if not table_exists else "append",
        index=False,
    )

    create_indexes(table_name)


def execute_parallel_processing(
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
            f"Processing {region_name} (S: 0, F: 0)", total=len(args_list)
        )

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_building_elevation, args) for args in args_list
            ]

            for future in futures:
                building_id, stats = future.result()
                progress.update(task_id, advance=1)

                if stats:
                    results.append(stats)
                    success_count += 1
                else:
                    fail_count += 1

                progress.update(
                    task_id,
                    description=f"Processing {region_name} (S: {success_count}, F: {fail_count})",
                )

    return results, success_count, fail_count


def process_region(region_name: str, workers: int, resolution: float) -> None:
    logger.info(f"Processing ground elevation for region: {region_name}")

    buildings = fetch_buildings_with_las(region_name)
    if not buildings:
        logger.warning(f"No buildings with LAS files found for {region_name}")
        return

    dtm_dir = OUTPUT_ROOT / region_name / "lidar" / "dtm"
    dtm_dir.mkdir(parents=True, exist_ok=True)

    args_list = [
        (bid, rid, las_file, dtm_dir, resolution) for bid, rid, las_file in buildings
    ]

    results, success_count, fail_count = execute_parallel_processing(
        args_list, workers, region_name
    )

    write_elevation_stats(results, buildings[0][1])
    logger.success(
        f"{region_name}: {success_count} buildings processed, {fail_count} failed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ground elevation statistics from clipped LiDAR"
    )
    parser.add_argument("--region", choices=REGIONS, help="Process single region")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count(), 20),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.1, help="DTM resolution in meters"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    regions = [args.region] if args.region else REGIONS
    for region in regions:
        process_region(region, args.workers, args.resolution)


if __name__ == "__main__":
    main()
