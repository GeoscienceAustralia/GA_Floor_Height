#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from floor_heights.config import CONFIG
from floor_heights.db.audit import audit_database
from floor_heights.db.ibis_client import connect as ibis_connect
from floor_heights.db.loader import convert_to_parquet, load_from_parquet


def check_environment() -> bool:
    dotenv_path = find_dotenv(filename=".env", raise_error_if_not_found=False)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
        logger.debug(f"Loaded environment from: {dotenv_path}")

    return True


def get_required_files() -> list[Path]:
    return [
        CONFIG.project_root / "data" / "all_aoi_ffh_v5_3a2a2ee6e864.gpkg",
        Path("data/raw/wagga/FramePosOptimised-wagga-wagga-rev2.csv"),
        Path("data/raw/wagga/tileset/48068_Wagga_Wagga_TileSet.shp"),
        Path("data/raw/tweed/FramePosOptimised-tweed-heads-rev2.csv"),
        Path("data/raw/tweed/tileset/48068_Tweed_Heads_TileSet.shp"),
        Path("data/raw/launceston/FramePosOptimised-launceston-rev2.csv"),
        Path("data/raw/launceston/tileset/48068_Launceston_TileSet.shp"),
    ]


def validate_shapefile(shp_path: Path) -> tuple[bool, list[str]]:
    missing = [ext for ext in [".shp", ".dbf", ".shx"] if not shp_path.with_suffix(ext).exists()]

    return len(missing) == 0, missing


def check_data_files() -> bool:
    raw_dir = Path("data/raw")

    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        return False

    required_files = get_required_files()
    missing_files = []

    for file_path in required_files:
        if file_path.suffix == ".shp":
            is_valid, missing_components = validate_shapefile(file_path)
            if not is_valid:
                missing_files.append(f"{file_path} (missing: {', '.join(missing_components)})")
        elif not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        logger.error("Missing required files:")
        for f in missing_files:
            logger.error(f"  - {f}")
        return False

    processed_dir = Path("data/processed")
    for region in ["wagga", "tweed", "launceston"]:
        (processed_dir / region).mkdir(parents=True, exist_ok=True)

    return True


def get_database_summary(db_path: Path) -> dict:
    summary = {}

    conn = ibis_connect(db_path, read_only=True)

    table_names = conn.list_tables()

    tables = {}
    for table_name in table_names:
        try:
            count = conn.table(table_name).count().execute()
            tables[table_name] = count
        except Exception:
            tables[table_name] = 0

    summary["tables"] = tables

    if "buildings" in table_names:
        buildings = conn.table("buildings")
        regions_data = buildings.group_by("region_name").agg(count=buildings.count()).order_by("region_name").execute()
        summary["regions"] = {row["region_name"]: row["count"] for _, row in regions_data.iterrows()}

    return summary


def run_pipeline(skip_convert: bool = False, skip_load: bool = False, skip_audit: bool = False) -> int:
    start_time = datetime.now()
    logger.info("Starting GA floor heights data pipeline")

    check_environment()

    if not check_data_files():
        return 1

    try:
        if not skip_convert:
            logger.info("Converting raw spatial data to GeoParquet format")
            convert_to_parquet()

        if not skip_load:
            logger.info("Loading GeoParquet files into DuckDB")
            db_path = Path("data/floor_heights.duckdb")
            load_from_parquet(db_path)

            summary = get_database_summary(db_path)
            logger.info(f"Created {len(summary['tables'])} tables in database")

            if "regions" in summary:
                total_buildings = sum(summary["regions"].values())
                logger.info(f"Loaded {total_buildings:,} buildings across {len(summary['regions'])} regions")

        if not skip_audit:
            logger.info("Running database audit")
            audit_database(save_markdown=False)

        duration = datetime.now() - start_time
        logger.info(f"Pipeline completed successfully in {duration}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    ga_project_dir = Path(__file__).resolve().parent.parent.parent.parent
    os.chdir(ga_project_dir)
    logger.info(f"Working directory: {Path.cwd()}")

    return run_pipeline(
        skip_convert="--skip-convert" in args, skip_load="--skip-load" in args, skip_audit="--skip-audit" in args
    )


if __name__ == "__main__":
    sys.exit(main())
