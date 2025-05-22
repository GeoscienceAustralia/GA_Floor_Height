#!/usr/bin/env python
"""Stage-02: Panorama trajectory data consolidation with 2D geometry normalization."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from geoalchemy2 import Geometry
from loguru import logger
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    func,
    select,
    Index,
    inspect,
    delete,
)

DB = os.getenv("DB_CONNECTION_STRING")
if not DB:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")

engine = create_engine(DB, future=True)
meta = MetaData()

regions_table = Table("regions", meta, autoload_with=engine)
panoramas_table = Table("panoramas", meta, autoload_with=engine)


def get_region_id(region_name: str) -> int:
    """Retrieve region ID, raising if not found."""
    stmt = select(regions_table.c.id).where(regions_table.c.name == region_name)
    with engine.connect() as conn:
        result = conn.execute(stmt).scalar_one_or_none()
        if result is None:
            raise ValueError(f"Region '{region_name}' not found in regions table")
        return int(result)


def fetch_trajectory_data(region_id: int) -> gpd.GeoDataFrame:
    """Panorama data with 2D hex-encoded geometry for Shapely compatibility."""
    columns = [col for col in panoramas_table.c if col.name != "geom"]

    # force 2D geometry to avoid Shapely EWKB parsing issues
    geometry_hex = func.encode(
        func.ST_AsBinary(func.ST_Force2D(panoramas_table.c.geom)), "hex"
    ).label("geom")

    stmt = select(*columns, geometry_hex).where(
        panoramas_table.c.region_id == region_id
    )

    with engine.connect() as conn:
        gdf = gpd.read_postgis(stmt, conn, geom_col="geom")

    if not gdf.empty and gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    elif not gdf.empty and gdf.crs != "EPSG:4326":
        logger.warning(
            f"Unexpected CRS {gdf.crs} for region_id {region_id}, overriding to EPSG:4326"
        )
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    return gdf


def write_trajectory_data(gdf: gpd.GeoDataFrame, region_id: int) -> None:
    """Replace trajectory_processed rows for region."""
    table_name = "trajectory_processed"
    inspector = inspect(engine)
    table_exists = inspector.has_table(table_name)

    if table_exists:
        with engine.begin() as conn:
            traj_tbl = Table(table_name, MetaData(), autoload_with=engine)
            conn.execute(delete(traj_tbl).where(traj_tbl.c.region_id == region_id))

    gdf.to_postgis(
        name=table_name,
        con=engine,
        if_exists="replace" if not table_exists else "append",
        index=False,
        dtype={"geom": Geometry("POINT", srid=4326)},
    )

    traj_tbl = Table(table_name, MetaData(), autoload_with=engine)
    Index(f"{table_name}_region_id_idx", traj_tbl.c.region_id).create(
        bind=engine, checkfirst=True
    )
    Index(f"{table_name}_geom_idx", traj_tbl.c.geom, postgresql_using="gist").create(
        bind=engine, checkfirst=True
    )


def process_region(region_name: str) -> None:
    """Process trajectory data for single region."""
    logger.info(f"Processing trajectory for region: {region_name}")

    region_id = get_region_id(region_name)
    gdf = fetch_trajectory_data(region_id)

    if gdf.empty:
        logger.warning(f"No trajectory data found for region: {region_name}")
        return

    processed_gdf = gdf.assign(processed_at=pd.Timestamp.now(tz="UTC"))
    write_trajectory_data(processed_gdf, region_id)

    logger.success(
        f"{region_name}: {len(processed_gdf):,} panoramas → trajectory_processed"
    )


def clear_existing_data() -> None:
    """Remove all existing trajectory_processed data."""
    inspector = inspect(engine)
    if not inspector.has_table("trajectory_processed"):
        return

    traj_tbl = Table("trajectory_processed", meta, autoload_with=engine)
    with engine.begin() as conn:
        conn.execute(traj_tbl.delete())
    logger.info("Cleared existing trajectory_processed table")


def load_config() -> tuple[dict, list[str]]:
    """Load common configuration and extract region choices."""
    config_path = Path("config") / "common.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())
    return cfg, cfg.get("regions", [])


def main() -> None:
    """CLI entry point for trajectory data processing."""
    common_cfg, region_choices = load_config()

    parser = argparse.ArgumentParser(
        description="Convert panorama trajectory data to trajectory_processed table"
    )
    parser.add_argument("--region", choices=region_choices)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    clear_existing_data()

    regions = [args.region] if args.region else region_choices
    for region in regions:
        try:
            process_region(region)
        except ValueError as e:
            logger.error(f"Configuration error for region {region}: {e}")
        except Exception as e:
            logger.error(f"Failed to process region {region}: {e}")
            logger.exception(f"Detailed traceback for {region}:")


if __name__ == "__main__":
    main()
