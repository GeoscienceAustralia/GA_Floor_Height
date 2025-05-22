"""Stage-01: Footprint-first residential building enrichment with fallback centroids."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import yaml
from geoalchemy2 import Geometry
from loguru import logger
from sqlalchemy import (
    MetaData,
    Table,
    case,
    create_engine,
    func,
    inspect,
    select,
    Index,
)
from sqlalchemy.engine import Engine

DB = os.getenv("DB_CONNECTION_STRING")
if not DB:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")


def select_core(
    engine: Engine, meta: MetaData, region: str, cfg: Dict
) -> gpd.GeoDataFrame:
    """Residential footprints enriched with building points or fallback centroids."""

    bp = Table("building_points", meta, autoload_with=engine)
    bf = Table("building_footprints", meta, autoload_with=engine)
    reg = Table("regions", meta, autoload_with=engine)

    with engine.connect() as conn:
        region_id = conn.execute(
            select(reg.c.id).where(reg.c.name == region)
        ).scalar_one()

        footprints_count = conn.execute(
            select(func.count())
            .select_from(bf)
            .where(bf.c.region_id == region_id, bf.c.property_type == "Residential")
        ).scalar_one()
        logger.info(f"{region}: {footprints_count:,} residential footprints found")

        points_count = conn.execute(
            select(func.count()).select_from(bp).where(bp.c.region_id == region_id)
        ).scalar_one()
        logger.info(f"{region}: {points_count:,} total building points in region")

        q = (
            select(
                bf.c.id.label("footprint_id"),
                bf.c.region_id,
                func.encode(func.ST_AsBinary(func.ST_Force2D(bf.c.geom)), "hex").label(
                    "geom"
                ),
                bp.c.id.label("point_id"),
                bp.c.source_point_id,
                func.coalesce(bp.c.property_type, bf.c.property_type).label(
                    "property_type"
                ),
                bp.c.floor_level_m,
                bp.c.ground_level_m,
                bp.c.floor_height_m,
                bp.c.wall_material,
                case(
                    (bp.c.id.is_not(None), "point_data"), else_="footprint_only"
                ).label("data_source"),
                func.coalesce(
                    func.encode(func.ST_AsBinary(func.ST_Force2D(bp.c.geom)), "hex"),
                    func.encode(
                        func.ST_AsBinary(func.ST_Force2D(func.ST_Centroid(bf.c.geom))),
                        "hex",
                    ),
                ).label("point_geom"),
            )
            .select_from(bf.outerjoin(bp, bf.c.id == bp.c.footprint_id))
            .where(
                bf.c.region_id == region_id,
                func.coalesce(bp.c.property_type, bf.c.property_type) == "Residential",
            )
        )

        gdf = (
            gpd.read_postgis(
                q.compile(engine, compile_kwargs={"literal_binds": True}),
                conn,
                geom_col="geom",
            )
            .set_crs(4326)
            .assign(processed_at=pd.Timestamp.utcnow())
        )

    gdf["point_id"] = gdf["point_id"].astype("Int64")

    # optional sampling for development/debugging
    sampling = cfg.get("common", {}).get("sampling", {})
    if sampling.get("enabled"):
        seed = sampling.get("random_seed", 42)
        if (n := sampling.get("sample_size")) and n < len(gdf):
            gdf = gdf.sample(n, random_state=seed)
        elif (frac := sampling.get("sample_fraction")) and 0 < frac < 1:
            gdf = gdf.sample(frac=frac, random_state=seed)

    logger.info(
        f"{region}: {len(gdf):,} buildings → {gdf['data_source'].value_counts().to_dict()}"
    )
    return gdf


def write_processed(
    engine: Engine, gdf: gpd.GeoDataFrame, region_id: int, meta: MetaData
) -> None:
    """Replace building_points_processed rows for region with dual geometries."""

    table_name = "building_points_processed"
    inspector = inspect(engine)
    table_exists = inspector.has_table(table_name)

    if table_exists:
        with engine.begin() as cn:
            bp_proc = Table(table_name, MetaData(), autoload_with=engine)
            cn.execute(bp_proc.delete().where(bp_proc.c.region_id == region_id))

    gdf_to_write = gdf.copy()
    gdf_to_write["point_geom"] = gpd.GeoSeries.from_wkb(
        gdf["point_geom"].apply(bytes.fromhex), crs=4326
    )

    gdf_to_write.to_postgis(
        table_name,
        con=engine,
        if_exists="replace" if not table_exists else "append",
        index=False,
        dtype={
            "geom": Geometry("MULTIPOLYGON", srid=4326),
            "point_geom": Geometry("POINT", srid=4326),
        },
    )

    bp_proc = Table(table_name, MetaData(), autoload_with=engine)
    Index("building_points_processed_region_id_idx", bp_proc.c.region_id).create(
        bind=engine, checkfirst=True
    )
    Index("building_points_processed_footprint_id_idx", bp_proc.c.footprint_id).create(
        bind=engine, checkfirst=True
    )
    Index(
        "building_points_processed_geom_idx", bp_proc.c.geom, postgresql_using="gist"
    ).create(bind=engine, checkfirst=True)
    Index(
        "building_points_processed_point_geom_idx",
        bp_proc.c.point_geom,
        postgresql_using="gist",
    ).create(bind=engine, checkfirst=True)


def main() -> None:
    """CLI entry point for single region or all regions from config."""

    common_cfg = yaml.safe_load(Path("config/common.yaml").read_text())
    region_choices: List[str] = common_cfg.get("regions", [])

    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=region_choices)
    ap.add_argument(
        "--log-level",
        default="info",
        choices=["trace", "debug", "info", "warning", "error", "critical"],
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    engine = create_engine(DB, future=True)
    meta = MetaData()

    # clear existing processed table – avoids accidental duplicates
    if inspect(engine).has_table("building_points_processed"):
        bp_proc = Table("building_points_processed", meta, autoload_with=engine)
        with engine.begin() as conn:
            conn.execute(bp_proc.delete())
        logger.info("Cleared existing building_points_processed table")

    regions_to_run = [args.region] if args.region else region_choices
    for region in regions_to_run:
        region_cfg = yaml.safe_load((Path("config") / f"{region}.yaml").read_text())
        region_cfg["common"] = common_cfg

        gdf = select_core(engine, meta, region, region_cfg)

        # fetch region_id once per region
        reg_tbl = Table("regions", meta, autoload_with=engine)
        with engine.connect() as conn:
            region_id = conn.execute(
                select(reg_tbl.c.id).where(reg_tbl.c.name == region)
            ).scalar_one()

        write_processed(engine, gdf, region_id, meta)
        logger.success(f"{region}: {len(gdf):,} rows → building_points_processed")


if __name__ == "__main__":
    main()
