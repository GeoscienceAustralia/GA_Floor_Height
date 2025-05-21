"""Data migration utilities for importing region data into the database."""

from __future__ import annotations

import sys

from typing import Iterable, Sequence

import psycopg2
import yaml
from dotenv import load_dotenv, find_dotenv
from .audit import audit_database
from loguru import logger
from psycopg2.extras import execute_values

from .constants import DSN, Dirs
from .initialize import init_schema
from .importers import (
    associate_tilesets,
    deduplicate_footprints,
    finalize_regions,
    footprints,
    link_points_to_footprints,
    panoramas,
    points,
    tileset,
)


# Mapping of region names to dataset specifics
_SHAPEFILES = {
    "wagga": Dirs.data / "wagga" / "Final_Wagga" / "Final_Wagga.shp",
    "tweed": Dirs.data
    / "tweed"
    / "BuildingFloorLevels_FloodStudyTweed"
    / "BuildingFloorLevels_FloodStudy.shp",
    "launceston": Dirs.data / "launceston" / "LC_Final" / "LC_Final.shp",
}

_TRAJECTORIES = {
    "wagga": Dirs.data
    / "wagga"
    / "FramePosOptimised-wagga-wagga-rev1.1_fixednaming.csv",
    "tweed": Dirs.data
    / "tweed"
    / "FramePosOptimised-tweedheads-rev1.1_fixednaming.csv",
    "launceston": Dirs.data
    / "launceston"
    / "FramePosOptimised-launceston-rev1.1_fixednaming.csv",
}

_TILESETS = {
    "wagga": Dirs.data / "wagga" / "tileset" / "48068_Wagga_Wagga_TileSet.shp",
    "tweed": Dirs.data / "tweed" / "tileset" / "48068_Tweed_Heads_TileSet.shp",
    "launceston": Dirs.data / "launceston" / "tileset" / "48068_Launceston_TileSet.shp",
}

_PKS = {"wagga": "UFI", "tweed": "GID", "launceston": "UFI"}
_IDS = {"wagga": 1, "launceston": 2, "tweed": 3}


def _fetch_one(cur: psycopg2.extensions.cursor, sql: str) -> int:
    """Execute SQL query and return the first column of the first row as an integer."""
    cur.execute(sql)
    return int(cur.fetchone()[0])


def _format_table(headers: Iterable[str], rows: Iterable[Iterable]) -> str:
    """Format rows of data as a text table with aligned columns."""
    str_rows = [tuple(map(str, r)) for r in rows]
    cols = list(zip(*([headers] + str_rows)))
    widths = [max(len(c) for c in col) for col in cols]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    body = [" | ".join(v.ljust(w) for v, w in zip(row, widths)) for row in str_rows]
    return "\n".join([line, sep, *body])


def _truncate_table(table: str) -> None:
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(f"TRUNCATE {table} RESTART IDENTITY CASCADE")
        conn.commit()


def _table_counts() -> list[tuple[str, int]]:
    tables = [
        "regions",
        "building_footprints",
        "building_points",
        "panoramas",
        "tileset_indexes",
    ]
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        return [(t, _fetch_one(cur, f"SELECT COUNT(*) FROM {t}")) for t in tables]


def _load_config(region: str) -> dict:
    cfg = yaml.safe_load((Dirs.root / "config" / f"{region}.yaml").read_text()) or {}
    rid = _IDS[region]
    return {
        "footprints": Dirs.data / "buildings.gpkg",
        "points": {
            "region_id": rid,
            "shp": _SHAPEFILES[region],
            "pk": _PKS[region],
            "colmap": cfg.get("column_mappings", {}),
        },
        "panoramas": {"region_id": rid, "csv": _TRAJECTORIES[region]},
        "tileset": {
            "tileset_id": rid,
            "shp": _TILESETS[region],
            "crs": int(cfg.get("crs", 4326)),
        },
    }


def _ensure_regions() -> None:
    """Insert required regions if they do not exist."""
    rows = [(rid, name) for name, rid in _IDS.items()]
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO regions (id, name) VALUES %s ON CONFLICT (id) DO NOTHING",
            rows,
        )
        conn.commit()


def _ensure_tilesets() -> None:
    """Insert required tileset records if they do not exist."""
    from .constants import CRS_LATLON

    rows = [
        (rid, rid, name, f"{name} LiDAR collection", "LAS", CRS_LATLON)
        for name, rid in _IDS.items()
    ]
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        execute_values(
            cur,
            """INSERT INTO tilesets
               (id, region_id, name, description, format, crs)
               VALUES %s ON CONFLICT (id) DO NOTHING""",
            rows,
        )
        conn.commit()


def main(region: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
    )
    cfg = _load_config(region)
    logger.info(f"running migration for region: {region} ...")

    logger.info("initializing schema...")
    init_schema()

    logger.info("ensuring regions exist...")
    _ensure_regions()

    logger.info("ensuring tilesets exist...")
    _ensure_tilesets()

    logger.info("importing building footprints...")
    _truncate_table("building_footprints")
    footprints(cfg["footprints"])
    deduplicate_footprints()

    logger.info("importing building points...")
    points(**cfg["points"])
    link_points_to_footprints(cfg["points"]["region_id"])

    logger.info("importing panoramas...")
    panoramas(**cfg["panoramas"])

    logger.info("importing tileset indexes...")
    tileset(**cfg["tileset"])
    finalize_regions()
    associate_tilesets(cfg["points"]["region_id"], cfg["tileset"]["tileset_id"])

    logger.success("migration complete")
    summary = _table_counts()
    logger.info("\n" + _format_table(["table", "rows"], summary))

    logger.info("running database audit...")
    audit_database("public")


def migrate_many(regions: Sequence[str]) -> None:
    """Run migrations for multiple ``regions`` in one database."""
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
    )
    logger.info("initializing schema...")
    init_schema()

    logger.info("ensuring regions exist...")
    _ensure_regions()

    logger.info("ensuring tilesets exist...")
    _ensure_tilesets()

    logger.info("importing building footprints ...")
    first_cfg = _load_config(regions[0])
    _truncate_table("building_footprints")
    footprints(first_cfg["footprints"])
    deduplicate_footprints()

    for region in regions:
        cfg = _load_config(region)
        logger.info(f"importing data for region: {region} ...")
        points(**cfg["points"])
        link_points_to_footprints(cfg["points"]["region_id"])
        panoramas(**cfg["panoramas"])
        tileset(**cfg["tileset"])

    finalize_regions()

    for region in regions:
        cfg = _load_config(region)
        associate_tilesets(cfg["points"]["region_id"], cfg["tileset"]["tileset_id"])

    logger.success("migration complete")
    summary = _table_counts()
    logger.info("\n" + _format_table(["table", "rows"], summary))

    logger.info("running database audit...")
    audit_database("public")


if __name__ == "__main__":
    import argparse

    dotenv_path = find_dotenv(filename=".env", usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)

    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--all", action="store_true", help="Run migrations for all regions"
    )
    parser.add_argument("--region", type=str, help="Specific region to migrate")
    args = parser.parse_args()

    if args.all:
        migrate_many(["wagga", "launceston", "tweed"])
    elif args.region:
        main(args.region)
    else:
        parser.print_help()
