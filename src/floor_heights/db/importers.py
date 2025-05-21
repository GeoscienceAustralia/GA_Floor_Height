"""Data import functions for the floor heights database."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import psycopg2

import geopandas as gpd
import pandas as pd
from loguru import logger

from .constants import CRS_LATLON, DSN
from ._utils import ensure_valid, nan_to_none, pg_copy, read_layers

PROPERTY_TYPE_MAP = {
    "C": "Commercial",
    "Commercial": "Commercial",
    "I": "Industrial",
    "Industrial": "Industrial",
    "R": "Residential",
    "Residential": "Residential",
    "Education": "Education",
    "Emergency": "Emergency",
    "Medical": "Medical",
    "Other": "Other",
    "Worship": "Worship",
}

WALL_MATERIAL_MAP = {
    "BK": "Brick",
    "BK - Brick": "Brick",
    "BKBD - Brick Board": "Brick",
    "BKR - Brick Rendered": "Brick",
    "BKV - Brick Veneer": "Brick",
    "BKVR - Brick Veneer Rendered": "Brick",
    "BRICK": "Brick",
    "Block": "Brick",
    "Brick": "Brick",
    "MBK - Masonry Brick": "Brick",
    "MBKR - Masonry Brick Rendered": "Brick",
    "MBKV - Masonry Brick Veneer": "Brick",
    "C - Concrete": "Concrete",
    "CB - Colorbond": "Colorbond",
    "CBKV - Concrete Brick Veneer": "Concrete",
    "CBL - Concrete Block": "Concrete",
    "CBLR - Concrete Block Rendered": "Concrete",
    "CLADDING": "Cladding",
    "CONCRETE": "Concrete",
    "CONR - Concrete Rendered": "Concrete",
    "CSHT - Cement Sheet": "Cement Sheet",
    "CSTR - Cement Sheet Rendered": "Cement Sheet",
    "CTP - Concrete Tilt Panel": "Concrete",
    "FC": "Fibre Cement",
    "FIBRO": "Fibre Cement",
    "Fibro": "Fibre Cement",
    "GI - Galvanised Iron": "Galvanised Iron",
    "Glass": "Glass",
    "IW - Imitation Weatherboard": "Weatherboard",
    "METC - Metal Cladding": "Metal",
    "Metal": "Metal",
    "OT": "Other",
    "OTHER": "Other",
    "Other": "Other",
    "PCP - Precast Concrete Panel": "Concrete",
    "Precast Concrete": "Concrete",
    "RENDER": "Rendered",
    "STON - Stone": "Stone",
    "Stone": "Stone",
    "TIMB - Timber": "Timber",
    "TIMBER": "Timber",
    "Timber": "Timber",
    "VB - Vertical Board": "Weatherboard",
    "WB": "Weatherboard",
    "WB - Weatherboard": "Weatherboard",
    "NO": None,
    "None": None,
    "YES": None,
}

ZONE_TO_PROPERTY_TYPE = {
    "General Residential": "Residential",
    "Inner Residential": "Residential",
    "Low Density Residential": "Residential",
    "Medium Density Residential": "Residential",
    "Environmental Living": "Residential",
    "Village": "Residential",
    "Commercial": "Commercial",
    "Industrial": "Industrial",
    "Mixed Use": "Mixed Use",
    "Urban Mixed Use": "Mixed Use",
    "Large Lot Residential": None,
    "Rural Activity Zone": None,
    "Rural Landscape": None,
}

RESIDENTIAL_ZONES = {
    "General Residential",
    "Inner Residential",
    "Large Lot Residential",
    "Low Density Residential",
    "Medium Density Residential",
    "Environmental Living",
    "Village",
    "Rural Activity Zone",
    "Rural Landscape",
    "Mixed Use",
    "Urban Mixed Use",
}

MAX_LINK_DIST = 5  # metres to snap points to a footprint


def footprints(gpkg: Path) -> None:
    """Import polygons into ``building_footprints``."""
    logger.info(f"Reading building footprints from: {gpkg}")
    gdf = read_layers(gpkg).to_crs(CRS_LATLON)
    logger.info(f"Found {len(gdf):,} building footprints, converting to EPSG:4326")
    ensure_valid(gdf, "footprints")

    gnaf = "gnaf_id"
    addr = "gnaf_address"
    logger.info(f"Using address column: {addr}, GNAF column: {gnaf}")

    logger.info(f"Preparing {len(gdf):,} footprints for database import")

    def _row(geom, record):
        zone = record.get("land_use_zone")
        ptype = ZONE_TO_PROPERTY_TYPE.get(zone)
        return (
            None,
            record.get("id"),
            record.get(gnaf) if gnaf else None,
            ([record.get(addr)] if addr and record.get(addr) else []),
            record.get("geocode_type"),
            zone,
            ptype,
            f"SRID={CRS_LATLON};{geom.wkt}",
            ptype == "Residential" or zone in RESIDENTIAL_ZONES,
        )

    rows: Iterable[tuple] = (
        _row(geom, record)
        for geom, record in zip(
            gdf.geometry, gdf.drop(columns="geometry").to_dict("records")
        )
    )

    pg_copy(
        "building_footprints",
        (
            "region_id",
            "external_id",
            "gnaf_id",
            "address",
            "geocode_type",
            "land_use_zone",
            "property_type",
            "geom",
            "is_residential",
        ),
        rows,
    )


def points(region_id: int, shp: Path, pk: str, colmap: dict[str, str]) -> None:
    """Import point geometries into ``building_points``."""
    logger.info(f"Reading building points from: {shp}")
    gdf = gpd.read_file(shp, engine="pyogrio", use_arrow=True).to_crs(CRS_LATLON)
    logger.info(f"Found {len(gdf):,} building points")
    ensure_valid(gdf, "points")
    if pk not in gdf.columns:
        raise ValueError(f"{pk!r} missing in {shp}")

    logger.info("Converting floor and ground level values")
    fl = pd.to_numeric(gdf[colmap["floor_level_m"]], errors="coerce")
    gl = pd.to_numeric(gdf[colmap["ground_level_m"]], errors="coerce")
    fh = fl - gl

    # Standardise categorical values
    if "property_type" in colmap:
        prop_col = colmap["property_type"]
        gdf[prop_col] = gdf[prop_col].map(lambda v: PROPERTY_TYPE_MAP.get(v, v))
    if "wall_material" in colmap:
        wall_col = colmap["wall_material"]
        gdf[wall_col] = gdf[wall_col].map(lambda v: WALL_MATERIAL_MAP.get(v, v))

    logger.info(f"Preparing {len(gdf):,} building points for import")

    rows: Iterable[tuple] = (
        (
            region_id,
            None,
            str(gdf.loc[i, pk]),
            nan_to_none(fl[i]),
            nan_to_none(gl[i]),
            nan_to_none(fh[i]),
            gdf.loc[i, colmap["property_type"]] if "property_type" in colmap else None,
            gdf.loc[i, colmap["wall_material"]] if "wall_material" in colmap else None,
            f"SRID={CRS_LATLON};{gdf.geometry[i].wkt}",
        )
        for i in range(len(gdf))
    )
    pg_copy(
        "building_points",
        (
            "region_id",
            "footprint_id",
            "source_point_id",
            "floor_level_m",
            "ground_level_m",
            "floor_height_m",
            "property_type",
            "wall_material",
            "geom",
        ),
        rows,
    )


_REQUIRED = [
    "ucid",
    "Systemtime_sec",
    "Frame_index",
    "Longitude_deg",
    "Latitude_deg",
    "Altitude_m",
    "Heading_deg",
    "Pitch_deg",
    "Roll_deg",
    "imgID",
]


def panoramas(region_id: int, csv: Path) -> None:
    """Import panorama metadata from ``csv``."""
    logger.info(f"Reading panorama metadata from: {csv}")
    df = pd.read_csv(csv)
    logger.info(f"Found {len(df):,} panorama entries")

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{csv}: missing {', '.join(missing)}")

    mask = (
        df.Latitude_deg.between(-90, 90)
        & df.Longitude_deg.between(-180, 180)
        & df.Altitude_m.notna()
    )
    d = df[mask]
    logger.info(f"Using {len(d):,} panoramas with valid coordinates")

    logger.info("Preparing panorama records for import")
    rows: Iterable[tuple] = (
        (
            region_id,
            r.ucid,
            r.Systemtime_sec,
            r.Frame_index,
            r.Longitude_deg,
            r.Latitude_deg,
            r.Altitude_m,
            r.get("LTP_x_m"),
            r.get("LTP_y_m"),
            r.get("LTP_z_m"),
            r.Roll_deg,
            r.Pitch_deg,
            r.Heading_deg,
            r.imgID,
            f"SRID=4326;POINT Z({r.Longitude_deg} {r.Latitude_deg} {r.Altitude_m})",
        )
        for _, r in d.iterrows()
    )
    pg_copy(
        "panoramas",
        (
            "region_id",
            "ucid",
            "system_time",
            "frame_index",
            "longitude_deg",
            "latitude_deg",
            "altitude_m",
            "ltp_x_m",
            "ltp_y_m",
            "ltp_z_m",
            "roll_deg",
            "pitch_deg",
            "heading_deg",
            "imgid",
            "geom",
        ),
        rows,
        conflict=("imgid",),
    )


def tileset(tileset_id: int, shp: Path, crs: int) -> None:
    """Import tileset footprints."""
    logger.info(f"Reading tileset from: {shp}")
    gdf = gpd.read_file(shp, engine="pyogrio", use_arrow=True)
    logger.info(f"Found {len(gdf)} tileset entries, initial CRS: {gdf.crs}")

    if gdf.crs is None:
        logger.info(f"Setting CRS to {crs} since dataset had no CRS information")
        gdf = gdf.set_crs(crs)
    gdf = gdf.to_crs(CRS_LATLON)
    ensure_valid(gdf, "tileset index")

    filename_col = next((c for c in gdf.columns if "file" in c.lower()), None)
    if filename_col is None:
        raise ValueError("no filename column containing 'file' found")
    logger.info(f"Using '{filename_col}' as filename column")

    logger.info(f"Preparing {len(gdf)} rows for insert")
    rows = [
        (
            tileset_id,
            r[filename_col],
            f"SRID=4326;{geom.wkt}",
        )
        for geom, r in zip(
            gdf.geometry, gdf.drop(columns="geometry").to_dict("records")
        )
    ]

    logger.info(f"Importing {len(rows)} tileset entries to database")
    pg_copy(
        "tileset_indexes",
        ("tileset_id", "file_name", "geom"),
        rows,
    )
    logger.info(f"Completed tileset import for tileset_id={tileset_id}")


def link_points_to_footprints(region_id: int) -> None:
    """Assign ``footprint_id`` for points in ``region_id``."""
    sql = f"""
        WITH c AS (
            SELECT bp.id,
                   COALESCE(
                       (SELECT bf.id FROM building_footprints bf
                        WHERE ST_Contains(bf.geom, bp.geom)
                          AND (bf.region_id IS NULL OR bf.region_id=%s)
                        ORDER BY bp.geom <-> bf.geom LIMIT 1),
                       (SELECT bf.id FROM building_footprints bf
                        WHERE ST_DWithin(bp.geom, bf.geom, {MAX_LINK_DIST})
                          AND (bf.region_id IS NULL OR bf.region_id=%s)
                        ORDER BY bp.geom <-> bf.geom LIMIT 1)
                   ) AS fid
            FROM building_points bp
            WHERE bp.region_id=%s AND bp.footprint_id IS NULL
        )
        UPDATE building_points bp
        SET footprint_id=c.fid
        FROM c
        WHERE bp.id=c.id AND c.fid IS NOT NULL;
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, (region_id, region_id, region_id))
        updated = cur.rowcount
        conn.commit()
    logger.info(f"linked {updated:,} points to footprints for region_id={region_id}")


def assign_footprint_regions() -> None:
    """Populate ``building_footprints.region_id`` using linked points."""
    sql = """
        WITH fp_regions AS (
            SELECT bf.id AS footprint_id, MAX(bp.region_id) AS region_id
            FROM building_footprints bf
            JOIN building_points bp ON bp.footprint_id = bf.id
            WHERE bf.region_id IS NULL
            GROUP BY bf.id
        )
        UPDATE building_footprints bf
        SET region_id = fr.region_id
        FROM fp_regions fr
        WHERE bf.id = fr.footprint_id;
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql)
        updated = cur.rowcount
        conn.commit()
    logger.info(f"updated region_id for {updated:,} building footprints")


def _update_region_geometries() -> None:
    """Generate convex hulls for regions based on panorama locations."""
    sql = """
        WITH hulls AS (
            SELECT region_id,
                   ST_Multi(ST_Buffer(ST_ConvexHull(ST_Collect(geom)), 0.0001)) AS hull
            FROM panoramas
            GROUP BY region_id
        )
        UPDATE regions r
        SET geom = h.hull
        FROM hulls h
        WHERE r.id = h.region_id;
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()


def _assign_regions_from_hulls() -> None:
    """Assign remaining footprints to the nearest region hull."""
    sql = """
        WITH region_centroids AS (
            SELECT id AS region_id, ST_Centroid(geom) AS cen
            FROM regions
            WHERE geom IS NOT NULL
        ),
        footprints AS (
            SELECT id AS fid, ST_Centroid(geom) AS cen
            FROM building_footprints
            WHERE region_id IS NULL
        ),
        nearest AS (
            SELECT DISTINCT ON (f.fid) f.fid, rc.region_id
            FROM footprints f
            CROSS JOIN region_centroids rc
            ORDER BY f.fid, f.cen <-> rc.cen
        )
        UPDATE building_footprints bf
        SET region_id = n.region_id
        FROM nearest n
        WHERE bf.id = n.fid AND bf.region_id IS NULL;
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()


def deduplicate_footprints() -> None:
    """Aggregate addresses then remove duplicate rows."""
    update_sql = """
        WITH unnested AS (
            SELECT CASE WHEN external_id IS NOT NULL THEN external_id 
                        ELSE 'geom_' || ST_GeoHash(geom, 10) END AS key,
                   id,
                   unnest(address) AS addr
            FROM building_footprints
        ), agg AS (
            SELECT key, MIN(id) AS keep_id,
                   ARRAY_AGG(DISTINCT addr) AS addresses
            FROM unnested
            GROUP BY key
        )
        UPDATE building_footprints bf
        SET address = agg.addresses
        FROM agg
        WHERE bf.id = agg.keep_id;
    """

    delete_sql = """
        DELETE FROM building_footprints bf
        USING (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY CASE WHEN external_id IS NOT NULL THEN external_id 
                                     ELSE 'geom_' || ST_GeoHash(geom, 10) END
                       ORDER BY id
                   ) AS rn
            FROM building_footprints
        ) d
        WHERE bf.id = d.id AND d.rn > 1;
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(update_sql)
        cur.execute(delete_sql)
        conn.commit()


def associate_tilesets(
    region_id: int, tileset_id: int, *, min_pct: float = 1.0
) -> None:
    """Populate ``building_tileset_associations`` for ``region_id``."""
    sql = """
        INSERT INTO building_tileset_associations (
            building_id, tileset_index_id, intersection_percent
        )
        SELECT bf.id, ti.id,
               100.0 * ST_Area(ST_Intersection(bf.geom, ti.geom)) / ST_Area(bf.geom)
        FROM building_footprints bf
        JOIN tileset_indexes ti ON ti.tileset_id = %s AND ST_Intersects(bf.geom, ti.geom)
        WHERE bf.region_id = %s
          AND ST_Area(ST_Intersection(bf.geom, ti.geom)) / ST_Area(bf.geom) * 100 >= %s
        ON CONFLICT (building_id, tileset_index_id) DO UPDATE
            SET intersection_percent = EXCLUDED.intersection_percent;
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, (tileset_id, region_id, min_pct))
        conn.commit()


def finalize_regions() -> None:
    """Assign region IDs using all available methods."""
    assign_footprint_regions()
    _update_region_geometries()
    _assign_regions_from_hulls()
