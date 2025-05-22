#!/usr/bin/env python
"""Stage-04: Clip panoramas to buildings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterable, Mapping

import argparse
import importlib.util
import os
import sys
import warnings

import geopandas as gpd
import pandas as pd
import yaml
from PIL import Image
from geoalchemy2 import Geometry
from loguru import logger
from sqlalchemy import Index, MetaData, Table, create_engine, func, select
from sqlalchemy.engine import Engine


warnings.filterwarnings("ignore", message="Did not recognize type 'geometry'")

DB_URL = os.getenv("DB_CONNECTION_STRING")
if not DB_URL:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")

CFG_PATH = Path(__file__).resolve().parents[3] / "config" / "common.yaml"
_common_cfg = yaml.safe_load(CFG_PATH.read_text()) if CFG_PATH.exists() else {}
OUTPUT_ROOT: Path = Path(_common_cfg.get("output_root", "output"))
REGIONS: tuple[str, ...] = tuple(_common_cfg.get("regions", ()))

CLIP_UPPER_PROP = 0.25
CLIP_LOWER_PROP = 0.60
ANGLE_EXTEND = 40.0
WORKERS = max(1, (cpu_count() or 1) - 1)
Image.MAX_IMAGE_PIXELS = None


_geo_path = Path(__file__).parent.parent / "utils" / "geometry.py"
_spec = importlib.util.spec_from_file_location("geometry", _geo_path)
assert _spec is not None
geo = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(geo)


engine: Engine = create_engine(DB_URL, future=True, pool_pre_ping=True)
_meta = MetaData()
_tbl_views = Table("panorama_candidate_views", _meta, autoload_with=engine)
_tbl_panos = Table("panoramas", _meta, autoload_with=engine)
_tbl_bldg = Table("building_footprints", _meta, autoload_with=engine)


def _region_folder(name: str) -> Path:
    return OUTPUT_ROOT / name.capitalize()


def _jpg(name: str) -> str:  # tiny utility to guarantee *.jpg*
    return name if name.lower().endswith(".jpg") else f"{name}.jpg"


def _pano_path(region: str, bid: int, pid: str) -> Path:
    return _region_folder(region) / "panoramas" / str(bid) / _jpg(pid)


def _clip_path(region: str, bid: int, pid: str) -> Path:
    return _region_folder(region) / "clips" / str(bid) / _jpg(pid)


def _ensure_gist_idx(table_name: str) -> None:
    tbl = Table(table_name, MetaData(), autoload_with=engine, extend_existing=True)
    idx = Index(f"{table_name}_geom_idx", tbl.c.geom, postgresql_using="gist")
    with engine.begin() as cn:
        idx.create(bind=cn, checkfirst=True)


def _chosen_views(region: str) -> gpd.GeoDataFrame:
    stmt = select(
        _tbl_views.c.building_id,
        _tbl_views.c.pano_id,
        _tbl_views.c.geom,
    ).where(_tbl_views.c.region == region, _tbl_views.c.is_chosen.is_(True))

    with engine.connect() as cn:
        gdf = gpd.read_postgis(stmt, cn, geom_col="geom").set_crs(4326)
    return gdf.drop_duplicates(subset=["building_id", "pano_id"])


def _panorama_meta(pano_ids: Iterable[str]) -> pd.DataFrame:
    stmt = select(
        _tbl_panos.c.imgid.label("pano_id"),
        _tbl_panos.c.latitude_deg.label("pano_lat"),
        _tbl_panos.c.longitude_deg.label("pano_lon"),
        _tbl_panos.c.heading_deg.label("pano_heading"),
    ).where(_tbl_panos.c.imgid.in_(list(pano_ids)))
    with engine.connect() as cn:
        return pd.read_sql(stmt, cn).set_index("pano_id")


def _building_centroids(bids: Iterable[int]) -> pd.DataFrame:
    cent = func.ST_AsText(func.ST_Centroid(_tbl_bldg.c.geom)).label("wkt")
    stmt = select(_tbl_bldg.c.id.label("building_id"), cent).where(
        _tbl_bldg.c.id.in_(list(map(int, bids)))
    )
    with engine.connect() as cn:
        df = pd.read_sql(stmt, cn)

    coords = (
        df.wkt.str.replace(r"^POINT\(", "", regex=True)
        .str.replace(r"\)", "", regex=True)
        .str.split(" ", expand=True)
        .astype(float)
        .rename(columns={0: "bldg_lon", 1: "bldg_lat"})
    )
    return pd.concat([df.drop(columns="wkt"), coords], axis=1).set_index("building_id")


def _horizontal_range(
    *,
    pano_lat: float,
    pano_lon: float,
    bldg_lat: float,
    bldg_lon: float,
    heading: float,
    width: int,
) -> tuple[float, float]:
    info: Mapping[str, tuple[float, float]] = geo.localize_house_in_panorama(
        lat_c=pano_lat,
        lon_c=pano_lon,
        lat_house=bldg_lat,
        lon_house=bldg_lon,
        beta_yaw_deg=heading,
        Wim=width,
        angle_extend=ANGLE_EXTEND,
    )
    range_data = info["horizontal_pixel_range_house"]
    return (float(range_data[0]), float(range_data[1]))


def _crop(img: Image.Image, h_range: tuple[float, float]) -> Image.Image:
    left, right = map(round, h_range)
    top = round(CLIP_UPPER_PROP * img.height)
    bottom = round(CLIP_LOWER_PROP * img.height)
    if left >= right or top >= bottom:
        raise ValueError("invalid crop window")
    return img.crop((int(left), int(top), int(right), int(bottom)))


@dataclass(frozen=True)
class ClipResult:
    status: str
    building_id: int | None = None
    pano_id: str | None = None
    region: str | None = None
    clip_left: float | None = None
    clip_right: float | None = None
    geom: object | None = None  # geometry column stays opaque


def _clip_row(region: str, row: pd.Series) -> ClipResult:
    out_path = _clip_path(region, row.building_id, row.pano_id)
    if out_path.exists():
        return ClipResult("skip")

    pano_file = _pano_path(region, row.building_id, row.pano_id)
    if not pano_file.exists():
        return ClipResult("missing")

    try:
        img = Image.open(pano_file)
        h_range = _horizontal_range(
            pano_lat=row.pano_lat,
            pano_lon=row.pano_lon,
            bldg_lat=row.bldg_lat,
            bldg_lon=row.bldg_lon,
            heading=row.pano_heading,
            width=img.width,
        )
        clip = _crop(img, h_range)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clip.save(out_path)
        return ClipResult(
            "success",
            building_id=row.building_id,
            pano_id=row.pano_id,
            region=region,
            clip_left=h_range[0],
            clip_right=h_range[1],
            geom=row.geom,
        )
    except ValueError:
        return ClipResult("clip_fail")
    except Exception as exc:  # noqa: BLE001 – counted failure only
        logger.debug(f"{region} {row.building_id}/{row.pano_id}: {exc}")
        return ClipResult("fail")


def _run_region(region: str) -> None:
    logger.info(f"── clipping {region} ──")

    chosen = _chosen_views(region)
    if chosen.empty:
        logger.warning(f"{region}: nothing to clip")
        return

    pano_df = _panorama_meta(chosen.pano_id.unique())
    bldg_df = _building_centroids(chosen.building_id.unique())
    df = (
        chosen.join(pano_df, on="pano_id")
        .join(bldg_df, on="building_id")
        .reset_index(drop=True)
    )

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        executor = partial(_clip_row, region)
        futures = [pool.submit(executor, row) for _, row in df.iterrows()]
        results = [fut.result() for fut in as_completed(futures)]

    success_results = [r for r in results if r.status == "success"]
    counts = {
        status: len([r for r in results if r.status == status])
        for status in ["success", "skip", "missing", "clip_fail", "fail"]
    }

    if success_results:
        gdf = gpd.GeoDataFrame(
            [r.__dict__ for r in success_results],
            geometry="geom",
            crs=4326,
        )
        gdf.to_postgis(
            "panorama_clipped_views",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=5_000,
            dtype={"geom": Geometry("GEOMETRY", srid=4326)},
        )
        _ensure_gist_idx("panorama_clipped_views")

    logger.success(f"{region}: {', '.join(f'{k}={v}' for k, v in counts.items())}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Clip panoramas per region")
    ap.add_argument("--region", choices=REGIONS)
    ap.add_argument(
        "--log-level",
        default="info",
        choices=["trace", "debug", "info", "warning", "error", "critical"],
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    # wipe existing rows for selected region(s) to avoid duplicates – only if table exists
    from sqlalchemy import inspect

    insp = inspect(engine)
    if insp.has_table("panorama_clipped_views"):
        with engine.begin() as cn:
            tbl = Table("panorama_clipped_views", MetaData(), autoload_with=engine)
            if args.region:
                cn.execute(tbl.delete().where(tbl.c.region == args.region))
            else:
                cn.execute(tbl.delete())

    for r in [args.region] if args.region else REGIONS:
        _run_region(r)

    logger.info("Stage‑04 complete")


if __name__ == "__main__":  # pragma: no cover
    main()
