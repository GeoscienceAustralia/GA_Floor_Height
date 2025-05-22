#!/usr/bin/env python
"""Stage-03a: Ray-casting panorama geometry harvesting with self-occlusion avoidance."""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, cast

import geopandas as gpd
import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import yaml
from geoalchemy2 import Geometry
from loguru import logger
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from shapely.ops import nearest_points
from sqlalchemy import (
    Index,
    MetaData,
    Table,
    create_engine,
    func,
    select,
    inspect,
)

project_root = Path(__file__).resolve().parents[3]
common_cfg_path = project_root / "config" / "common.yaml"
common_cfg = (
    yaml.safe_load(common_cfg_path.read_text()) if common_cfg_path.exists() else {}
)
region_choices = common_cfg.get("regions", [])

DB = os.getenv("DB_CONNECTION_STRING")
if not DB:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")

MAX_DIST_M = 40
DIRECT_TOL_DEG = 15.0
MIN_OBL_DEG = 45.0
MAX_OBL_DEG = 50
WRITE_CHUNK = 5_000
EPS = 1e-8

# refined blocking thresholds for self-occlusion detection
EPS_LEN = 0.5  # metres: ignore intersections very near midpoint
EPS_AREA = 1e-4  # m²: treat overlaps > EPS_AREA as "self"

engine = create_engine(DB, future=True, pool_pre_ping=True)
meta = MetaData()

tbl_regions = Table("regions", meta, autoload_with=engine)
tbl_footprints = Table("building_footprints", meta, autoload_with=engine)
tbl_traj = Table("trajectory_processed", meta, autoload_with=engine)
tbl_bp_proc = Table("building_points_processed", meta, autoload_with=engine)


def edge_midpoints_and_normals(poly: sg.Polygon) -> List[Dict]:
    """Extract edge midpoints and outward-facing normals from polygon."""
    centre = np.array(poly.centroid.coords[0][:2])
    coords = list(poly.exterior.coords)
    out: List[Dict] = []

    for i in range(len(coords) - 1):
        p1, p2 = np.array(coords[i][:2]), np.array(coords[i + 1][:2])
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue
        norm = np.array([vec[1], -vec[0]]) / length
        mid = (p1 + p2) / 2
        if np.dot(norm, centre - mid) > 0:
            norm = -norm
        out.append(
            dict(
                p1=sg.Point(p1),
                p2=sg.Point(p2),
                mid=sg.Point(mid),
                norm=norm,
                length=length,
            )
        )
    return out


def signed_angle(norm: np.ndarray, vec: np.ndarray) -> float:
    """Compute signed angle between normal and vector in degrees."""
    return math.degrees(math.atan2(norm[0] * vec[1] - norm[1] * vec[0], norm.dot(vec)))


def sector_polygon(
    centre: sg.Point,
    normal: np.ndarray,
    ang_min: float,
    ang_max: float,
    radius: float,
    n_pts: int = 16,
) -> sg.Polygon:
    """Generate viewing sector polygon for ray search."""

    def _vec(a_deg: float) -> np.ndarray:
        θ = math.radians(a_deg)
        rot = np.array([[math.cos(θ), -math.sin(θ)], [math.sin(θ), math.cos(θ)]])
        return np.asarray(normal @ rot.T)

    arc = [
        centre.coords[0] + radius * _vec(a)
        for a in np.linspace(ang_min, ang_max, n_pts)
    ]
    return sg.Polygon([centre.coords[0], *arc]).buffer(0)


def _search_area_wkt(fp: gpd.GeoDataFrame, crs: int) -> str:
    """Create buffered search area WKT for trajectory queries."""
    hull = so.unary_union(fp.geometry).convex_hull.buffer(MAX_DIST_M)
    geom = gpd.GeoSeries([hull], crs=crs).to_crs(4326).iloc[0]
    return cast(str, geom.wkt)


def read_footprints(region_id: int, crs: int) -> gpd.GeoDataFrame:
    """Load all building footprints for region in target CRS."""
    stmt = select(tbl_footprints.c.id, tbl_footprints.c.geom).where(
        tbl_footprints.c.region_id == region_id
    )
    with engine.connect() as cn:
        gdf = gpd.read_postgis(stmt, cn, geom_col="geom")
    return gdf.set_crs(4326).to_crs(crs)


def read_trajectories(region_id: int, search_wkt: str, crs: int) -> gpd.GeoDataFrame:
    """Load trajectory points within search area."""
    area = func.ST_GeomFromText(search_wkt, 4326)
    stmt = select(tbl_traj.c.imgid, tbl_traj.c.geom).where(
        tbl_traj.c.region_id == region_id,
        func.ST_Intersects(tbl_traj.c.geom, area),
    )
    with engine.connect() as cn:
        gdf = gpd.read_postgis(stmt, cn, geom_col="geom")
    return gdf.set_crs(4326).to_crs(crs)


def processed_footprint_ids(region_id: int) -> set[int]:
    """Get processed building footprint IDs for region."""
    stmt = select(tbl_bp_proc.c.footprint_id).where(
        tbl_bp_proc.c.region_id == region_id
    )
    with engine.connect() as cn:
        return {fid for (fid,) in cn.execute(stmt)}


def write_table(gdf: gpd.GeoDataFrame, name: str, region: str) -> None:
    """Write geodataframe to PostGIS table with spatial index."""
    if gdf.empty:
        logger.warning(f"{region}: nothing to write → {name}")
        return

    gdf = gdf.to_crs(4326)
    if gdf.geometry.name != "geom":
        gdf = gdf.rename_geometry("geom")

    with engine.begin() as cn:
        gdf.to_postgis(
            name=name,
            con=cn,
            if_exists="append",
            index=False,
            chunksize=WRITE_CHUNK,
            dtype={"geom": Geometry("GEOMETRY", srid=4326)},
        )

    idx = Index(
        f"{name}_geom_idx",
        Table(name, MetaData(), autoload_with=engine).c.geom,
        postgresql_using="gist",
    )
    with engine.begin() as cn:
        idx.create(bind=cn, checkfirst=True)

    logger.info(f"{region}: wrote {len(gdf):,} rows → {name}")


# multiprocessing globals
TRAJ: gpd.GeoDataFrame
TRAJ_SINDEX: gpd.sindex.SpatialIndex
FP_GEOM: gpd.GeoSeries
FP_SINDEX: gpd.sindex.SpatialIndex
ID_LOOKUP: np.ndarray
REGION: str


def _init_worker(
    traj: gpd.GeoDataFrame, fp_geom: gpd.GeoSeries, id_lookup: np.ndarray, region: str
) -> None:
    """Initialize worker process globals."""
    global TRAJ, TRAJ_SINDEX, FP_GEOM, FP_SINDEX, ID_LOOKUP, REGION
    TRAJ = traj
    TRAJ_SINDEX = traj.sindex
    FP_GEOM = fp_geom
    FP_SINDEX = fp_geom.sindex
    ID_LOOKUP = id_lookup
    REGION = region


def _ray_blocked(ray: sg.LineString, building_id: int, hull: sg.Polygon) -> bool:
    """Check if ray is blocked by another footprint (avoids self-occlusion)."""
    try:
        idxs = list(FP_SINDEX.intersection(ray.bounds))
    except Exception:
        idxs = []

    target_len = ray.length
    for i in idxs:
        fid = ID_LOOKUP[i]
        if fid == building_id:
            continue

        geom = FP_GEOM.iloc[i]

        # skip self-overlap / façade skirt
        if geom.intersection(hull).area > EPS_AREA:
            continue

        if not ray.crosses(geom) and not ray.within(geom):
            continue

        inter_pt, _ = nearest_points(ray, geom)
        d_inter = ray.project(inter_pt)

        if d_inter + EPS_LEN < target_len:
            return True
    return False


def _process_single_footprint(
    data: Tuple[int, sg.base.BaseGeometry],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Process single building footprint to generate viewing rays."""
    building_id, geom = data
    hull = so.orient(geom.convex_hull, sign=1.0)

    bldg_geoms: List[Dict] = [
        dict(
            building_id=building_id,
            region=REGION,
            geom_type="footprint_hull",
            geom=hull,
        )
    ]
    derived_geoms: List[Dict] = []
    cand_views: List[Dict] = []

    search_specs = {
        "direct": (-DIRECT_TOL_DEG, DIRECT_TOL_DEG),
        "oblique_left": (MIN_OBL_DEG, MAX_OBL_DEG),
        "oblique_right": (-MAX_OBL_DEG, -MIN_OBL_DEG),
    }

    for e_idx, edge in enumerate(edge_midpoints_and_normals(hull)):
        bldg_geoms.extend(
            [
                dict(
                    building_id=building_id,
                    region=REGION,
                    geom_type="edge",
                    edge_idx=e_idx,
                    geom=sg.LineString([edge["p1"], edge["p2"]]),
                ),
                dict(
                    building_id=building_id,
                    region=REGION,
                    geom_type="vertex",
                    edge_idx=e_idx,
                    geom=edge["p1"],
                ),
                dict(
                    building_id=building_id,
                    region=REGION,
                    geom_type="vertex",
                    edge_idx=e_idx,
                    geom=edge["p2"],
                ),
                dict(
                    building_id=building_id,
                    region=REGION,
                    geom_type="midpoint",
                    edge_idx=e_idx,
                    geom=edge["mid"],
                ),
            ]
        )

        for vtype, (a_min, a_max) in search_specs.items():
            derived_geoms.append(
                dict(
                    building_id=building_id,
                    region=REGION,
                    geom_type="search_sector",
                    view_type=vtype,
                    edge_idx=e_idx,
                    geom=sector_polygon(
                        edge["mid"], edge["norm"], a_min, a_max, MAX_DIST_M
                    ),
                )
            )

        buf = edge["mid"].buffer(MAX_DIST_M)
        idx = TRAJ_SINDEX.query(buf, predicate="intersects")
        if idx.size == 0:
            continue

        subset = TRAJ.iloc[idx]
        cands = subset[subset.geometry.within(buf)]
        if cands.empty:
            continue

        best: Dict[str, int | None] = {
            "direct": None,
            "oblique_left": None,
            "oblique_right": None,
        }

        for _, crow in cands.iterrows():
            pano_pt = crow.geom
            if hull.contains(pano_pt):
                continue

            vec = np.array([pano_pt.x - edge["mid"].x, pano_pt.y - edge["mid"].y])
            if np.dot(edge["norm"], vec) <= 0:
                continue

            ray = sg.LineString(
                [(pano_pt.x, pano_pt.y), (edge["mid"].x, edge["mid"].y)]
            )
            if _ray_blocked(ray, building_id, hull):
                continue

            dist = ray.length
            if dist > MAX_DIST_M or dist < 1e-6:
                continue

            ang = signed_angle(edge["norm"], vec)
            a_abs = abs(ang)
            if a_abs > MAX_OBL_DEG + EPS:
                continue

            if a_abs <= DIRECT_TOL_DEG + EPS:
                vtype = "direct"
            elif ang >= MIN_OBL_DEG - EPS:
                vtype = "oblique_left"
            elif ang <= -MIN_OBL_DEG + EPS:
                vtype = "oblique_right"
            else:
                continue

            cand_views.append(
                dict(
                    building_id=building_id,
                    region=REGION,
                    edge_idx=e_idx,
                    pano_id=crow.imgid,
                    view_type=vtype,
                    angle=ang,
                    distance=dist,
                    is_chosen=False,
                    geom=ray,
                )
            )
            idx_new = len(cand_views) - 1
            prev = best[vtype]
            if prev is None or dist < cand_views[prev]["distance"]:
                best[vtype] = idx_new

        for i in best.values():
            if i is not None:
                cand_views[i]["is_chosen"] = True

    return bldg_geoms, derived_geoms, cand_views


def process_region(region: str) -> None:
    """Process panorama geometry harvesting for single region."""
    logger.info(f"Processing {region}")

    region_cfg_path = project_root / "config" / f"{region}.yaml"
    if not region_cfg_path.exists():
        raise FileNotFoundError(f"Region config not found: {region_cfg_path}")

    region_cfg = yaml.safe_load(region_cfg_path.read_text())
    crs = region_cfg.get("crs")
    if not crs:
        raise ValueError(f"CRS not specified in config for region: {region}")

    with engine.connect() as cn:
        region_id = cn.execute(
            select(tbl_regions.c.id).where(tbl_regions.c.name == region)
        ).scalar_one()

    target_ids = processed_footprint_ids(region_id)
    if not target_ids:
        logger.warning(f"{region}: no processed footprints")
        return

    fp_all = read_footprints(region_id, crs)
    fp_target = fp_all[fp_all.id.isin(target_ids)].reset_index(drop=True)
    if fp_target.empty:
        logger.warning(f"{region}: matching geometries not found")
        return

    search_wkt = _search_area_wkt(fp_target, crs)
    traj = read_trajectories(region_id, search_wkt, crs)
    if traj.empty:
        logger.warning(f"{region}: no panoramas within {MAX_DIST_M} m")
        return

    id_lookup = fp_all["id"].to_numpy()
    tasks = list(zip(fp_target["id"].values, fp_target.geometry.values))
    if not tasks:
        logger.warning(f"{region}: nothing to process")
        return

    workers = min(len(tasks), cpu_count() or 1)
    logger.info(f"{region}: using {workers} worker(s)")

    results: List[Tuple[List[Dict], List[Dict], List[Dict]]] = []
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(traj, fp_all.geometry, id_lookup, region),
        ) as pool,
        Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            transient=False,
        ) as prog,
    ):
        tid = prog.add_task(f"[cyan]{region}", total=len(tasks))
        for res in pool.map(_process_single_footprint, tasks):
            results.append(res)
            prog.advance(tid)

    bldg_geoms, derived_geoms, cand_views = [], [], []
    for b, d, c in results:
        bldg_geoms.extend(b)
        derived_geoms.extend(d)
        cand_views.extend(c)

    write_table(
        gpd.GeoDataFrame(bldg_geoms, geometry="geom", crs=crs),
        "panorama_building_geoms",
        region,
    )
    write_table(
        gpd.GeoDataFrame(derived_geoms, geometry="geom", crs=crs),
        "panorama_derived_geoms",
        region,
    )
    write_table(
        gpd.GeoDataFrame(cand_views, geometry="geom", crs=crs),
        "panorama_candidate_views",
        region,
    )

    logger.success(f"{region}: {len(cand_views):,} rays written")


def main() -> None:
    """CLI entry point for panorama geometry harvesting."""
    ap = argparse.ArgumentParser(description="Harvest panorama geometries per region")
    ap.add_argument("--region", choices=region_choices)
    ap.add_argument(
        "--log-level",
        default="info",
        choices=["trace", "debug", "info", "warning", "error", "critical"],
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    tables = [
        "panorama_building_geoms",
        "panorama_derived_geoms",
        "panorama_candidate_views",
    ]
    insp = inspect(engine)

    with engine.begin() as cn:
        for t in tables:
            if insp.has_table(t):
                tbl = Table(t, MetaData(), autoload_with=engine)
                if args.region:
                    cn.execute(tbl.delete().where(tbl.c.region == args.region))
                else:
                    cn.execute(tbl.delete())
                logger.info(f"Cleared {t} for region(s): {args.region or 'all'}")

    regs = [args.region] if args.region else region_choices
    for reg in regs:
        try:
            process_region(reg)
        except Exception as exc:
            logger.error(f"{reg}: {exc}")
            logger.exception("Traceback:")

    logger.info("Done.")


if __name__ == "__main__":
    main()
