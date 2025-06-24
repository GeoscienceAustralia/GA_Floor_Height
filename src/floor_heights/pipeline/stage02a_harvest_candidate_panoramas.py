#!/usr/bin/env python
"""Stage-02a: Ray-casting panorama geometry harvesting with self-occlusion avoidance."""

from __future__ import annotations

import math
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import shapely.wkt
from loguru import logger
from shapely.ops import nearest_points

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from floor_heights.config import CONFIG, REGIONS
from floor_heights.db.schemas import (
    BatchWriter,
    Stage02aBuildingGeomRecord,
    Stage02aBuildingRecord,
    Stage02aCandidateViewRecord,
    Stage02aDerivedGeomRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import get_processed_stage_ids, read_table
from floor_heights.utils.progress import processing_progress

MAX_DIST_M = 40
DIRECT_TOL_DEG = 15.0
MIN_SEMI_OBL_DEG = 20.0
MAX_SEMI_OBL_DEG = 40.0
MIN_OBL_DEG = 45.0
MAX_OBL_DEG = 50
EPS = 1e-8

EPS_LEN = 0.5
EPS_AREA = 1e-4


def edge_midpoints_and_normals(poly: sg.Polygon) -> list[dict]:
    """Extract edge midpoints and outward-facing normals from polygon."""
    centre = np.array(poly.centroid.coords[0][:2])
    coords = list(poly.exterior.coords)
    out: list[dict] = []

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
            {
                "p1": sg.Point(p1),
                "p2": sg.Point(p2),
                "mid": sg.Point(mid),
                "norm": norm,
                "length": length,
            }
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
        for a in np.linspace(ang_min, ang_max, CONFIG.panorama.sector_polygon_points)
    ]
    return sg.Polygon([centre.coords[0], *arc]).buffer(0)


def _search_area_wkt(fp: gpd.GeoDataFrame, crs: int) -> str:
    """Create buffered search area WKT for trajectory queries."""
    hull = so.unary_union(fp.geometry).convex_hull.buffer(MAX_DIST_M)
    geom = gpd.GeoSeries([hull], crs=crs).to_crs(4326).iloc[0]
    return geom.wkt


def read_footprints(region_name: str, crs: int) -> gpd.GeoDataFrame:
    """Load all building footprints for region in target CRS."""
    gdf = read_table("buildings", region=region_name, as_geo=True, geom_col="footprint_geom", crs="EPSG:7844")
    gdf = gdf.rename(columns={"footprint_geom": "geometry"})
    gdf = gdf.set_geometry("geometry")
    return gdf.to_crs(crs)


def read_trajectories(region_name: str, search_wkt: str, crs: int) -> gpd.GeoDataFrame:
    """Load trajectory points within search area.

    Note: panoramas table uses 'geometry' column and is in WGS84
    """
    gdf = read_table("panoramas", region=region_name, as_geo=True, geom_col="geometry", crs="EPSG:4326")

    gdf = gdf.rename(columns={"imgID": "imgid", "geometry": "geom"})
    gdf = gdf.set_geometry("geom")

    search_geom = shapely.wkt.loads(search_wkt)
    gdf = gdf[gdf.geom.intersects(search_geom)]

    return gdf.to_crs(crs)


TRAJ: gpd.GeoDataFrame
TRAJ_SINDEX: gpd.sindex.SpatialIndex
FP_GEOM: gpd.GeoSeries
FP_SINDEX: gpd.sindex.SpatialIndex
ID_LOOKUP: np.ndarray
GNAF_LOOKUP: np.ndarray
REGION: str


def _init_worker(
    traj: gpd.GeoDataFrame, fp_geom: gpd.GeoSeries, id_lookup: np.ndarray, gnaf_lookup: np.ndarray, region: str
) -> None:
    """Initialize worker process globals."""
    global TRAJ, TRAJ_SINDEX, FP_GEOM, FP_SINDEX, ID_LOOKUP, GNAF_LOOKUP, REGION
    TRAJ = traj
    TRAJ_SINDEX = traj.sindex
    FP_GEOM = fp_geom
    FP_SINDEX = fp_geom.sindex
    ID_LOOKUP = id_lookup
    GNAF_LOOKUP = gnaf_lookup
    REGION = region


def _ray_blocked(ray: sg.LineString, row_id: int, hull: sg.Polygon) -> bool:
    """Check if ray is blocked by another footprint (avoids self-occlusion)."""
    try:
        idxs = list(FP_SINDEX.intersection(ray.bounds))
    except Exception:
        idxs = []

    target_len = ray.length
    for i in idxs:
        fid = ID_LOOKUP[i]
        if fid == row_id:
            continue

        geom = FP_GEOM.iloc[i]

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
    data: tuple[int, str, str, sg.base.BaseGeometry],
) -> tuple[
    list[Stage02aBuildingGeomRecord],
    list[Stage02aDerivedGeomRecord],
    list[Stage02aCandidateViewRecord],
]:
    """Process single building footprint to generate viewing rays."""
    row_id, building_id, gnaf_id, footprint = data
    gnaf_id = gnaf_id if gnaf_id else ""
    hull = so.orient(footprint.convex_hull, sign=1.0)

    bldg_geoms: list[Stage02aBuildingGeomRecord] = [
        Stage02aBuildingGeomRecord(
            id=row_id,
            building_id=building_id,
            region_name=REGION,
            gnaf_id=gnaf_id,
            geom_type="footprint_hull",
            view_type="",
            edge_idx=-1,
            geom_wkt=hull.wkt,
        )
    ]
    derived_geoms: list[Stage02aDerivedGeomRecord] = []
    cand_views: list[Stage02aCandidateViewRecord] = []

    search_specs = {
        "direct": (-DIRECT_TOL_DEG, DIRECT_TOL_DEG),
        "semi_oblique_left": (MIN_SEMI_OBL_DEG, MAX_SEMI_OBL_DEG),
        "semi_oblique_right": (-MAX_SEMI_OBL_DEG, -MIN_SEMI_OBL_DEG),
        "oblique_left": (MIN_OBL_DEG, MAX_OBL_DEG),
        "oblique_right": (-MAX_OBL_DEG, -MIN_OBL_DEG),
    }

    for e_idx, edge in enumerate(edge_midpoints_and_normals(hull)):
        bldg_geoms.extend(
            [
                Stage02aBuildingGeomRecord(
                    id=row_id,
                    building_id=building_id,
                    region_name=REGION,
                    gnaf_id=gnaf_id,
                    geom_type="edge",
                    view_type="",
                    edge_idx=e_idx,
                    geom_wkt=sg.LineString([edge["p1"], edge["p2"]]).wkt,
                ),
                Stage02aBuildingGeomRecord(
                    id=row_id,
                    building_id=building_id,
                    region_name=REGION,
                    gnaf_id=gnaf_id,
                    geom_type="vertex",
                    view_type="",
                    edge_idx=e_idx,
                    geom_wkt=edge["p1"].wkt,
                ),
                Stage02aBuildingGeomRecord(
                    id=row_id,
                    building_id=building_id,
                    region_name=REGION,
                    gnaf_id=gnaf_id,
                    geom_type="vertex",
                    view_type="",
                    edge_idx=e_idx,
                    geom_wkt=edge["p2"].wkt,
                ),
                Stage02aBuildingGeomRecord(
                    id=row_id,
                    building_id=building_id,
                    region_name=REGION,
                    gnaf_id=gnaf_id,
                    geom_type="midpoint",
                    view_type="",
                    edge_idx=e_idx,
                    geom_wkt=edge["mid"].wkt,
                ),
            ]
        )

        for vtype, (a_min, a_max) in search_specs.items():
            derived_geoms.append(
                Stage02aDerivedGeomRecord(
                    id=row_id,
                    building_id=building_id,
                    region_name=REGION,
                    gnaf_id=gnaf_id,
                    geom_type="search_sector",
                    view_type=vtype,
                    edge_idx=e_idx,
                    geom_wkt=sector_polygon(edge["mid"], edge["norm"], a_min, a_max, MAX_DIST_M).wkt,
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

        best: dict[str, int | None] = {
            "direct": None,
            "semi_oblique_left": None,
            "semi_oblique_right": None,
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

            ray = sg.LineString([(pano_pt.x, pano_pt.y), (edge["mid"].x, edge["mid"].y)])
            if _ray_blocked(ray, row_id, hull):
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
            elif MIN_SEMI_OBL_DEG - EPS <= ang <= MAX_SEMI_OBL_DEG + EPS:
                vtype = "semi_oblique_left"
            elif -MAX_SEMI_OBL_DEG - EPS <= ang <= -MIN_SEMI_OBL_DEG + EPS:
                vtype = "semi_oblique_right"
            elif ang >= MIN_OBL_DEG - EPS:
                vtype = "oblique_left"
            elif ang <= -MIN_OBL_DEG + EPS:
                vtype = "oblique_right"
            else:
                continue

            cand_views.append(
                Stage02aCandidateViewRecord(
                    id=row_id,
                    building_id=building_id,
                    region_name=REGION,
                    gnaf_id=gnaf_id,
                    pano_id=crow.imgid,
                    edge_idx=e_idx,
                    view_type=vtype,
                    distance=dist,
                    angle=ang,
                    is_chosen=False,
                    ray_wkt=ray.wkt,
                    pano_lat=crow.Latitude_deg,
                    pano_lon=crow.Longitude_deg,
                    pano_heading=crow.Heading_deg,
                    footprint_geom_mga=footprint.wkt,
                )
            )
            idx_new = len(cand_views) - 1
            prev = best[vtype]
            if prev is None or dist < cand_views[prev].distance:
                best[vtype] = idx_new

        for i in best.values():
            if i is not None:
                cand_views[i].is_chosen = True

    return bldg_geoms, derived_geoms, cand_views


def process_region(region: str, sample_size: int | None = None) -> None:
    """Process panorama geometry harvesting for single region."""
    logger.info(f"Processing {region}")

    region_config = CONFIG.regions.get(region)
    if not region_config:
        raise ValueError(f"Region not configured: {region}")
    crs = region_config.crs_projected

    fp_all = read_footprints(region, crs)
    if fp_all.empty:
        logger.warning(f"{region}: no buildings found")
        return

    stage01_processed = get_processed_stage_ids("stage01_clips", region)
    if not stage01_processed:
        logger.warning(f"{region}: no buildings from stage01 found. Run stage01 first.")
        return

    fp_all = fp_all[fp_all["id"].isin(stage01_processed)]
    logger.info(f"Found {len(fp_all)} buildings from stage01")

    processed_ids = get_processed_stage_ids("stage02a_buildings", region)
    logger.info(f"Found {len(processed_ids)} already processed in stage02a")

    fp_target = fp_all[~fp_all["id"].isin(processed_ids)]

    if sample_size:
        fp_target = fp_target.head(sample_size)
        logger.info(f"Processing sample of {len(fp_target)} buildings")

    if fp_target.empty:
        logger.info(f"{region}: all buildings already processed")
        return

    search_wkt = _search_area_wkt(fp_target, crs)
    traj = read_trajectories(region, search_wkt, crs)
    if traj.empty:
        logger.warning(f"{region}: no panoramas within {MAX_DIST_M} m")
        return

    id_lookup = fp_all["id"].to_numpy()
    gnaf_lookup = fp_all["gnaf_id"].to_numpy()
    tasks = list(
        zip(
            fp_target["id"].values,
            fp_target["building_id"].values,
            fp_target["gnaf_id"].values,
            fp_target.geometry.values,
            strict=False,
        )
    )
    if not tasks:
        logger.warning(f"{region}: nothing to process")
        return

    workers = min(len(tasks), cpu_count() or 1)
    logger.info(f"{region}: using {workers} worker(s)")

    db_lock = threading.Lock()

    total_bldg_geoms = 0
    total_derived_geoms = 0
    total_cand_views = 0
    buildings_processed = 0

    writers = {}

    def process_building_results(
        task_data: tuple[str, str, str, sg.base.BaseGeometry],
        result: tuple[
            list[Stage02aBuildingGeomRecord],
            list[Stage02aDerivedGeomRecord],
            list[Stage02aCandidateViewRecord],
        ],
    ) -> None:
        """Process results for a single building."""
        nonlocal total_bldg_geoms, total_derived_geoms, total_cand_views, buildings_processed

        row_id, building_id, gnaf_id, _ = task_data
        bldg_geoms, derived_geoms, cand_views = result

        chosen_count = sum(1 for view in cand_views if view.is_chosen)
        writers["buildings"].add(
            Stage02aBuildingRecord(
                id=row_id,
                building_id=building_id,
                region_name=region,
                gnaf_id=gnaf_id if gnaf_id else "",
                candidate_count=len(cand_views),
                chosen_count=chosen_count,
            )
        )

        for geom in bldg_geoms:
            writers["building_geoms"].add(geom)

        for geom in derived_geoms:
            writers["derived_geoms"].add(geom)

        for view in cand_views:
            writers["candidate_views"].add(view)

        with db_lock:
            total_bldg_geoms += len(bldg_geoms)
            total_derived_geoms += len(derived_geoms)
            total_cand_views += len(cand_views)
            buildings_processed += 1

    with processing_progress(f"{region} harvesting", len(tasks)) as prog:
        writers["buildings"] = BatchWriter("stage02a_buildings", batch_size=5000, progress_tracker=prog)

        writers["building_geoms"] = BatchWriter("stage02a_building_geoms", batch_size=100000, progress_tracker=prog)

        writers["derived_geoms"] = BatchWriter("stage02a_derived_geoms", batch_size=100000, progress_tracker=prog)

        writers["candidate_views"] = BatchWriter("stage02a_candidate_views", batch_size=500000, progress_tracker=prog)

        with (
            ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(traj, fp_all.geometry, id_lookup, gnaf_lookup, region),
            ) as pool,
            writers["buildings"],
            writers["building_geoms"],
            writers["derived_geoms"],
            writers["candidate_views"],
        ):
            futures = {pool.submit(_process_single_footprint, task): i for i, task in enumerate(tasks)}

            for future in as_completed(futures):
                try:
                    i = futures[future]
                    result = future.result()

                    process_building_results(tasks[i], result)
                    prog.update("suc", 1)

                except Exception as e:
                    import traceback

                    logger.error(f"Error processing building: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    prog.update("fail", 1)

    logger.success(
        f"{region}: Saved {total_cand_views:,} candidate views, {total_bldg_geoms:,} building geoms, {total_derived_geoms:,} derived geoms for {buildings_processed} buildings"
    )


def run_stage(region: str | None = None, workers: int = -1, sample_size: int | None = None) -> None:
    """Run stage02a with the given parameters.

    Args:
        region: Single region to process (None for all)
        workers: Number of workers (-1 for CPU count)
        sample_size: Process only first N buildings per region
    """
    initialize_all_stage_tables()

    try:
        if region:
            process_region(region, sample_size)
        else:
            logger.info(f"Processing {len(REGIONS)} regions: {', '.join(REGIONS)}")
            for r in REGIONS:
                process_region(r, sample_size)
        logger.info("Stage-02a complete")
    except Exception as e:
        logger.error(f"Stage-02a failed: {e}")
        raise
