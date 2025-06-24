#!/usr/bin/env python
"""Stage-03: Clip panoramas to buildings using visibility-based bounds and DSM heights."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from PIL import Image
from shapely.geometry import LineString, Point
from shapely.wkt import loads as wkt_loads

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from floor_heights.config import CONFIG, REGIONS
from floor_heights.db.schemas import (
    BatchWriter,
    Stage03ClipRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import read_table, validate_file_exists_and_valid
from floor_heights.utils.progress import processing_progress

CLIP_UPPER_PROP = CONFIG.constants.clip_upper_prop
CLIP_LOWER_PROP = CONFIG.constants.clip_lower_prop
ANGLE_EXTEND = CONFIG.constants.angle_extend
Image.MAX_IMAGE_PIXELS = CONFIG.constants.max_image_pixels

_geo_path = Path(__file__).parent.parent / "utils" / "geometry.py"
_spec = importlib.util.spec_from_file_location("geometry", _geo_path)
assert _spec is not None
geo = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(geo)


def _region_folder(name: str) -> Path:
    return CONFIG.output_root / name.capitalize()


def _jpg(name: str) -> str:
    return name if name.lower().endswith(".jpg") else f"{name}.jpg"


def _clip_path(
    region: str, row_id: str, bid: str, pid: str, edge_idx: int, view_type: str, gnaf_id: str | None = None
) -> Path:
    """Create clip path following stage03 pattern (returns relative path)."""
    base_pid = pid[:-4] if pid.lower().endswith(".jpg") else pid
    filename = f"{base_pid}_edge{edge_idx}_{view_type}"
    gnaf_id = gnaf_id if gnaf_id and pd.notna(gnaf_id) else "NO_GNAF"
    return Path(region.capitalize()) / "clips" / f"{row_id}_{bid}_{gnaf_id}" / _jpg(filename)


def get_downloaded_views(region: str) -> pd.DataFrame:
    """Get successfully downloaded views from stage02b."""
    df = read_table("stage02b_downloads", region=region)

    if df.empty:
        return pd.DataFrame()

    candidates = read_table("stage02a_candidate_views", region=region, filters={"is_chosen": True})

    merged = df.merge(candidates, on=["id", "building_id", "gnaf_id", "pano_id", "edge_idx", "view_type"], how="inner")

    return merged


def get_region_crs(region: str) -> int:
    """Get the projected CRS for a region."""
    return CONFIG.regions[region].crs_projected


def _calculate_visibility_based_clipping_bounds(
    pano_lat: float,
    pano_lon: float,
    building_polygon_wkt: str,
    heading: float,
    img_width: int,
    geo_module,
    region_crs: int,
) -> tuple[float, float]:
    """Calculate clipping bounds based on visible vertices of the building.

    Works in MGA coordinates for precision, but uses lat/lon for panorama calculations.
    """
    adjusted_heading = -heading
    if adjusted_heading < 0:
        adjusted_heading += 360

    building_polygon = wkt_loads(building_polygon_wkt)

    pano_gdf = gpd.GeoDataFrame([{"geometry": Point(pano_lon, pano_lat)}], crs=CONFIG.crs.wgs84)
    pano_mga = pano_gdf.to_crs(region_crs).geometry[0]

    building_gdf = gpd.GeoDataFrame([{"geometry": building_polygon}], crs=region_crs)
    building_latlon = building_gdf.to_crs(CONFIG.crs.wgs84).geometry[0]
    vertices_latlon = list(building_latlon.exterior.coords[:-1])
    vertices_mga = list(building_polygon.exterior.coords[:-1])

    visible_pixels = []

    for vertex_mga, vertex_latlon in zip(vertices_mga, vertices_latlon, strict=False):
        vertex_point_mga = Point(vertex_mga)
        ray = LineString([pano_mga, vertex_point_mga])

        ray_coords = list(ray.coords)
        shortened_ray = LineString(
            [
                ray_coords[0],
                (
                    ray_coords[0][0] + CONFIG.panorama.ray_shortening_factor * (ray_coords[1][0] - ray_coords[0][0]),
                    ray_coords[0][1] + CONFIG.panorama.ray_shortening_factor * (ray_coords[1][1] - ray_coords[0][1]),
                ),
            ]
        )

        intersection = shortened_ray.intersection(building_polygon)

        is_visible = intersection.is_empty or (
            intersection.geom_type == "Point"
            and vertex_point_mga.distance(intersection) > CONFIG.panorama.visibility_distance_threshold
        )

        if is_visible:
            result = geo_module.localize_house_in_panorama(
                lat_c=pano_lat,
                lon_c=pano_lon,
                lat_house=vertex_latlon[1],
                lon_house=vertex_latlon[0],
                beta_yaw_deg=adjusted_heading,
                wim=img_width,
                angle_extend=0,
            )
            visible_pixels.append(result["horizontal_pixel_house"])

    if not visible_pixels:
        logger.warning("No visible vertices found, using all vertices")
        visible_pixels = [
            geo_module.localize_house_in_panorama(
                lat_c=pano_lat,
                lon_c=pano_lon,
                lat_house=vertex[1],
                lon_house=vertex[0],
                beta_yaw_deg=adjusted_heading,
                wim=img_width,
                angle_extend=0,
            )["horizontal_pixel_house"]
            for vertex in vertices_latlon
        ]

    left_x = min(visible_pixels)
    right_x = max(visible_pixels)

    span = right_x - left_x
    buffer = max(CONFIG.clipping.min_buffer_pixels, int(span * CONFIG.clipping.span_buffer_percent))

    left_x = max(0, left_x - buffer)
    right_x = min(img_width, right_x + buffer)

    return (float(left_x), float(right_x))


def _crop(img: Image.Image, h_range: tuple[float, float], v_range: tuple[int, int] | None = None) -> Image.Image:
    left, right = map(round, h_range)

    if v_range is not None:
        top, bottom = v_range
    else:
        top = round(CLIP_UPPER_PROP * img.height)
        bottom = round(CLIP_LOWER_PROP * img.height)

    if left >= right or top >= bottom:
        raise ValueError("invalid crop window")
    return img.crop((int(left), int(top), int(right), int(bottom)))


def _clip_row_worker(
    args_tuple,
) -> tuple[str, Stage03ClipRecord | None]:
    """Worker function for process pool.

    Returns:
        Tuple of (status, Stage03ClipRecord or None)
        Status can be: 'success', 'skip', 'missing', 'clip_fail', 'fail'
    """
    region, row_data, geo_path, processed_set = args_tuple

    _spec = importlib.util.spec_from_file_location("geometry", geo_path)
    assert _spec is not None
    geo = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(geo)

    class RowLike:
        def __init__(self, data):
            self._data = data

        def __getattr__(self, name):
            return self._data.get(name)

    row = RowLike(row_data)

    row_id = str(row.id)
    gnaf_id = row.gnaf_id if pd.notna(row.gnaf_id) else "NO_GNAF"

    out_path = _clip_path(region, row_id, row.building_id, row.pano_id, row.edge_idx, row.view_type, row.gnaf_id)
    abs_out_path = CONFIG.output_root / out_path

    clip_key = (row_id, row.pano_id, row.edge_idx, row.view_type)
    if clip_key in processed_set and validate_file_exists_and_valid(
        abs_out_path, file_type="image", min_size_bytes=1000
    ):
        return ("skip", None)

    pano_file = CONFIG.output_root / row.download_path

    if not validate_file_exists_and_valid(pano_file, file_type="image", min_size_bytes=1000):
        logger.debug(f"Panorama file not found or corrupted: {pano_file}")
        return ("missing", None)

    try:
        with Image.open(pano_file) as img:
            img = img.copy()

            h_range = _calculate_visibility_based_clipping_bounds(
                pano_lat=row.pano_lat,
                pano_lon=row.pano_lon,
                building_polygon_wkt=row.footprint_geom_mga,
                heading=row.pano_heading,
                img_width=img.width,
                geo_module=geo,
                region_crs=get_region_crs(region),
            )

            clip_top = round(CLIP_UPPER_PROP * img.height)
            clip_bottom = round(CLIP_LOWER_PROP * img.height)
            v_range = (clip_top, clip_bottom)

            clip = _crop(img, h_range, v_range)
            abs_out_path.parent.mkdir(parents=True, exist_ok=True)
            clip.save(abs_out_path, quality=CONFIG.clipping.jpeg_quality, optimize=CONFIG.clipping.jpeg_optimize)
            clip.close()

            if not validate_file_exists_and_valid(abs_out_path, file_type="image", min_size_bytes=1000):
                logger.error(f"Saved clip appears corrupted: {abs_out_path}")
                abs_out_path.unlink(missing_ok=True)
                return ("fail", None)

        record = Stage03ClipRecord(
            id=row_id,
            building_id=row.building_id,
            region_name=region,
            gnaf_id=gnaf_id,
            pano_id=row.pano_id,
            edge_idx=row.edge_idx,
            view_type=row.view_type,
            clip_path=str(out_path),
            clip_left=h_range[0],
            clip_right=h_range[1],
            clip_top=float(clip_top),
            clip_bottom=float(clip_bottom),
        )
        return ("success", record)
    except ValueError as ve:
        logger.error(f"Clip failed for {row.building_id}/{row.pano_id}: {ve}")
        return ("clip_fail", None)
    except Exception as exc:
        logger.error(f"{region} {row.building_id}/{row.pano_id}: {exc}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return ("fail", None)


def get_processed_clips(region: str) -> set[tuple]:
    """Get set of already processed clips as (id, pano_id, edge_idx, view_type) tuples."""
    try:
        df = read_table("stage03_clips", region=region)

        if df.empty:
            return set()

        return set(
            zip(
                df["id"],
                df["pano_id"],
                df["edge_idx"],
                df["view_type"],
                strict=False,
            )
        )
    except Exception:
        return set()


def process_region(region: str, workers: int = -1) -> None:
    """Process panorama clipping for single region."""
    logger.info(f"Processing {region}")

    df = get_downloaded_views(region)
    if df.empty:
        logger.warning(f"{region}: no downloaded views found")
        return

    df["clip_key"] = df.apply(lambda r: (r["id"], r["pano_id"], int(r["edge_idx"]), r["view_type"]), axis=1)
    unique_clips = df["clip_key"].nunique()
    logger.info(f"Found {unique_clips} unique clips to process ({len(df)} total rows)")

    processed_clips = get_processed_clips(region)
    logger.info(f"Found {len(processed_clips)} already processed clips")

    clips_to_check = []
    not_in_db = 0
    file_missing = 0

    for row in df.itertuples(index=False):
        clip_key = (row.id, row.pano_id, int(row.edge_idx), row.view_type)
        out_path = _clip_path(
            region,
            str(row.id),
            row.building_id,
            row.pano_id,
            row.edge_idx,
            row.view_type,
            getattr(row, "gnaf_id", None),
        )
        abs_out_path = CONFIG.output_root / out_path

        in_db = clip_key in processed_clips
        file_exists = validate_file_exists_and_valid(abs_out_path, file_type="image", min_size_bytes=1000)

        if not in_db or not file_exists:
            clips_to_check.append(row._asdict())
            if not in_db:
                not_in_db += 1
            if in_db and not file_exists:
                file_missing += 1

    logger.debug(f"Clips not in DB: {not_in_db}, Files missing: {file_missing}")

    if not clips_to_check:
        logger.success(f"{region}: All clips already processed")
        return

    df_to_process = pd.DataFrame(clips_to_check)
    logger.info(f"{region}: {len(df_to_process)} clips to process")

    if workers <= 0:
        workers = CONFIG.constants.default_workers
        if workers <= 0:
            workers = mp.cpu_count()

    logger.info(f"Using {workers} worker(s)")

    geo_path = Path(__file__).parent.parent / "utils" / "geometry.py"

    with (
        processing_progress(f"Clipping {region}", len(df_to_process)) as prog,
        BatchWriter("stage03_clips", batch_size=5000, progress_tracker=prog) as writer,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {}
        for row in df_to_process.itertuples(index=False):
            args = (region, row._asdict(), geo_path, processed_clips)
            future = pool.submit(_clip_row_worker, args)
            futures[future] = row

        for future in as_completed(futures):
            try:
                row = futures[future]
                status, record_obj = future.result()

                if status == "success" and record_obj:
                    writer.add(record_obj)
                    prog.update("suc", 1)
                elif status == "skip":
                    prog.update("skp", 1)
                elif status == "missing":
                    prog.update("fail", 1)
                else:
                    prog.update("fail", 1)

            except Exception as e:
                logger.debug(f"Worker failed: {e}")
                prog.update("fail", 1)

    logger.success(f"{region}: {prog.get_summary()}")


def run_stage(region: str | None = None, workers: int = -1) -> None:
    """Run stage03 with the given parameters.

    Args:
        region: Single region to process (None for all)
        workers: Number of workers (-1 for default)
    """
    initialize_all_stage_tables()

    try:
        if region:
            process_region(region, workers)
        else:
            logger.info(f"Processing {len(REGIONS)} regions: {', '.join(REGIONS)}")
            for r in REGIONS:
                process_region(r, workers)
        logger.info("Stage-03 complete")
    except Exception as e:
        logger.error(f"Stage-03 failed: {e}")
        raise


if __name__ == "__main__":
    run_stage()
