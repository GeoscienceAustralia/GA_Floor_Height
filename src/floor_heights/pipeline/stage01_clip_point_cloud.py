#!/usr/bin/env python
"""Stage-01: Clip LiDAR tiles to residential footprints."""

from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import geopandas as gpd
from loguru import logger
from shapely.geometry import Polygon

from floor_heights.config import CONFIG, REGIONS
from floor_heights.db.schemas import (
    BatchWriter,
    Stage01ClipRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import (
    find_lidar_tiles,
    get_processed_stage_ids,
    las_path,
    read_table,
    validate_file_exists_and_valid,
)
from floor_heights.utils.progress import clip_progress

load_dotenv(find_dotenv(usecwd=True))

if not CONFIG.db_path.exists():
    raise ValueError(f"DuckDB database not found at {CONFIG.db_path}")

BUFFER_M = CONFIG.constants.buffer_m
LIDAR_DATA_ROOT = CONFIG.lidar_data_root


def load_tile_index(region: str) -> gpd.GeoDataFrame:
    """Load tileset index for the region and transform to projected CRS."""
    gdf = read_table("tilesets", region=region, as_geo=True, geom_col="geometry", crs=CONFIG.crs.geographic)

    gdf = gdf.rename(columns={"filename": "FileName"})
    gdf = gdf.to_crs(CONFIG.regions[region].crs_projected)

    return gdf


def residential_fp(region_name: str, target_crs: int) -> gpd.GeoDataFrame:
    """Load residential building footprints for the region."""
    gdf = read_table(
        "buildings",
        region=region_name,
        filters={"property_type": "Residential"},
        as_geo=True,
        geom_col="footprint_geom",
        crs=CONFIG.crs.geographic,
    )

    gdf = gdf.rename(columns={"footprint_geom": "geometry"})
    gdf = gdf.set_geometry("geometry")

    return gdf.to_crs(target_crs)


def pdal_pipeline(las_paths: list[Path], poly: Polygon, out_path: Path, crs: int) -> str:
    """Generate PDAL pipeline JSON for clipping LAS files."""
    readers = [{"type": "readers.las", "filename": str(p), "override_srs": f"EPSG:{crs}"} for p in las_paths]
    merge = [{"type": "filters.merge"}] if len(las_paths) > 1 else []
    crop = [{"type": "filters.crop", "polygon": poly.wkt}]

    writer = [
        {
            "type": "writers.las",
            "filename": str(out_path),
            "extra_dims": CONFIG.pdal.extra_dims,
            "forward": CONFIG.pdal.forward,
            "a_srs": f"EPSG:{crs}",
            "scale_x": CONFIG.pdal.scale_x,
            "scale_y": CONFIG.pdal.scale_y,
            "scale_z": CONFIG.pdal.scale_z,
            "offset_x": CONFIG.pdal.offset_x,
            "offset_y": CONFIG.pdal.offset_y,
            "offset_z": CONFIG.pdal.offset_z,
        }
    ]

    return json.dumps({"pipeline": readers + merge + crop + writer}, separators=(",", ":"))


def run_region(
    region: str,
    revision: str = "rev2",
    lidar_source: str = "s3",
    sample_size: int | None = None,
    workers: int = -1,
) -> None:
    """Process all residential footprints in a region.

    Args:
        region: Region name to process.
        revision: LiDAR revision identifier.
        lidar_source: Source for LiDAR files (``"local"`` or ``"s3"``).
        sample_size: If provided, limit processing to the first ``sample_size``
            buildings.
        workers: Number of worker threads to use (-1 to use ``cpu_count``).
    """
    logger.info(f"── Clipping {region} {f'({revision})' if revision else ''} [source: {lidar_source}] ──")

    tiles = load_tile_index(region)
    tiles_gdf = tiles

    target_crs = CONFIG.regions[region].crs_projected
    fps = residential_fp(region, target_crs)

    fps_len = len(fps)
    if fps_len == 0:
        logger.warning(f"{region}: no residential footprints; skipping")
        return

    if sample_size:
        unique_buildings = fps.drop_duplicates(subset=["building_id"], keep="first")
        if len(unique_buildings) < sample_size:
            logger.warning(f"Only {len(unique_buildings)} unique building_ids available, requested {sample_size}")
        fps = unique_buildings.head(sample_size)
        fps_len = len(fps)
        logger.info(f"Sampling {fps_len} buildings with unique building_ids")

    processed_ids = get_processed_stage_ids("stage01_clips", region)
    logger.info(f"Found {len(processed_ids)} already processed buildings")

    num_workers = mp.cpu_count() if workers < 1 else workers
    logger.info(f"Using {num_workers} worker(s)")

    db_lock = threading.Lock()

    def get_local_tiles(tile_hits) -> list[Path]:
        """Get paths to local LAS/LAZ tiles."""
        filenames = list(tile_hits["FileName"])
        use_s3 = lidar_source == "s3"
        return find_lidar_tiles(filenames, region, revision, use_s3=use_s3)

    def execute_pdal_clip(las_paths: list[Path], polygon: Polygon, out_path: Path) -> bool:
        """Execute PDAL pipeline to clip point cloud."""
        pipe = pdal_pipeline(las_paths, polygon, out_path, target_crs)
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as tmp:
            tmp.write(pipe)
            tmp.flush()
            result = subprocess.run(["pdal", "pipeline", tmp.name], capture_output=True)
            if result.returncode != 0:
                stderr = result.stderr.decode()
                if "Warning" in stderr and validate_file_exists_and_valid(
                    out_path, file_type="las", min_size_bytes=100
                ):
                    logger.debug(f"PDAL warning (but valid file created): {stderr}")
                    return True
                logger.error(f"PDAL failed: {stderr}")
                out_path.unlink(missing_ok=True)
                return False
            return True

    def clip_one(row: Any) -> tuple[str, Stage01ClipRecord | None]:
        """Process a single building footprint.

        Returns:
            Tuple of (status, Stage01ClipRecord or None)
        """
        row_id = row.id
        building_id = row.building_id
        gnaf_id = row.gnaf_id if hasattr(row, "gnaf_id") else None

        buf = row.geometry.simplify(CONFIG.pdal.simplification_tolerance).buffer(
            BUFFER_M, join_style=CONFIG.pdal.buffer_join_style
        )

        x1, y1, x2, y2 = buf.bounds
        hits = tiles_gdf.cx[x1:x2, y1:y2][tiles_gdf.cx[x1:x2, y1:y2].geometry.intersects(buf)]

        if hits.empty:
            return ("nt", None)

        las_paths = get_local_tiles(hits)
        if not las_paths:
            return ("mt", None)

        tile_count = len(las_paths)

        output_ext = ".laz" if any(p.suffix == ".laz" for p in las_paths) else ".las"
        out_path = las_path(region, row_id, building_id, gnaf_id, revision, output_ext, lidar_source)
        abs_path = CONFIG.output_root / out_path
        if row_id in processed_ids and validate_file_exists_and_valid(abs_path, file_type="las"):
            return ("skp", None)

        if execute_pdal_clip(las_paths, buf, abs_path):
            record = Stage01ClipRecord(
                id=row_id,
                building_id=building_id,
                region_name=region,
                gnaf_id=gnaf_id,
                clip_path=str(out_path),
                tile_count=tile_count,
            )
            return ("suc", record)

        return ("fail", None)

    with (
        clip_progress(region, fps_len) as prog,
        BatchWriter("stage01_clips", batch_size=100, progress_tracker=prog) as writer,
        ThreadPoolExecutor(max_workers=num_workers) as pool,
    ):
        future_to_row = {pool.submit(clip_one, row): row for _, row in fps.iterrows()}

        for fut in as_completed(future_to_row):
            try:
                future_to_row[fut]
                status, record_obj = fut.result()

                prog.update(status, 1)

                if status == "suc" and record_obj:
                    with db_lock:
                        writer.add(record_obj)
                        processed_ids.add(record_obj.id)

            except Exception as e:
                logger.error(f"Error processing building: {e}")
                prog.update("fail", 1)

    logger.info(f"{region}: {prog.get_summary()}")


def run_stage(
    region: str | None = None,
    workers: int = -1,
    revision: str = "rev2",
    lidar_source: str = "local",
    sample_size: int | None = None,
) -> None:
    """Run stage01 with the given parameters.

    Args:
        region: Single region to process (None for all)
        workers: Number of workers (-1 for CPU count)
        revision: Revision to use (e.g., 'rev1', 'rev2')
        lidar_source: Source for LiDAR files ('local' or 's3')
    """
    initialize_all_stage_tables()

    try:
        if region:
            run_region(region, revision, lidar_source, sample_size, workers)
        else:
            logger.info(f"Processing {len(REGIONS)} regions: {', '.join(REGIONS)}")
            for r in REGIONS:
                run_region(r, revision, lidar_source, sample_size, workers)
        logger.info("Stage-01 complete")
    except Exception as e:
        logger.error(f"Stage-01 failed: {e}")
        raise
