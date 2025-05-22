#!/usr/bin/env python
"""Stage-05: Clip LiDAR tiles to residential footprints."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import geopandas as gpd
import yaml
from botocore.exceptions import ClientError
from dotenv import find_dotenv, load_dotenv
from geoalchemy2 import Geometry  # noqa: F401
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from shapely.geometry import Polygon
from sqlalchemy import MetaData, Table, create_engine, select

load_dotenv(find_dotenv(usecwd=True))

DB = os.getenv("DB_CONNECTION_STRING")
if not DB:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")

cfg_path = Path(__file__).resolve().parents[3] / "config" / "common.yaml"
cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
OUTPUT_ROOT = Path(cfg.get("output_root", "output")).expanduser()
REGIONS: List[str] = cfg.get("regions", [])

S3_BUCKET = "frontiersi-p127-floor-height-woolpert"
AWS_REGION = "ap-southeast-2"
BUFFER_M = 5.0

SPEC: Dict[str, Dict[str, Any]] = {
    "wagga": {
        "tile_index": "01_WaggaWagga/03_Ancillary/01_TileIndex/rev1/48068_Wagga_Wagga_TileSet.shp",
        "lidar_prefix": "01_WaggaWagga/02_MLSPointCloud/rev1/",
        "crs": 28355,
    },
    "tweed": {
        "tile_index": "02_TweedHeads/03_Ancillary/01_TileIndex/rev1/48068_Tweed_Heads_TileSet.shp",
        "lidar_prefix": "02_TweedHeads/02_MLSPointCloud/rev1/",
        "crs": 28356,
    },
    "launceston": {
        "tile_index": "03_Launceston/03_Ancillary/01_TileIndex/rev1/48068_Launceston_TileSet.shp",
        "lidar_prefix": "03_Launceston/02_MLSPointCloud/rev1/",
        "crs": 28355,
    },
}

console = Console(file=sys.stderr, force_terminal=True, force_interactive=True)
engine = create_engine(DB, future=True, pool_pre_ping=True)
meta = MetaData()

_tbl_regions = Table("regions", meta, autoload_with=engine)
_tbl_bp_proc = Table("building_points_processed", meta, autoload_with=engine)
_tbl_footprints = Table("building_footprints", meta, autoload_with=engine)

S3 = boto3.client("s3", region_name=AWS_REGION)

_download_lock = threading.Lock()
_in_progress: Dict[str, threading.Event] = {}


def _s3_size(key: str) -> int:
    """Return size (bytes) of existing key or raise FileNotFoundError."""
    try:
        response = S3.head_object(Bucket=S3_BUCKET, Key=key)
        return int(response["ContentLength"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise FileNotFoundError(key) from None
        raise


def atomic_download(key: str, dest: Path) -> None:
    """Download key to dest exactly once across many threads."""
    with _download_lock:
        evt = _in_progress.get(key)
        if evt is None:
            evt = threading.Event()
            _in_progress[key] = evt
            first = True
        else:
            first = False

    if not first:
        evt.wait()
        if not dest.exists():
            raise FileNotFoundError(key)
        return

    try:
        size = _s3_size(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == size:
            return
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with open(tmp, "wb") as fh:
            S3.download_fileobj(S3_BUCKET, key, fh)
        if tmp.stat().st_size != size:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Incomplete download for {key}")
        tmp.replace(dest)
    finally:
        evt.set()
        with _download_lock:
            _in_progress.pop(key, None)


def out_dir(region: str, kind: str) -> Path:
    d = OUTPUT_ROOT / region.capitalize() / "lidar" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_tile_index(region: str) -> gpd.GeoDataFrame:
    sql = (
        "SELECT ti.file_name, ti.geom "
        "FROM tileset_indexes ti "
        "JOIN tilesets ts ON ts.id = ti.tileset_id "
        "JOIN regions r   ON r.id  = ts.region_id "
        "WHERE r.name = %s"
    )
    with engine.connect() as cn:
        gdf = gpd.read_postgis(sql, cn, geom_col="geom", params=(region,))
    gdf = (
        gdf.rename(columns={"file_name": "FileName"})
        .set_crs(4326)
        .to_crs(SPEC[region]["crs"])
    )
    gdf.sindex
    return gdf


def residential_fp(region_id: int, target_crs: int) -> gpd.GeoDataFrame:
    ids_subq = select(_tbl_bp_proc.c.footprint_id).where(
        _tbl_bp_proc.c.region_id == region_id
    )
    stmt = select(_tbl_footprints.c.id, _tbl_footprints.c.geom).where(
        _tbl_footprints.c.id.in_(ids_subq)
    )
    with engine.connect() as cn:
        gdf = gpd.read_postgis(stmt, cn, geom_col="geom")
    return gdf.set_crs(4326).to_crs(target_crs)


def pdal_pipeline(las_paths: List[Path], poly: Polygon, out_path: Path) -> str:
    steps: List[Dict[str, str]] = [
        {"type": "readers.las", "filename": str(p)} for p in las_paths
    ]
    if len(las_paths) > 1:
        steps.append({"type": "filters.merge"})
    steps.append({"type": "filters.crop", "polygon": poly.wkt})
    steps.append(
        {"type": "writers.las", "filename": str(out_path), "extra_dims": "all"}
    )
    return json.dumps({"pipeline": steps}, separators=(",", ":"))


def run_region(region: str, workers: Optional[int] = None) -> None:
    logger.info(f"── Clipping {region} ──")
    tiles = load_tile_index(region)

    with engine.connect() as cn:
        rid = cn.execute(
            select(_tbl_regions.c.id).where(_tbl_regions.c.name == region)
        ).scalar_one()

    target_crs = SPEC[region]["crs"]
    fps = residential_fp(rid, target_crs)
    if fps.empty:
        logger.warning(f"{region}: no residential footprints; skipping")
        return

    originals = out_dir(region, "original")
    clipped = out_dir(region, "clipped")

    workers = workers or min(mp.cpu_count(), 64)
    logger.info(f"Using {workers} worker(s)")

    counters = {
        "suc": 0,
        "skp": 0,
        "nt": 0,
        "mt": 0,
        "fail": 0,
    }

    def clip_one(row: Any) -> str:
        bid = int(row.id)
        buf = row.geom.simplify(0.1).buffer(BUFFER_M, join_style=2)
        x1, y1, x2, y2 = buf.bounds
        candidates = tiles.cx[x1:x2, y1:y2]
        hits = candidates[candidates.geometry.intersects(buf)]
        if hits.empty:
            return "nt"

        las_paths: List[Path] = []
        for _, t in hits.iterrows():
            stem = re.sub(r"(\d+)_\s+(\d+)", r"\1_\2", Path(t.FileName).stem)
            key = f"{SPEC[region]['lidar_prefix']}{stem}.las"
            local = originals / f"{stem}.las"
            try:
                atomic_download(key, local)
            except FileNotFoundError:
                return "mt"
            las_paths.append(local)

        out_path = clipped / f"{bid}.las"
        if out_path.exists():
            return "skp"

        pipe = pdal_pipeline(las_paths, buf, out_path)
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as tmp:
            tmp.write(pipe)
            tmp.flush()
            try:
                subprocess.run(
                    ["pdal", "pipeline", tmp.name],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                return "suc"
            except subprocess.CalledProcessError as e:
                logger.error(f"{region}:{bid} PDAL failed → {e}")
                out_path.unlink(missing_ok=True)
                return "fail"

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TimeRemainingColumn(compact=True),
        TextColumn(" • [green]✓{task.fields[suc]}"),
        TextColumn(" • [yellow]➟{task.fields[skp]}"),
        TextColumn(" • [cyan]⊘{task.fields[nt]}"),
        TextColumn(" • [magenta]⊖{task.fields[mt]}"),
        TextColumn(" • [red]✗{task.fields[fail]}"),
        console=console,
        refresh_per_second=2,
    ) as prog:
        tid = prog.add_task(
            f"Clipping {region}", total=len(fps), suc=0, skp=0, nt=0, mt=0, fail=0
        )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(clip_one, row) for _, row in fps.iterrows()]
            for fut in as_completed(futures):
                status = fut.result()
                counters[status] += 1
                prog.advance(tid)
                prog.update(
                    tid,
                    suc=counters["suc"],
                    skp=counters["skp"],
                    nt=counters["nt"],
                    mt=counters["mt"],
                    fail=counters["fail"],
                )

    logger.info(
        f"{region}: ✓{counters['suc']} ➟{counters['skp']} ⊘{counters['nt']} ⊖{counters['mt']} ✗{counters['fail']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clip LiDAR tiles to residential footprints from S3"
    )
    ap.add_argument(
        "--region", help="Single region to process (default = all from config)"
    )
    ap.add_argument(
        "--workers", type=int, help="Parallel workers (default = min(cpu, 64))"
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log verbosity",
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    try:
        if args.region:
            run_region(args.region, args.workers)
        else:
            logger.info(f"Processing {len(REGIONS)} regions: {', '.join(REGIONS)}")
            for r in REGIONS:
                run_region(r, args.workers)
        logger.info("Stage‑05 complete")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
