#!/usr/bin/env python
"""Stage-02b: Download panoramas flagged is_chosen."""

from __future__ import annotations

import multiprocessing as mp
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from floor_heights.config import CONFIG, REGIONS
from floor_heights.db.schemas import (
    BatchWriter,
    Stage02bDownloadRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import read_table, validate_file_exists_and_valid
from floor_heights.utils.progress import processing_progress


@dataclass(frozen=True)
class RegionSpec:
    prefix: str


SPEC: Mapping[str, RegionSpec] = {
    "tweed": RegionSpec("02_TweedHeads/01_StreetViewImagery/"),
    "wagga": RegionSpec("01_WaggaWagga/01_StreetViewImagery/"),
    "launceston": RegionSpec("03_Launceston/01_StreetViewImagery/"),
}


def extract_ucid(pano_id: str) -> str:
    """Extract UCID from panorama ID."""
    import re

    stem = pano_id.removesuffix(".jpg")
    match = re.search(r"_(\d+)-", stem)
    if match:
        return match.group(1)
    raise ValueError(f"No UCID found in '{pano_id}'.")


@cache
def _s3_head(client: Any, key: str) -> int:
    """Get file size from S3 HEAD request."""
    return client.head_object(Bucket=CONFIG.constants.s3_bucket, Key=key)["ContentLength"]


def resolve_s3_key(client: Any, region: str, pano_id: str) -> str:
    """Resolve S3 key for a panorama ID."""
    spec = SPEC[region]
    ucid = extract_ucid(pano_id)
    prefix = spec.prefix

    def candidate(name: str) -> str:
        return f"{prefix}{ucid}/Panoramas/{name}"

    names = []
    if region == "wagga" and "NoRoad" in pano_id:
        names.append(pano_id.replace("NoRoad.jpg", "OutsideAOI.jpg"))
    names.append(pano_id)

    for name in names:
        key = candidate(name)
        try:
            _s3_head(client, key)
            return key
        except ClientError:
            continue

    raise FileNotFoundError(f"No object found for '{pano_id}' in {region}.")


def get_candidates_to_download(region: str) -> pd.DataFrame:
    """Get candidate views marked as chosen from stage02a."""

    df = read_table("stage02a_candidate_views", region=region)

    chosen = df[df["is_chosen"]].copy()

    if chosen.empty:
        return pd.DataFrame()

    return chosen


def get_processed_downloads(region: str) -> set[tuple]:
    """Get set of already processed downloads as (id, pano_id, edge_idx, view_type) tuples."""
    try:
        df = read_table("stage02b_downloads", region=region)

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


def create_download_filename(row: pd.Series, region: str) -> Path:
    """Create standardized download filename and directory (returns relative path)."""
    row_id = row["id"]
    bid = str(row["building_id"])
    gnaf_id = str(row.get("gnaf_id", "NO_GNAF"))
    if pd.isna(row.get("gnaf_id")) or gnaf_id == "":
        gnaf_id = "NO_GNAF"
    pid = row["pano_id"]
    edge_idx = row["edge_idx"]
    view_type = row["view_type"]

    name = pid if pid.endswith(".jpg") else f"{pid}.jpg"
    stem = name.removesuffix(".jpg")
    new_name = f"{stem}_edge{edge_idx}_{view_type}.jpg"

    dir_name = f"{row_id}_{bid}_{gnaf_id}"
    return Path(region.capitalize()) / "panoramas" / dir_name / new_name


def download_once(client: Any, key: str, dest: Path) -> tuple[bool, int]:
    """Download file from S3 if it doesn't exist."""
    abs_dest = CONFIG.output_root / dest
    abs_dest.parent.mkdir(parents=True, exist_ok=True)
    if validate_file_exists_and_valid(abs_dest, file_type="image", min_size_bytes=1000):
        return False, abs_dest.stat().st_size

    size = _s3_head(client, key)
    with abs_dest.open("wb") as fh:
        client.download_fileobj(CONFIG.constants.s3_bucket, key, fh)

    if not validate_file_exists_and_valid(abs_dest, file_type="image", min_size_bytes=1000):
        logger.error(f"Downloaded file appears corrupted: {abs_dest}")
        abs_dest.unlink(missing_ok=True)
        raise ValueError("Downloaded file validation failed")

    return True, size


def _format_bytes(n: int) -> str:
    """Format bytes in human readable format."""
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    val = float(n)
    for unit in units:
        if val < 1024:
            return f"{val:.1f}{unit}"
        val = val / 1024
    return f"{val:.1f}EB"


def process_region(region: str, workers: int = -1) -> None:
    """Process panorama downloads for a single region."""
    logger.info(f"Processing {region}")

    candidates = get_candidates_to_download(region)
    if candidates.empty:
        logger.warning(f"{region}: no chosen candidates found")
        return

    processed = get_processed_downloads(region)

    candidates["key"] = list(
        zip(candidates["id"], candidates["pano_id"], candidates["edge_idx"], candidates["view_type"], strict=False)
    )

    download_tasks = []
    for _idx, row in candidates.iterrows():
        if row["key"] not in processed:
            download_tasks.append(row)
        else:
            dest = create_download_filename(row, region)
            abs_dest = CONFIG.output_root / dest
            if not validate_file_exists_and_valid(abs_dest, file_type="image", min_size_bytes=1000):
                download_tasks.append(row)

    if not download_tasks:
        logger.info(f"{region}: all files already downloaded")
        return

    logger.info(f"{region}: {len(download_tasks)} files to download")

    if workers <= 0:
        workers = CONFIG.constants.default_workers
        if workers <= 0:
            workers = mp.cpu_count()

    s3 = boto3.Session(region_name=CONFIG.constants.s3_region).client("s3")

    def process_download(row: pd.Series) -> tuple:
        """Process a single download."""
        dest = create_download_filename(row, region)
        abs_dest = CONFIG.output_root / dest

        try:
            if validate_file_exists_and_valid(abs_dest, file_type="image", min_size_bytes=1000):
                return ("exists", str(dest))

            key = resolve_s3_key(s3, region, row["pano_id"])
            fresh, size = download_once(s3, key, dest)

            return ("success" if fresh else "exists", str(dest))
        except Exception as e:
            logger.debug(f"Download failed for {row['pano_id']}: {e}")
            return ("failed", None)

    with (
        processing_progress(f"{region} downloading", len(download_tasks)) as prog,
        ThreadPoolExecutor(max_workers=workers) as pool,
        BatchWriter("stage02b_downloads", batch_size=5000, progress_tracker=prog) as writer,
    ):
        futures = {}
        for task in download_tasks:
            future = pool.submit(process_download, task)
            futures[future] = task

        for future in as_completed(futures):
            try:
                row = futures[future]
                status, download_path = future.result()

                if status in ["success", "exists"] and download_path:
                    writer.add(
                        Stage02bDownloadRecord(
                            id=str(row["id"]),
                            building_id=row["building_id"],
                            region_name=region,
                            gnaf_id=row.get("gnaf_id", ""),
                            pano_id=row["pano_id"],
                            edge_idx=row["edge_idx"],
                            view_type=row["view_type"],
                            download_path=download_path,
                        )
                    )

                    if status == "success":
                        prog.update("suc", 1)
                    else:
                        prog.update("skp", 1)
                else:
                    prog.update("fail", 1)

            except Exception as e:
                logger.error(f"Error processing download result: {e}")
                prog.update("fail", 1)

    logger.success(f"{region}: {prog.get_summary()}")


def run_stage(region: str | None = None, workers: int = -1) -> None:
    """Run stage02b with the given parameters.

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
        logger.info("Stage-02b complete")
    except Exception as e:
        logger.error(f"Stage-02b failed: {e}")
        raise


if __name__ == "__main__":
    run_stage()
