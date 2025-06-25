#!/usr/bin/env python
"""Download required AWS data files for the floor heights pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger

S3_BUCKET = "s3://frontiersi-p127-floor-height-woolpert/"


AWS_FILES: dict[str, dict[str, list[str]]] = {
    "wagga": {
        "trajectory": ["01_WaggaWagga/03_Ancillary/02_TrajectoryFiles/rev2/FramePosOptimised-wagga-wagga-rev2.csv"],
        "tileset": [
            "01_WaggaWagga/03_Ancillary/01_TileIndex/rev1/48068_Wagga_Wagga_TileSet.shp",
            "01_WaggaWagga/03_Ancillary/01_TileIndex/rev1/48068_Wagga_Wagga_TileSet.shx",
            "01_WaggaWagga/03_Ancillary/01_TileIndex/rev1/48068_Wagga_Wagga_TileSet.dbf",
            "01_WaggaWagga/03_Ancillary/01_TileIndex/rev1/48068_Wagga_Wagga_TileSet.prj",
            "01_WaggaWagga/03_Ancillary/01_TileIndex/rev1/48068_Wagga_Wagga_TileSet.cpg",
        ],
    },
    "tweed": {
        "trajectory": ["02_TweedHeads/03_Ancillary/02_TrajectoryFiles/FramePosOptimised-tweed-heads-rev2.csv"],
        "tileset": [
            "02_TweedHeads/03_Ancillary/01_TileIndex/48068_Tweed_Heads_TileSet.shp",
            "02_TweedHeads/03_Ancillary/01_TileIndex/48068_Tweed_Heads_TileSet.shx",
            "02_TweedHeads/03_Ancillary/01_TileIndex/48068_Tweed_Heads_TileSet.dbf",
            "02_TweedHeads/03_Ancillary/01_TileIndex/48068_Tweed_Heads_TileSet.prj",
            "02_TweedHeads/03_Ancillary/01_TileIndex/48068_Tweed_Heads_TileSet.cpg",
        ],
    },
    "launceston": {
        "trajectory": ["03_Launceston/03_Ancillary/02_TrajectoryFiles/FramePosOptimised-launceston-rev2.csv"],
        "tileset": [
            "03_Launceston/03_Ancillary/01_TileIndex/rev1/48068_Launceston_TileSet.shp",
            "03_Launceston/03_Ancillary/01_TileIndex/rev1/48068_Launceston_TileSet.shx",
            "03_Launceston/03_Ancillary/01_TileIndex/rev1/48068_Launceston_TileSet.dbf",
            "03_Launceston/03_Ancillary/01_TileIndex/rev1/48068_Launceston_TileSet.prj",
            "03_Launceston/03_Ancillary/01_TileIndex/rev1/48068_Launceston_TileSet.cpg",
        ],
    },
}


def ensure_data_directories() -> Path:
    """Ensure data directory structure exists."""
    data_root = Path("data/raw")
    data_root.mkdir(parents=True, exist_ok=True)

    for region in AWS_FILES:
        region_dir = data_root / region
        region_dir.mkdir(exist_ok=True)
        (region_dir / "tileset").mkdir(exist_ok=True)

    return data_root


def download_file(s3_path: str, local_path: Path) -> bool:
    """Download a single file from S3.

    Args:
        s3_path: S3 path relative to bucket
        local_path: Local destination path

    Returns:
        True if successful, False otherwise
    """
    s3_url = f"{S3_BUCKET}{s3_path}"

    if local_path.exists():
        logger.info(f"File already exists: {local_path}")
        return True

    logger.info(f"Downloading: {s3_path} -> {local_path}")

    try:
        cmd = ["aws", "s3", "cp", s3_url, str(local_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Failed to download {s3_path}: {result.stderr}")
            return False

        logger.success(f"Downloaded: {local_path.name}")
        return True

    except Exception as e:
        logger.error(f"Error downloading {s3_path}: {e}")
        return False


def download_region_data(region: str, data_root: Path) -> bool:
    """Download all required files for a region.

    Args:
        region: Region name
        data_root: Root data directory

    Returns:
        True if all downloads successful, False otherwise
    """
    if region not in AWS_FILES:
        logger.error(f"Unknown region: {region}")
        return False

    region_files = AWS_FILES[region]
    region_dir = data_root / region
    success = True

    for trajectory_file in region_files["trajectory"]:
        filename = Path(trajectory_file).name
        local_path = region_dir / filename
        if not download_file(trajectory_file, local_path):
            success = False

    for tileset_file in region_files["tileset"]:
        filename = Path(tileset_file).name
        local_path = region_dir / "tileset" / filename
        if not download_file(tileset_file, local_path):
            success = False

    return success


def download_aws_data(
    region: str | None = None,
    dry_run: bool = False,
) -> None:
    """Download AWS data files required for the pipeline.

    Args:
        region: Single region to process (None for all)
        dry_run: If True, show what would be downloaded without actually downloading
    """
    logger.info("── Download AWS Data Files ──")

    if dry_run:
        logger.info("DRY RUN MODE - No files will be downloaded")

    data_root = ensure_data_directories()

    regions_to_process = [region] if region else list(AWS_FILES.keys())

    total_files = 0
    for r in regions_to_process:
        logger.info(f"\nProcessing region: {r}")

        if dry_run:
            region_files = AWS_FILES[r]
            trajectory_count = len(region_files["trajectory"])
            tileset_count = len(region_files["tileset"])
            total_files += trajectory_count + tileset_count
            logger.info(f"Would download {trajectory_count} trajectory file(s)")
            logger.info(f"Would download {tileset_count} tileset file(s)")
            continue

        success = download_region_data(r, data_root)

        if success:
            logger.success(f"✓ All files downloaded for {r}")
        else:
            logger.error(f"✗ Some downloads failed for {r}")

    if dry_run:
        logger.info(f"\nTotal files to download: {total_files}")
    else:
        logger.info("\nDownload complete")
        logger.info(f"Data files are in: {data_root.absolute()}")


if __name__ == "__main__":
    import typer

    app = typer.Typer()
    app.command()(download_aws_data)
    app()
