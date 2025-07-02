"""Download required AWS data files for the floor heights pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from loguru import logger

from floor_heights.config import CONFIG

load_dotenv()

S3_BUCKET_NAME = "frontiersi-p127-floor-height-woolpert"

SHAPEFILE_EXTENSIONS = [".shp", ".shx", ".dbf", ".prj", ".cpg"]


def ensure_data_directories() -> Path:
    """Ensure data directory structure exists."""
    data_root = Path("data/raw")
    data_root.mkdir(parents=True, exist_ok=True)

    for region in CONFIG.regions:
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
    if local_path.exists():
        logger.info(f"File already exists: {local_path}")
        return True

    logger.info(f"Downloading: {s3_path} -> {local_path}")

    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
        )

        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3_client.download_file(S3_BUCKET_NAME, s3_path, str(local_path))

        logger.success(f"Downloaded: {local_path.name}")
        return True

    except NoCredentialsError:
        logger.error("AWS credentials not found. Please check your .env file.")
        return False
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "403":
            logger.error(f"Access denied to {s3_path}. Check your AWS credentials and permissions.")
        elif error_code == "404":
            logger.error(f"File not found: {s3_path}")
        else:
            logger.error(f"Failed to download {s3_path}: {e}")
        return False
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
    if region not in CONFIG.regions:
        logger.error(f"Unknown region: {region}")
        return False

    region_config = CONFIG.regions[region]
    region_dir = data_root / region
    success = True

    if region_config.trajectory_path:
        trajectory_path = region_config.trajectory_path
        filename = Path(trajectory_path).name
        local_path = region_dir / filename
        if not download_file(trajectory_path, local_path):
            success = False

    if region_config.tile_index_path:
        tile_index_path = Path(region_config.tile_index_path)
        base_name = tile_index_path.stem

        for ext in SHAPEFILE_EXTENSIONS:
            s3_path = str(tile_index_path.with_suffix(ext))
            filename = base_name + ext
            local_path = region_dir / "tileset" / filename
            if not download_file(s3_path, local_path):
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

    regions_to_process = [region] if region else list(CONFIG.regions.keys())

    total_files = 0
    for r in regions_to_process:
        logger.info(f"\nProcessing region: {r}")

        if dry_run:
            region_config = CONFIG.regions[r]
            files_count = 0
            if region_config.trajectory_path:
                files_count += 1
                logger.info("Would download trajectory file")
            if region_config.tile_index_path:
                files_count += len(SHAPEFILE_EXTENSIONS)
                logger.info(f"Would download {len(SHAPEFILE_EXTENSIONS)} tileset files")
            total_files += files_count
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
