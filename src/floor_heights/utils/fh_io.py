"""Common I/O utilities for the floor heights pipeline.

This module provides canonical helpers for:
- Generating consistent file paths (LAS, clips)
- Reading from DuckDB with GeoPandas
- Managing stage result tables
- Tracking processed IDs across stages
- Validating file integrity
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd
import torch
from loguru import logger
from PIL import Image

from floor_heights.config import CONFIG
from floor_heights.db.ibis_client import connect
from floor_heights.db.reader import DuckDBReader


def las_path(
    region: str,
    row_id: int,
    building_id: str,
    gnaf_id: str | None = None,
    revision: str = "rev2",
    ext: str = ".las",
    source: str = "local",
) -> Path:
    """Generate consistent LAS file path for clipped point clouds.

    Args:
        region: Region name
        row_id: Row ID from database
        building_id: Building ID
        gnaf_id: Optional GNAF ID
        revision: Optional revision (e.g., 'rev1', 'rev2')
        ext: File extension (.las or .laz)
        source: Source type ('local' or 's3')

    Returns:
        Path to the LAS file (relative to CONFIG.output_root)
    """
    rel_path = Path(region.capitalize()) / "lidar" / f"{revision}-{source}" / "clipped"

    abs_dir = CONFIG.output_root / rel_path
    abs_dir.mkdir(parents=True, exist_ok=True)

    gnaf_part = gnaf_id if gnaf_id and pd.notna(gnaf_id) else "NO_GNAF"
    filename = f"{row_id}_{building_id}_{gnaf_part}{ext}"

    return rel_path / filename


def clip_path(stage: str, region: str, name: str, ext: str = "jpg") -> Path:
    """Generate path for clipped images or other outputs.

    Args:
        stage: Stage name
        region: Region name
        name: Base filename without extension
        ext: File extension (default: jpg)

    Returns:
        Path to the clip file
    """
    output_dir = CONFIG.output_root / region.capitalize() / stage / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{name}.{ext}"


def read_table(
    table: str,
    columns: list[str] | None = None,
    region: str | None = None,
    filters: dict[str, any] | None = None,
    as_geo: bool = False,
    geom_col: str = "geometry",
    crs: str | int | None = None,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Read table from DuckDB with optional filters."""
    with DuckDBReader(CONFIG.db_path) as r:
        t = r.table(table)

        if region:
            key = "name" if table == "regions" else "region_name"
            t = t.filter(t[key] == region)

        if filters:
            for col, val in filters.items():
                t = t.filter(t[col].isnull()) if val is None else t.filter(t[col] == val)

        if columns:
            t = t[columns]

        if as_geo and geom_col in t.columns:
            t = t.mutate(_geom_wkt=t[geom_col].as_text())

            df = t.execute()

            from shapely import wkt

            df["geometry"] = df["_geom_wkt"].apply(wkt.loads)

            cols_to_drop = ["_geom_wkt"]
            if geom_col != "geometry":
                cols_to_drop.append(geom_col)

            gdf = gpd.GeoDataFrame(df.drop(cols_to_drop, axis=1), geometry="geometry", crs=crs)
            return gdf

        return t.execute()


def ensure_stage_table(table_name: str, schema: dict[str, str]) -> None:
    """Ensure a stage result table exists with the given schema.

    Args:
        table_name: Name of the table to create
        schema: Dictionary mapping column names to SQL types
    """
    conn = connect(CONFIG.db_path, read_only=False)

    if table_name not in conn.list_tables():
        type_map = {
            "str": "VARCHAR",
            "int32": "INTEGER",
            "int64": "BIGINT",
            "float32": "REAL",
            "float64": "DOUBLE",
            "datetime64[ns]": "TIMESTAMP",
            "bool": "BOOLEAN",
        }

        columns = []
        for col, dtype in schema.items():
            sql_type = type_map.get(dtype, dtype.upper())
            columns.append(f"{col} {sql_type}")

        create_sql = f"CREATE TABLE {table_name} ({', '.join(columns)})"

        conn.raw_sql(create_sql)
        logger.info(f"Created {table_name} table")


def save_stage_result_batch(table_name: str, records: list[dict]) -> None:
    """Save multiple stage processing results to the database efficiently.

    Uses DuckDB's native INSERT for optimal performance.
    Only appends records - does not support upsert.

    Args:
        table_name: Name of the stage table
        records: List of dictionaries containing the result data
    """
    if not records:
        return

    from datetime import datetime

    for record in records:
        if "processed_at" not in record:
            record["processed_at"] = datetime.now()

    from floor_heights.db.schemas import get_stage_schema

    schema = get_stage_schema(table_name)
    ensure_stage_table(table_name, schema)

    conn = connect(CONFIG.db_path, read_only=False)
    conn.insert(table_name, pd.DataFrame(records))


def get_processed_stage_ids(table_name: str, region: str) -> set[int]:
    """Get set of processed IDs from a stage table.

    Since we only write successful records to the database,
    this returns all IDs present in the table for the given region.

    Args:
        table_name: Name of the stage table (e.g., 'stage01_clips')
        region: Region to filter by

    Returns:
        Set of row IDs that have been successfully processed
    """
    conn = connect(CONFIG.db_path, read_only=True)

    if table_name not in conn.list_tables():
        return set()

    table = conn.table(table_name)

    filtered = table.filter(table.region_name == region)

    ids_df = filtered.select("id").distinct().execute()

    return set(ids_df["id"].values)


def export_tables_to_geoparquet(output_dir: Path | None = None, tables: list[str] | None = None) -> None:
    """Export DuckDB tables to GeoParquet files.

    Args:
        output_dir: Directory to write files (default: CONFIG.output_root / 'exports')
        tables: List of table names to export (default: all tables)
    """
    import geopandas as gpd
    from shapely import wkt

    from floor_heights.db import DuckDBReader

    if output_dir is None:
        output_dir = CONFIG.output_root / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    with DuckDBReader(CONFIG.db_path) as reader:
        all_tables = reader._con.list_tables()
        tables_to_export = tables if tables else all_tables

        for table_name in tables_to_export:
            if table_name not in all_tables:
                logger.warning(f"Table '{table_name}' not found, skipping")
                continue

            logger.info(f"Exporting {table_name}...")

            table = reader.table(table_name)
            df = table.execute()

            if df.empty:
                logger.warning(f"Table '{table_name}' is empty, skipping")
                continue

            geom_cols = [col for col in df.columns if "geom" in col.lower() or col == "geometry"]
            wkt_cols = [col for col in df.columns if "wkt" in col.lower()]

            if geom_cols or wkt_cols:
                if wkt_cols:
                    geom_col = wkt_cols[0]
                    df["geometry"] = df[geom_col].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
                    gdf = gpd.GeoDataFrame(df, geometry="geometry")
                else:
                    geom_col = geom_cols[0]
                    gdf = gpd.GeoDataFrame(df, geometry=geom_col)

                output_path = output_dir / f"{table_name}.parquet"
                gdf.to_parquet(output_path)
                logger.info(f"  → {output_path} ({len(gdf):,} rows)")
            else:
                output_path = output_dir / f"{table_name}.parquet"
                df.to_parquet(output_path)
                logger.info(f"  → {output_path} ({len(df):,} rows)")


def find_lidar_file(base_dir: Path, stem: str) -> Path | None:
    """Find LiDAR file with either .las or .laz extension.

    Args:
        base_dir: Directory to search in
        stem: File stem (without extension)

    Returns:
        Path to found file or None
    """
    for ext in [".las", ".laz"]:
        path = base_dir / f"{stem}{ext}"
        if validate_file_exists_and_valid(path, file_type="las"):
            return path
    return None


def get_lidar_dir(region: str, revision: str = "rev2", use_s3: bool = False) -> Path:
    """Get the directory containing LiDAR files for a region.

    Args:
        region: Region name
        revision: Optional revision (defaults to 'rev2')
        use_s3: If True, return S3 cache directory in output

    Returns:
        Path to LiDAR directory
    """
    if use_s3 or not CONFIG.lidar_data_root:
        return CONFIG.output_root / region.capitalize() / "lidar" / f"{revision}-s3" / "original"
    else:
        return CONFIG.lidar_data_root / revision / region


def find_lidar_tiles(tile_filenames: list[str], region: str, revision: str = "rev2", use_s3: bool = True) -> list[Path]:
    """Find LiDAR tiles (LAS or LAZ) for given filenames.

    Args:
        tile_filenames: List of tile filenames to find
        region: Region name
        revision: Revision (defaults to 'rev2')
        use_s3: If True, download from S3; if False, use local files

    Returns:
        List of paths to files
    """
    local_dir = get_lidar_dir(region, revision, use_s3=use_s3)

    if use_s3:
        return _download_from_s3(tile_filenames, region, revision, local_dir)
    else:
        paths = [
            local_path for filename in tile_filenames if (local_path := find_lidar_file(local_dir, Path(filename).stem))
        ]
        return paths


def _download_from_s3(filenames: list[str], region: str, revision: str, local_dir: Path) -> list[Path]:
    """Download LiDAR files from S3 if not already cached.

    Returns list of file paths (downloaded or already cached).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import boto3
    from dotenv import load_dotenv

    load_dotenv(override=True)

    from botocore.config import Config

    s3_config = Config(
        region_name=CONFIG.constants.s3_region,
        max_pool_connections=100,
        retries={"max_attempts": 3, "mode": "adaptive"},
        tcp_keepalive=True,
    )

    s3_client = boto3.client("s3", config=s3_config)
    bucket = CONFIG.constants.s3_bucket

    s3_prefixes = {
        "wagga": "01_WaggaWagga/02_MLSPointCloud",
        "tweed": "02_TweedHeads/02_MLSPointCloud",
        "launceston": "03_Launceston/02_MLSPointCloud",
    }

    if region not in s3_prefixes:
        logger.warning(f"No S3 prefix configured for region {region}")
        return []

    s3_prefix = f"{s3_prefixes[region]}/{revision}/"
    paths = []
    download_lock = threading.Lock()

    def download_file(filename: str) -> Path | None:
        """Download a single file from S3 if not cached."""
        stem = Path(filename).stem
        s3_keys = [f"{s3_prefix}{stem}.las"]

        if "_" in stem:
            parts = stem.rsplit("_", 1)
            s3_keys.append(f"{s3_prefix}{parts[0]}_ {parts[1]}.las")

        local_path = local_dir / f"{stem}.las"

        if validate_file_exists_and_valid(local_path, file_type="las"):
            return local_path
        laz_path = local_path.with_suffix(".laz")
        if validate_file_exists_and_valid(laz_path, file_type="las"):
            return laz_path

        local_dir.mkdir(parents=True, exist_ok=True)

        for s3_key in s3_keys:
            try:
                logger.debug(f"Trying {s3_key}")
                s3_client.download_file(bucket, s3_key, str(local_path))
                logger.debug(f"Downloaded {s3_key}")
                return local_path
            except s3_client.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code in ["NoSuchKey", "404"]:
                    continue
                else:
                    logger.error(f"S3 error {error_code}: {s3_key}")
                    return None
            except Exception as e:
                logger.error(f"Download failed: {e}")
                local_path.unlink(missing_ok=True)
                return None

        logger.warning(f"File not found in S3 with any variant: {stem}")
        return None

    import os

    cpu_count = os.cpu_count() or 8

    optimal_workers = min(max(cpu_count * 2, 16), min(len(filenames), 64))

    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        futures = {executor.submit(download_file, f): f for f in filenames}

        for future in as_completed(futures):
            result = future.result()
            if result:
                with download_lock:
                    paths.append(result)

    return paths


def validate_file_exists_and_valid(
    file_path: Path | str,
    file_type: str | None = None,
    min_size_bytes: int = 1,
) -> bool:
    """Validate that a file exists and is not corrupted.

    Args:
        file_path: Path to the file to validate
        file_type: Optional file type hint ('image', 'las', 'model', 'db')
        min_size_bytes: Minimum file size in bytes (default 1)

    Returns:
        True if file exists and appears valid, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False

    if not file_path.is_file():
        logger.warning(f"Path exists but is not a file: {file_path}")
        return False

    try:
        size = file_path.stat().st_size
        if size < min_size_bytes:
            logger.warning(f"File too small ({size} bytes): {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error checking file size: {e}")
        return False

    if file_type:
        return validate_file_type(file_path, file_type)

    return True


def validate_file_type(file_path: Path, file_type: str) -> bool:
    """Validate specific file types for corruption.

    Args:
        file_path: Path to the file
        file_type: Type of file ('image', 'las', 'model', 'db')

    Returns:
        True if file appears valid, False otherwise
    """
    try:
        if file_type == "image":
            return validate_image_file(file_path)
        elif file_type == "las":
            return validate_las_file(file_path)
        elif file_type == "model":
            return validate_model_file(file_path)
        elif file_type == "db":
            return validate_db_file(file_path)
        else:
            logger.warning(f"Unknown file type for validation: {file_type}")
            return True
    except Exception as e:
        logger.error(f"Error validating {file_type} file {file_path}: {e}")
        return False


def validate_image_file(file_path: Path) -> bool:
    """Validate an image file can be opened and read.

    Args:
        file_path: Path to image file

    Returns:
        True if image is valid, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            img.verify()

        with Image.open(file_path) as img:
            _ = img.size

        return True
    except Exception as e:
        logger.error(f"Invalid image file {file_path}: {e}")
        return False


def validate_las_file(file_path: Path) -> bool:
    """Validate a LAS/LAZ point cloud file using PDAL.

    Args:
        file_path: Path to LAS/LAZ file

    Returns:
        True if file is valid, False otherwise
    """
    try:
        cmd = ["pdal", "info", str(file_path), "--metadata"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            logger.error(f"PDAL validation failed for {file_path}: {result.stderr}")
            return False

        try:
            info = json.loads(result.stdout)

            if "metadata" not in info:
                logger.error(f"No metadata in PDAL output for {file_path}")
                return False
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from PDAL for {file_path}")
            return False

        return True

    except subprocess.TimeoutExpired:
        logger.error(f"PDAL validation timed out for {file_path}")
        return False
    except Exception as e:
        logger.error(f"Error validating LAS file {file_path}: {e}")
        return False


def validate_model_file(file_path: Path) -> bool:
    """Validate a PyTorch model file.

    Args:
        file_path: Path to model file

    Returns:
        True if model file is valid, False otherwise
    """
    try:
        _ = torch.load(file_path, map_location="cpu", weights_only=False)
        return True
    except Exception as e:
        logger.error(f"Invalid model file {file_path}: {e}")
        return False


def validate_db_file(file_path: Path) -> bool:
    """Validate a database file (basic check).

    Args:
        file_path: Path to database file

    Returns:
        True if database file appears valid, False otherwise
    """
    try:
        with file_path.open("rb") as f:
            header = f.read(8)

            if len(header) < 8:
                logger.error(f"Database file too small: {file_path}")
                return False

        return True
    except Exception as e:
        logger.error(f"Error validating database file {file_path}: {e}")
        return False


def compute_file_hash(file_path: Path, algorithm: str = "md5") -> str | None:
    """Compute hash of a file for integrity checking.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use ('md5', 'sha256')

    Returns:
        Hex digest of file hash, or None if error
    """
    try:
        hash_func = hashlib.new(algorithm)
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {e}")
        return None
