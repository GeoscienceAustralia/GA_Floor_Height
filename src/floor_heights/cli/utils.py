"""Utility functions for the CLI."""

import subprocess

from loguru import logger
from rich.console import Console

from floor_heights.config import CONFIG

console = Console()


def check_stage_config(stage: str | None = None) -> bool:
    """Check configuration for a specific stage or all stages.

    Returns True if configuration is valid, False otherwise.
    """
    errors = []
    warnings = []

    if not CONFIG.project_root.exists():
        errors.append("Project root missing")
        return False

    if not CONFIG.output_root.exists():
        warnings.append("Output directory missing - will be created")
        try:
            CONFIG.output_root.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            errors.append(f"Cannot create output directory: {e}")
            return False

    if not CONFIG.db_path.exists():
        errors.append("Database not found")
        return False

    if stage == "stage01":
        try:
            result = subprocess.run(["pdal", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                errors.append("PDAL not available")
        except FileNotFoundError:
            errors.append("PDAL not found in PATH")

    return len(errors) == 0


def run_stage_direct(stage_name: str, **kwargs) -> None:
    """Run a stage by directly importing and calling its run_stage function.

    This avoids subprocess overhead for better performance.
    """
    import importlib

    from floor_heights.cli.stages import STAGE_DEFINITIONS

    stage_def = None
    for sd in STAGE_DEFINITIONS:
        if sd.name == stage_name:
            stage_def = sd
            break

    if not stage_def:
        raise ValueError(f"Unknown stage: {stage_name}")

    try:
        module = importlib.import_module(stage_def.module_import_path)
        run_func = getattr(module, stage_def.function_name)

        import inspect

        sig = inspect.signature(run_func)
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        if stage_name == "stage01" and "lidar_source" not in filtered_kwargs:
            import os

            lidar_root = os.getenv("FH_LIDAR_DATA_ROOT")
            filtered_kwargs["lidar_source"] = "local" if lidar_root else "s3"

        run_func(**filtered_kwargs)

    except Exception as e:
        logger.error(f"Failed to run {stage_name}: {e}")
        raise


def validate_database_tables() -> dict[str, list[str]]:
    """Check database for required tables.

    Returns dict with 'present' and 'missing' table lists.
    """
    try:
        import duckdb

        conn = duckdb.connect(str(CONFIG.db_path), read_only=True)
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]

        required_tables = ["buildings", "panoramas", "tilesets"]
        present = [t for t in required_tables if t in table_names]
        missing = [t for t in required_tables if t not in table_names]

        conn.close()
        return {"present": present, "missing": missing}

    except Exception as e:
        logger.error(f"Database validation failed: {e}")
        return {"present": [], "missing": []}
