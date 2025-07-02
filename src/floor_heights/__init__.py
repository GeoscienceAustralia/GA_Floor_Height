"""Floor Heights package for estimating building heights."""

from __future__ import annotations

import os
from pathlib import Path

# Set YOLO/Ultralytics directories before any imports
_project_root = Path(__file__).parent.parent.parent
_cache_dir = _project_root / "weights" / ".cache"
_cache_dir.mkdir(parents=True, exist_ok=True)

os.environ["YOLO_CONFIG_DIR"] = str(_cache_dir)
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(_cache_dir)
os.environ["ULTRALYTICS_WEIGHTS_DIR"] = str(_cache_dir)


def __getattr__(name):
    """Lazy import to avoid config loading at import time."""
    if name in [
        "load_from_parquet",
        "convert_to_parquet",
        "PROPERTY_TYPE_MAP",
        "WALL_MATERIAL_MAP",
        "REGION_CONFIGS",
        "audit_database",
        "DuckDBReader",
    ]:
        import floor_heights.db as db

        return getattr(db, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PROPERTY_TYPE_MAP",
    "REGION_CONFIGS",
    "WALL_MATERIAL_MAP",
    "DuckDBReader",
    "Path",
    "audit_database",
    "convert_to_parquet",
    "load_from_parquet",
]

__version__ = "0.1.0"
