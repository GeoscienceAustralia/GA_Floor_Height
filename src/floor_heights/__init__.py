"""Floor Heights package for estimating building heights."""

from __future__ import annotations

from pathlib import Path


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
