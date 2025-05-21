"""Public API for the database submodule."""

from __future__ import annotations

from pathlib import Path

from .initialize import init_schema
from .importers import (
    footprints,
    panoramas,
    points,
    tileset,
    deduplicate_footprints,
    associate_tilesets,
    finalize_regions,
)
from .pipeline import run_all
from .audit import audit_database


def migrate(region):
    from .migrate import main

    return main(region)


def migrate_many(regions):
    from .migrate import migrate_many as _migrate_many

    return _migrate_many(regions)


__all__ = [
    "init_schema",
    "footprints",
    "points",
    "panoramas",
    "tileset",
    "deduplicate_footprints",
    "associate_tilesets",
    "finalize_regions",
    "run_all",
    "audit_database",
    "migrate",
    "migrate_many",
    "Path",
]
