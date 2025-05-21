"""Composition helpers for orchestrating database setup."""

from __future__ import annotations

from functools import reduce
from typing import Any, Callable

from .initialize import init_schema
from .importers import (
    associate_tilesets,
    deduplicate_footprints,
    finalize_regions,
    footprints,
    panoramas,
    points,
    tileset,
)


def _compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose single-argument functions left-to-right."""
    return reduce(lambda f, g: lambda x: g(f(x)), funcs)


def run_all(cfg: dict) -> None:
    """Run the entire setup pipeline with ``cfg``."""
    _compose(
        lambda _: init_schema(),
        lambda _: footprints(cfg["footprints"]),
        lambda _: deduplicate_footprints(),
        lambda _: points(**cfg["points"]),
        lambda _: panoramas(**cfg["panoramas"]),
        lambda _: tileset(**cfg["tileset"]),
        lambda _: finalize_regions(),
        lambda _: associate_tilesets(
            cfg["points"]["region_id"], cfg["tileset"]["tileset_id"]
        ),
    )(None)
