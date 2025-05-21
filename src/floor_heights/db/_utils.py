"""Private helper utilities for the DB module."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from .constants import DSN


@lru_cache(maxsize=None)
def db() -> psycopg2.extensions.connection:
    """Return a cached database connection."""
    return psycopg2.connect(DSN)


def read_layers(path: Path) -> gpd.GeoDataFrame:
    """Load all layers of a GeoPackage using pyogrio."""
    from pyogrio import list_layers, read_dataframe

    layers = [layer_info[0] for layer_info in list_layers(path)]
    frames = (
        read_dataframe(path, layer=layer_name, use_arrow=True) for layer_name in layers
    )
    first = next(frames)
    rest = list(frames)
    return gpd.GeoDataFrame(pd.concat([first, *rest], ignore_index=True), crs=first.crs)


def ensure_valid(gdf: gpd.GeoDataFrame, label: str) -> None:
    """Raise if any geometries are invalid."""
    invalid = (~gdf.is_valid).sum()
    if invalid:
        raise ValueError(f"{label}: {invalid} invalid geometries")


def nan_to_none(x: float | int | None) -> float | None:
    """Convert NaN values to ``None`` for database insertion."""
    return None if (pd.isna(x) or (isinstance(x, float) and np.isnan(x))) else float(x)


def pg_copy(
    table: str,
    cols: tuple[str, ...],
    rows: Iterable[tuple],
    conflict: tuple[str, ...] | None = None,
) -> int:
    """Bulk insert rows with optional upsert. Returns number of rows inserted."""
    from loguru import logger

    rows_list = list(rows)
    row_count = len(rows_list)

    if row_count == 0:
        logger.info(f"No rows to insert into table {table}")
        return 0

    tmpl = "(" + ",".join(["%s"] * len(cols)) + ")"
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES %s" + (
        ""
        if conflict is None
        else " ON CONFLICT ("
        + ",".join(conflict)
        + ") DO UPDATE SET "
        + ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in conflict)
    )

    logger.info(f"Inserting {row_count:,} rows into table '{table}'")

    with db().cursor() as cur:
        execute_values(cur, sql, rows_list, template=tmpl)
    db().commit()

    logger.success(f"Successfully inserted {row_count:,} rows into table '{table}'")
    return row_count


def load_json(path: str | Path) -> dict[str, object]:
    """Load a JSON file from disk."""
    result: dict[str, object] = json.loads(Path(path).read_text())
    return result
