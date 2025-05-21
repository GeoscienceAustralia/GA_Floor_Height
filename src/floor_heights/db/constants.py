"""Immutable configuration for database access."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BASE: Final[Path] = Path(__file__).resolve().parents[3]
DATA: Final[Path] = BASE / "data"
SQL_DIR: Final[Path] = BASE / "sql"

CRS_LATLON: Final[int] = 4326
CRS_PLANAR: Final[int] = 3857


def get_dsn() -> str:
    """Get database connection string from environment."""
    db_str = os.environ.get("DB_CONNECTION_STRING")
    if not db_str:
        print(
            "WARNING: DB_CONNECTION_STRING not found in environment. Using default local connection."
        )
        return "postgresql://postgres:postgres@localhost:5432/floor_heights"
    return db_str


DSN: str = get_dsn()


@dataclass(frozen=True)
class Dirs:
    """Project directories."""

    root: Path = BASE
    data: Path = DATA
    sql: Path = SQL_DIR
