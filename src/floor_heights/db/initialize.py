"""Schema initialization functions."""

from __future__ import annotations

import psycopg2
from loguru import logger

from .constants import DSN
from .schema import DROP_ALL, ddl_text


def init_schema() -> None:
    """Drop and recreate all user tables."""
    logger.info("Dropping all existing tables...")
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(DROP_ALL)
        logger.info("Creating new database schema...")
        cur.execute(ddl_text())
