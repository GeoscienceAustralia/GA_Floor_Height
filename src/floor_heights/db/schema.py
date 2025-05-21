"""Pure SQL strings consumed by :func:`initialize.init_schema`."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from .constants import Dirs

# load *.sql deterministically (001_xxx.sql, 002_…, …)
DDL_FILES: Final[list[Path]] = sorted(list(Dirs.sql.glob("*.sql")))


def ddl_text() -> str:
    """Concatenate bundled SQL files."""
    return "\n\n".join(p.read_text() for p in DDL_FILES)


DROP_ALL: Final[str] = """
DO $$
DECLARE r record;
BEGIN
  FOR r IN (
      SELECT tablename
      FROM pg_tables
      WHERE schemaname = current_schema()
        AND tablename NOT LIKE 'spatial_ref_sys')
  LOOP
    EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', r.tablename);
  END LOOP;
END$$;
"""
