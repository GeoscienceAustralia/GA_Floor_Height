from __future__ import annotations

from pathlib import Path

import ibis

from .ibis_client import connect as _connect


class DuckDBReader:
    def __init__(self, db_path: str | Path, *, read_only: bool = True):
        self.db_path, self.read_only = Path(db_path), read_only
        self._con: ibis.BaseBackend | None = None

    def __enter__(self):
        self._con = _connect(self.db_path, read_only=self.read_only)
        return self

    def __exit__(self, *_):
        if self._con is not None:
            self._con.disconnect()
            self._con = None

    def table(self, name: str):
        return self._con.table(name)

    def read_parquet(self, path: str | Path):
        return self._con.read_parquet(path)
