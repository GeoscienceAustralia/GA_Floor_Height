from pathlib import Path

import ibis


def connect(db_path: str | Path, *, read_only: bool = True):
    conn = ibis.duckdb.connect(
        database=str(db_path),
        read_only=read_only,
    )
    conn.raw_sql("INSTALL spatial;")
    conn.raw_sql("LOAD spatial;")
    return conn
