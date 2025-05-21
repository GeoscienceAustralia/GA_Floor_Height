"""Deep-dive QA report for the floor-heights database."""

from __future__ import annotations
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import pandas as pd
from loguru import logger

from rich.console import Console
from .constants import DSN
from rich.table import Table
from rich import box
from tabulate import tabulate

DB = DSN

CRITICAL = False  # becomes True when a red‑flag is raised

console = Console()


def flag(msg: str, level: str = "warning") -> None:
    """
    Emit a colourised log message and record critical status.

    level can be:
        • "yellow"  – logged as WARNING
        • "red"     – logged as ERROR (sets CRITICAL = True)
        • any other valid Loguru level name / number
    """
    global CRITICAL
    level_lower = level.lower()
    if level_lower == "yellow":
        logger.warning(msg)
    elif level_lower == "red":
        CRITICAL = True
        logger.error(msg)
    else:
        logger.log(level.upper() if isinstance(level, str) else level, msg)


def list_user_tables(cur: psycopg2.extensions.cursor, schema: str) -> List[str]:
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type   = 'BASE TABLE'
          AND table_name NOT IN ('spatial_ref_sys')
        ORDER BY table_name
        """,
        (schema,),
    )
    return [r[0] for r in cur.fetchall()]


def get_cols(cur: psycopg2.extensions.cursor, schema: str, table: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT column_name, data_type, is_nullable, udt_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return pd.DataFrame(
        cur.fetchall(), columns=["col", "data_type", "is_nullable", "udt"]
    )


def quick_stats(
    cur: psycopg2.extensions.cursor, table: str, col: str
) -> Dict[str, Any]:
    """Return null_count & distinct_count; min/max for numerics."""
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    total = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL")
    nulls = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table}")
    distinct = cur.fetchone()[0]

    stats = {"total": total, "nulls": nulls, "distinct": distinct}

    # Check if column is boolean by checking its data type
    cur.execute(
        """
        SELECT data_type 
        FROM information_schema.columns 
        WHERE table_name = %s AND column_name = %s
        """,
        (table, col),
    )
    data_type = cur.fetchone()[0]

    # Skip min/max for boolean columns
    if data_type == "boolean":
        stats["min"] = "-"
        stats["max"] = "-"
    else:
        # min/max for numeric & timestamp
        cur.execute(
            f"""
            SELECT
              MIN({col})::text,
              MAX({col})::text
            FROM {table}
            WHERE {col} IS NOT NULL
            """
        )
        mn, mx = cur.fetchone()
        stats["min"] = mn
        stats["max"] = mx

    return stats


def check_geometry(
    cur: psycopg2.extensions.cursor, table: str, col: str = "geom"
) -> Tuple[int, int, List[int]]:
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE NOT ST_IsValid({col})")
    invalid = cur.fetchone()[0]
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL")
    nulls = cur.fetchone()[0]
    cur.execute(f"SELECT DISTINCT ST_SRID({col}) FROM {table}")
    srids = [r[0] for r in cur.fetchall()]
    return invalid, nulls, srids


def orphan_building_points(cur):
    cur.execute(
        """
        SELECT COUNT(*)
        FROM building_points bp
        LEFT JOIN building_footprints bf ON bf.id = bp.footprint_id
        WHERE bp.footprint_id IS NOT NULL
          AND bf.id IS NULL
        """
    )
    return cur.fetchone()[0]


def duplicate_source_ids(cur):
    cur.execute(
        """
        SELECT region_id, source_point_id, COUNT(*)
        FROM building_points
        GROUP BY region_id, source_point_id
        HAVING COUNT(*) > 1
        LIMIT 5
        """
    )
    rows = cur.fetchall()
    return len(rows), rows


def orphan_extension_rows(cur: psycopg2.extensions.cursor, ext_table: str) -> int:
    cur.execute(
        f"""
        SELECT COUNT(*)
        FROM {ext_table} ext
        LEFT JOIN building_points bp ON bp.id = ext.building_point_id
        WHERE ext.building_point_id IS NOT NULL
          AND bp.id IS NULL
        """
    )
    result = cur.fetchone()[0]
    return int(result)


def orphan_tilesets_region_id(cur: psycopg2.extensions.cursor) -> int:
    """Check for orphaned tilesets rows (where region_id points to non-existent region)."""
    cur.execute(
        """
        SELECT COUNT(*)
        FROM tilesets t
        LEFT JOIN regions r ON r.id = t.region_id
        WHERE t.region_id IS NOT NULL
          AND r.id IS NULL
        """
    )
    result = cur.fetchone()[0]
    return int(result)


def orphan_tileset_indexes_tileset_id(cur: psycopg2.extensions.cursor) -> int:
    """Check for orphaned tileset_indexes rows (where tileset_id points to non-existent tileset)."""
    cur.execute(
        """
        SELECT COUNT(*)
        FROM tileset_indexes ti
        LEFT JOIN tilesets t ON t.id = ti.tileset_id
        WHERE ti.tileset_id IS NOT NULL
          AND t.id IS NULL
        """
    )
    result = cur.fetchone()[0]
    return int(result)


def orphan_bldg_tileset_assoc_bldg_id(cur: psycopg2.extensions.cursor) -> int:
    """Check for orphaned building_tileset_associations (where building_id points to non-existent building)."""
    cur.execute(
        """
        SELECT COUNT(*)
        FROM building_tileset_associations bta
        LEFT JOIN building_footprints bf ON bf.id = bta.building_id
        WHERE bta.building_id IS NOT NULL
          AND bf.id IS NULL
        """
    )
    result = cur.fetchone()[0]
    return int(result)


def orphan_bldg_tileset_assoc_tile_idx_id(cur: psycopg2.extensions.cursor) -> int:
    """Check for orphaned building_tileset_associations (where tileset_index_id points to non-existent tile)."""
    cur.execute(
        """
        SELECT COUNT(*)
        FROM building_tileset_associations bta
        LEFT JOIN tileset_indexes ti ON ti.id = bta.tileset_index_id
        WHERE bta.tileset_index_id IS NOT NULL
          AND ti.id IS NULL
        """
    )
    result = cur.fetchone()[0]
    return int(result)


def get_table_df(cur: psycopg2.extensions.cursor, table: str) -> pd.DataFrame:
    cur.execute(f"SELECT * FROM {table}")
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=colnames)
    # Convert array columns to strings to avoid unhashable type errors
    result_df: pd.DataFrame = df.apply(
        lambda x: x.apply(str) if x.apply(lambda v: isinstance(v, list)).any() else x
    )
    return result_df


def describe_columns_df(df: pd.DataFrame) -> Tuple[Table, Optional[Table]]:
    """
    Returns a tuple of (stats_table, cat_table or None) using rich.Table.
    """
    stats = Table(box=box.MINIMAL, show_lines=False)
    stats.add_column("column", style="bold cyan")
    stats.add_column("dtype", style="dim")
    stats.add_column("null%", justify="right")
    stats.add_column("distinct", justify="right")
    stats.add_column("min / first", overflow="fold")
    stats.add_column("max / last", overflow="fold")

    cat = Table(box=box.SIMPLE_HEAVY)
    cat.add_column("categorical column", style="magenta")
    cat.add_column("distinct values", overflow="fold")

    total = len(df)
    for col in df.columns:
        s = df[col]
        nulls = s.isna().sum()
        pct_null = 100 * nulls / total if total else 0
        distinct = s.nunique(dropna=True)

        # numeric / datetime get min/max; others first/last
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            _min = s.min(skipna=True)
            _max = s.max(skipna=True)
        else:
            non_null = s.dropna()
            if non_null.empty:
                _min, _max = ("-", "-")
            else:
                _min, _max = (non_null.iloc[0], non_null.iloc[-1])

        stats.add_row(
            str(col),
            str(s.dtype),
            f"{pct_null:5.1f}",
            f"{distinct:,}",
            textwrap.shorten(str(_min), 40),
            textwrap.shorten(str(_max), 40),
        )

        # collect low-cardinality categorical columns
        if (not pd.api.types.is_numeric_dtype(s)) and 0 < distinct <= 15:
            vals = ", ".join(sorted(map(str, s.dropna().unique())))
            cat.add_row(str(col), vals)

    cat_table = cat if len(cat.rows) else None
    return stats, cat_table


def format_column_stats(column_series: pd.Series, total: int) -> dict:
    """Extract statistics for a single column."""
    nulls = column_series.isna().sum()
    pct_null = 100 * nulls / total if total else 0
    distinct = column_series.nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(
        column_series
    ) or pd.api.types.is_datetime64_any_dtype(column_series):
        min_val = column_series.min(skipna=True)
        max_val = column_series.max(skipna=True)
    else:
        non_null = column_series.dropna()
        min_val, max_val = (
            (non_null.iloc[0], non_null.iloc[-1]) if not non_null.empty else ("-", "-")
        )

    return {
        "dtype": str(column_series.dtype),
        "null_pct": f"{pct_null:5.1f}",
        "distinct": f"{distinct:,}",
        "min": textwrap.shorten(str(min_val), 40, placeholder="[...]"),
        "max": textwrap.shorten(str(max_val), 40, placeholder="[...]"),
    }


def extract_categorical_columns(df: pd.DataFrame) -> dict:
    """Identify low-cardinality categorical columns."""
    return {
        col: sorted(map(str, df[col].dropna().unique()))
        for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
        and 0 < df[col].nunique(dropna=True) <= 15
        and col != "geom"
    }


def create_markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    """Generate markdown table lines from headers and rows."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def check_pg_stat_statements(cur: psycopg2.extensions.cursor) -> bool:
    """Check if pg_stat_statements extension is available."""
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
            );
        """)
        return bool(cur.fetchone()[0])
    except Exception:
        return False


def audit_query_performance(cur: psycopg2.extensions.cursor) -> None:
    """Audit query performance using pg_stat_statements."""
    logger.info("\n╭─ Query Performance Analysis")

    if not check_pg_stat_statements(cur):
        logger.warning("│ pg_stat_statements extension not available")
        logger.info("╰──────────────────────────────────────────────")
        return

    # Slowest queries
    logger.info("│ Slowest queries by mean execution time:")
    cur.execute("""
        SELECT 
            substring(query, 1, 60) AS query_preview,
            round(mean_exec_time::numeric, 2) AS mean_ms,
            round(total_exec_time::numeric / 1000, 2) AS total_sec,
            calls
        FROM pg_stat_statements
        WHERE query NOT LIKE '%pg_stat_statements%'
          AND query NOT LIKE 'BEGIN%'
          AND query NOT LIKE 'COMMIT%'
        ORDER BY mean_exec_time DESC
        LIMIT 5;
    """)
    rows = cur.fetchall()
    if rows:
        console.print(
            tabulate(
                rows,
                headers=["Query", "Mean(ms)", "Total(s)", "Calls"],
                tablefmt="grid",
            )
        )

    # Spatial query performance
    logger.info("│\n│ Spatial query performance:")
    cur.execute("""
        SELECT 
            substring(query, 1, 60) AS query_preview,
            round(mean_exec_time::numeric, 2) AS mean_ms,
            calls,
            shared_blks_hit + shared_blks_read AS total_blocks,
            round(shared_blks_hit::numeric / NULLIF(shared_blks_hit + shared_blks_read, 0) * 100, 2) AS cache_hit_pct
        FROM pg_stat_statements
        WHERE (query LIKE '%ST_%' OR query LIKE '%geom%')
          AND query NOT LIKE '%pg_stat_statements%'
        ORDER BY mean_exec_time DESC
        LIMIT 5;
    """)
    rows = cur.fetchall()
    if rows:
        console.print(
            tabulate(
                rows,
                headers=["Query", "Mean(ms)", "Calls", "Blocks", "Cache%"],
                tablefmt="grid",
            )
        )

    # Poor cache hit queries
    logger.info("│\n│ Queries with poor cache hit rates:")
    cur.execute("""
        SELECT 
            substring(query, 1, 60) AS query_preview,
            shared_blks_read AS disk_reads,
            shared_blks_hit AS cache_hits,
            round(shared_blks_hit::numeric / NULLIF(shared_blks_hit + shared_blks_read, 0) * 100, 2) AS cache_hit_pct
        FROM pg_stat_statements
        WHERE shared_blks_hit + shared_blks_read > 1000  -- Only queries with significant I/O
          AND query NOT LIKE '%pg_stat_statements%'
        ORDER BY cache_hit_pct ASC NULLS LAST
        LIMIT 5;
    """)
    rows = cur.fetchall()
    if rows:
        for row in rows:
            if row[3] is not None and row[3] < 50:  # Cache hit rate below 50%
                flag(f"│ Poor cache hit: {row[0][:40]}... (cache={row[3]}%)", "yellow")
        console.print(
            tabulate(
                rows,
                headers=["Query", "Disk Reads", "Cache Hits", "Cache%"],
                tablefmt="grid",
            )
        )

    logger.info("╰──────────────────────────────────────────────")


def audit_table(cur: psycopg2.extensions.cursor, schema: str, table: str) -> None:
    logger.info(f"\n╭─ {table}")
    cols = get_cols(cur, schema, table)
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    total = cur.fetchone()[0]
    logger.info(f"│ rows: {total:,}")

    for _, row in cols.iterrows():
        name = row.col
        st = quick_stats(cur, table, name)
        pct_null = 100 * st["nulls"] / st["total"] if st["total"] else 0
        msg = (
            f"│ {name:<25} null%={pct_null:5.1f}  "
            f"distinct={st['distinct']:<7}  min={st['min'] or '-':<12}  max={st['max'] or '-':<12}"
        )

        if st["nulls"] == st["total"]:
            flag(msg + "  ← ALL NULL", "red")
        elif pct_null > 95:
            flag(msg + "  ← >95 % NULL", "yellow")
        else:
            logger.info(msg)

    # geometry checks
    if "geom" in cols.col.values:
        invalid, gnulls, srids = check_geometry(cur, table)
        if invalid or len(srids) != 1:
            flag(f"│ geometry issues: invalid={invalid:,}  SRIDs={srids}", "yellow")
        else:
            logger.info(f"│ geometry: SRID={srids[0]} invalid={invalid}")

    # Value distribution table (added for all tables)
    df = get_table_df(cur, table)
    if len(df) == 0:
        logger.info("│ (empty table)")
        logger.info("╰──────────────────────────────────────────────")
        return

    console.rule(f"[bold white]{table}[/]", style="green")
    stats, cats = describe_columns_df(df)
    console.print(stats)
    if cats is not None:
        console.print("\n[bold]Low-cardinality columns:[/]")
        console.print(cats)
    logger.info("╰──────────────────────────────────────────────")


def audit_extension_table(cur: psycopg2.extensions.cursor, ext_table: str) -> None:
    logger.info(f"\n╭─ {ext_table} (extension table)")
    # Orphan check
    orphans = orphan_extension_rows(cur, ext_table)
    if orphans:
        flag(
            f"❌ Orphaned {ext_table}.building_point_id → building_points: {orphans:,}",
            "red",
        )
    else:
        logger.info(f"✅ No orphaned {ext_table}.building_point_id")

    # Value distribution table
    df = get_table_df(cur, ext_table)
    logger.info(f"│ rows: {len(df):,}")
    if len(df) == 0:
        logger.info("│ (empty table)")
        logger.info("╰──────────────────────────────────────────────")
        return

    console.rule(f"[bold white]{ext_table}[/]", style="green")
    stats, cats = describe_columns_df(df)
    console.print(stats)
    if cats is not None:
        console.print("\n[bold]Low-cardinality columns:[/]")
        console.print(cats)
    logger.info("╰──────────────────────────────────────────────")


def audit_database(schema: str) -> None:
    global CRITICAL
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    )

    logger.info(f"Connecting → {DB}")
    with psycopg2.connect(DB) as conn, conn.cursor() as cur:
        tables = list_user_tables(cur, schema)
        # Standard audit for all tables except extension tables
        ext_tables = [
            "wagga_building_extensions",
            "tweed_building_extensions",
            "launceston_building_extensions",
        ]
        for t in tables:
            if t not in ext_tables:
                audit_table(cur, schema, t)
        # Special audit for extension tables
        for ext in ext_tables:
            if ext in tables:
                audit_extension_table(cur, ext)

        # cross-table checks
        orphans = orphan_building_points(cur)
        if orphans:
            flag(f"\n❌ Orphaned building_points → footprints: {orphans:,}", "red")
        else:
            logger.info("\n✅ No orphaned building_points.footprint_id")

        dups, rows = duplicate_source_ids(cur)
        if dups:
            flag(f"❌ Duplicate (region_id, source_point_id) rows: {dups}", "red")
            for r in rows:
                flag(f"   region={r[0]}  source_id={r[1]}  count={r[2]}", "yellow")
        else:
            logger.info("✅ No duplicate (region_id, source_point_id) pairs")

        # Tileset cross-table checks
        for check_name, check_func in [
            ("tilesets.region_id → regions", orphan_tilesets_region_id),
            (
                "tileset_indexes.tileset_id → tilesets",
                orphan_tileset_indexes_tileset_id,
            ),
            (
                "building_tileset_associations.building_id → building_footprints",
                orphan_bldg_tileset_assoc_bldg_id,
            ),
            (
                "building_tileset_associations.tileset_index_id → tileset_indexes",
                orphan_bldg_tileset_assoc_tile_idx_id,
            ),
        ]:
            orphans = check_func(cur)
            if orphans:
                flag(f"❌ Orphaned {check_name}: {orphans:,}", "red")
            else:
                logger.info(f"✅ No orphaned {check_name}")

        # Query performance audit
        audit_query_performance(cur)

    if CRITICAL:
        logger.error("\nAudit finished with RED flags.")
        raise RuntimeError("database audit failed")
    else:
        logger.success("\nAudit finished – no critical issues.")
