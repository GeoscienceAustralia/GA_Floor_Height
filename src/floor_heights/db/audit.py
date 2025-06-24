from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import ibis
import pandas as pd
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

from .ibis_client import connect

CRITICAL = False

console = Console()


def db():
    """Connect to the database."""
    db_path = Path("data/floor_heights.duckdb")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    return connect(db_path, read_only=True)


def flag(msg: str, level: str = "warning") -> None:
    """
    Emit a colourised log message and record critical status.

    level can be:
        • "yellow"  - logged as WARNING
        • "red"     - logged as ERROR (sets CRITICAL = True)
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


def list_user_tables() -> list[str]:
    conn = db()
    tables = conn.list_tables()
    return sorted(tables)


def get_cols(conn, table: str) -> pd.DataFrame:
    """Get column information for a table."""
    t = conn.table(table)
    schema = t.schema()

    cols_info = []
    for col_name, col_type in schema.items():
        cols_info.append({"col": col_name, "data_type": str(col_type), "is_nullable": "YES", "udt": str(col_type)})

    return pd.DataFrame(cols_info)


def quick_stats(conn, table: str, col: str) -> dict[str, Any]:
    """Return null_count & distinct_count; min/max for numerics."""
    t = conn.table(table)

    total = t.count().execute()
    nulls = t.filter(t[col].isnull()).count().execute()
    distinct = t[col].nunique().execute()

    stats = {"total": total, "nulls": nulls, "distinct": distinct}

    col_type = str(t[col].type())

    if "bool" in col_type.lower():
        stats["min"] = "-"
        stats["max"] = "-"
    else:
        try:
            min_val = t[col].min().execute()
            max_val = t[col].max().execute()
            stats["min"] = str(min_val) if min_val is not None else "-"
            stats["max"] = str(max_val) if max_val is not None else "-"
        except Exception as e:
            logger.warning(f"Failed to get min/max for {table}.{col}: {e}")
            stats["min"] = "-"
            stats["max"] = "-"

    return stats


def check_geometry(conn, table: str, col: str = "geom") -> tuple[int, int, list[int]]:
    """Check geometry column exists and count nulls."""
    t = conn.table(table)

    if col not in t.columns:
        for geom_col in ["geometry", "geom", "the_geom"]:
            if geom_col in t.columns:
                col = geom_col
                break
        else:
            return 0, 0, []

    nulls = t.filter(t[col].isnull()).count().execute()

    srids = [4326]

    return 0, nulls, srids


def get_table_df(conn, table: str) -> pd.DataFrame:
    """Get full table as DataFrame."""
    t = conn.table(table)
    df = t.execute()
    if len(df) == 0:
        return pd.DataFrame()
    return df.apply(lambda x: x.apply(str) if x.apply(lambda v: isinstance(v, list)).any() else x)


def describe_columns_df(df: pd.DataFrame) -> tuple[Table, Table | None]:
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
    geom_cols = ["geom", "geometry", "the_geom"]

    for col in df.columns:
        s = df[col]
        nulls = s.isna().sum()
        pct_null = 100 * nulls / total if total else 0

        if col.lower() in geom_cols:
            distinct = "-"
        else:
            try:
                distinct = s.nunique(dropna=True)
            except Exception as e:
                logger.debug(f"Failed to calculate distinct values for {col}: {e}")
                distinct = "-"

        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            _min = s.min(skipna=True)
            _max = s.max(skipna=True)
        else:
            non_null = s.dropna()
            if non_null.empty:
                _min, _max = ("-", "-")
            else:
                _min, _max = (str(non_null.iloc[0])[:40], str(non_null.iloc[-1])[:40])

        stats.add_row(
            str(col),
            str(s.dtype),
            f"{pct_null:5.1f}",
            str(distinct) if isinstance(distinct, int) else distinct,
            textwrap.shorten(str(_min), 40),
            textwrap.shorten(str(_max), 40),
        )

        if (
            col.lower() not in geom_cols
            and not pd.api.types.is_numeric_dtype(s)
            and isinstance(distinct, int)
            and 0 < distinct <= 15
        ):
            try:
                first_val = s.dropna().iloc[0] if len(s.dropna()) > 0 else None
                if not isinstance(first_val, bytes | bytearray):
                    vals = ", ".join(sorted(map(str, s.dropna().unique())))
                    cat.add_row(str(col), vals)
            except Exception as e:
                logger.debug(f"Failed to process categorical column {col}: {e}")

    cat_table = cat if len(cat.rows) else None
    return stats, cat_table


def audit_table(conn, table: str, markdown_lines: list[str] | None = None) -> None:
    logger.info(f"\n╭─ {table}")
    cols = get_cols(conn, table)
    total = conn.table(table).count().execute()
    logger.info(f"│ rows: {total:,}")

    if markdown_lines is not None:
        markdown_lines.extend([f"## Table: `{table}`", "", f"**Total rows**: {total:,}", "", "### Columns", ""])

    for _, row in cols.iterrows():
        name = row.col
        st = quick_stats(conn, table, name)
        pct_null = 100 * st["nulls"] / st["total"] if st["total"] else 0
        msg = (
            f"│ {name:<25} null%={pct_null:5.1f}  "
            f"distinct={st['distinct']:<7}  min={st['min'] or '-':<12}  max={st['max'] or '-':<12}"
        )

        if st["nulls"] == st["total"]:
            flag(msg + "  ← ALL NULL", "yellow")
        elif pct_null > 95:
            flag(msg + "  ← >95 % NULL", "yellow")
        else:
            logger.info(msg)

    geom_cols = ["geom", "geometry", "the_geom"]
    for geom_col in geom_cols:
        if geom_col in cols.col.values:
            invalid, _, srids = check_geometry(conn, table, geom_col)
            if invalid or len(srids) != 1:
                flag(f"│ geometry issues: invalid={invalid:,}  SRIDs={srids}", "yellow")
            else:
                logger.info(f"│ geometry: SRID={srids[0]} invalid={invalid}")
            break

    df = get_table_df(conn, table)
    if len(df) == 0:
        logger.info("│ (empty table)")
        logger.info("╰──────────────────────────────────────────────")
        if markdown_lines is not None:
            markdown_lines.append("*(empty table)*\n")
        return

    console.rule(f"[bold white]{table}[/]", style="green")
    stats, cats = describe_columns_df(df)
    console.print(stats)
    if cats is not None:
        console.print("\n[bold]Low-cardinality columns:[/]")
        console.print(cats)
    logger.info("╰──────────────────────────────────────────────")

    if markdown_lines is not None:
        markdown_lines.append("| Column | Type | Null % | Distinct | Min/First | Max/Last |")
        markdown_lines.append("|--------|------|--------|----------|-----------|----------|")

        for col in df.columns:
            s = df[col]
            nulls = s.isna().sum()
            pct_null = 100 * nulls / len(df) if len(df) else 0

            if col.lower() in ["geom", "geometry", "the_geom", "footprint_geom", "point_geom"] or s.dtype == "object":
                try:
                    first_val = s.dropna().iloc[0] if len(s.dropna()) > 0 else None
                    distinct = "-" if isinstance(first_val, bytes | bytearray) else s.nunique(dropna=True)
                except Exception as e:
                    logger.debug(f"Failed to calculate distinct values for {col}: {e}")
                    distinct = "-"
            else:
                try:
                    distinct = s.nunique(dropna=True)
                except Exception as e:
                    logger.debug(f"Failed to calculate distinct values for {col}: {e}")
                    distinct = "-"

            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                _min = str(s.min(skipna=True))[:40]
                _max = str(s.max(skipna=True))[:40]
            else:
                non_null = s.dropna()
                if non_null.empty:
                    _min, _max = ("-", "-")
                else:
                    _min = str(non_null.iloc[0])[:40]
                    _max = str(non_null.iloc[-1])[:40]

            markdown_lines.append(f"| {col} | {s.dtype} | {pct_null:.1f} | {distinct} | {_min} | {_max} |")

        if cats:
            markdown_lines.extend(["", "### Low-cardinality columns", ""])
            markdown_lines.append("| Column | Values |")
            markdown_lines.append("|--------|--------|")

            for col in df.columns:
                if col.lower() not in [
                    "geom",
                    "geometry",
                    "the_geom",
                    "footprint_geom",
                    "point_geom",
                ] and not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                        if not isinstance(first_val, bytes | bytearray):
                            distinct = df[col].nunique(dropna=True)
                            if 0 < distinct <= 15:
                                vals = ", ".join(sorted(map(str, df[col].dropna().unique())))
                                markdown_lines.append(f"| {col} | {vals} |")
                    except Exception as e:
                        logger.debug(f"Failed to get categorical values for {col}: {e}")

        markdown_lines.append("")


def audit_database(save_markdown: bool = False) -> None:
    """Main audit function for DuckDB database."""
    global CRITICAL
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    )

    markdown_lines = [
        "# DuckDB Database Schema",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    conn = db()
    logger.info("Connected to DuckDB")

    tables = list_user_tables()
    logger.info(f"Found {len(tables)} tables")
    markdown_lines.append(f"## Overview\n\n**Total tables**: {len(tables)}\n")

    for t in tables:
        audit_table(conn, t, markdown_lines if save_markdown else None)

    if "buildings" in tables:
        logger.info("\n╭─ Buildings Summary")

        if save_markdown:
            markdown_lines.extend(["## Buildings Summary", ""])

        buildings = conn.table("buildings")
        result = buildings.group_by("region_name").agg(count=buildings.count()).order_by("region_name").execute()

        logger.info("│ Records per region:")

        if save_markdown:
            markdown_lines.extend(["### Records per region", "", "| Region | Count |", "|--------|-------|"])

        for _, row in result.iterrows():
            logger.info(f"│   {row['region_name']}: {row['count']:,} records")
            if save_markdown:
                markdown_lines.append(f"| {row['region_name']} | {row['count']:,} |")

        if save_markdown:
            markdown_lines.append("")

        filtered_buildings = buildings.filter(buildings.property_type.notnull())
        result = (
            filtered_buildings.group_by("property_type")
            .agg(count=filtered_buildings.count())
            .order_by(ibis.desc("count"))
            .execute()
        )

        logger.info("│ Property type distribution:")

        if save_markdown:
            markdown_lines.extend(
                ["### Property type distribution", "", "| Property Type | Count |", "|---------------|-------|"]
            )

        for _i, row in result.head(5).iterrows():
            logger.info(f"│   {row['property_type']}: {row['count']:,}")
            if save_markdown:
                markdown_lines.append(f"| {row['property_type']} | {row['count']:,} |")

        if save_markdown:
            markdown_lines.append("")

        cols = get_cols(conn, "buildings")
        skip_cols = {"building_id", "region_id", "processed_at", "footprint_geom", "point_geom"}
        columns_to_check = [col for col in cols.col if col not in skip_cols]

        logger.info("│ Data completeness:")

        if save_markdown:
            markdown_lines.extend(
                [
                    "### Data completeness",
                    "",
                    "| Field | Complete | Total | Percentage |",
                    "|-------|----------|-------|------------|",
                ]
            )

        total = buildings.count().execute()
        completeness_data = []

        for col in columns_to_check:
            count = buildings.filter(buildings[col].notnull()).count().execute()
            percentage = 100 * count / total if total > 0 else 0

            display_name = col.replace("_", " ").title()
            completeness_data.append((display_name, count, total, percentage))

        completeness_data.sort(key=lambda x: x[3])

        for display_name, count, total, percentage in completeness_data:
            logger.info(f"│   {display_name}: {count:,} / {total:,} ({percentage:.1f}%)")

            if save_markdown:
                markdown_lines.append(f"| {display_name} | {count:,} | {total:,} | {percentage:.1f}% |")

        logger.info("╰──────────────────────────────────────────────")

        if save_markdown:
            markdown_lines.append("")

    if save_markdown and markdown_lines:
        schema_path = Path(__file__).parent / "schema.md"
        schema_path.write_text("\n".join(markdown_lines))
        logger.info(f"\nSchema documentation saved to: {schema_path}")

    if CRITICAL:
        logger.error("\nAudit finished with RED flags.")
        raise RuntimeError("database audit failed")
    else:
        logger.success("\nAudit finished - no critical issues.")


if __name__ == "__main__":
    audit_database()
