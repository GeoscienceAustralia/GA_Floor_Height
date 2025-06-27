"""Database management commands for the Floor Heights CLI."""

import contextlib
import time
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

from floor_heights.config import CONFIG
from floor_heights.db.audit import audit_database
from floor_heights.db.ibis_client import connect as ibis_connect
from floor_heights.db.run_pipeline import run_pipeline as run_db_pipeline

console = Console()


def create_db_app() -> typer.Typer:
    """Create the database subcommand app."""
    app = typer.Typer(help="Database management utilities")

    @app.callback(invoke_without_command=True)
    def db_callback(ctx: typer.Context) -> None:
        """Show help when no subcommand is provided."""
        if ctx.invoked_subcommand is None:
            console.print(ctx.get_help())
            raise typer.Exit(0)

    @app.command("pipeline", help="Run full database processing pipeline")
    def pipeline(
        skip_convert: bool = typer.Option(
            False,
            "--skip-convert",
            help="Skip the raw-to-parquet conversion step",
        ),
        skip_load: bool = typer.Option(
            False,
            "--skip-load",
            help="Skip the parquet-to-duckdb loading step",
        ),
        skip_audit: bool = typer.Option(
            False,
            "--skip-audit",
            help="Skip the audit step",
        ),
    ) -> None:
        """Run the complete database processing pipeline.

        This executes the full data processing pipeline:
        1. Convert raw spatial data to GeoParquet format
        2. Load GeoParquet files into DuckDB
        3. Run comprehensive audit and generate schema documentation

        The pipeline validates all required data files before starting and
        provides detailed progress reporting throughout the process.
        """
        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(CONFIG.project_root)
            logger.info(f"Working directory: {Path.cwd()}")

            result = run_db_pipeline(skip_convert=skip_convert, skip_load=skip_load, skip_audit=skip_audit)

            if result == 0:
                console.print("[green]✓ Pipeline completed successfully![/green]")
            else:
                console.print("[red]✗ Pipeline failed[/red]")
                raise typer.Exit(result)

        finally:
            os.chdir(original_cwd)

    @app.command("audit", help="Audit the database and generate schema documentation")
    def audit(
        save_markdown: bool = typer.Option(
            True,
            "--save-markdown/--no-save-markdown",
            help="Save audit results as markdown documentation",
        ),
    ) -> None:
        """Run comprehensive database audit.

        This command analyzes the database structure and content, generating:
        - Schema documentation with table structures
        - Row counts and data statistics
        - Relationship mapping between tables
        - Detailed markdown documentation (if enabled)
        """
        try:
            if not CONFIG.db_path.exists():
                console.print(f"[red]Database not found at: {CONFIG.db_path}[/red]")
                console.print("Run 'fh db pipeline' to create the database first")
                raise typer.Exit(1)

            console.print("[cyan]Running database audit...[/cyan]")
            audit_database(save_markdown=save_markdown)

            if save_markdown:
                schema_path = CONFIG.db_path.parent / "schema.md"
                console.print(f"[green]✓ Schema documentation saved to: {schema_path}[/green]")

            console.print("[green]✓ Database audit complete![/green]")

        except Exception as e:
            console.print(f"[red]Error during audit: {e}[/red]")
            raise typer.Exit(1) from e

    @app.command("info", help="Show database information and statistics")
    def info() -> None:
        """Display information about the current database.

        Shows:
        - Database location and size
        - Table names and row counts
        - Buildings by region breakdown
        """
        try:
            if not CONFIG.db_path.exists():
                console.print(f"[yellow]Database not found at: {CONFIG.db_path}[/yellow]")
                console.print("Run 'fh db pipeline' to create the database")
                raise typer.Exit(0)

            db_size = CONFIG.db_path.stat().st_size / (1024 * 1024)
            console.print("\n[bold]Database Information:[/bold]")
            console.print(f"  Location: {CONFIG.db_path}")
            console.print(f"  Size: {db_size:.1f} MB")

            conn = ibis_connect(CONFIG.db_path, read_only=True)

            tables = conn.list_tables()

            console.print("\n[bold]Tables:[/bold]")
            for table_name in sorted(tables):
                try:
                    count = conn.table(table_name).count().execute()
                    console.print(f"  • {table_name}: {count:,} rows")
                except Exception:
                    console.print(f"  • {table_name}: [yellow]Error reading[/yellow]")

            if "buildings" in tables:
                with contextlib.suppress(Exception):
                    buildings = conn.table("buildings")
                    regions_data = (
                        buildings.group_by("region_name").agg(count=buildings.count()).order_by("region_name").execute()
                    )

                    if not regions_data.empty:
                        console.print("\n[bold]Buildings by region:[/bold]")
                        total = 0
                        for row in regions_data.itertuples(index=False):
                            console.print(f"  • {row.region_name}: {row.count:,} buildings")
                            total += row.count
                        console.print(f"  [dim]Total: {total:,} buildings[/dim]")

            console.print()

        except Exception as e:
            console.print(f"[red]Error reading database: {e}[/red]")
            raise typer.Exit(1) from e

    @app.command("show", help="Display table contents")
    def show(
        table: str | None = typer.Argument(None, help="Table name to display"),
        limit: int = typer.Option(10, "--limit", "-n", help="Number of rows to show"),
        where: str | None = typer.Option(None, "--where", "-w", help="Filter condition (e.g., 'region_name=wagga')"),
    ) -> None:
        """Display table contents in a formatted view."""

        import duckdb

        max_retries = 3
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                conn = ibis_connect(CONFIG.db_path, read_only=True)
                break
            except duckdb.IOException as e:
                if "lock" in str(e).lower() and attempt < max_retries - 1:
                    console.print(f"[yellow]Database locked, retrying in {retry_delay:.1f}s...[/yellow]")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                console.print(f"[red]Failed to connect to database: {e}[/red]")
                console.print(
                    "[yellow]The database may be locked by another process. Please try again in a moment.[/yellow]"
                )
                return

        tables = conn.list_tables()

        if not table:
            console.print("\n[bold]Available tables:[/bold]")
            for t in sorted(tables):
                console.print(f"  • {t}")
            return

        if table not in tables:
            console.print(f"[red]Table '{table}' not found[/red]")
            return

        from rich.table import Table as RichTable

        t = conn.table(table)

        if where and "=" in where:
            col, val = where.split("=", 1)
            t = t.filter(t[col.strip()] == val.strip())

        df = t.limit(limit).execute()

        max_cols = 15
        cols = list(df.columns[:max_cols])

        rich_table = RichTable(title=f"{table} (showing {len(df)}/{t.count().execute()} rows)")

        for col in cols:
            rich_table.add_column(col, style="cyan" if col == "id" else None)

        for i in range(len(df)):
            rich_table.add_row(*[str(df[col].iloc[i]) for col in cols])

        console.print(rich_table)

        if len(df.columns) > max_cols:
            console.print(f"\n[dim]Showing {max_cols} of {len(df.columns)} columns[/dim]")

    @app.command("drop", help="Drop a database table")
    def drop(
        table: str = typer.Argument(..., help="Table name to drop"),
        force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    ) -> None:
        """Drop a table from the database."""
        if not force and not typer.confirm(f"Are you sure you want to drop table '{table}'?"):
            console.print("[yellow]Aborted[/yellow]")
            return

        try:
            import duckdb

            conn = duckdb.connect(str(CONFIG.db_path))
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.close()
            console.print(f"[green]✓ Dropped table '{table}'[/green]")
        except Exception as e:
            console.print(f"[red]Error dropping table: {e}[/red]")
            raise typer.Exit(1) from e

    @app.command("query", help="Execute a SQL query")
    def query(
        sql: str = typer.Argument(..., help="SQL query to execute"),
        output: Path | None = typer.Option(None, "--output", "-o", help="Save results to CSV file"),
    ) -> None:
        """Execute a SQL query against the database."""
        try:
            conn = ibis_connect(CONFIG.db_path, read_only=True)

            result = conn.sql(sql).execute()

            if output:
                result.to_csv(output, index=False)
                console.print(f"[green]✓ Results saved to {output}[/green]")
            else:
                from rich.table import Table as RichTable

                display_limit = 100
                display_df = result.head(display_limit)

                table = RichTable(title=f"Query Results ({len(display_df)} of {len(result)} rows)")

                for col in display_df.columns:
                    table.add_column(col)

                for i in range(len(display_df)):
                    table.add_row(*[str(display_df[col].iloc[i]) for col in display_df.columns])

                console.print(table)

                if len(result) > display_limit:
                    console.print(f"\n[dim]Showing first {display_limit} rows. Use --output to save all results.[/dim]")

        except Exception as e:
            console.print(f"[red]Query error: {e}[/red]")
            raise typer.Exit(1) from e

    @app.command("export", help="Export tables to Parquet/GeoParquet format")
    def export(
        table: str | None = typer.Argument(None, help="Table name to export (or 'all' for all tables)"),
        output_dir: Path | None = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Directory to save Parquet files",
        ),
    ) -> None:
        """Export database tables to Parquet/GeoParquet format.

        This command exports tables from DuckDB to Parquet files. Tables with
        geometry columns will automatically be exported as GeoParquet.

        Examples:
            # Export a specific table
            fh db export buildings

            # Export all tables
            fh db export all

            # Export to a specific directory
            fh db export buildings --output-dir /path/to/exports

            # Export all tables (also saves to exports folder)
            fh db export all
        """
        try:
            if not CONFIG.db_path.exists():
                console.print(f"[red]Database not found at: {CONFIG.db_path}[/red]")
                console.print("Run 'fh db pipeline' to create the database first")
                raise typer.Exit(1)

            if output_dir is None:
                output_dir = CONFIG.db_path.parent / "exports"

            output_dir.mkdir(parents=True, exist_ok=True)

            conn = ibis_connect(CONFIG.db_path, read_only=True)

            if table is None:
                console.print("\n[bold]Available tables:[/bold]")
                tables = conn.list_tables()
                for t in sorted(tables):
                    console.print(f"  • {t}")
                console.print("\n[yellow]Specify a table name or 'all' to export[/yellow]")
                raise typer.Exit(0)

            tables_to_export = []
            if table.lower() == "all":
                tables_to_export = conn.list_tables()
            else:
                if table not in conn.list_tables():
                    console.print(f"[red]Table '{table}' not found[/red]")
                    raise typer.Exit(1)

                tables_to_export = [table]

            console.print(f"\n[cyan]Exporting {len(tables_to_export)} table(s) to {output_dir}...[/cyan]")

            con = conn.con

            with contextlib.suppress(Exception):
                con.execute("LOAD spatial")

            for table_name in tables_to_export:
                try:
                    console.print(f"\nExporting {table_name}...")

                    row_count = conn.table(table_name).count().execute()

                    output_path = output_dir / f"{table_name}.parquet"
                    con.execute(f"COPY {table_name} TO '{output_path}' (FORMAT parquet)")

                    file_size = output_path.stat().st_size / (1024 * 1024)
                    console.print(f"  [green]✓ Exported {row_count:,} rows ({file_size:.1f} MB)[/green]")

                except Exception as e:
                    console.print(f"  [red]✗ Error exporting {table_name}: {e}[/red]")
                    continue
            console.print(f"\n[green]✓ Export complete! Files saved to: {output_dir}[/green]")

        except Exception as e:
            console.print(f"[red]Export error: {e}[/red]")
            raise typer.Exit(1) from e

    return app
