"""Floor Heights Pipeline CLI using Typer - Refactored Version.

This module provides a unified command-line interface for all pipeline stages.
"""

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console

from floor_heights.cli.commands import register_subcommands
from floor_heights.cli.stages import (
    STAGE_DEFINITIONS,
    get_stage_map,
    register_stage_commands,
)
from floor_heights.cli.utils import (
    run_stage_direct,
    validate_database_tables,
)
from floor_heights.config import CONFIG

load_dotenv()

app = typer.Typer(
    name="fh",
    help="Floor Heights Pipeline - Extract building floor heights from LiDAR and imagery",
    add_completion=False,
)

console = Console()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    import sys

    logger.remove()
    logger.add(sys.stderr, level=log_level)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set log verbosity",
        case_sensitive=False,
    ),
) -> None:
    """Floor Heights Pipeline - Process LiDAR and imagery data to extract building heights."""
    setup_logging(log_level)

    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)


register_stage_commands(app)


@app.command("stages", help="List all pipeline stages")
@app.command("list", help="List all pipeline stages", hidden=True)
def list_stages() -> None:
    """List all available pipeline stages with their aliases."""
    console.print("\n[bold]Pipeline Stages:[/bold]")

    for stage_def in STAGE_DEFINITIONS:
        console.print(f"\n  [cyan]{stage_def.name}[/cyan] - {stage_def.help}")
        aliases = [stage_def.number, *stage_def.aliases]
        console.print(f"  Aliases: [dim]{', '.join(aliases)}[/dim]")

    console.print("\n[dim]Example usage:[/dim]")
    console.print("  fh 1 --region wagga      # Run stage 1 for Wagga")
    console.print("  fh clip -r wagga         # Same as above")
    console.print("  fh download              # Download panoramas for all regions")
    console.print()


@app.command("list-regions", help="List available regions")
@app.command("regions", help="List available regions", hidden=True)
def list_regions() -> None:
    """List all configured regions."""
    console.print("\n[bold]Available regions:[/bold]")
    for name, config in CONFIG.regions.items():
        console.print(f"  • {name} - {config.name} (CRS: EPSG:{config.crs_projected})")
    console.print()


@app.command("run", help="Run multiple pipeline stages sequentially")
def run(
    stages: list[str] = typer.Argument(..., help="Stage identifiers to run (e.g., 1 2a 2b 3)"),
    region: str | None = typer.Option(
        None,
        "--region",
        "-r",
        help="Single region to process (default: all from config)",
    ),
    screen: bool = typer.Option(
        False,
        "--screen",
        "-s",
        help="Run in a screen session for background execution",
    ),
    screen_name: str = typer.Option(
        "fh-pipeline",
        "--screen-name",
        help="Name for the screen session",
    ),
    use_subprocess: bool = typer.Option(
        False,
        "--subprocess",
        help="Use subprocess calls for isolation (slower)",
    ),
) -> None:
    """Run multiple pipeline stages sequentially.

    By default, stages are run in-process for better performance.
    Use --subprocess if you need process isolation.

    Examples:
        # Run stages 1 through 3 for all regions
        fh run 1 2a 2b 3

        # Run stages for specific region
        fh run 1 2a 2b 3 4a 4b -r wagga

        # Run in screen session (can disconnect)
        fh run 1 2a 2b 3 4a 4b --screen

        # Run with subprocess isolation
        fh run 1 2a 2b --subprocess
    """

    stage_map = get_stage_map()

    normalized_stages = []
    for stage in stages:
        if stage.lower() not in stage_map:
            console.print(f"[red]Error: Invalid stage '{stage}'[/red]")
            console.print(f"[dim]Valid stages: {', '.join(sorted(set(stage_map.keys())))}[/dim]")
            raise typer.Exit(1)
        normalized_stages.append(stage_map[stage.lower()])

    unique_stages = []
    seen = set()
    for stage in normalized_stages:
        if stage not in seen:
            unique_stages.append(stage)
            seen.add(stage)

    if screen:
        import shlex

        cmd_parts = ["fh", "run", *stages]
        if region:
            cmd_parts.extend(["-r", region])
        if use_subprocess:
            cmd_parts.append("--subprocess")

        screen_cmd = [
            "screen",
            "-dmS",
            screen_name,
            "bash",
            "-c",
            f"cd {CONFIG.project_root} && {' '.join(shlex.quote(p) for p in cmd_parts)}; exec bash",
        ]

        console.print(f"[cyan]Starting pipeline in screen session '{screen_name}'...[/cyan]")

        import subprocess

        result = subprocess.run(screen_cmd)

        if result.returncode == 0:
            console.print("[green]✓ Pipeline started in background[/green]")
            console.print("\n[dim]To view progress:[/dim]")
            console.print(f"  screen -r {screen_name}")
            console.print("\n[dim]To detach from screen:[/dim]")
            console.print("  Press Ctrl+A then D")
            console.print("\n[dim]To list screen sessions:[/dim]")
            console.print("  screen -ls")
        else:
            console.print("[red]Failed to start screen session[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[bold]Running stages: {', '.join(unique_stages)}[/bold]")
        if region:
            console.print(f"[dim]Region: {region}[/dim]")
        else:
            console.print(f"[dim]Regions: all ({', '.join(CONFIG.region_names)})[/dim]")
        console.print()

        failed_stages = []

        for i, stage in enumerate(unique_stages, 1):
            console.print(f"\n[cyan]{'=' * 60}[/cyan]")
            console.print(f"[cyan]Stage {i}/{len(unique_stages)}: {stage}[/cyan]")
            console.print(f"[cyan]{'=' * 60}[/cyan]\n")

            try:
                if use_subprocess:
                    import subprocess

                    cmd = ["fh", stage]
                    if region:
                        cmd.extend(["-r", region])

                    result = subprocess.run(cmd, cwd=CONFIG.project_root)
                    if result.returncode != 0:
                        raise Exception(f"Stage {stage} failed with exit code {result.returncode}")
                else:
                    run_stage_direct(stage, region=region)

                console.print(f"\n[green]✓ {stage} completed successfully[/green]")
            except Exception as e:
                console.print(f"\n[red]✗ {stage} failed: {e}[/red]")
                failed_stages.append(stage)

                if i < len(unique_stages) and not typer.confirm("\nContinue with remaining stages?", default=False):
                    break

        console.print(f"\n[cyan]{'=' * 60}[/cyan]")
        console.print("[bold]Pipeline Summary:[/bold]")
        console.print(f"  • Completed: {len(unique_stages) - len(failed_stages)}/{len(unique_stages)} stages")
        if failed_stages:
            console.print(f"  • Failed: {', '.join(failed_stages)}")
        console.print(f"[cyan]{'=' * 60}[/cyan]\n")

        if failed_stages:
            raise typer.Exit(1)


@app.command("check", help="Validate pipeline configuration")
def check_config(
    stage: str | None = typer.Option(None, "--stage", "-s", help="Check requirements for specific stage"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix common issues"),
) -> None:
    """Validate pipeline configuration and requirements."""
    import os
    import subprocess

    errors = []
    warnings = []

    console.print("[bold cyan]Configuration Validation[/bold cyan]\n")

    if CONFIG.project_root.exists():
        console.print("[green]✓[/green] Project root exists")
    else:
        console.print("[red]✗[/red] Project root missing")
        errors.append("project_root")

    if CONFIG.output_root.exists():
        console.print("[green]✓[/green] Output directory exists")
    else:
        console.print("[yellow]⚠[/yellow] Output directory missing")
        warnings.append("output_dir")
        if fix:
            CONFIG.output_root.mkdir(parents=True, exist_ok=True)
            console.print("  [dim]→ Created output directory[/dim]")

    if CONFIG.db_path.exists():
        db_status = validate_database_tables()
        if db_status["missing"]:
            console.print(f"[yellow]⚠[/yellow] Database missing tables: {', '.join(db_status['missing'])}")
            warnings.append("missing_tables")
        else:
            console.print(f"[green]✓[/green] Database exists ({len(db_status['present'])} tables)")
    else:
        console.print("[yellow]⚠[/yellow] Database not found")
        warnings.append("no_database")

    lidar_root = os.getenv("FH_LIDAR_DATA_ROOT")
    if lidar_root and not lidar_root.startswith("/path/to/"):
        lidar_path = Path(lidar_root)
        if lidar_path.exists():
            console.print(f"[green]✓[/green] Local LiDAR: {lidar_path}")
        else:
            console.print(f"[red]✗[/red] LiDAR root missing: {lidar_path}")
            errors.append("lidar_root")
    else:
        console.print("[green]✓[/green] LiDAR source: S3 (default)")

    console.print(f"[green]✓[/green] Regions: {', '.join(CONFIG.region_names)}")

    if stage:
        console.print(f"\n[bold]Stage {stage} Requirements:[/bold]")

        if stage in ["1", "stage01"]:
            try:
                result = subprocess.run(["pdal", "--version"], capture_output=True)
                if result.returncode == 0:
                    console.print("[green]✓[/green] PDAL available")
                else:
                    console.print("[red]✗[/red] PDAL error")
                    errors.append("pdal")
            except FileNotFoundError:
                console.print("[red]✗[/red] PDAL not found")
                errors.append("pdal_missing")

        elif stage in ["4a", "stage04a"]:
            yolo_path = CONFIG.yolo_model_path
            if yolo_path.exists():
                console.print(f"[green]✓[/green] YOLO model: {yolo_path}")
            else:
                console.print(f"[red]✗[/red] YOLO model not found: {yolo_path}")
                errors.append("yolo_model")

    console.print("\n[bold]Summary:[/bold]")
    if errors:
        console.print(f"[red]✗ {len(errors)} error(s) found[/red]")
        console.print(f"[red]Errors: {', '.join(errors)}[/red]")
    else:
        console.print("[green]✅ Configuration valid - ready to run![/green]")

    if warnings:
        console.print(f"[yellow]⚠ {len(warnings)} warning(s)[/yellow]")
        console.print(f"[yellow]Warnings: {', '.join(warnings)}[/yellow]")

    console.print()

    if errors:
        raise typer.Exit(1)


@app.command("download-data", help="Download required AWS data files")
def download_data(
    region: str | None = typer.Option(
        None,
        "--region",
        "-r",
        help="Single region to download (default: all)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be downloaded without actually downloading",
    ),
) -> None:
    """Download required trajectory and tileset files from AWS S3.

    This downloads essential files needed before running the pipeline:
    - Trajectory files (FramePosOptimised CSV files)
    - Tile index shapefiles (with .shp, .shx, .dbf, .prj, .cpg extensions)

    Files are organized in the data/raw directory structure:
    - data/raw/{region}/FramePosOptimised-{region}-rev2.csv
    - data/raw/{region}/tileset/*.shp (and related files)

    Examples:
        # Download files for all regions
        fh download-data

        # Download files for a specific region
        fh download-data -r wagga

        # Preview what would be downloaded
        fh download-data --dry-run
    """
    from floor_heights.utils.s3_utils import download_aws_data

    try:
        download_aws_data(region=region, dry_run=dry_run)
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("info", help="Show pipeline configuration")
def show_info() -> None:
    """Display current pipeline configuration."""
    console.print("[bold cyan]Floor Heights Pipeline Configuration[/bold cyan]\n")

    console.print("[bold]Paths:[/bold]")
    console.print(f"  Project root: {CONFIG.project_root}")
    console.print(f"  Output root:  {CONFIG.output_root}")
    console.print(f"  Database:     {CONFIG.db_path}")
    console.print()

    console.print("[bold]Regions:[/bold]")
    for name, region_config in CONFIG.regions.items():
        console.print(f"  {name}:")
        console.print(f"    Full name: {region_config.name}")
        console.print(f"    CRS: EPSG:{region_config.crs_projected}")
    console.print()

    console.print("[bold]Environment:[/bold]")
    import os

    lidar_root = os.getenv("FH_LIDAR_DATA_ROOT")
    if lidar_root:
        console.print(f"  LiDAR source: Local ({lidar_root})")
    else:
        console.print("  LiDAR source: S3")

    console.print(f"  S3 bucket: {CONFIG.constants.s3_bucket}")
    console.print(f"  S3 region: {CONFIG.constants.s3_region}")
    console.print()

    console.print("[bold]Stage Configuration:[/bold]")
    console.print(f"  Default workers: {CONFIG.constants.default_workers}")
    console.print(f"  YOLO model: {CONFIG.yolo_model_path}")
    console.print(f"  YOLO confidence: {CONFIG.object_detection.confidence_threshold}")
    console.print()


register_subcommands(app)


if __name__ == "__main__":
    app()
