"""LiDAR Quality Assurance CLI commands."""

import multiprocessing as mp
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from floor_heights.config import CONFIG
from floor_heights.qa import LidarQAPipeline

console = Console()


def create_qa_app() -> typer.Typer:
    """Create the QA subcommand group."""
    app = typer.Typer(help="LiDAR quality assurance and validation")

    @app.command("run", help="Run quality assessment on LiDAR tiles")
    def run_qa(
        input_dirs: list[Path] = typer.Argument(
            None, help="Input directories containing LiDAR tiles (defaults to all regions)"
        ),
        output: Path = typer.Option(
            CONFIG.project_root / "data" / "exports", "--output", "-o", help="Output directory for QA reports"
        ),
        pattern: str = typer.Option("*.las", "--pattern", "-p", help="File pattern to match (e.g., *.las, *.laz)"),
        workers: int = typer.Option(
            mp.cpu_count(), "--workers", "-w", help=f"Number of parallel workers (default: all {mp.cpu_count()} cores)"
        ),
        region: str | None = typer.Option(None, "--region", "-r", help="Process specific region only"),
    ) -> None:
        """Run LiDAR quality assessment on tile data.

        Examples:
            # Run QA on all regions
            fh qa run

            # Run QA on specific region
            fh qa run -r wagga

            # Run QA on custom directories
            fh qa run /path/to/lidar1 /path/to/lidar2

            # Use custom output directory
            fh qa run -o /custom/output/path
        """

        if not input_dirs:
            if region:
                input_dirs = [CONFIG.lidar_data_root / "rev2" / region]
            else:
                input_dirs = [CONFIG.lidar_data_root / "rev2" / r for r in CONFIG.region_names]

        valid_dirs = [d for d in input_dirs if d.exists()]

        if not valid_dirs:
            console.print("[red]Error: No valid input directories found[/red]")
            for d in input_dirs:
                if not d.exists():
                    console.print(f"  [dim]Missing: {d}[/dim]")
            raise typer.Exit(1)

        output.mkdir(parents=True, exist_ok=True)

        console.print("[bold cyan]LiDAR Quality Assessment[/bold cyan]")
        console.print(f"Output: {output}")
        console.print(f"Workers: {workers}")
        console.print(f"Pattern: {pattern}")
        console.print()

        pipeline = LidarQAPipeline(output)

        total_issues = 0
        all_summaries = {}

        for input_dir in valid_dirs:
            console.print(f"[cyan]Processing {input_dir.name}...[/cyan]")

            try:
                df = pipeline.process_directory(input_dir, pattern=pattern, max_workers=workers)

                if not df.empty:
                    summary = pipeline.generate_summary_report(df)
                    all_summaries[input_dir.name] = summary

                    _display_summary(input_dir.name, summary)
                    total_issues += summary["tiles_needing_correction"]

                    region_report = output / f"qa_report_{input_dir.name}.parquet"
                    df.to_parquet(region_report, index=False)
                    console.print(f"[green]✓[/green] Saved report to {region_report}")
                else:
                    console.print(f"[yellow]No tiles found in {input_dir}[/yellow]")

            except Exception as e:
                console.print(f"[red]Error processing {input_dir}: {e}[/red]")
                logger.exception("QA processing error")

        if all_summaries:
            console.print("\n[bold]Overall Summary:[/bold]")
            console.print(f"Processed {len(all_summaries)} regions")
            console.print(f"Total tiles needing correction: {total_issues}")
            console.print("\n[green]✅ QA processing complete![/green]")
            console.print(f"Check {output} for detailed reports")

    @app.command("summary", help="Display summary of existing QA reports")
    def show_summary(
        report_dir: Path = typer.Argument(
            CONFIG.project_root / "data" / "exports", help="Directory containing QA reports"
        ),
        region: str | None = typer.Option(None, "--region", "-r", help="Show summary for specific region"),
    ) -> None:
        """Display summary of existing QA reports.

        Examples:
            # Show all QA summaries
            fh qa summary

            # Show summary for specific region
            fh qa summary -r wagga
        """
        import json

        import pandas as pd

        if region:
            summary_file = report_dir / "qa_summary.json"
            parquet_file = report_dir / f"qa_report_{region}.parquet"

            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                console.print(f"\n[bold cyan]QA Report: {region}[/bold cyan]")
                console.print(f"Report: {parquet_file}")

                summary = {
                    "total_tiles": len(df),
                    "tiles_needing_correction": df["needs_correction"].sum(),
                    "average_density": df["density"].mean(),
                    "quality_level_distribution": df["quality_level"].value_counts().to_dict(),
                    "severity_distribution": {
                        "info": df["issues_info"].sum(),
                        "warning": df["issues_warning"].sum(),
                        "error": df["issues_error"].sum(),
                        "critical": df["issues_critical"].sum(),
                    },
                }

                issue_types = {}
                for types in df["issue_types"]:
                    if types:
                        for t in types.split(","):
                            issue_types[t] = issue_types.get(t, 0) + 1
                summary["issue_type_frequency"] = issue_types

                _display_summary(region, summary)
            else:
                console.print(f"[red]No QA report found for {region}[/red]")
        else:
            summary_file = report_dir / "qa_summary.json"
            if summary_file.exists():
                with summary_file.open() as f:
                    summary = json.load(f)
                _display_summary("All Regions", summary)
            else:
                parquet_files = list(report_dir.glob("qa_report_*.parquet"))
                if parquet_files:
                    console.print("[bold cyan]Available QA Reports:[/bold cyan]")
                    for pf in sorted(parquet_files):
                        region_name = pf.stem.replace("qa_report_", "")
                        df = pd.read_parquet(pf)
                        console.print(f"\n[cyan]{region_name}:[/cyan]")
                        console.print(f"  Total tiles: {len(df)}")
                        console.print(f"  Needing correction: {df['needs_correction'].sum()}")
                        console.print(f"  Average density: {df['density'].mean():.2f} pts/m²")
                else:
                    console.print("[yellow]No QA reports found[/yellow]")

    @app.command("issues", help="List tiles with specific issues")
    def list_issues(
        report_path: Path = typer.Argument(..., help="Path to QA report parquet file"),
        issue_type: str | None = typer.Option(
            None, "--type", "-t", help="Filter by issue type (e.g., striping, outliers, low_density)"
        ),
        severity: str | None = typer.Option(
            None, "--severity", "-s", help="Filter by severity (info, warning, error, critical)"
        ),
        limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of tiles to display"),
    ) -> None:
        """List tiles with specific quality issues.

        Examples:
            # List all tiles with issues
            fh qa issues data/exports/qa_report_wagga.parquet

            # List tiles with striping issues
            fh qa issues data/exports/qa_report_wagga.parquet -t striping

            # List tiles with critical issues
            fh qa issues data/exports/qa_report_wagga.parquet -s critical
        """
        import pandas as pd

        if not report_path.exists():
            console.print(f"[red]Report not found: {report_path}[/red]")
            raise typer.Exit(1)

        df = pd.read_parquet(report_path)

        if issue_type:
            mask = df["issue_types"].str.contains(issue_type, na=False)
            df = df[mask]

        if severity:
            severity_col = f"issues_{severity}"
            if severity_col in df.columns:
                df = df[df[severity_col] > 0]
            else:
                console.print(f"[red]Invalid severity: {severity}[/red]")
                console.print("[dim]Valid options: info, warning, error, critical[/dim]")
                raise typer.Exit(1)

        df = df[df["needs_correction"]]

        if df.empty:
            console.print("[green]No tiles found matching criteria[/green]")
            return

        df["total_issues"] = df["issues_warning"] + df["issues_error"] + df["issues_critical"]
        df = df.sort_values("total_issues", ascending=False)

        table = Table(title=f"Tiles with Issues ({len(df)} found, showing {min(limit, len(df))})")
        table.add_column("Tile", style="cyan")
        table.add_column("Density", justify="right")
        table.add_column("Issues", justify="right")
        table.add_column("Types", style="yellow")

        for _, row in df.head(limit).iterrows():
            issues_str = f"W:{row['issues_warning']} E:{row['issues_error']} C:{row['issues_critical']}"
            table.add_row(
                row["tile_name"],
                f"{row['density']:.1f}",
                issues_str,
                row["issue_types"][:50] + "..." if len(row["issue_types"]) > 50 else row["issue_types"],
            )

        console.print(table)

    return app


def _display_summary(name: str, summary: dict) -> None:
    """Display a formatted summary."""
    console.print(f"\n[bold]{name.upper()} SUMMARY[/bold]")
    console.print(f"Total tiles: {summary['total_tiles']}")
    console.print(f"Tiles needing correction: {summary['tiles_needing_correction']}")
    console.print(f"Average density: {summary['average_density']:.2f} pts/m²")

    if summary.get("quality_level_distribution"):
        console.print("\nQuality Levels:")
        for level, count in summary["quality_level_distribution"].items():
            if level:
                console.print(f"  {level}: {count}")

    if summary.get("severity_distribution"):
        console.print("\nIssue Severity:")
        for sev, count in summary["severity_distribution"].items():
            console.print(f"  {sev}: {count}")

    if summary.get("issue_type_frequency"):
        console.print("\nMost Common Issues:")
        sorted_issues = sorted(summary["issue_type_frequency"].items(), key=lambda x: x[1], reverse=True)
        for issue, count in sorted_issues[:5]:
            console.print(f"  {issue}: {count}")
