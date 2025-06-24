"""YOLO detection utility commands for the Floor Heights CLI."""

import random
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table as RichTable

from floor_heights.config import CONFIG

console = Console()


def create_yolo_app() -> typer.Typer:
    """Create the YOLO subcommand app."""
    app = typer.Typer(help="YOLO detection utilities")

    @app.callback(invoke_without_command=True)
    def yolo_callback(ctx: typer.Context) -> None:
        """Show help when no subcommand is provided."""
        if ctx.invoked_subcommand is None:
            console.print(ctx.get_help())
            raise typer.Exit(0)

    @app.command("export", help="Export clips containing specific object classes")
    def export(
        region: str = typer.Argument(..., help="Region name to export from"),
        classes: str | None = typer.Option(
            None,
            "--classes",
            "-c",
            help="Comma-separated class names (e.g., 'Front Door,Window')",
        ),
        output_dir: Path | None = typer.Option(
            None,
            "--output",
            "-o",
            help="Output directory (default: output/exports/clips_by_class)",
        ),
        limit: int | None = typer.Option(
            None,
            "--limit",
            "-l",
            help="Maximum number of clips to export",
        ),
        confidence: float = typer.Option(
            0.5,
            "--confidence",
            help="Minimum confidence threshold",
        ),
        annotations: bool = typer.Option(
            True,
            "--annotations/--no-annotations",
            help="Export detection annotations as JSON",
        ),
        visualize: bool = typer.Option(
            False,
            "--visualize",
            "-v",
            help="Create visualized images with bounding boxes",
        ),
        random_sample: bool = typer.Option(
            False,
            "--random",
            "-r",
            help="Randomly sample clips (when using --limit)",
        ),
    ) -> None:
        """Export clips containing specific object classes.

        Examples:
            # Export specific classes
            fh yolo export wagga -c "Front Door,Window"

            # Export rare classes with visualizations
            fh yolo export wagga -c "Foundation,Stairs" -v --limit 50

            # Export high-confidence examples
            fh yolo export wagga -c "Front Door" --confidence 0.8
        """
        from floor_heights.utils.yolo_utils import export_clips_by_class

        if classes:
            class_list = [c.strip() for c in classes.split(",")]
        else:
            console.print("[red]Error: --classes option is required[/red]")
            console.print("[dim]Example: fh yolo export wagga --classes 'Front Door,Window'[/dim]")
            return

        export_clips_by_class(
            region=region,
            class_names=class_list,
            export_dir=output_dir,
            include_annotations=annotations,
            include_visualizations=visualize,
            confidence_threshold=confidence,
            limit=limit,
            random_sample=random_sample,
        )

    @app.command("stats", help="Show detection statistics by class")
    def stats(
        region: str = typer.Argument(..., help="Region name"),
        confidence: float = typer.Option(
            0.5,
            "--confidence",
            help="Minimum confidence threshold",
        ),
    ) -> None:
        """Display object detection statistics for a region.

        Shows:
        - Total detections per class
        - Number of unique clips and buildings
        - Confidence score statistics
        - Average bounding box sizes
        """
        from floor_heights.utils.yolo_utils import get_class_statistics

        stats_df = get_class_statistics(region, min_confidence=confidence)

        if stats_df.empty:
            console.print("[yellow]No detection data found for this region[/yellow]")
            return

        table = RichTable(title=f"Detection Statistics for {region}")
        table.add_column("Class", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Clips", justify="right")
        table.add_column("Buildings", justify="right")
        table.add_column("Avg Conf", justify="right")
        table.add_column("Min Conf", justify="right")
        table.add_column("Max Conf", justify="right")
        table.add_column("Avg Area", justify="right")

        for row in stats_df.itertuples(index=False):
            table.add_row(
                row.class_name,
                str(row.total_detections),
                str(row.unique_clips),
                str(row.unique_buildings),
                f"{row.avg_confidence:.3f}",
                f"{row.min_confidence:.3f}",
                f"{row.max_confidence:.3f}",
                f"{int(row.avg_bbox_area):,}",
            )

        console.print(table)

    @app.command("view", help="View detection results in terminal (auto-detects image protocol)")
    def view(
        region: str | None = typer.Argument(None, help="Region name (random if not specified)"),
        building_id: str | None = typer.Option(
            None,
            "--building",
            "-b",
            help="Specific building ID to view",
        ),
        classes: str | None = typer.Option(
            None,
            "--classes",
            "-c",
            help="Comma-separated class names to filter",
        ),
        confidence: float = typer.Option(
            0.5,
            "--confidence",
            help="Minimum confidence threshold",
        ),
        max_images: int = typer.Option(
            10,
            "--max",
            "-m",
            help="Maximum number of images to display",
        ),
        no_boxes: bool = typer.Option(
            False,
            "--no-boxes",
            help="Don't draw bounding boxes",
        ),
    ) -> None:
        """View detection results directly in the terminal using sixel protocol.

        Supports terminals with sixel graphics:
        - VSCode integrated terminal
        - WezTerm
        - mlterm
        - Other sixel-compatible terminals

        Examples:
            # View detections for a random region
            fh yolo view

            # View detections for a specific region
            fh yolo view wagga

            # View specific building
            fh yolo view wagga -b "NSW2951808594"

            # View only doors and windows
            fh yolo view wagga -c "Front Door,Window"

            # View high-confidence detections
            fh yolo view wagga --confidence 0.8
        """
        from floor_heights.utils.yolo_utils import SIXEL_AVAILABLE, view_detection_results

        if not SIXEL_AVAILABLE:
            console.print("[yellow]⚠️  Sixel library not installed[/yellow]")
            console.print("\nTo view images in terminal, install:")
            console.print("  pip install sixel")
            console.print("\nSupported terminals:")
            console.print("  • VSCode (with terminal.integrated.enableImages: true)")
            console.print("  • WezTerm (with enable_sixel: true)")
            console.print("  • mlterm")
            console.print("  • And other sixel-compatible terminals")
            return

        if region is None:
            available_regions = list(CONFIG.regions.keys())
            region = random.choice(available_regions)
            console.print(f"[cyan]No region specified, randomly selected: {region}[/cyan]")

        class_list = None
        if classes:
            class_list = [c.strip() for c in classes.split(",")]

        view_detection_results(
            region=region,
            building_id=building_id,
            class_names=class_list,
            confidence_threshold=confidence,
            max_images=max_images,
            show_annotations=not no_boxes,
        )

    @app.command("find-rare", help="Find high-confidence examples of rare classes")
    def find_rare(
        region: str = typer.Argument(..., help="Region name"),
        classes: str = typer.Argument(..., help="Comma-separated rare class names (e.g., 'Foundation,Stairs')"),
        max_examples: int = typer.Option(
            10,
            "--max",
            "-m",
            help="Maximum examples per class",
        ),
        confidence: float = typer.Option(
            0.7,
            "--confidence",
            help="Minimum confidence threshold",
        ),
        export: bool = typer.Option(
            False,
            "--export",
            "-e",
            help="Export found examples",
        ),
    ) -> None:
        """Find high-confidence examples of rare classes.

        Useful for finding training examples of underrepresented classes.

        Examples:
            # Find foundation and stairs examples
            fh yolo find-rare wagga "Foundation,Stairs"

            # Find and export examples
            fh yolo find-rare wagga "Foundation,Stairs" --export
        """
        from floor_heights.utils.yolo_utils import export_clips_by_class, find_rare_examples

        class_list = [c.strip() for c in classes.split(",")]

        examples = find_rare_examples(
            region=region, class_names=class_list, max_examples=max_examples, min_confidence=confidence
        )

        if examples.empty:
            console.print(f"[yellow]No examples found for classes: {', '.join(class_list)}[/yellow]")
            return

        table = RichTable(title="High-Confidence Examples of Rare Classes")
        table.add_column("Building ID", style="cyan")
        table.add_column("Class", style="green")
        table.add_column("Confidence", justify="right")
        table.add_column("Pano ID")
        table.add_column("View")

        for row in examples.itertuples(index=False):
            table.add_row(
                row.building_id,
                row.class_name,
                f"{row.confidence:.3f}",
                row.pano_id,
                row.view_type,
            )

        console.print(table)
        console.print(
            f"\n[dim]Found {len(examples)} examples across {examples['building_id'].nunique()} buildings[/dim]"
        )

        if export:
            console.print("\n[cyan]Exporting examples...[/cyan]")

            export_clips_by_class(
                region=region,
                class_names=class_list,
                export_dir=CONFIG.output_root / "exports" / "rare_examples",
                include_visualizations=True,
                confidence_threshold=confidence,
            )

    return app
