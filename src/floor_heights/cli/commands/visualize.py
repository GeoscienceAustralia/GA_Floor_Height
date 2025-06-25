"""Visualization commands for the floor heights CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from floor_heights.config import CONFIG, PipelineConfig
from floor_heights.db.reader import DuckDBReader
from floor_heights.utils.visualization import PotreeConverter, PotreeViewer


def create_visualize_app() -> typer.Typer:
    """Create the visualization subcommand app."""
    app = typer.Typer()

    @app.command("potree")
    def potree(
        region: Annotated[str, typer.Option("--region", "-r", help="Region name to visualize")],
        building_id: Annotated[
            list[str], typer.Option("--building-id", "-b", help="Specific building ID(s) to visualize")
        ] = None,
        limit: Annotated[int, typer.Option("--limit", "-l", help="Maximum number of clips to visualize")] = None,
        port: Annotated[int, typer.Option("--port", "-p", help="Port for HTTP server")] = 8080,
        potree_converter: Annotated[
            str, typer.Option("--potree-converter", help="Path to PotreeConverter executable")
        ] = "/home/ubuntu/GA-floor-height/tools/PotreeConverter_linux_x64/PotreeConverter",
        temp_dir: Annotated[Path, typer.Option("--temp-dir", help="Temporary directory for conversions")] = None,
        skip_conversion: Annotated[
            bool, typer.Option("--skip-conversion", help="Skip conversion if Potree files already exist")
        ] = False,
    ) -> None:
        """Visualize stage01 clipped point clouds using Potree.

        This command converts LAS/LAZ files to Potree format and serves them
        via a local web server for interactive 3D visualization.

        Examples:
            fh visualize potree --region wagga --limit 5
            fh visualize potree --region launceston --building-id 12345
        """
        visualize_region_clips(CONFIG, region, building_id, limit, port, potree_converter, temp_dir, skip_conversion)

    return app


def visualize_region_clips(
    config: PipelineConfig,
    region: str,
    building_ids: list[str] | None = None,
    limit: int | None = None,
    port: int = 8080,
    potree_converter: str = "PotreeConverter",
    temp_dir: Path | None = None,
    skip_conversion: bool = False,
) -> None:
    """
    Visualize clipped point clouds from stage01 using Potree.

    Args:
        config: Configuration object
        region: Region name
        building_ids: Specific building IDs to visualize (all if None)
        limit: Maximum number of clips to visualize
        port: Port for HTTP server
        potree_converter: Path to PotreeConverter executable
        temp_dir: Temporary directory for conversions (auto if None)
        skip_conversion: Skip conversion if Potree files already exist
    """
    
    with DuckDBReader(config.db_path) as db:
        table = db.table("stage01_clips")
        query = table.filter(table.region_name == region)

        if building_ids:
            query = query.filter(table.building_id.isin(building_ids))

        if limit:
            query = query.limit(limit)

        clips = query.execute().to_dict(orient="records")

    if not clips:
        logger.warning(f"No clips found for region {region}")
        return

    logger.info(f"Found {len(clips)} clips to visualize")

    
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="potree_"))
    else:
        temp_dir.mkdir(parents=True, exist_ok=True)

    output_dir = temp_dir / "potree_output"
    output_dir.mkdir(exist_ok=True)

    
    converter = PotreeConverter(potree_converter)
    viewer = PotreeViewer(port)

    
    potree_dirs = []

    for clip in clips:
        
        clip_file = config.output_root / clip["clip_path"]

        if not clip_file.exists():
            logger.warning(f"Clip file not found: {clip_file}")
            continue

        
        clip_output = output_dir / f"{clip['building_id']}_{clip['gnaf_id']}"

        if skip_conversion and clip_output.exists():
            logger.info(f"Skipping conversion for {clip['building_id']} (already exists)")
        else:
            try:
                converter.convert(clip_file, clip_output, title=f"Building {clip['building_id']} ({clip['gnaf_id']})")

            except Exception as e:
                logger.error(f"Failed to convert {clip_file}: {e}")
                continue

        potree_dirs.append(clip_output)

    if not potree_dirs:
        logger.error("No clips successfully converted")
        return

    logger.success(f"Successfully converted {len(potree_dirs)} point clouds")
    logger.info(f"Temporary files saved to: {temp_dir}")

    
    viewer.serve_potree_directories(potree_dirs)
