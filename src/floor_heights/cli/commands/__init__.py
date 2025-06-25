"""CLI subcommands for Floor Heights pipeline."""

import typer

from floor_heights.cli.commands.config import create_config_app
from floor_heights.cli.commands.database import create_db_app
from floor_heights.cli.commands.visualize import create_visualize_app
from floor_heights.cli.commands.yolo import create_yolo_app


def register_subcommands(app: typer.Typer) -> None:
    """Register all subcommand groups with the main CLI app."""
    app.add_typer(create_db_app(), name="db", help="Database management utilities")
    app.add_typer(create_yolo_app(), name="yolo", help="YOLO detection utilities")
    app.add_typer(create_config_app(), name="config", help="Set configuration values")
    app.add_typer(create_visualize_app(), name="visualize", help="Visualization utilities")
