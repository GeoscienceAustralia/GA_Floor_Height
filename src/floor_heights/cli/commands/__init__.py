"""CLI subcommands for Floor Heights pipeline."""

import typer

from floor_heights.cli.commands.config import create_config_app
from floor_heights.cli.commands.database import create_db_app
from floor_heights.cli.commands.qa import create_qa_app
from floor_heights.cli.commands.viewer import create_viewer_app
from floor_heights.cli.commands.yolo import create_yolo_app


def register_subcommands(app: typer.Typer) -> None:
    """Register all subcommand groups with the main CLI app."""
    app.add_typer(create_db_app(), name="db", help="Database management utilities")
    app.add_typer(create_yolo_app(), name="yolo", help="YOLO detection utilities")
    app.add_typer(create_config_app(), name="config", help="Set configuration values")
    app.add_typer(create_viewer_app(), name="viewer", help="LiDAR point cloud viewer")
    app.add_typer(create_qa_app(), name="qa", help="LiDAR quality assurance")
