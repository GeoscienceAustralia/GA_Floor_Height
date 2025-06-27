"""Stage definitions for the Floor Heights pipeline CLI."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import typer
from loguru import logger


@dataclass
class StageParameter:
    """Definition of a CLI parameter for a stage."""

    name: str
    type: type
    default: Any = None
    help: str = ""
    short: str | None = None
    is_option: bool = True


@dataclass
class StageDefinition:
    """Definition of a pipeline stage."""

    number: str
    name: str
    aliases: list[str]
    help: str
    description: str
    module_path: str
    function_name: str = "run_stage"
    parameters: list[StageParameter] = field(default_factory=list)

    @property
    def module_import_path(self) -> str:
        """Get the full import path for the stage module."""
        return f"floor_heights.pipeline.{self.module_path}"


REGION_PARAM = StageParameter(
    name="region",
    type=str | None,
    default=None,
    help="Single region to process (default: all from config)",
    short="-r",
)

WORKERS_PARAM = StageParameter(
    name="workers", type=int, default=-1, help="Number of worker threads/processes (-1 for auto)", short="-w"
)

SAMPLE_PARAM = StageParameter(
    name="sample_size",
    type=int | None,
    default=None,
    help="Process only first N buildings per region (for testing)",
    short="-s",
)


STAGE_DEFINITIONS = [
    StageDefinition(
        number="1",
        name="stage01",
        aliases=["clip"],
        help="Clip LiDAR tiles to residential footprints",
        description="""Stage 01: Clip LiDAR point clouds to building footprints.

This stage:
- Loads residential building footprints from the database
- Finds intersecting LiDAR tiles for each building
- Clips point clouds to building footprints with a buffer
- Saves clipped LAS files and tracking metadata""",
        module_path="stage01_clip_point_cloud",
        parameters=[
            REGION_PARAM,
            WORKERS_PARAM,
            StageParameter(name="revision", type=str, default="rev2", help="LiDAR revision to use (default: 'rev2')"),
            SAMPLE_PARAM,
            StageParameter(
                name="dry_run", type=bool, default=False, help="Show what would be processed without actually running"
            ),
        ],
    ),
    StageDefinition(
        number="2a",
        name="stage02a",
        aliases=["harvest"],
        help="Harvest candidate panoramas from Street View",
        description="""Stage 02a: Harvest candidate Street View panoramas.

This stage:
- Loads building footprints and panorama locations
- Performs ray-casting to find viewing angles
- Avoids self-occlusion from other buildings
- Outputs candidate panorama-to-building view rays""",
        module_path="stage02a_harvest_candidate_panoramas",
        parameters=[
            REGION_PARAM,
            SAMPLE_PARAM,
        ],
    ),
    StageDefinition(
        number="2b",
        name="stage02b",
        aliases=["download"],
        help="Download panorama images",
        description="""Stage 02b: Download Street View panorama images.

This stage:
- Reads chosen panorama candidates from stage02a
- Downloads panorama images from S3
- Organizes images by building_id and gnaf_id
- Tracks download status in database""",
        module_path="stage02b_download_candidate_panoramas",
        parameters=[
            REGION_PARAM,
        ],
    ),
    StageDefinition(
        number="3",
        name="stage03",
        aliases=["clip-panos"],
        help="Clip panoramas to building views",
        description="""Stage 03: Clip panoramas to show only building facades.

This stage:
- Reads downloaded panoramas from stage02b
- Calculates horizontal bounds using visibility-based ray casting
- Optionally uses LiDAR data for vertical bounds
- Clips panoramas to show only the target building
- Saves clipped images and tracking metadata""",
        module_path="stage03_clip_panoramas",
        parameters=[
            REGION_PARAM,
        ],
    ),
    StageDefinition(
        number="4a",
        name="stage04a",
        aliases=["detect"],
        help="Run object detection on clipped panoramas",
        description="""Stage 04a: Object detection on clipped panoramas.

This stage:
- Loads YOLO model for building feature detection
- Processes clipped panoramas from stage03
- Detects doors, windows, and other building features
- Saves detection results to database
- Optionally creates visualization images""",
        module_path="stage04a_object_detection",
        parameters=[
            REGION_PARAM,
            WORKERS_PARAM,
            StageParameter(
                name="conf", type=float, default=0.25, help="Object detection confidence threshold", short="-c"
            ),
            StageParameter(name="visualize", type=bool, default=False, help="Create visualization images", short="-v"),
        ],
    ),
    StageDefinition(
        number="4b",
        name="stage04b",
        aliases=["best-view"],
        help="Select best view with SigLIP occlusion scoring",
        description="""Stage 04b: Select best panorama view with SigLIP occlusion scoring.

This stage:
- Loads object detections from stage04a
- Uses SigLIP2 model to assess occlusion and view quality
- Scores views based on detection confidence and occlusion
- Selects best and closest direct views for each building
- Saves selected views with scoring metadata""",
        module_path="stage04b_best_view_selection",
        parameters=[
            REGION_PARAM,
            WORKERS_PARAM,
            StageParameter(
                name="batch_size", type=int, default=100, help="Batch size for database operations", short="-b"
            ),
        ],
    ),
    StageDefinition(
        number="5",
        name="stage05",
        aliases=["project"],
        help="Project point clouds to facade rasters",
        description="""Stage 05: Project LiDAR point clouds onto building facades.

This stage:
- Loads best views from stage04b
- Projects 3D LiDAR points onto 2D facade plane
- Creates rasters for depth, intensity, and classification
- Saves projected rasters and metadata""",
        module_path="stage05_project_point_cloud_to_facade",
        parameters=[
            REGION_PARAM,
            WORKERS_PARAM,
            StageParameter(
                name="projection_mode", type=str, default="normal", help="Point cloud projection mode", short="-m"
            ),
        ],
    ),
    StageDefinition(
        number="6",
        name="stage06",
        aliases=["ground"],
        help="Extract ground elevation from clipped LiDAR",
        description="""Stage 06: Extract ground elevation from LiDAR data.

This stage:
- Analyzes clipped LiDAR point clouds from stage01
- Identifies ground points using classification and heuristics
- Estimates ground elevation for each building
- Saves ground elevation data with confidence scores""",
        module_path="stage06_extract_ground_elevation",
        parameters=[
            REGION_PARAM,
            WORKERS_PARAM,
            StageParameter(
                name="method",
                type=str,
                default="auto",
                help="Ground extraction method (auto/percentile/classification)",
            ),
            StageParameter(
                name="confidence_threshold",
                type=float,
                default=0.5,
                help="Minimum confidence score for ground elevation",
            ),
        ],
    ),
    StageDefinition(
        number="7",
        name="stage07",
        aliases=["ffh", "floors"],
        help="Estimate First Floor Heights from features and LiDAR",
        description="""Stage 07: Estimate First Floor Heights (FFH) from detected features and LiDAR.

This stage combines:
- Object detection results from stage04a/04b (doors, windows, stairs, foundations)
- LiDAR point cloud projections from stage05 (elevation, depth, classification rasters)
- Ground elevations from stage06 (gap-filling ground levels)

To estimate First Floor Heights (FFH) using multiple methods:
1. FFH1: Floor feature to ground feature (when both detected)
2. FFH2: Floor feature to nearest ground area from LiDAR
3. FFH3: Floor feature to ground elevation from DTM""",
        module_path="stage07_estimate_floor_height",
        parameters=[
            REGION_PARAM,
            WORKERS_PARAM,
            StageParameter(
                name="projection_mode", type=str, default="normal", help="Point cloud projection mode from stage05"
            ),
            SAMPLE_PARAM,
            StageParameter(name="skip_existing", type=bool, default=True, help="Skip already processed buildings"),
        ],
    ),
    StageDefinition(
        number="8",
        name="stage08",
        aliases=["validate", "validation"],
        help="Validate FFH results against Frontier SI ground truth",
        description="""Stage 08: Validate FFH results against Frontier SI ground truth.

This stage compares the estimated floor heights from stage07 with Frontier SI
LIDAR-derived ground truth values to assess the accuracy of the pipeline.

The validation:
- Uses only Frontier SI validation data (501 LIDAR-derived measurements)
- Calculates RMSE, MAE, bias, and correlation for each FFH method
- Generates scatter plots and error distribution plots
- Creates a detailed validation report
- Saves results to the database for further analysis""",
        module_path="stage08_validate_results",
        parameters=[
            StageParameter(name="skip_existing", type=bool, default=False, help="Skip if validation already exists"),
        ],
    ),
    StageDefinition(
        number="9a",
        name="stage09a",
        aliases=["lidar-stats"],
        help="Extract LiDAR statistics from clipped point clouds",
        description="""Stage 09a: Extract statistics from clipped LiDAR files.

Reads LAS files from stage01 and computes metrics for analysis:
- Point density and coverage
- Height distribution and percentiles
- Classification features (ground, building, vegetation)
- Building structure indicators
- Return pattern analysis
- Features for FFH prediction

Statistics are stored in the database for analysis, ML models,
and quality assessment.""",
        module_path="stage09a_compute_lidar_stats",
        parameters=[
            REGION_PARAM,
            WORKERS_PARAM,
            SAMPLE_PARAM,
        ],
    ),
]


def create_stage_command(stage_def: StageDefinition) -> Callable:
    """Create a Typer command function for a stage definition."""

    def stage_command(**kwargs):
        """Generic stage command handler."""

        if stage_def.number == "1" and kwargs.get("dry_run", False):
            from rich.console import Console

            console = Console()
            region = kwargs.get("region")
            if region:
                console.print(f"[cyan]Would process region: {region}[/cyan]")
            else:
                from floor_heights.config import CONFIG

                console.print(f"[cyan]Would process regions: {', '.join(CONFIG.region_names)}[/cyan]")
            return

        if stage_def.number == "1":
            from rich.console import Console

            console = Console()
            console.print("[dim]Checking configuration...[/dim]")
            from floor_heights.cli.utils import check_stage_config

            if not check_stage_config("stage01"):
                console.print("[red]❌ Configuration errors found. Run 'fh check --stage 1' for details.[/red]")
                raise typer.Exit(1) from None

            import os

            from dotenv import load_dotenv

            load_dotenv(override=True)
            lidar_root = os.getenv("FH_LIDAR_DATA_ROOT")
            lidar_source = "local" if lidar_root else "s3"
            console.print(f"[dim]Using LiDAR source: {lidar_source}[/dim]")
            kwargs["lidar_source"] = lidar_source

        import importlib

        try:
            module = importlib.import_module(stage_def.module_import_path)
            run_func = getattr(module, stage_def.function_name)

            import inspect

            sig = inspect.signature(run_func)
            valid_params = set(sig.parameters.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

            if "conf" in kwargs and "conf_threshold" in valid_params:
                filtered_kwargs["conf_threshold"] = kwargs["conf"]
                filtered_kwargs.pop("conf", None)

            run_func(**filtered_kwargs)

        except ImportError as e:
            logger.error(f"Failed to import stage module: {e}")
            raise typer.Exit(1) from e
        except Exception as e:
            logger.error(f"Stage {stage_def.name} failed: {e}")
            raise typer.Exit(1) from e

    stage_command.__name__ = stage_def.name
    stage_command.__doc__ = stage_def.description

    return stage_command


def register_stage_commands(app: typer.Typer) -> None:
    """Register all stage commands with the Typer app."""
    for stage_def in STAGE_DEFINITIONS:
        command_func = create_stage_command(stage_def)

        params = []
        for param in stage_def.parameters:
            if param.is_option:
                option_args = [f"--{param.name}"]
                if param.short:
                    option_args.append(param.short)

                params.append(typer.Option(param.default, *option_args, help=param.help))
            else:
                params.append(typer.Argument(param.default, help=param.help))

        for i, alias in enumerate([stage_def.name, f"{stage_def.number}", *stage_def.aliases]):
            is_hidden = i > 0

            def make_command(sd=stage_def, ps=params, cmd_func=command_func):
                def wrapper(**kwargs):
                    return cmd_func(**kwargs)

                wrapper.__doc__ = sd.description
                wrapper.__name__ = sd.name

                import inspect

                sig_params = []
                for i, p in enumerate(ps):
                    param_name = sd.parameters[i].name
                    sig_params.append(
                        inspect.Parameter(
                            param_name, inspect.Parameter.KEYWORD_ONLY, default=p, annotation=sd.parameters[i].type
                        )
                    )

                wrapper.__signature__ = inspect.Signature(sig_params)
                return wrapper

            app.command(alias, help=stage_def.help, hidden=is_hidden)(make_command())


def get_stage_map() -> dict[str, str]:
    """Get mapping of all stage aliases to canonical stage names."""
    stage_map = {}
    for stage_def in STAGE_DEFINITIONS:
        canonical_name = stage_def.name

        stage_map[stage_def.number] = canonical_name
        stage_map[canonical_name] = canonical_name
        for alias in stage_def.aliases:
            stage_map[alias] = canonical_name
    return stage_map
