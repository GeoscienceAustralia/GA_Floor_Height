"""Floor Heights Pipeline CLI package."""

import sys

from rich.console import Console


def app():
    """Wrapped app entry point with nice error handling."""
    try:
        from floor_heights.cli.main import app as _app

        return _app()
    except ValueError as e:
        console = Console()
        console.print("\n[red bold]Configuration Error[/red bold]\n")
        console.print(f"[red]{e!s}[/red]\n")
        console.print(
            "[yellow]Please check your .env file and ensure all required environment variables are set correctly.[/yellow]"
        )
        console.print("\nFor more information, check the .env.example file for reference.\n")
        sys.exit(1)
    except ImportError as e:
        console = Console()
        console.print("\n[red bold]Import Error[/red bold]\n")
        console.print(f"[red]{e!s}[/red]\n")
        console.print("[yellow]Please ensure all dependencies are installed:[/yellow]")
        console.print("  pip install -e .")
        sys.exit(1)
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        console = Console()
        console.print("\n[red bold]Initialization Error[/red bold]\n")
        console.print(f"[red]{type(e).__name__}: {e!s}[/red]\n")
        console.print("[yellow]An error occurred while starting the Floor Heights pipeline.[/yellow]")
        console.print("\nRun with --debug flag for full traceback.\n")
        sys.exit(1)


try:
    from floor_heights.cli.main import app as _original_app
    from floor_heights.cli.stages import (
        STAGE_DEFINITIONS,
        get_stage_map,
        register_stage_commands,
    )
    from floor_heights.cli.utils import (
        check_stage_config,
        run_stage_direct,
        validate_database_tables,
    )
except Exception:
    pass

__all__ = [
    "STAGE_DEFINITIONS",
    "app",
    "check_stage_config",
    "get_stage_map",
    "register_stage_commands",
    "run_stage_direct",
    "validate_database_tables",
]
