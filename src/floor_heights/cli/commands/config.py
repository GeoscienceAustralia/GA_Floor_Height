"""Configuration management commands for the Floor Heights CLI."""

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def create_config_app() -> typer.Typer:
    """Create the config subcommand app."""
    app = typer.Typer(help="Set configuration values")

    @app.callback(invoke_without_command=True)
    def config_callback(
        ctx: typer.Context,
        lidar_root: str | None = typer.Option(None, "--lidar-root", help="Set local LiDAR data root path"),
        aws_key: str | None = typer.Option(None, "--aws-key", help="Set AWS access key ID"),
        aws_secret: str | None = typer.Option(None, "--aws-secret", help="Set AWS secret access key"),
        sample_size: int | None = typer.Option(None, "--sample-size", help="Set sample size per region (0 for all)"),
    ) -> None:
        """Set configuration values in .env file."""

        if ctx.invoked_subcommand is None and not any([lidar_root, aws_key, aws_secret, sample_size]):
            console.print(ctx.get_help())
            raise typer.Exit(0)

        if any([lidar_root, aws_key, aws_secret, sample_size]):
            env_path = Path(".env")
            lines = []
            if env_path.exists():
                with env_path.open() as f:
                    lines = f.readlines()

            updates = {}
            if lidar_root:
                updates["FH_LIDAR_DATA_ROOT"] = lidar_root
            if aws_key:
                updates["AWS_ACCESS_KEY_ID"] = aws_key
            if aws_secret:
                updates["AWS_SECRET_ACCESS_KEY"] = aws_secret
            if sample_size is not None:
                updates["FH_SAMPLE_SIZE"] = str(sample_size)

            for key, value in updates.items():
                found = False
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{key}="):
                        lines[i] = f"{key}={value}\n"
                        found = True
                        break
                if not found:
                    if lines and not lines[-1].endswith("\n"):
                        lines[-1] += "\n"
                    lines.append(f"{key}={value}\n")

            with env_path.open("w") as f:
                f.writelines(lines)

            for key in updates:
                console.print(f"[green]✓ Set {key}[/green]")

    @app.command("show", help="Show current configuration values")
    def show() -> None:
        """Display current configuration values from .env file."""
        env_path = Path(".env")

        if not env_path.exists():
            console.print("[yellow]No .env file found[/yellow]")
            console.print("Create one with 'fh config --lidar-root /path/to/lidar'")
            return

        console.print("[bold]Current configuration:[/bold]\n")

        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if (
                    line
                    and not line.startswith("#")
                    and any(key in line for key in ["FH_", "AWS_", "ANTHROPIC_", "HF_"])
                ):
                    if "SECRET" in line or "KEY" in line or "TOKEN" in line:
                        key, value = line.split("=", 1)
                        masked = "*" * len(value) if len(value) < 10 else "*" * (len(value) - 2) + value[-2:]
                        console.print(f"  {key}={masked}")
                    else:
                        console.print(f"  {line}")

        console.print()

    @app.command("init", help="Initialize configuration with interactive setup")
    def init() -> None:
        """Interactive configuration setup."""
        console.print("[bold]Floor Heights Pipeline Configuration Setup[/bold]\n")

        env_path = Path(".env")
        if env_path.exists() and not typer.confirm("Configuration file already exists. Overwrite?", default=False):
            console.print("[yellow]Aborted[/yellow]")
            return

        config_lines = ["# Floor Heights Pipeline Configuration\n"]

        console.print("[cyan]LiDAR Data Source:[/cyan]")
        use_local = typer.confirm("Do you have local LiDAR data?", default=False)

        if use_local:
            lidar_path = typer.prompt("Enter path to LiDAR data directory")
            config_lines.append(f"FH_LIDAR_DATA_ROOT={lidar_path}\n")
        else:
            console.print("Will use S3 for LiDAR data")
            config_lines.append("# FH_LIDAR_DATA_ROOT=/path/to/lidar\n")

        console.print("\n[cyan]AWS Credentials (for S3 access):[/cyan]")
        if typer.confirm("Configure AWS credentials?", default=True):
            aws_key = typer.prompt("AWS Access Key ID")
            aws_secret = typer.prompt("AWS Secret Access Key", hide_input=True)
            aws_region = typer.prompt("AWS Region", default="ap-southeast-2")

            config_lines.extend(
                [
                    "\n# AWS Configuration\n",
                    f"AWS_ACCESS_KEY_ID={aws_key}\n",
                    f"AWS_SECRET_ACCESS_KEY={aws_secret}\n",
                    f"AWS_DEFAULT_REGION={aws_region}\n",
                ]
            )

        console.print("\n[cyan]Optional Configuration:[/cyan]")

        if typer.confirm("Configure sample size for testing?", default=False):
            sample_size = typer.prompt("Sample size per region (0 for all)", type=int, default=100)
            config_lines.append(f"\n# Testing\nFH_SAMPLE_SIZE={sample_size}\n")

        with env_path.open("w") as f:
            f.writelines(config_lines)

        console.print("\n[green]✓ Configuration saved to .env[/green]")
        console.print("\n[dim]You can update these values anytime with:[/dim]")
        console.print("  fh config --lidar-root /new/path")
        console.print("  fh config --aws-key NEW_KEY")

    return app
