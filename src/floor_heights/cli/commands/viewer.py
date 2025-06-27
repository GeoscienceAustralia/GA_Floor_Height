"""Viewer command for launching the deck.gl LiDAR visualization."""

import os
import subprocess
import time
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

app = typer.Typer(
    name="viewer",
    help="Launch interactive LiDAR point cloud viewer",
    rich_markup_mode="markdown",
)


def check_node_installed() -> bool:
    """Check if Node.js is installed."""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_dependencies(viz_dir: Path) -> bool:
    """Install npm dependencies if needed."""
    node_modules = viz_dir / "node_modules"
    if not node_modules.exists():
        console.print("[yellow]Installing viewer dependencies...[/yellow]")
        try:
            subprocess.run(["npm", "install"], cwd=viz_dir, check=True)
            console.print("[green]✓ Dependencies installed[/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to install dependencies: {e}[/red]")
            return False
    return True


@app.callback()
def callback():
    """LiDAR Point Cloud Viewer."""
    pass


@app.command()
def launch(
    clip_id: str | None = typer.Argument(None, help="Specific clip ID to visualize (e.g., '57238_GANSW706146768')"),
    region: str | None = typer.Option(None, "--region", "-r", help="Region name (wagga, launceston, tweed)"),
    port: int = typer.Option(8000, "--port", "-p", help="Port for the API server"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically"),
    dev: bool = typer.Option(False, "--dev", help="Run in development mode"),
):
    """Launch the interactive LiDAR point cloud viewer.

    Examples:
        fh viewer launch
        fh viewer launch 57238_GANSW706146768
        fh viewer launch --region tweed
    """

    if not check_node_installed():
        console.print(
            Panel(
                "[red]Node.js is not installed![/red]\n\n"
                "Please install Node.js (v16 or later) to use the viewer:\n"
                "  • Ubuntu/Debian: sudo apt install nodejs npm\n"
                "  • Or visit: https://nodejs.org/",
                title="Missing Dependency",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    viz_dir = Path(__file__).parent.parent.parent / "visualization"
    server_script = viz_dir / "server.py"

    if not viz_dir.exists():
        console.print("[red]Visualization directory not found![/red]")
        raise typer.Exit(1)

    if not install_dependencies(viz_dir):
        raise typer.Exit(1)

    console.print("[yellow]Checking for existing viewer processes...[/yellow]")
    try:
        subprocess.run(["pkill", "-f", "uvicorn.*server:app"], capture_output=True)

        subprocess.run(["pkill", "-f", "npm run start"], capture_output=True)
        subprocess.run(["pkill", "-f", "vite"], capture_output=True)

        time.sleep(1)
    except Exception:
        pass

    console.print(
        Panel(
            f"[green]Starting LiDAR Point Cloud Viewer[/green]\n\n"
            f"  • API Server: http://localhost:{port}\n"
            f"  • Web Interface: http://localhost:3000\n\n"
            f"[yellow]Press Ctrl+C to stop[/yellow]",
            title="Floor Heights Viewer",
            border_style="green",
        )
    )

    import sys

    python_cmd = sys.executable

    env = {**os.environ, "PORT": str(port)}
    if clip_id:
        env["INITIAL_CLIP_ID"] = clip_id
    if region:
        env["INITIAL_REGION"] = region

    api_process = subprocess.Popen(
        [python_cmd, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", str(port)], cwd=viz_dir, env=env
    )

    time.sleep(2)

    try:
        npm_cmd = "npm run start" if not dev else "npm run dev"
        frontend_env = {**env, "VITE_API_PORT": str(port)}
        frontend_process = subprocess.Popen(npm_cmd.split(), cwd=viz_dir, env=frontend_env)

        if open_browser:
            time.sleep(3)
            webbrowser.open("http://localhost:3000")

        frontend_process.wait()

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        api_process.terminate()
        if "frontend_process" in locals():
            frontend_process.terminate()

        time.sleep(1)

        if api_process.poll() is None:
            api_process.kill()
        if "frontend_process" in locals() and frontend_process.poll() is None:
            frontend_process.kill()

        console.print("[green]✓ Viewer stopped[/green]")


@app.command()
def build(
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory for build"),
):
    """Build the viewer for production deployment."""
    viz_dir = Path(__file__).parent.parent.parent / "visualization"

    if not viz_dir.exists():
        console.print("[red]Visualization directory not found![/red]")
        raise typer.Exit(1)

    if not install_dependencies(viz_dir):
        raise typer.Exit(1)

    console.print("[yellow]Building viewer for production...[/yellow]")

    try:
        subprocess.run(["npm", "run", "build"], cwd=viz_dir, check=True)

        dist_dir = viz_dir / "dist"
        if output_dir:
            import shutil

            shutil.copytree(dist_dir, output_dir, dirs_exist_ok=True)
            console.print(f"[green]✓ Built to: {output_dir}[/green]")
        else:
            console.print(f"[green]✓ Built to: {dist_dir}[/green]")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        raise typer.Exit(1) from e


def create_viewer_app() -> typer.Typer:
    """Create and return the viewer Typer app."""
    return app
