"""LiDAR point cloud viewer using deck.gl web interface."""

import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlparse

import laspy
import numpy as np
from loguru import logger
from rich.console import Console

console = Console()


class LiDARViewerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the LiDAR viewer."""
    
    def __init__(self, *args, output_dir: Path = None, **kwargs):
        self.output_dir = output_dir or Path("output")
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        
        if path == "/":
            self.serve_html()
        elif path == "/api/files":
            self.serve_file_list()
        elif path.startswith("/api/pointcloud/"):
            filename = path.replace("/api/pointcloud/", "")
            self.serve_pointcloud(filename)
        else:
            self.send_error(404)
    
    def serve_html(self):
        """Serve the main HTML page."""
        html_path = Path(__file__).parent / "web" / "index.html"
        
        if not html_path.exists():
            self.send_error(404, "HTML template not found")
            return
        
        with open(html_path, "rb") as f:
            content = f.read()
        
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
    
    def serve_file_list(self):
        """Serve list of available LAS files."""
        las_files = []
        

        for region_dir in self.output_dir.iterdir():
            if not region_dir.is_dir():
                continue
            
            clipped_dir = region_dir / "lidar" / "rev2-local" / "clipped"
            if clipped_dir.exists():
                for las_file in clipped_dir.glob("*.las"):

                    relative_path = las_file.relative_to(self.output_dir)
                    las_files.append(str(relative_path))
        

        las_files.sort()
        

        if len(las_files) > 100:
            logger.warning(f"Found {len(las_files)} LAS files, showing first 100")
            las_files = las_files[:100]
        
        response = {"files": las_files, "count": len(las_files)}
        
        self.send_json_response(response)
    
    def serve_pointcloud(self, filename: str):
        """Serve point cloud data as JSON."""
        try:

            filename = filename.replace("..", "").strip("/")
            las_path = self.output_dir / filename
            
            if not las_path.exists() or not las_path.suffix == ".las":
                self.send_error(404, "LAS file not found")
                return
            

            logger.info(f"Loading point cloud: {las_path}")
            
            with laspy.open(las_path) as las_file:
                las_data = las_file.read()
                

                points = np.vstack((las_data.x, las_data.y, las_data.z)).T
                

                center = points.mean(axis=0)
                points_centered = points - center
                

                point_list = []
                

                num_points = len(points)
                if num_points > 500000:
                    logger.warning(f"Sampling {num_points} points down to 500k for performance")
                    indices = np.random.choice(num_points, 500000, replace=False)
                else:
                    indices = np.arange(num_points)
                

                has_intensity = hasattr(las_data, 'intensity')
                has_classification = hasattr(las_data, 'classification')
                has_color = hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue')
                

                for i in indices:
                    point_dict = {
                        "position": points_centered[i].tolist()
                    }
                    
                    if has_intensity:
                        point_dict["intensity"] = int(las_data.intensity[i])
                    
                    if has_classification:
                        point_dict["classification"] = int(las_data.classification[i])
                    
                    if has_color:
                        point_dict["color"] = [
                            int(las_data.red[i] >> 8),
                            int(las_data.green[i] >> 8),
                            int(las_data.blue[i] >> 8)
                        ]
                    
                    point_list.append(point_dict)
                

                bounds = {
                    "minX": float(points_centered[:, 0].min()),
                    "maxX": float(points_centered[:, 0].max()),
                    "minY": float(points_centered[:, 1].min()),
                    "maxY": float(points_centered[:, 1].max()),
                    "minZ": float(points_centered[:, 2].min()),
                    "maxZ": float(points_centered[:, 2].max())
                }
                

                stats = {
                    "pointCount": len(indices),
                    "totalPoints": num_points,
                    "minX": bounds["minX"],
                    "maxX": bounds["maxX"],
                    "minY": bounds["minY"],
                    "maxY": bounds["maxY"],
                    "minZ": bounds["minZ"],
                    "maxZ": bounds["maxZ"]
                }
                

                if has_classification:
                    classifications = {}
                    unique_classes, counts = np.unique(las_data.classification[indices], return_counts=True)
                    for cls, count in zip(unique_classes, counts):
                        classifications[int(cls)] = int(count)
                    stats["classifications"] = classifications
                
                response = {
                    "points": point_list,
                    "bounds": bounds,
                    "stats": stats,
                    "hasColor": has_color,
                    "center": center.tolist()
                }
                
                self.send_json_response(response)
                
        except Exception as e:
            logger.error(f"Error loading point cloud: {e}")
            self.send_error(500, str(e))
    
    def send_json_response(self, data: dict[str, Any]):
        """Send JSON response."""
        content = json.dumps(data).encode()
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)
    
    def log_message(self, format, *args):
        """Override to use loguru."""
        logger.debug(f"{self.address_string()} - {format % args}")


def run_viewer(port: int = 8080, output_dir: Path = None, open_browser: bool = True):
    """Run the LiDAR viewer web server.
    
    Args:
        port: Port to run the server on
        output_dir: Directory containing LAS files
        open_browser: Whether to automatically open the browser
    """
    if output_dir is None:
        output_dir = Path("output")
    
    if not output_dir.exists():
        raise ValueError(f"Output directory not found: {output_dir}")
    

    def handler(*args, **kwargs):
        LiDARViewerHandler(*args, output_dir=output_dir, **kwargs)
    

    handler.output_dir = output_dir
    

    server = HTTPServer(("", port), handler)
    
    console.print(f"\n[green]Starting LiDAR viewer server on port {port}[/green]")
    console.print(f"[dim]Output directory: {output_dir}[/dim]")
    console.print(f"\n[cyan]Open your browser to: http://localhost:{port}[/cyan]")
    console.print("[dim]Press Ctrl+C to stop the server[/dim]\n")
    

    if open_browser:
        def open_browser_delayed():
            import time
            time.sleep(1)
            webbrowser.open(f"http://localhost:{port}")
        
        Thread(target=open_browser_delayed, daemon=True).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")
        server.shutdown()
        console.print("[green]Server stopped.[/green]")


if __name__ == "__main__":

    run_viewer()