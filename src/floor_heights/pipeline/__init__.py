"""Pipeline stages for floor height estimation."""

from .stage01_clip_point_cloud import run_stage as stage01_run

__all__ = [
    "stage01_run",
]
