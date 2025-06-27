"""Base classes and utilities for LiDAR Quality Assurance.

This module provides the foundation for comprehensive LiDAR data quality assessment,
including detection of geometric errors, noise, classification issues, and other artifacts.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pdal
from loguru import logger


class QualityLevel(Enum):
    """USGS 3DEP Quality Levels for LiDAR data."""

    QL0 = (0.05, 8.0, 0.35)
    QL1 = (0.10, 8.0, 0.35)
    QL2 = (0.10, 2.0, 0.71)
    QL3 = (0.20, 0.5, 1.41)


class IssueType(Enum):
    """Types of quality issues that can be detected."""

    STRIPING = "striping"
    MISALIGNMENT = "misalignment"
    LOW_DENSITY = "low_density"
    DENSITY_VARIATION = "density_variation"
    DATA_VOID = "data_void"

    OUTLIERS = "outliers"
    NOISE = "noise"
    GHOSTING = "ghosting"
    BLOOMING = "blooming"

    MISCLASSIFICATION = "misclassification"
    UNCLASSIFIED = "unclassified"
    CLASSIFICATION_NOISE = "classification_noise"

    INTENSITY_BANDING = "intensity_banding"
    BLIND_SPOTS = "blind_spots"
    EDGE_ARTIFACTS = "edge_artifacts"

    WEATHER_NOISE = "weather_noise"
    SOLAR_NOISE = "solar_noise"
    WATER_ABSORPTION = "water_absorption"


class IssueSeverity(Enum):
    """Severity levels for detected issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """Represents a detected quality issue in the LiDAR data."""

    issue_type: IssueType
    severity: IssueSeverity
    description: str
    metrics: dict[str, Any] = field(default_factory=dict)
    affected_points: np.ndarray | None = None
    correction_possible: bool = True
    correction_method: str | None = None


@dataclass
class QualityMetrics:
    """Core quality metrics for a LiDAR tile."""

    point_count: int = 0
    area: float = 0.0
    density: float = 0.0
    density_variation: float = 0.0

    vertical_accuracy: float | None = None
    horizontal_accuracy: float | None = None
    relative_accuracy: float | None = None

    coverage_ratio: float = 0.0
    void_ratio: float = 0.0

    classified_ratio: float = 0.0
    ground_points_ratio: float = 0.0
    building_points_ratio: float = 0.0
    noise_points_ratio: float = 0.0

    striping_score: float = 0.0
    alignment_score: float = 0.0

    outlier_ratio: float = 0.0
    noise_level: float = 0.0

    intensity_mean: float = 0.0
    intensity_std: float = 0.0
    intensity_consistency: float = 0.0


@dataclass
class TileQAReport:
    """Complete QA report for a single LiDAR tile."""

    tile_path: Path
    metrics: QualityMetrics
    issues: list[QualityIssue] = field(default_factory=list)
    quality_level: QualityLevel | None = None
    processing_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: QualityIssue):
        """Add a quality issue to the report."""
        self.issues.append(issue)

    def get_severity_counts(self) -> dict[IssueSeverity, int]:
        """Get count of issues by severity."""
        counts = dict.fromkeys(IssueSeverity, 0)
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts

    def needs_correction(self) -> bool:
        """Check if tile needs correction based on issues."""
        severity_counts = self.get_severity_counts()
        return severity_counts[IssueSeverity.CRITICAL] > 0 or severity_counts[IssueSeverity.ERROR] > 0


class QualityChecker(ABC):
    """Abstract base class for quality checkers."""

    @abstractmethod
    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Check for quality issues in the point cloud."""
        pass


class LidarTileLoader:
    """Utility class for loading LiDAR tiles."""

    @staticmethod
    def load_tile(tile_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        """Load a LiDAR tile and extract metadata."""
        try:
            pipeline_json = {"pipeline": [{"type": "readers.las", "filename": str(tile_path)}]}

            pipeline = pdal.Pipeline(json.dumps(pipeline_json))
            pipeline.execute()

            arrays = pipeline.arrays[0]

            metadata = {
                "filename": tile_path.name,
                "path": str(tile_path),
                "point_count": len(arrays),
                "has_intensity": "Intensity" in arrays.dtype.names,
                "has_classification": "Classification" in arrays.dtype.names,
                "has_returns": "NumberOfReturns" in arrays.dtype.names,
                "has_rgb": all(c in arrays.dtype.names for c in ["Red", "Green", "Blue"]),
                "bounds": {
                    "min_x": float(arrays["X"].min()),
                    "max_x": float(arrays["X"].max()),
                    "min_y": float(arrays["Y"].min()),
                    "max_y": float(arrays["Y"].max()),
                    "min_z": float(arrays["Z"].min()),
                    "max_z": float(arrays["Z"].max()),
                },
            }

            if metadata["has_classification"]:
                unique_classes = np.unique(arrays["Classification"])
                metadata["classifications"] = unique_classes.tolist()

            return arrays, metadata

        except Exception as e:
            logger.error(f"Error loading tile {tile_path}: {e}")
            raise


def calculate_point_density_grid(points: np.ndarray, cell_size: float = 1.0) -> tuple[np.ndarray, float, float]:
    """Calculate point density on a regular grid.

    Returns:
        grid: 2D density grid
        mean_density: Average points per square meter
        density_variation: Coefficient of variation of density
    """
    x = points["X"]
    y = points["Y"]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_bins = int(np.ceil((x_max - x_min) / cell_size))
    y_bins = int(np.ceil((y_max - y_min) / cell_size))

    hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])

    density_grid = hist / (cell_size * cell_size)

    non_zero_cells = density_grid[density_grid > 0]
    if len(non_zero_cells) > 0:
        mean_density = non_zero_cells.mean()
        density_std = non_zero_cells.std()
        density_variation = density_std / mean_density if mean_density > 0 else 0
    else:
        mean_density = 0
        density_variation = 0

    return density_grid, float(mean_density), float(density_variation)


def identify_data_voids(density_grid: np.ndarray, threshold: float = 0.1) -> tuple[np.ndarray, float]:
    """Identify data voids in the density grid.

    Returns:
        void_mask: Boolean mask of void cells
        void_ratio: Ratio of void area to total area
    """
    void_mask = density_grid < threshold
    void_ratio = np.sum(void_mask) / void_mask.size
    return void_mask, float(void_ratio)


def estimate_noise_level(points: np.ndarray, k_neighbors: int = 10) -> float:
    """Estimate noise level using local point statistics."""
    from scipy.spatial import cKDTree

    coords = np.column_stack((points["X"], points["Y"], points["Z"]))
    tree = cKDTree(coords)

    n_samples = min(10000, len(points))
    sample_indices = np.random.choice(len(points), n_samples, replace=False)

    distances = []
    for idx in sample_indices:
        _, indices = tree.query(coords[idx], k=k_neighbors + 1)
        neighbor_coords = coords[indices[1:]]

        centroid = neighbor_coords.mean(axis=0)
        centered = neighbor_coords - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]

        dist = abs(np.dot(coords[idx] - centroid, normal))
        distances.append(dist)

    distances = np.array(distances)
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    noise_level = 1.4826 * mad

    return float(noise_level)
