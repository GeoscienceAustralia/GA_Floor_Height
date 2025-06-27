"""Noise and outlier detection for LiDAR data.

Implements various algorithms for detecting and characterizing noise,
outliers, and sensor artifacts in point clouds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .lidar_qa_base import IssueSeverity, IssueType, QualityChecker, QualityIssue, estimate_noise_level


class StatisticalOutlierChecker(QualityChecker):
    """Statistical Outlier Removal (SOR) based detection."""

    def __init__(self, k_neighbors: int = 20, std_multiplier: float = 2.0, max_outlier_ratio: float = 0.02):
        self.k_neighbors = k_neighbors
        self.std_multiplier = std_multiplier
        self.max_outlier_ratio = max_outlier_ratio

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Detect statistical outliers."""
        issues = []

        coords = np.column_stack((point_cloud["X"], point_cloud["Y"], point_cloud["Z"]))

        tree = cKDTree(coords)

        n_points = len(coords)
        if n_points > 100000:
            sample_size = 50000
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
        else:
            sample_indices = np.arange(n_points)

        mean_distances = []
        for idx in sample_indices:
            distances, _ = tree.query(coords[idx], k=self.k_neighbors + 1)
            mean_dist = np.mean(distances[1:])
            mean_distances.append(mean_dist)

        mean_distances = np.array(mean_distances)

        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + self.std_multiplier * global_std

        outlier_mask = mean_distances > threshold
        outlier_ratio = np.sum(outlier_mask) / len(outlier_mask)

        estimated_outlier_ratio = outlier_ratio

        if estimated_outlier_ratio > self.max_outlier_ratio:
            severity = (
                IssueSeverity.ERROR if estimated_outlier_ratio > self.max_outlier_ratio * 2 else IssueSeverity.WARNING
            )

            outlier_indices = sample_indices[outlier_mask]

            issues.append(
                QualityIssue(
                    issue_type=IssueType.OUTLIERS,
                    severity=severity,
                    description=f"High outlier ratio detected ({estimated_outlier_ratio * 100:.1f}%)",
                    metrics={
                        "outlier_ratio": float(estimated_outlier_ratio),
                        "threshold": float(threshold),
                        "mean_distance": float(global_mean),
                        "std_distance": float(global_std),
                        "num_outliers": int(estimated_outlier_ratio * n_points),
                    },
                    affected_points=outlier_indices,
                    correction_possible=True,
                    correction_method="statistical_outlier_removal",
                )
            )

        return issues


class RadiusOutlierChecker(QualityChecker):
    """Radius-based outlier detection for isolated points."""

    def __init__(self, search_radius: float = 0.5, min_neighbors: int = 5):
        self.search_radius = search_radius
        self.min_neighbors = min_neighbors

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Detect isolated points."""
        issues = []

        coords = np.column_stack((point_cloud["X"], point_cloud["Y"], point_cloud["Z"]))

        tree = cKDTree(coords)

        n_points = len(coords)
        sample_size = min(30000, n_points)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)

        isolated_count = 0
        isolated_indices = []

        for idx in sample_indices:
            neighbors = tree.query_ball_point(coords[idx], self.search_radius)
            if len(neighbors) < self.min_neighbors:
                isolated_count += 1
                isolated_indices.append(idx)

        isolated_ratio = isolated_count / sample_size

        if isolated_ratio > 0.01:
            issues.append(
                QualityIssue(
                    issue_type=IssueType.OUTLIERS,
                    severity=IssueSeverity.WARNING,
                    description=f"Isolated points detected ({isolated_ratio * 100:.1f}%)",
                    metrics={
                        "isolated_ratio": float(isolated_ratio),
                        "search_radius": self.search_radius,
                        "min_neighbors": self.min_neighbors,
                        "estimated_isolated_count": int(isolated_ratio * n_points),
                    },
                    affected_points=np.array(isolated_indices[:1000]),
                    correction_possible=True,
                    correction_method="radius_outlier_removal",
                )
            )

        return issues


class NoiseEstimator(QualityChecker):
    """Estimate overall noise level and detect noisy regions."""

    def __init__(self, max_noise_level: float = 0.05, k_neighbors: int = 15):
        self.max_noise_level = max_noise_level
        self.k_neighbors = k_neighbors

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Estimate noise level."""
        issues = []

        noise_level = estimate_noise_level(point_cloud, self.k_neighbors)

        if noise_level > self.max_noise_level:
            severity = IssueSeverity.ERROR if noise_level > self.max_noise_level * 2 else IssueSeverity.WARNING

            issues.append(
                QualityIssue(
                    issue_type=IssueType.NOISE,
                    severity=severity,
                    description=f"High noise level detected ({noise_level:.3f}m)",
                    metrics={
                        "noise_level": noise_level,
                        "max_allowed": self.max_noise_level,
                        "signal_to_noise": float(metadata["bounds"]["max_z"] - metadata["bounds"]["min_z"])
                        / noise_level,
                    },
                    correction_possible=True,
                    correction_method="noise_filtering",
                )
            )

        x = point_cloud["X"]
        y = point_cloud["Y"]

        grid_size = 10.0
        x_bins = np.arange(x.min(), x.max() + grid_size, grid_size)
        y_bins = np.arange(y.min(), y.max() + grid_size, grid_size)

        high_noise_cells = []

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (x >= x_bins[i]) & (x < x_bins[i + 1]) & (y >= y_bins[j]) & (y < y_bins[j + 1])

                if np.sum(mask) > 100:
                    cell_points = point_cloud[mask]
                    cell_noise = estimate_noise_level(cell_points, min(10, np.sum(mask) // 10))

                    if cell_noise > self.max_noise_level * 2:
                        high_noise_cells.append(
                            {
                                "x": (x_bins[i] + x_bins[i + 1]) / 2,
                                "y": (y_bins[j] + y_bins[j + 1]) / 2,
                                "noise_level": cell_noise,
                            }
                        )

        if len(high_noise_cells) > 5:
            issues.append(
                QualityIssue(
                    issue_type=IssueType.NOISE,
                    severity=IssueSeverity.WARNING,
                    description=f"Localized high-noise regions detected ({len(high_noise_cells)} cells)",
                    metrics={
                        "high_noise_cell_count": len(high_noise_cells),
                        "max_cell_noise": max(c["noise_level"] for c in high_noise_cells),
                        "cells": high_noise_cells[:10],
                    },
                    correction_possible=True,
                    correction_method="adaptive_noise_filtering",
                )
            )

        return issues


class GhostingDetector(QualityChecker):
    """Detect ghosting artifacts from multi-path reflections."""

    def __init__(self, intensity_threshold: float = 0.8, elevation_threshold: float = 0.5):
        self.intensity_threshold = intensity_threshold
        self.elevation_threshold = elevation_threshold

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Detect ghosting artifacts."""
        issues = []

        if not metadata.get("has_intensity", False):
            return issues

        intensity = point_cloud["Intensity"]
        z = point_cloud["Z"]

        high_intensity_mask = intensity > np.percentile(intensity, 95)

        if np.sum(high_intensity_mask) > 10:
            coords = np.column_stack((point_cloud["X"], point_cloud["Y"], point_cloud["Z"]))

            tree = cKDTree(coords)

            ghost_candidates = []
            high_intensity_indices = np.where(high_intensity_mask)[0]

            sample_size = min(1000, len(high_intensity_indices))
            sample_indices = np.random.choice(high_intensity_indices, sample_size, replace=False)

            for idx in sample_indices:
                neighbors_idx = tree.query_ball_point(coords[idx], r=2.0)
                if len(neighbors_idx) > 10:
                    neighbor_z = z[neighbors_idx]
                    z_diff = z[idx] - np.median(neighbor_z)

                    if z_diff > self.elevation_threshold:
                        ghost_candidates.append(
                            {"index": int(idx), "z_diff": float(z_diff), "intensity": float(intensity[idx])}
                        )

            if len(ghost_candidates) > 5:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.GHOSTING,
                        severity=IssueSeverity.WARNING,
                        description=f"Potential ghosting artifacts detected ({len(ghost_candidates)} points)",
                        metrics={
                            "ghost_candidate_count": len(ghost_candidates),
                            "sample_size": sample_size,
                            "estimated_total": int(len(ghost_candidates) * len(high_intensity_indices) / sample_size),
                            "examples": ghost_candidates[:5],
                        },
                        correction_possible=True,
                        correction_method="ghost_point_removal",
                    )
                )

        return issues


class BloomingDetector(QualityChecker):
    """Detect blooming artifacts on reflective surfaces."""

    def __init__(self, intensity_threshold: float = 0.9, density_multiplier: float = 3.0):
        self.intensity_threshold = intensity_threshold
        self.density_multiplier = density_multiplier

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Detect blooming artifacts."""
        issues = []

        if not metadata.get("has_intensity", False):
            return issues

        intensity = point_cloud["Intensity"]

        intensity_threshold = np.percentile(intensity, 99)
        high_intensity_mask = intensity > intensity_threshold

        if np.sum(high_intensity_mask) > 50:
            coords = np.column_stack((point_cloud["X"], point_cloud["Y"], point_cloud["Z"]))

            tree = cKDTree(coords)

            avg_density = len(coords) / (
                (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
            )

            blooming_regions = []
            high_intensity_indices = np.where(high_intensity_mask)[0]

            sample_size = min(500, len(high_intensity_indices))
            sample_indices = np.random.choice(high_intensity_indices, sample_size, replace=False)

            for idx in sample_indices:
                neighbors = tree.query_ball_point(coords[idx], r=0.5)
                local_density = len(neighbors) / (np.pi * 0.5**2)

                if local_density > avg_density * self.density_multiplier:
                    blooming_regions.append(
                        {
                            "center": coords[idx].tolist(),
                            "local_density": float(local_density),
                            "density_ratio": float(local_density / avg_density),
                            "intensity": float(intensity[idx]),
                        }
                    )

            if len(blooming_regions) > 3:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.BLOOMING,
                        severity=IssueSeverity.WARNING,
                        description=f"Blooming artifacts detected ({len(blooming_regions)} regions)",
                        metrics={
                            "blooming_region_count": len(blooming_regions),
                            "avg_density": float(avg_density),
                            "max_density_ratio": max(r["density_ratio"] for r in blooming_regions),
                            "examples": blooming_regions[:5],
                        },
                        correction_possible=True,
                        correction_method="blooming_correction",
                    )
                )

        return issues


class WeatherNoiseDetector(QualityChecker):
    """Detect noise from weather conditions (rain, fog, snow)."""

    def __init__(self, low_intensity_threshold: float = 0.2, scatter_threshold: float = 0.8):
        self.low_intensity_threshold = low_intensity_threshold
        self.scatter_threshold = scatter_threshold

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Detect weather-induced noise."""
        issues = []

        z = point_cloud["Z"]

        z_percentiles = np.percentile(z, [50, 90, 95, 99])

        high_z_mask = z > z_percentiles[2] + 2.0

        if np.sum(high_z_mask) > 100:
            high_z_points = point_cloud[high_z_mask]

            x_high = high_z_points["X"]
            y_high = high_z_points["Y"]

            coords_high = np.column_stack((x_high, y_high))
            if len(coords_high) > 10:
                tree = cKDTree(coords_high)
                nn_distances = []

                sample_size = min(500, len(coords_high))
                for i in range(sample_size):
                    if i < len(coords_high):
                        dist, _ = tree.query(coords_high[i], k=2)
                        nn_distances.append(dist[1])

                avg_nn_distance = np.mean(nn_distances)

                area = (x_high.max() - x_high.min()) * (y_high.max() - y_high.min())
                expected_nn_distance = 0.5 * np.sqrt(area / len(coords_high))

                scatter_ratio = avg_nn_distance / expected_nn_distance if expected_nn_distance > 0 else 0

                weather_score = scatter_ratio
                if metadata.get("has_intensity", False):
                    intensity_high = high_z_points["Intensity"]
                    intensity_all = point_cloud["Intensity"]

                    intensity_ratio = np.median(intensity_high) / np.median(intensity_all)
                    if intensity_ratio < self.low_intensity_threshold:
                        weather_score *= 2

                if weather_score > self.scatter_threshold:
                    issues.append(
                        QualityIssue(
                            issue_type=IssueType.WEATHER_NOISE,
                            severity=IssueSeverity.WARNING,
                            description=f"Potential weather noise detected ({np.sum(high_z_mask)} points)",
                            metrics={
                                "affected_point_count": int(np.sum(high_z_mask)),
                                "scatter_ratio": float(scatter_ratio),
                                "height_above_surface": float(z[high_z_mask].mean() - z_percentiles[1]),
                                "weather_score": float(weather_score),
                            },
                            correction_possible=True,
                            correction_method="weather_noise_filtering",
                        )
                    )

        return issues
