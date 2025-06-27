"""Geometric quality checks for LiDAR data.

Detects issues like striping, misalignment, density variations, and data voids.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .lidar_qa_base import (
    IssueSeverity,
    IssueType,
    QualityChecker,
    QualityIssue,
    calculate_point_density_grid,
    identify_data_voids,
)


class DensityChecker(QualityChecker):
    """Check for point density issues."""

    def __init__(self, min_density: float = 2.0, max_variation: float = 0.5, void_threshold: float = 0.1):
        self.min_density = min_density
        self.max_variation = max_variation
        self.void_threshold = void_threshold

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Check for density-related issues."""
        issues = []

        density_grid, mean_density, density_variation = calculate_point_density_grid(point_cloud, cell_size=1.0)

        if mean_density < self.min_density:
            severity = IssueSeverity.CRITICAL if mean_density < self.min_density / 2 else IssueSeverity.ERROR
            issues.append(
                QualityIssue(
                    issue_type=IssueType.LOW_DENSITY,
                    severity=severity,
                    description=f"Point density ({mean_density:.2f} pts/m²) below minimum ({self.min_density} pts/m²)",
                    metrics={
                        "mean_density": mean_density,
                        "min_required": self.min_density,
                        "ratio": mean_density / self.min_density,
                    },
                    correction_possible=False,
                )
            )

        if density_variation > self.max_variation:
            issues.append(
                QualityIssue(
                    issue_type=IssueType.DENSITY_VARIATION,
                    severity=IssueSeverity.WARNING,
                    description=f"High density variation (CV={density_variation:.2f})",
                    metrics={"coefficient_of_variation": density_variation, "max_allowed": self.max_variation},
                    correction_possible=True,
                    correction_method="density_normalization",
                )
            )

        void_mask, void_ratio = identify_data_voids(density_grid, self.void_threshold)
        if void_ratio > 0.05:
            severity = IssueSeverity.ERROR if void_ratio > 0.1 else IssueSeverity.WARNING
            issues.append(
                QualityIssue(
                    issue_type=IssueType.DATA_VOID,
                    severity=severity,
                    description=f"Data voids detected ({void_ratio * 100:.1f}% of area)",
                    metrics={"void_ratio": void_ratio, "void_count": np.sum(void_mask), "total_cells": void_mask.size},
                    correction_possible=True,
                    correction_method="interpolation_or_fusion",
                )
            )

        return issues


class StripingChecker(QualityChecker):
    """Detect striping artifacts between flight lines."""

    def __init__(self, max_offset: float = 0.1, overlap_threshold: float = 0.2):
        self.max_offset = max_offset
        self.overlap_threshold = overlap_threshold

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Check for striping between swaths."""
        issues = []

        x = point_cloud["X"]
        z = point_cloud["Z"]

        x_bins = np.linspace(x.min(), x.max(), 100)
        x_indices = np.digitize(x, x_bins)

        z_profile = []
        for i in range(1, len(x_bins)):
            mask = x_indices == i
            if np.sum(mask) > 10:
                z_profile.append(np.median(z[mask]))

        if len(z_profile) > 10:
            z_profile = np.array(z_profile)
            z_detrended = z_profile - np.mean(z_profile)

            fft = np.fft.fft(z_detrended)
            power = np.abs(fft) ** 2

            dominant_freq_idx = np.argmax(power[1 : len(power) // 2]) + 1
            dominant_power = power[dominant_freq_idx]

            if dominant_power > np.mean(power) * 10:
                stripe_spacing = len(z_profile) * (x.max() - x.min()) / (100 * dominant_freq_idx)

                stripe_amplitude = np.std(z_detrended)

                if stripe_amplitude > self.max_offset:
                    issues.append(
                        QualityIssue(
                            issue_type=IssueType.STRIPING,
                            severity=IssueSeverity.ERROR,
                            description=f"Striping detected with amplitude {stripe_amplitude:.3f}m",
                            metrics={
                                "stripe_amplitude": float(stripe_amplitude),
                                "stripe_spacing": float(stripe_spacing),
                                "max_allowed_offset": self.max_offset,
                            },
                            correction_possible=True,
                            correction_method="strip_alignment",
                        )
                    )

        return issues


class AlignmentChecker(QualityChecker):
    """Check for alignment issues in overlapping areas."""

    def __init__(self, max_horizontal_error: float = 0.25, max_vertical_error: float = 0.10):
        self.max_horizontal_error = max_horizontal_error
        self.max_vertical_error = max_vertical_error

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Check for alignment issues."""
        issues = []

        has_gps_time = "GpsTime" in point_cloud.dtype.names

        if has_gps_time:
            gps_times = point_cloud["GpsTime"]
            time_diff = np.diff(np.sort(gps_times))

            gap_threshold = np.percentile(time_diff, 95)
            large_gaps = np.where(time_diff > gap_threshold * 10)[0]

            if len(large_gaps) > 0:
                sorted_indices = np.argsort(gps_times)
                z_sorted = point_cloud["Z"][sorted_indices]

                for gap_idx in large_gaps:
                    before_idx = max(0, gap_idx - 100)
                    after_idx = min(len(z_sorted), gap_idx + 100)

                    z_before = z_sorted[before_idx:gap_idx]
                    z_after = z_sorted[gap_idx:after_idx]

                    if len(z_before) > 10 and len(z_after) > 10:
                        z_diff = np.median(z_after) - np.median(z_before)

                        if abs(z_diff) > self.max_vertical_error:
                            issues.append(
                                QualityIssue(
                                    issue_type=IssueType.MISALIGNMENT,
                                    severity=IssueSeverity.ERROR,
                                    description=f"Vertical misalignment detected ({z_diff:.3f}m)",
                                    metrics={"vertical_offset": float(z_diff), "max_allowed": self.max_vertical_error},
                                    correction_possible=True,
                                    correction_method="vertical_adjustment",
                                )
                            )

        x = point_cloud["X"]
        y = point_cloud["Y"]
        z = point_cloud["Z"]

        a_matrix = np.column_stack([x, y, np.ones_like(x)])
        plane_params, residuals, _, _ = np.linalg.lstsq(a_matrix, z, rcond=None)

        z_predicted = a_matrix @ plane_params
        z_residuals = z - z_predicted

        residual_std = np.std(z_residuals)
        if residual_std > 0.5:
            issues.append(
                QualityIssue(
                    issue_type=IssueType.MISALIGNMENT,
                    severity=IssueSeverity.WARNING,
                    description=f"Possible geometric distortion (residual std={residual_std:.3f}m)",
                    metrics={"residual_std": float(residual_std), "plane_params": plane_params.tolist()},
                    correction_possible=True,
                    correction_method="geometric_correction",
                )
            )

        return issues


class EdgeArtifactChecker(QualityChecker):
    """Detect edge artifacts and scan angle issues."""

    def __init__(self, max_scan_angle: float = 20.0):
        self.max_scan_angle = max_scan_angle

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Check for edge-related artifacts."""
        issues = []

        if "ScanAngleRank" in point_cloud.dtype.names:
            scan_angles = point_cloud["ScanAngleRank"]

            extreme_angle_mask = np.abs(scan_angles) > self.max_scan_angle
            extreme_ratio = np.sum(extreme_angle_mask) / len(scan_angles)

            if extreme_ratio > 0.1:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.EDGE_ARTIFACTS,
                        severity=IssueSeverity.WARNING,
                        description=f"High proportion of points at extreme scan angles ({extreme_ratio * 100:.1f}%)",
                        metrics={
                            "extreme_angle_ratio": float(extreme_ratio),
                            "max_scan_angle": float(np.max(np.abs(scan_angles))),
                            "threshold": self.max_scan_angle,
                        },
                        correction_possible=True,
                        correction_method="edge_filtering",
                    )
                )

            if "Intensity" in point_cloud.dtype.names:
                intensity = point_cloud["Intensity"]

                center_mask = np.abs(scan_angles) < 5
                edge_mask = np.abs(scan_angles) > 15

                if np.sum(center_mask) > 100 and np.sum(edge_mask) > 100:
                    center_intensity = np.mean(intensity[center_mask])
                    edge_intensity = np.mean(intensity[edge_mask])

                    intensity_ratio = edge_intensity / center_intensity if center_intensity > 0 else 0

                    if intensity_ratio < 0.5 or intensity_ratio > 2.0:
                        issues.append(
                            QualityIssue(
                                issue_type=IssueType.INTENSITY_BANDING,
                                severity=IssueSeverity.WARNING,
                                description=f"Intensity variation between center and edge (ratio={intensity_ratio:.2f})",
                                metrics={
                                    "center_intensity": float(center_intensity),
                                    "edge_intensity": float(edge_intensity),
                                    "intensity_ratio": float(intensity_ratio),
                                },
                                correction_possible=True,
                                correction_method="intensity_normalization",
                            )
                        )

        return issues


class BlindSpotChecker(QualityChecker):
    """Detect sensor blind spots and near-field issues."""

    def __init__(self, min_range: float = 0.5):
        self.min_range = min_range

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Check for blind spots."""
        issues = []

        x = point_cloud["X"]
        y = point_cloud["Y"]

        x_center = (x.min() + x.max()) / 2
        y_center = (y.min() + y.max()) / 2

        distances = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

        near_field_mask = distances < self.min_range
        near_field_ratio = np.sum(near_field_mask) / len(distances)

        if near_field_ratio < 0.001:
            issues.append(
                QualityIssue(
                    issue_type=IssueType.BLIND_SPOTS,
                    severity=IssueSeverity.INFO,
                    description=f"Potential blind spot detected (min range ~{np.min(distances):.1f}m)",
                    metrics={
                        "min_observed_range": float(np.min(distances)),
                        "expected_min_range": self.min_range,
                        "near_field_point_ratio": float(near_field_ratio),
                    },
                    correction_possible=False,
                )
            )

        return issues
