"""Classification quality checks for LiDAR data.

Detects misclassification, unclassified points, and other classification issues.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .lidar_qa_base import IssueSeverity, IssueType, QualityChecker, QualityIssue

COMMON_CLASS_CODES = {
    0: "Never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Noise",
}


class ClassificationCompletenessChecker(QualityChecker):
    """Check for unclassified or minimally classified data."""

    def __init__(self, max_unclassified_ratio: float = 0.02, min_ground_ratio: float = 0.05):
        self.max_unclassified_ratio = max_unclassified_ratio
        self.min_ground_ratio = min_ground_ratio

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Check classification completeness."""
        issues = []

        if not metadata.get("has_classification", False):
            issues.append(
                QualityIssue(
                    issue_type=IssueType.UNCLASSIFIED,
                    severity=IssueSeverity.CRITICAL,
                    description="No classification data available",
                    metrics={"has_classification": False},
                    correction_possible=True,
                    correction_method="automatic_classification",
                )
            )
            return issues

        classification = point_cloud["Classification"]
        unique_classes, counts = np.unique(classification, return_counts=True)
        class_distribution = dict(zip(unique_classes, counts, strict=False))
        total_points = len(classification)

        unclassified_count = class_distribution.get(0, 0) + class_distribution.get(1, 0)
        unclassified_ratio = unclassified_count / total_points

        if unclassified_ratio > self.max_unclassified_ratio:
            severity = IssueSeverity.ERROR if unclassified_ratio > 0.1 else IssueSeverity.WARNING
            issues.append(
                QualityIssue(
                    issue_type=IssueType.UNCLASSIFIED,
                    severity=severity,
                    description=f"High ratio of unclassified points ({unclassified_ratio * 100:.1f}%)",
                    metrics={
                        "unclassified_ratio": float(unclassified_ratio),
                        "unclassified_count": int(unclassified_count),
                        "max_allowed_ratio": self.max_unclassified_ratio,
                    },
                    correction_possible=True,
                    correction_method="classification_refinement",
                )
            )

        ground_count = class_distribution.get(2, 0)
        ground_ratio = ground_count / total_points

        if ground_ratio < self.min_ground_ratio:
            issues.append(
                QualityIssue(
                    issue_type=IssueType.MISCLASSIFICATION,
                    severity=IssueSeverity.WARNING,
                    description=f"Low ground point ratio ({ground_ratio * 100:.1f}%)",
                    metrics={
                        "ground_ratio": float(ground_ratio),
                        "ground_count": int(ground_count),
                        "expected_min_ratio": self.min_ground_ratio,
                    },
                    correction_possible=True,
                    correction_method="ground_classification",
                )
            )

        class_summary = {}
        for class_id, count in class_distribution.items():
            class_name = COMMON_CLASS_CODES.get(class_id, f"Class {class_id}")
            class_summary[class_name] = {"count": int(count), "ratio": float(count / total_points)}

        issues.append(
            QualityIssue(
                issue_type=IssueType.UNCLASSIFIED,
                severity=IssueSeverity.INFO,
                description=f"Classification distribution: {len(unique_classes)} classes used",
                metrics={"unique_classes": len(unique_classes), "distribution": class_summary},
                correction_possible=False,
            )
        )

        return issues


class MisclassificationDetector(QualityChecker):
    """Detect potential misclassifications using geometric rules."""

    def __init__(self, height_threshold: float = 0.5, planarity_threshold: float = 0.1):
        self.height_threshold = height_threshold
        self.planarity_threshold = planarity_threshold

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Detect misclassifications."""
        issues = []

        if not metadata.get("has_classification", False):
            return issues

        classification = point_cloud["Classification"]
        z = point_cloud["Z"]

        ground_mask = classification == 2
        if np.sum(ground_mask) > 10:
            ground_z = z[ground_mask]
            z_min = z.min()

            high_ground_mask = ground_z > (z_min + 5.0)
            high_ground_ratio = np.sum(high_ground_mask) / len(ground_z)

            if high_ground_ratio > 0.01:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.MISCLASSIFICATION,
                        severity=IssueSeverity.WARNING,
                        description="Potential ground misclassification (high elevation)",
                        metrics={
                            "high_ground_ratio": float(high_ground_ratio),
                            "max_ground_height": float(ground_z.max() - z_min),
                            "affected_points": int(np.sum(high_ground_mask)),
                        },
                        correction_possible=True,
                        correction_method="ground_reclassification",
                    )
                )

        building_mask = classification == 6
        if np.sum(building_mask) > 10 and np.sum(ground_mask) > 10:
            building_z = z[building_mask]
            ground_z_median = np.median(z[ground_mask])

            low_building_mask = building_z < (ground_z_median + self.height_threshold)
            low_building_ratio = np.sum(low_building_mask) / len(building_z)

            if low_building_ratio > 0.1:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.MISCLASSIFICATION,
                        severity=IssueSeverity.WARNING,
                        description="Potential building misclassification (low elevation)",
                        metrics={
                            "low_building_ratio": float(low_building_ratio),
                            "height_threshold": self.height_threshold,
                            "affected_points": int(np.sum(low_building_mask)),
                        },
                        correction_possible=True,
                        correction_method="building_reclassification",
                    )
                )

        veg_classes = [3, 4, 5]
        for veg_class in veg_classes:
            veg_mask = classification == veg_class
            if np.sum(veg_mask) > 100:
                veg_z = z[veg_mask]
                expected_heights = {3: (0.0, 0.5), 4: (0.5, 2.0), 5: (2.0, None)}

                min_h, max_h = expected_heights[veg_class]
                if np.sum(ground_mask) > 10:
                    height_above_ground = veg_z - np.median(z[ground_mask])

                    out_of_range = 0
                    if min_h is not None:
                        out_of_range += np.sum(height_above_ground < min_h)
                    if max_h is not None:
                        out_of_range += np.sum(height_above_ground > max_h)

                    out_of_range_ratio = out_of_range / len(height_above_ground)

                    if out_of_range_ratio > 0.2:
                        veg_name = COMMON_CLASS_CODES.get(veg_class, f"Class {veg_class}")
                        issues.append(
                            QualityIssue(
                                issue_type=IssueType.MISCLASSIFICATION,
                                severity=IssueSeverity.WARNING,
                                description=f"{veg_name} height inconsistency",
                                metrics={
                                    "class": veg_class,
                                    "out_of_range_ratio": float(out_of_range_ratio),
                                    "expected_range": expected_heights[veg_class],
                                    "actual_range": (
                                        float(height_above_ground.min()),
                                        float(height_above_ground.max()),
                                    ),
                                },
                                correction_possible=True,
                                correction_method="vegetation_reclassification",
                            )
                        )

        return issues


class ClassificationNoiseDetector(QualityChecker):
    """Detect noise in classification (isolated class changes)."""

    def __init__(self, min_cluster_size: int = 10, search_radius: float = 1.0):
        self.min_cluster_size = min_cluster_size
        self.search_radius = search_radius

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Detect classification noise."""
        issues = []

        if not metadata.get("has_classification", False):
            return issues

        classification = point_cloud["Classification"]
        coords = np.column_stack((point_cloud["X"], point_cloud["Y"], point_cloud["Z"]))

        tree = cKDTree(coords)

        n_points = len(coords)
        sample_size = min(20000, n_points)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)

        isolated_count = 0
        class_change_count = 0

        for idx in sample_indices:
            neighbor_indices = tree.query_ball_point(coords[idx], self.search_radius)

            if len(neighbor_indices) > self.min_cluster_size:
                neighbor_classes = classification[neighbor_indices]

                unique_classes, counts = np.unique(neighbor_classes, return_counts=True)
                majority_class = unique_classes[np.argmax(counts)]

                if classification[idx] != majority_class:
                    class_change_count += 1

                    same_class_count = np.sum(neighbor_classes == classification[idx])
                    if same_class_count < 3:
                        isolated_count += 1

        noise_ratio = class_change_count / sample_size
        isolated_ratio = isolated_count / sample_size

        if noise_ratio > 0.05:
            issues.append(
                QualityIssue(
                    issue_type=IssueType.CLASSIFICATION_NOISE,
                    severity=IssueSeverity.WARNING,
                    description=f"High classification noise detected ({noise_ratio * 100:.1f}%)",
                    metrics={
                        "noise_ratio": float(noise_ratio),
                        "isolated_ratio": float(isolated_ratio),
                        "estimated_noisy_points": int(noise_ratio * n_points),
                        "search_radius": self.search_radius,
                    },
                    correction_possible=True,
                    correction_method="classification_smoothing",
                )
            )

        return issues


class EdgeClassificationChecker(QualityChecker):
    """Check classification quality at class boundaries."""

    def __init__(self, edge_width: float = 0.5, min_edge_points: int = 100):
        self.edge_width = edge_width
        self.min_edge_points = min_edge_points

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Check edge classification quality."""
        issues = []

        if not metadata.get("has_classification", False):
            return issues

        classification = point_cloud["Classification"]
        coords = np.column_stack((point_cloud["X"], point_cloud["Y"], point_cloud["Z"]))

        ground_mask = classification == 2
        building_mask = classification == 6

        if np.sum(ground_mask) < 100 or np.sum(building_mask) < 100:
            return issues

        ground_coords = coords[ground_mask]
        building_coords = coords[building_mask]

        ground_tree = cKDTree(ground_coords)

        sample_size = min(5000, len(building_coords))
        sample_indices = np.random.choice(len(building_coords), sample_size, replace=False)

        edge_points = []
        mixed_edge_points = 0

        for i in sample_indices:
            neighbors = ground_tree.query_ball_point(building_coords[i], self.edge_width)

            if len(neighbors) > 0:
                edge_points.append(i)

                building_tree = cKDTree(building_coords)
                building_neighbors = building_tree.query_ball_point(building_coords[i], self.edge_width)

                mixing_ratio = len(neighbors) / (len(neighbors) + len(building_neighbors))
                if 0.3 < mixing_ratio < 0.7:
                    mixed_edge_points += 1

        if len(edge_points) > self.min_edge_points:
            edge_quality = 1.0 - (mixed_edge_points / len(edge_points))

            if edge_quality < 0.7:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.MISCLASSIFICATION,
                        severity=IssueSeverity.WARNING,
                        description=f"Poor edge classification quality (score={edge_quality:.2f})",
                        metrics={
                            "edge_quality_score": float(edge_quality),
                            "edge_point_count": len(edge_points),
                            "mixed_edge_ratio": float(mixed_edge_points / len(edge_points)),
                        },
                        correction_possible=True,
                        correction_method="edge_refinement",
                    )
                )

        return issues


class NoisePointValidator(QualityChecker):
    """Validate noise classification (class 7)."""

    def __init__(self, isolation_radius: float = 2.0, max_neighbors: int = 5):
        self.isolation_radius = isolation_radius
        self.max_neighbors = max_neighbors

    def check(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> list[QualityIssue]:
        """Validate noise point classification."""
        issues = []

        if not metadata.get("has_classification", False):
            return issues

        classification = point_cloud["Classification"]
        noise_mask = classification == 7

        if np.sum(noise_mask) < 10:
            return issues

        coords = np.column_stack((point_cloud["X"], point_cloud["Y"], point_cloud["Z"]))

        tree = cKDTree(coords)
        noise_coords = coords[noise_mask]

        non_isolated_count = 0
        sample_size = min(1000, len(noise_coords))
        sample_indices = np.random.choice(len(noise_coords), sample_size, replace=False)

        for i in sample_indices:
            neighbors = tree.query_ball_point(noise_coords[i], self.isolation_radius)
            if len(neighbors) > self.max_neighbors:
                non_isolated_count += 1

        non_isolated_ratio = non_isolated_count / sample_size

        if non_isolated_ratio > 0.3:
            issues.append(
                QualityIssue(
                    issue_type=IssueType.MISCLASSIFICATION,
                    severity=IssueSeverity.WARNING,
                    description=f"Noise points not properly isolated ({non_isolated_ratio * 100:.1f}%)",
                    metrics={
                        "non_isolated_ratio": float(non_isolated_ratio),
                        "total_noise_points": int(np.sum(noise_mask)),
                        "isolation_radius": self.isolation_radius,
                        "max_neighbors": self.max_neighbors,
                    },
                    correction_possible=True,
                    correction_method="noise_reclassification",
                )
            )

        return issues
