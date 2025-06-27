"""Main LiDAR QA pipeline for processing tiles and generating reports.

This module orchestrates the quality assessment of LiDAR tiles, runs all checks,
and generates comprehensive reports in Parquet format.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .classification_checks import (
    ClassificationCompletenessChecker,
    ClassificationNoiseDetector,
    EdgeClassificationChecker,
    MisclassificationDetector,
    NoisePointValidator,
)
from .geometric_checks import AlignmentChecker, BlindSpotChecker, DensityChecker, EdgeArtifactChecker, StripingChecker
from .lidar_qa_base import (
    LidarTileLoader,
    QualityLevel,
    QualityMetrics,
    TileQAReport,
    calculate_point_density_grid,
    identify_data_voids,
)
from .noise_detection import (
    BloomingDetector,
    GhostingDetector,
    NoiseEstimator,
    RadiusOutlierChecker,
    StatisticalOutlierChecker,
    WeatherNoiseDetector,
)


class LidarQAPipeline:
    """Main pipeline for LiDAR quality assessment."""

    def __init__(self, output_dir: Path):
        """Initialize the QA pipeline.

        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkers = [
            DensityChecker(min_density=2.0, max_variation=0.5),
            StripingChecker(max_offset=0.1),
            AlignmentChecker(max_horizontal_error=0.25, max_vertical_error=0.10),
            EdgeArtifactChecker(max_scan_angle=20.0),
            BlindSpotChecker(min_range=0.5),
            StatisticalOutlierChecker(k_neighbors=20, std_multiplier=2.0),
            RadiusOutlierChecker(search_radius=0.5, min_neighbors=5),
            NoiseEstimator(max_noise_level=0.05),
            GhostingDetector(intensity_threshold=0.8),
            BloomingDetector(intensity_threshold=0.9),
            WeatherNoiseDetector(low_intensity_threshold=0.2),
            ClassificationCompletenessChecker(max_unclassified_ratio=0.02),
            MisclassificationDetector(height_threshold=0.5),
            ClassificationNoiseDetector(min_cluster_size=10),
            EdgeClassificationChecker(edge_width=0.5),
            NoisePointValidator(isolation_radius=2.0),
        ]

    def process_tile(self, tile_path: Path) -> TileQAReport:
        """Process a single LiDAR tile.

        Args:
            tile_path: Path to the LiDAR tile

        Returns:
            QA report for the tile
        """
        start_time = time.time()

        try:
            point_cloud, metadata = LidarTileLoader.load_tile(tile_path)
        except Exception as e:
            logger.error(f"Failed to load {tile_path}: {e}")

            report = TileQAReport(
                tile_path=tile_path,
                metrics=QualityMetrics(),
                processing_time=time.time() - start_time,
                metadata={"error": str(e), "error_type": "load_error"},
            )
            return report

        metrics = self._calculate_metrics(point_cloud, metadata)

        report = TileQAReport(tile_path=tile_path, metrics=metrics, processing_time=0, metadata=metadata)

        for checker in self.checkers:
            try:
                issues = checker.check(point_cloud, metadata)
                for issue in issues:
                    report.add_issue(issue)
            except Exception as e:
                logger.warning(f"Checker {checker.__class__.__name__} failed on {tile_path}: {e}")

        report.quality_level = self._determine_quality_level(metrics)

        report.processing_time = time.time() - start_time

        return report

    def _calculate_metrics(self, point_cloud: np.ndarray, metadata: dict[str, Any]) -> QualityMetrics:
        """Calculate quality metrics for the point cloud."""
        metrics = QualityMetrics()

        metrics.point_count = len(point_cloud)

        x_range = metadata["bounds"]["max_x"] - metadata["bounds"]["min_x"]
        y_range = metadata["bounds"]["max_y"] - metadata["bounds"]["min_y"]
        metrics.area = x_range * y_range

        density_grid, mean_density, density_variation = calculate_point_density_grid(point_cloud)
        metrics.density = mean_density
        metrics.density_variation = density_variation

        void_mask, void_ratio = identify_data_voids(density_grid)
        metrics.void_ratio = void_ratio
        metrics.coverage_ratio = 1.0 - void_ratio

        if metadata.get("has_classification", False):
            classification = point_cloud["Classification"]
            unique_classes, counts = np.unique(classification, return_counts=True)
            class_dist = dict(zip(unique_classes, counts, strict=False))

            total = len(classification)
            unclassified = class_dist.get(0, 0) + class_dist.get(1, 0)
            metrics.classified_ratio = 1.0 - (unclassified / total)
            metrics.ground_points_ratio = class_dist.get(2, 0) / total
            metrics.building_points_ratio = class_dist.get(6, 0) / total
            metrics.noise_points_ratio = class_dist.get(7, 0) / total

        if metadata.get("has_intensity", False):
            intensity = point_cloud["Intensity"]
            metrics.intensity_mean = float(np.mean(intensity))
            metrics.intensity_std = float(np.std(intensity))

            if metrics.intensity_mean > 0:
                metrics.intensity_consistency = 1.0 - (metrics.intensity_std / metrics.intensity_mean)

        return metrics

    def _determine_quality_level(self, metrics: QualityMetrics) -> QualityLevel | None:
        """Determine the quality level based on metrics."""

        if metrics.density >= 8.0:
            return QualityLevel.QL1
        elif metrics.density >= 2.0:
            return QualityLevel.QL2
        elif metrics.density >= 0.5:
            return QualityLevel.QL3
        else:
            return None

    def process_directory(self, directory: Path, pattern: str = "*.las", max_workers: int = 4) -> pd.DataFrame:
        """Process all LiDAR tiles in a directory.

        Args:
            directory: Directory containing LiDAR tiles
            pattern: File pattern to match
            max_workers: Number of parallel workers

        Returns:
            DataFrame with QA results
        """

        tiles = list(directory.rglob(pattern))
        if pattern == "*.las":
            tiles.extend(list(directory.rglob("*.laz")))

        logger.info(f"Found {len(tiles)} tiles to process")

        if not tiles:
            return pd.DataFrame()

        reports = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {executor.submit(self.process_tile, tile): tile for tile in tiles}

            with tqdm(total=len(tiles), desc="Processing tiles") as pbar:
                for future in as_completed(future_to_tile):
                    tile = future_to_tile[future]
                    try:
                        report = future.result()
                        reports.append(report)
                    except Exception as e:
                        logger.error(f"Failed to process {tile}: {e}")
                    pbar.update(1)

        df = self._reports_to_dataframe(reports)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"lidar_qa_report_{timestamp}.parquet"
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved QA report to {output_file}")

        return df

    def _reports_to_dataframe(self, reports: list[TileQAReport]) -> pd.DataFrame:
        """Convert list of reports to a DataFrame."""
        rows = []

        for report in reports:
            row = {
                "tile_path": str(report.tile_path),
                "tile_name": report.tile_path.name,
                "processing_time": report.processing_time,
                "quality_level": report.quality_level.name if report.quality_level else None,
                "needs_correction": report.needs_correction(),
                "point_count": report.metrics.point_count,
                "area": report.metrics.area,
                "density": report.metrics.density,
                "density_variation": report.metrics.density_variation,
                "coverage_ratio": report.metrics.coverage_ratio,
                "void_ratio": report.metrics.void_ratio,
                "classified_ratio": report.metrics.classified_ratio,
                "ground_points_ratio": report.metrics.ground_points_ratio,
                "building_points_ratio": report.metrics.building_points_ratio,
                "noise_points_ratio": report.metrics.noise_points_ratio,
                "outlier_ratio": report.metrics.outlier_ratio,
                "noise_level": report.metrics.noise_level,
                "intensity_mean": report.metrics.intensity_mean,
                "intensity_std": report.metrics.intensity_std,
                "issues_info": sum(1 for i in report.issues if i.severity.name == "INFO"),
                "issues_warning": sum(1 for i in report.issues if i.severity.name == "WARNING"),
                "issues_error": sum(1 for i in report.issues if i.severity.name == "ERROR"),
                "issues_critical": sum(1 for i in report.issues if i.severity.name == "CRITICAL"),
                "issue_types": ",".join(sorted({i.issue_type.value for i in report.issues})),
                "has_intensity": report.metadata.get("has_intensity", False),
                "has_classification": report.metadata.get("has_classification", False),
                "has_returns": report.metadata.get("has_returns", False),
                "has_rgb": report.metadata.get("has_rgb", False),
                "min_x": report.metadata["bounds"]["min_x"],
                "max_x": report.metadata["bounds"]["max_x"],
                "min_y": report.metadata["bounds"]["min_y"],
                "max_y": report.metadata["bounds"]["max_y"],
                "min_z": report.metadata["bounds"]["min_z"],
                "max_z": report.metadata["bounds"]["max_z"],
            }

            issue_flags = {
                "has_striping": any(i.issue_type.value == "striping" for i in report.issues),
                "has_misalignment": any(i.issue_type.value == "misalignment" for i in report.issues),
                "has_low_density": any(i.issue_type.value == "low_density" for i in report.issues),
                "has_data_voids": any(i.issue_type.value == "data_void" for i in report.issues),
                "has_outliers": any(i.issue_type.value == "outliers" for i in report.issues),
                "has_noise": any(i.issue_type.value == "noise" for i in report.issues),
                "has_misclassification": any(i.issue_type.value == "misclassification" for i in report.issues),
                "has_unclassified": any(i.issue_type.value == "unclassified" for i in report.issues),
            }
            row.update(issue_flags)

            rows.append(row)

        return pd.DataFrame(rows)

    def generate_summary_report(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a summary report from the QA results."""
        summary = {
            "total_tiles": int(len(df)),
            "tiles_needing_correction": int(df["needs_correction"].sum()),
            "average_density": float(df["density"].mean()),
            "quality_level_distribution": df["quality_level"].value_counts().to_dict(),
            "issue_type_frequency": {},
            "severity_distribution": {
                "info": int(df["issues_info"].sum()),
                "warning": int(df["issues_warning"].sum()),
                "error": int(df["issues_error"].sum()),
                "critical": int(df["issues_critical"].sum()),
            },
        }

        for issue_types in df["issue_types"]:
            if issue_types:
                for issue_type in issue_types.split(","):
                    summary["issue_type_frequency"][issue_type] = summary["issue_type_frequency"].get(issue_type, 0) + 1

        summary_file = self.output_dir / "qa_summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved summary to {summary_file}")

        return summary


def main():
    """Main entry point for the QA pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="LiDAR Quality Assurance Pipeline")
    parser.add_argument("input_dir", type=Path, help="Directory containing LiDAR tiles")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/GA-floor-height/data/exports"),
        help="Output directory for reports",
    )
    parser.add_argument("--pattern", default="*.las", help="File pattern to match")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    pipeline = LidarQAPipeline(args.output)

    df = pipeline.process_directory(args.input_dir, args.pattern, args.workers)

    if not df.empty:
        summary = pipeline.generate_summary_report(df)

        logger.info(f"Processed {len(df)} tiles")
        logger.info(f"Tiles needing correction: {summary['tiles_needing_correction']}")
        logger.info(f"Average density: {summary['average_density']:.2f} pts/m²")


if __name__ == "__main__":
    main()
