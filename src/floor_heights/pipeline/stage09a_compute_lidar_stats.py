#!/usr/bin/env python
"""Stage-09a: Extract statistics from clipped LiDAR files.

Reads LAS files from stage01 and computes metrics for analysis:
- Point density and coverage
- Height distribution and percentiles
- Classification-based features
- Building structure indicators
- Features for FFH prediction
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pdal
from loguru import logger
from scipy.spatial import ConvexHull
from shapely.vectorized import contains

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from floor_heights.config import CONFIG
from floor_heights.db.schemas import (
    BatchWriter,
    Stage09aLidarStatsRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import get_processed_stage_ids, read_table
from floor_heights.utils.progress import lidar_stats_progress

STAGE_NAME = "stage09a_lidar_stats"
STAGE_DESCRIPTION = "Extract LiDAR point cloud statistics"


def compute_lidar_stats(las_path: Path, footprint_area: float, footprint_polygon: Any = None) -> dict[str, Any]:
    """Extract statistics from a clipped LAS file."""
    try:
        pipeline_json = {"pipeline": [{"type": "readers.las", "filename": str(las_path)}]}
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()

        arrays = pipeline.arrays[0]
        point_count = len(arrays)

        if point_count == 0:
            logger.warning(f"No points found in {las_path}")
            return create_empty_stats()

        x = arrays["X"]
        y = arrays["Y"]
        z = arrays["Z"]

        z_min = float(z.min())
        z_max = float(z.max())
        z_range = z_max - z_min
        z_mean = float(z.mean())
        z_std = float(z.std())
        z_rel = z - z_min

        stats = create_empty_stats()

        stats.update(
            {
                "point_count": point_count,
                "footprint_area": footprint_area,
                "point_density": point_count / footprint_area if footprint_area > 0 else 0,
                "z_min": z_min,
                "z_max": z_max,
                "z_range": z_range,
                "z_mean": z_mean,
                "z_std": z_std,
                "building_height": z_range,
            }
        )

        percentiles = np.percentile(z, [10, 25, 50, 75, 90])
        stats.update({f"z_p{p}": float(percentiles[i]) for i, p in enumerate([10, 25, 50, 75, 90])})

        height_bands = [(0, 3), (3, 6), (6, 9), (9, 12), (12, float("inf"))]
        height_labels = ["pts_0_3m", "pts_3_6m", "pts_6_9m", "pts_9_12m", "pts_above_12m"]

        for label, (lower, upper) in zip(height_labels, height_bands, strict=False):
            mask = (z_rel >= lower) & (z_rel < upper) if upper != float("inf") else z_rel >= lower
            stats[label] = int(np.sum(mask))

        if point_count > 3:
            try:
                hull = ConvexHull(np.column_stack((x, y)))
                stats["convex_hull_area"] = hull.volume

                if footprint_polygon:
                    points_in_footprint = np.sum(contains(footprint_polygon, x, y))
                    stats["coverage_ratio"] = points_in_footprint / point_count
                    stats["spatial_coverage"] = hull.volume / footprint_area if footprint_area > 0 else 0
            except Exception as e:
                logger.debug(f"ConvexHull failed: {e}")

        if "NumberOfReturns" in arrays.dtype.names:
            single_returns = np.sum(arrays["NumberOfReturns"] == 1)
            stats["returns_single"] = int(single_returns)
            stats["returns_multiple"] = int(point_count - single_returns)

        if "Intensity" in arrays.dtype.names:
            intensity = arrays["Intensity"]
            stats["intensity_mean"] = float(intensity.mean())
            stats["intensity_std"] = float(intensity.std())

        if "Classification" in arrays.dtype.names:
            classification = arrays["Classification"]
            unique_classes, counts = np.unique(classification, return_counts=True)
            class_counts = dict(zip(unique_classes, counts, strict=False))

            class_names = ["never", "unassigned", "ground", "low_veg", "med_veg", "high_veg", "building", "noise"]
            for i, name in enumerate(class_names):
                stats[f"class_{i}_{name}"] = int(class_counts.get(i, 0))

            ground_count = class_counts.get(2, 0)
            stats["ground_point_count"] = int(ground_count)
            if ground_count > 0:
                ground_z = z[classification == 2]
                stats["ground_z_mean"] = float(ground_z.mean())
                stats["ground_z_std"] = float(ground_z.std())

            stats["noise_point_count"] = int(class_counts.get(7, 0))
            stats["vegetation_proximity_count"] = int(sum(class_counts.get(i, 0) for i in [3, 4, 5]))

            building_count = class_counts.get(6, 0)
            if building_count > 3:
                building_z = z[classification == 6]
                stats["roof_z_variance"] = float(np.var(building_z))

        if point_count > 10:
            xy_extent = max(x.max() - x.min(), y.max() - y.min())
            stats["verticality_score"] = float(min(z_range / xy_extent if xy_extent > 0 else 0, 10.0))
            stats["planarity_score"] = float(1.0 / (1.0 + z_std))

        if "Classification" in arrays.dtype.names and class_counts.get(6, 0) > 10:
            building_mask = classification == 6
            building_z = z[building_mask]
            building_z_rel = building_z - building_z.min()

            hist, _ = np.histogram(building_z_rel, bins=np.arange(0, building_z_rel.max() + 0.5, 0.5))
            if len(hist) > 2:
                is_peak = (hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]) & (hist[1:-1] > 10)
                peaks = (np.where(is_peak)[0] + 1) * 0.5
                stats["building_height_peaks"] = len(peaks)
                stats["building_height_regularity"] = float(np.std(np.diff(peaks))) if len(peaks) > 1 else 0.0

            for h_min, h_max in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 20), (20, 30), (30, 50)]:
                band_count = np.sum((building_z_rel >= h_min) & (building_z_rel < h_max))
                stats[f"building_density_{h_min}_{h_max}m"] = float(band_count / (h_max - h_min))

            if class_counts.get(6, 0) > 100:
                building_x = x[building_mask]
                building_y = y[building_mask]
                x_p10, x_p90 = np.percentile(building_x, [10, 90])
                y_p10, y_p90 = np.percentile(building_y, [10, 90])

                edge_mask = (building_x < x_p10) | (building_x > x_p90) | (building_y < y_p10) | (building_y > y_p90)

                if np.sum(edge_mask) > 20:
                    edge_z = building_z[edge_mask]
                    z_min_building = building_z.min()

                    facade_scores = [
                        0.8
                        for h in range(0, int(building_z_rel.max()), 3)
                        if np.sum((edge_z >= z_min_building + h) & (edge_z < z_min_building + h + 3)) > 10
                    ]
                    stats["facade_alignment_score"] = float(np.mean(facade_scores)) if facade_scores else 0.0

        if "NumberOfReturns" in arrays.dtype.names:
            multi_returns = arrays["NumberOfReturns"] > 1
            for h_min, h_max in [(0, 10), (10, 20), (20, 30), (30, 50)]:
                height_mask = (z_rel >= h_min) & (z_rel < h_max)
                mask_count = np.sum(height_mask)
                stats[f"multi_return_ratio_{h_min}_{h_max}m"] = float(
                    np.sum(multi_returns[height_mask]) / mask_count if mask_count > 0 else 0
                )

        if "Classification" in arrays.dtype.names:
            for class_id, class_name in [(2, "ground"), (6, "building")]:
                if class_counts.get(class_id, 0) > 3:
                    class_z = z[classification == class_id]
                    stats[f"{class_name}_height_variance"] = float(np.var(class_z))
                    p25, p75 = np.percentile(class_z, [25, 75])
                    stats[f"{class_name}_height_iqr"] = float(p75 - p25)

        return stats

    except Exception as e:
        logger.error(f"Error computing stats for {las_path}: {e}")
        return create_empty_stats()


def create_empty_stats() -> dict[str, Any]:
    """Create empty stats dictionary with all required fields."""
    stats = {
        "point_count": 0,
        "footprint_area": 0.0,
        "point_density": 0.0,
        "building_height": 0.0,
        "convex_hull_area": 0.0,
        "coverage_ratio": 0.0,
        "spatial_coverage": 0.0,
        "z_min": 0.0,
        "z_max": 0.0,
        "z_range": 0.0,
        "z_mean": 0.0,
        "z_std": 0.0,
        "z_p10": 0.0,
        "z_p25": 0.0,
        "z_p50": 0.0,
        "z_p75": 0.0,
        "z_p90": 0.0,
        "pts_0_3m": 0,
        "pts_3_6m": 0,
        "pts_6_9m": 0,
        "pts_9_12m": 0,
        "pts_above_12m": 0,
        "returns_single": 0,
        "returns_multiple": 0,
        "intensity_mean": 0.0,
        "intensity_std": 0.0,
        "ground_point_count": 0,
        "ground_z_mean": 0.0,
        "ground_z_std": 0.0,
        "noise_point_count": 0,
        "roof_z_variance": 0.0,
        "vegetation_proximity_count": 0,
        "verticality_score": 0.0,
        "planarity_score": 0.0,
        "building_height_peaks": 0,
        "building_height_regularity": 0.0,
        "facade_alignment_score": 0.0,
    }

    height_bands = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 20), (20, 30), (30, 50)]
    for h_min, h_max in height_bands:
        stats[f"building_density_{h_min}_{h_max}m"] = 0.0

    for h_min, h_max in [(0, 10), (10, 20), (20, 30), (30, 50)]:
        stats[f"multi_return_ratio_{h_min}_{h_max}m"] = 0.0

    for class_name in ["ground", "building"]:
        stats[f"{class_name}_height_variance"] = 0.0
        stats[f"{class_name}_height_iqr"] = 0.0

    for i in range(8):
        class_names = {
            0: "never",
            1: "unassigned",
            2: "ground",
            3: "low_veg",
            4: "med_veg",
            5: "high_veg",
            6: "building",
            7: "noise",
        }
        class_name = class_names.get(i, f"class_{i}")
        stats[f"class_{i}_{class_name}"] = 0

    return stats


def process_building(
    row: Any, footprint_area: float, footprint_geom: Any
) -> tuple[int, Stage09aLidarStatsRecord | None, str]:
    """Process statistics for one clipped LAS file."""
    try:
        las_path = CONFIG.output_root / row.output_path
        if not las_path.exists():
            logger.warning(f"LAS file not found: {las_path}")
            return row.id, None, "fail"

        stats = compute_lidar_stats(las_path, footprint_area, footprint_geom)

        if stats["point_count"] == 0:
            record = Stage09aLidarStatsRecord(
                id=row.id,
                building_id=row.building_id,
                region_name=row.region_name,
                gnaf_id=row.gnaf_id if row.gnaf_id else "",
                las_path=row.output_path,
                **stats,
            )
            return row.id, record, "empty"

        record = Stage09aLidarStatsRecord(
            id=row.id,
            building_id=row.building_id,
            region_name=row.region_name,
            gnaf_id=row.gnaf_id if row.gnaf_id else "",
            las_path=row.output_path,
            **stats,
        )

        return row.id, record, "suc"

    except Exception as e:
        logger.error(f"Error processing {row.building_id}: {e}")
        return row.id, None, "fail"


def run_region(region: str, sample_size: int | None = None, workers: int = -1) -> None:
    """Process statistics for all clipped LAS files in a region."""
    logger.info(f"══ Processing {region} ══")

    processed_ids = get_processed_stage_ids(STAGE_NAME, region)

    clips = read_table(
        "stage01_clips",
        columns=["id", "building_id", "gnaf_id", "region_name", "clip_path"],
        region=region,
    )

    if len(clips) == 0:
        logger.warning(f"{region}: no successful clips found; skipping")
        return

    clips["output_path"] = clips["clip_path"]

    fps = read_table(
        "buildings",
        columns=["id", "building_id", "footprint_geom"],
        region=region,
        as_geo=True,
        geom_col="footprint_geom",
        crs="EPSG:7844",
    )
    fps.rename(columns={"footprint_geom": "geometry"}, inplace=True)
    fps.set_geometry("geometry", inplace=True)

    target_crs = CONFIG.get_region(region).crs_projected
    fps = fps.to_crs(target_crs)

    pending = clips[~clips["id"].isin(processed_ids)]

    if len(pending) == 0:
        logger.info(f"{region}: All buildings already processed")
        return

    if sample_size:
        pending = pending.head(sample_size)

    total = len(pending)
    logger.info(f"Processing {total} clips")

    if workers <= 0:
        workers = CONFIG.constants.default_workers
        if workers <= 0:
            workers = mp.cpu_count()
    logger.info(f"Using {workers} worker(s)")

    counters = {"success": 0, "fail": 0}
    results_lock = threading.Lock()

    tasks = []
    for _, row in pending.iterrows():
        fp_match = fps[fps["id"] == row.id]
        if len(fp_match) == 0:
            logger.warning(f"No footprint found for id {row.id}")
            footprint_geom = None
            footprint_area = 0.0
        else:
            footprint_geom = fp_match.iloc[0].geometry
            footprint_area = footprint_geom.area

        tasks.append((row, footprint_area, footprint_geom))

    with (
        lidar_stats_progress(f"Processing {region} LiDAR stats", total) as prog,
        BatchWriter(STAGE_NAME, batch_size=100, progress_tracker=prog) as writer,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        future_to_task = {pool.submit(process_building, task[0], task[1], task[2]): i for i, task in enumerate(tasks)}

        for fut in as_completed(future_to_task):
            try:
                _, record, status = fut.result()

                if record:
                    with results_lock:
                        writer.add(record)

                    if status == "empty":
                        prog.update("empty", 1)
                    else:
                        counters["success"] += 1
                        prog.update("suc", 1)
                else:
                    counters["fail"] += 1
                    prog.update("fail", 1)

            except Exception as e:
                logger.error(f"Error: {e}")
                counters["fail"] += 1
                prog.update("fail", 1)

    logger.info(f"{region}: Processed {counters['success']} clips successfully, {counters['fail']} failed")


def run_stage(
    regions: list[str] | None = None,
    workers: int | None = None,
    sample: int | None = None,
) -> None:
    """Run stage 09a for specified regions.

    Args:
        regions: List of regions to process (default: all from config)
        workers: Number of parallel workers
        sample: Process only first N buildings per region
    """
    initialize_all_stage_tables()

    if not regions:
        regions = CONFIG.region_names

    for region in regions:
        try:
            run_region(region, sample, workers)
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            break
        except Exception as e:
            logger.error(f"Failed to process {region}: {e}")
            continue


def main():
    """Main entry point for the stage."""
    import argparse

    parser = argparse.ArgumentParser(description=STAGE_DESCRIPTION)
    parser.add_argument("--region", type=str, help="Specific region to process")
    parser.add_argument("--sample", type=int, help="Process only N samples per region")
    parser.add_argument("--workers", type=int, default=-1, help="Number of workers")

    args = parser.parse_args()

    try:
        if args.region:
            if args.region not in CONFIG.regions:
                logger.error(f"Unknown region: {args.region}")
                logger.info(f"Available regions: {', '.join(CONFIG.regions)}")
                sys.exit(1)
            run_stage([args.region], args.workers, args.sample)
        else:
            run_stage(None, args.workers, args.sample)

        logger.info(f"{STAGE_NAME} complete")
    except Exception as e:
        logger.error(f"{STAGE_NAME} failed: {e}")
        raise


if __name__ == "__main__":
    main()
