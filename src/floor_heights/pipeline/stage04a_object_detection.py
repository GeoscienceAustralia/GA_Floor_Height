#!/usr/bin/env python
"""Stage-04a: YOLOv8 object detection on clipped panoramas.

Detects building features (doors, windows, etc.) in clipped panorama images
and stores results in database tables for further analysis.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if mp.get_start_method(allow_none=True) != "spawn":
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn", force=True)

import pandas as pd
import torch
from loguru import logger
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from floor_heights.config import CONFIG, REGIONS
from floor_heights.db.schemas import (
    BatchWriter,
    Stage04aClipRecord,
    Stage04aDetectionRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import read_table, validate_file_exists_and_valid
from floor_heights.utils.progress import detection_progress
from floor_heights.utils.visualization import (
    OBJECT_CLASS_NAMES,
    create_object_detection_visualization,
    get_visualization_path,
)

MODEL_PATH = CONFIG.yolo_model_path

if not validate_file_exists_and_valid(MODEL_PATH, file_type="model", min_size_bytes=1000000):
    raise FileNotFoundError(f"YOLO model weights not found or corrupted at: {MODEL_PATH}")

CONF_THRESHOLD = CONFIG.object_detection.confidence_threshold
BATCH_SIZE = CONFIG.object_detection.batch_size


def check_gpu_availability() -> tuple[bool, int, list[str]]:
    """Check if GPU is available and return GPU count and device info."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_info.append(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

        logger.info(f"GPU available: {gpu_count} device(s) detected")
        for info in gpu_info:
            logger.info(f"  {info}")
        return True, gpu_count, gpu_info
    else:
        logger.info("No GPU available, will use CPU")
        return False, 0, []


def get_clipped_views(region: str) -> pd.DataFrame:
    """Get successfully clipped views from stage03."""
    df = read_table("stage03_clips", region=region)

    if df.empty:
        return pd.DataFrame()

    return df


def get_processed_clips(region: str) -> set[tuple]:
    """Get set of already processed clips as (id, pano_id, edge_idx, view_type) tuples."""
    try:
        df = read_table("stage04a_clips", region=region)
        if df.empty:
            return set()

        return set(
            zip(
                df["id"],
                df["pano_id"],
                df["edge_idx"],
                df["view_type"],
                strict=False,
            )
        )
    except Exception:
        return set()


def _get_clip_path(
    region: str, row_id: int, building_id: str, gnaf_id: str, pano_id: str, edge_idx: int, view_type: str
) -> Path:
    """Get path to clipped panorama image (following stage03 pattern, returns relative path)."""
    base_pid = pano_id[:-4] if pano_id.lower().endswith(".jpg") else pano_id
    filename = f"{base_pid}_edge{edge_idx}_{view_type}.jpg"
    gnaf_id = gnaf_id if gnaf_id and pd.notna(gnaf_id) else "NO_GNAF"
    return Path(region.capitalize()) / "clips" / f"{row_id}_{building_id}_{gnaf_id}" / filename


def _run_detection(model: YOLO, image_path: Path, conf_threshold: float) -> list[dict[str, Any]]:
    """Run YOLO detection on a single image."""
    detections = []

    try:
        results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            verbose=False,
            batch=1,
            device=model.device,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        image = Image.open(image_path)
        img_width, img_height = image.size

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            detection = {
                "class_id": cls_id,
                "class_name": OBJECT_CLASS_NAMES.get(cls_id, f"Unknown-{cls_id}"),
                "confidence": conf,
                "bbox_x1": float(x1),
                "bbox_y1": float(y1),
                "bbox_x2": float(x2),
                "bbox_y2": float(y2),
                "bbox_width": float(x2 - x1),
                "bbox_height": float(y2 - y1),
                "bbox_area": float((x2 - x1) * (y2 - y1)),
                "bbox_center_x": float(x1 + (x2 - x1) / 2),
                "bbox_center_y": float(y1 + (y2 - y1) / 2),
                "image_width": img_width,
                "image_height": img_height,
            }
            detections.append(detection)

    except Exception as e:
        logger.error(f"Detection failed for {image_path}: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

    return detections


def _batch_detection_worker(args_tuple):
    """Worker function that processes a batch of views efficiently."""
    batch_data, model_path, device, conf_threshold, create_visualizations = args_tuple

    import sys
    from pathlib import Path

    import torch
    from loguru import logger
    from ultralytics import YOLO

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    results = []

    try:
        model = YOLO(str(model_path))
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
            model.to(device)

        for region, row_data in batch_data:
            try:
                result = _process_single_view_simple(model, region, row_data, conf_threshold, create_visualizations)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item in batch: {e}")
                results.append(("fail", row_data.get("id"), row_data, None, None))

        return results

    except Exception as e:
        logger.error(f"Batch worker error on {device}: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return [("fail", row_data[1].get("id"), row_data[1], None, None) for row_data in batch_data]


def _process_single_view_simple(
    model, region, row_data, conf_threshold, create_visualizations
) -> tuple[str, str, dict[str, Any], list[dict[str, Any]] | None, str | None]:
    """Process a single clipped view for object detection with pre-loaded model.

    Returns:
        Tuple of (status, row_id, row_data, detections, clip_path)
    """

    class RowLike:
        def __init__(self, data):
            self._data = data

        def __getattr__(self, name):
            return self._data.get(name)

    row = RowLike(row_data)

    row_id = str(row.id)
    gnaf_id = row.gnaf_id if pd.notna(row.gnaf_id) else "NO_GNAF"

    if hasattr(row, "clip_path") and row.clip_path:
        clip_path = Path(row.clip_path)
    else:
        clip_path = _get_clip_path(
            region, row_id, row.building_id, row.gnaf_id, row.pano_id, row.edge_idx, row.view_type
        )

    abs_clip_path = CONFIG.output_root / clip_path

    if not validate_file_exists_and_valid(abs_clip_path, file_type="image", min_size_bytes=1000):
        return ("missing", row_id, row_data, None, None)

    try:
        detections = _run_detection(model, abs_clip_path, conf_threshold)

        if create_visualizations and detections:
            viz_path = get_visualization_path(
                CONFIG.output_root,
                region,
                row.building_id,
                gnaf_id,
                "stage04a_detections",
                f"{row.pano_id[:-4] if row.pano_id.lower().endswith('.jpg') else row.pano_id}_edge{row.edge_idx}_{row.view_type}_detections.jpg",
            )

            viz_img = create_object_detection_visualization(
                image_path=abs_clip_path,
                detections=detections,
            )
            viz_img.save(viz_path, quality=CONFIG.object_detection.visualization_quality, optimize=True)

            if not validate_file_exists_and_valid(viz_path, file_type="image", min_size_bytes=1000):
                logger.error(f"Saved visualization appears corrupted: {viz_path}")
                viz_path.unlink(missing_ok=True)

        return ("success", row_id, row_data, detections, str(clip_path))

    except Exception as e:
        logger.error(f"Detection failed for {clip_path}: {e}")
        return ("fail", row_id, row_data, None, None)


def process_region(
    region: str, conf_threshold: float = CONF_THRESHOLD, create_visualizations: bool = False, workers: int = -1
) -> None:
    """Process object detection for single region."""
    logger.info(f"Processing {region}")

    has_gpu, gpu_count, gpu_info = check_gpu_availability()

    df = get_clipped_views(region)
    if df.empty:
        logger.warning(f"{region}: no clipped views found")
        return

    logger.info(f"Found {len(df)} clipped views to process")

    processed_clips = get_processed_clips(region)
    logger.info(f"Found {len(processed_clips)} already processed clips")

    df["clip_key"] = list(zip(df["id"], df["pano_id"], df["edge_idx"], df["view_type"], strict=False))
    mask = ~df["clip_key"].isin(processed_clips)
    clips_to_process = df[mask].drop(columns=["clip_key"])

    if clips_to_process.empty:
        logger.info(f"{region}: All clips already processed, calculating statistics...")

        try:
            detections_df = read_table("stage04a_detections", region=region)
            clips_df = read_table("stage04a_clips", region=region)

            if not detections_df.empty and not clips_df.empty:
                successful_clips = len(clips_df)
                class_counts = detections_df["class_id"].value_counts().to_dict()

                avg_door = class_counts.get(0, 0) / successful_clips
                avg_found = class_counts.get(1, 0) / successful_clips
                avg_garage = class_counts.get(2, 0) / successful_clips
                avg_stairs = class_counts.get(3, 0) / successful_clips
                avg_window = class_counts.get(4, 0) / successful_clips

                logger.success(
                    f"{region}: All clips already processed | "
                    f"Averages per clip - Door:{avg_door:.3f} Found:{avg_found:.3f} "
                    f"Garage:{avg_garage:.3f} Stairs:{avg_stairs:.3f} Window:{avg_window:.3f}"
                )
            else:
                logger.success(f"{region}: All clips already processed")
        except Exception:
            logger.success(f"{region}: All clips already processed")

        return

    df_to_process = clips_to_process
    logger.info(f"{region}: {len(df_to_process)} clips to process")

    if workers <= 0:
        if has_gpu:
            workers = gpu_count * CONFIG.object_detection.workers_per_gpu
        else:
            workers = CONFIG.constants.default_workers
            if workers <= 0:
                workers = mp.cpu_count()

    workers = min(workers, len(df_to_process))

    if has_gpu:
        logger.info(f"Using {workers} worker(s) across {gpu_count} GPU(s)")
    else:
        logger.info(f"Using {workers} worker(s) with CPU")

    clips_with_detections = 0
    total_detections = 0

    progress_lock = threading.Lock()
    processed_clips = 0

    class_totals = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    with (
        detection_progress(f"Detecting {region}", len(df_to_process)) as prog,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        items_per_batch = CONFIG.object_detection.items_per_batch
        batches = []
        current_batch = []

        for row in df_to_process.itertuples(index=False):
            current_batch.append((region, row._asdict()))
            if len(current_batch) >= items_per_batch:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        logger.info(f"Created {len(batches)} batches of ~{items_per_batch} items each")
        logger.info(f"Each GPU will process ~{len(batches) // gpu_count} batches")

        futures = {}
        for i, batch in enumerate(batches):
            device = f"cuda:{i % gpu_count}" if has_gpu else "cpu"

            future = pool.submit(
                _batch_detection_worker, (batch, MODEL_PATH, device, conf_threshold, create_visualizations)
            )
            futures[future] = i

        with (
            BatchWriter("stage04a_clips", batch_size=5000, progress_tracker=prog) as clip_writer,
            BatchWriter("stage04a_detections", batch_size=50000, progress_tracker=prog) as detection_writer,
        ):
            for future in as_completed(futures):
                try:
                    batch_results = future.result()

                    for status, row_id, row_data, detections, clip_path in batch_results:
                        if status == "success" and row_data and clip_path:
                            clip_writer.add(
                                Stage04aClipRecord(
                                    id=str(row_id),
                                    building_id=row_data.get("building_id"),
                                    region_name=region,
                                    gnaf_id=row_data.get("gnaf_id", ""),
                                    pano_id=row_data.get("pano_id"),
                                    edge_idx=row_data.get("edge_idx"),
                                    view_type=row_data.get("view_type"),
                                    clip_path=clip_path,
                                    detection_count=len(detections) if detections else 0,
                                )
                            )

                            if detections:
                                clips_with_detections += 1
                                total_detections += len(detections)

                                for detection in detections:
                                    class_id = detection["class_id"]
                                    if class_id in class_totals:
                                        class_totals[class_id] += 1

                                for _, detection in enumerate(detections):
                                    if detection and "class_id" in detection and "confidence" in detection:
                                        detection_writer.add(
                                            Stage04aDetectionRecord(
                                                id=row_id,
                                                building_id=row_data.get("building_id"),
                                                region_name=region,
                                                gnaf_id=row_data.get("gnaf_id", ""),
                                                pano_id=row_data.get("pano_id"),
                                                edge_idx=row_data.get("edge_idx"),
                                                view_type=row_data.get("view_type"),
                                                clip_path=clip_path,
                                                **detection,
                                            )
                                        )

                            with progress_lock:
                                processed_clips += 1

                                if processed_clips > 0:
                                    prog.fields["avg_door"] = round(class_totals[0] / processed_clips, 2)
                                    prog.fields["avg_found"] = round(class_totals[1] / processed_clips, 2)
                                    prog.fields["avg_garage"] = round(class_totals[2] / processed_clips, 2)
                                    prog.fields["avg_stairs"] = round(class_totals[3] / processed_clips, 2)
                                    prog.fields["avg_window"] = round(class_totals[4] / processed_clips, 2)
                                    if prog.task_id is not None:
                                        prog.progress.update(prog.task_id, **prog.fields)

                            prog.update("suc", 1)
                        elif status == "missing":
                            prog.update("fail", 1)
                        elif status == "skip":
                            prog.update("skp", 1)
                        else:
                            prog.update("fail", 1)

                except Exception as e:
                    logger.error(f"Worker task failed: {e}")
                    prog.update("fail", 1)

    logger.success(f"{region}: {prog.get_summary()}")
    logger.info(
        f"{region}: Found {total_detections} total detections in {clips_with_detections}/{len(df_to_process)} clips ({clips_with_detections / len(df_to_process) * 100:.1f}%)"
    )


def run_stage(
    region: str | None = None, workers: int = -1, conf_threshold: float = CONF_THRESHOLD, visualize: bool = False
) -> None:
    """Run stage04a with the given parameters.

    Args:
        region: Single region to process (None for all)
        workers: Number of workers (-1 for default)
        conf_threshold: Object detection confidence threshold
        visualize: Whether to create visualization images
    """
    initialize_all_stage_tables()

    try:
        if region:
            process_region(region, conf_threshold, visualize, workers)
        else:
            logger.info(f"Processing {len(REGIONS)} regions: {', '.join(REGIONS)}")
            for r in REGIONS:
                process_region(r, conf_threshold, visualize, workers)
        logger.info("Stage-04a complete")
    except Exception as e:
        logger.error(f"Stage-04a failed: {e}")
        raise


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    run_stage()
