"""Stage-08: YOLOv8 object detection with FFH-aligned best view selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from loguru import logger
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlalchemy import (
    Index,
    MetaData,
    Table,
    create_engine,
    inspect,
    select,
    update,
)
from ultralytics import YOLO

DB = os.getenv("DB_CONNECTION_STRING")
if not DB:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")

MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
if not MODEL_PATH:
    raise ValueError("YOLO_MODEL_PATH environment variable is required")

cfg_path = Path(__file__).resolve().parents[3] / "config" / "common.yaml"
common_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
OUTPUT_ROOT = Path(common_cfg.get("output_root", "output"))
REGIONS = common_cfg.get("regions", [])

engine = create_engine(DB, future=True, pool_pre_ping=True)
meta = MetaData()

CLASS_NAMES = {
    0: "Front Door",
    1: "Foundation",
    2: "Garage Door",
    3: "Stairs",
    4: "Window",
}

FRONTDOOR_STANDARDS = {
    "width_m": 0.82,
    "height_m": 2.04,
    "area_m2": 1.67,
    "ratio": 0.40,
}

WEIGHTS = {"area_m2": 1, "ratio": 1, "confidence": 1, "x_location": 1, "y_location": 1}

FFH_PRIORITY_CLASSES = ["Front Door", "Foundation", "Garage Door", "Stairs"]


def get_device() -> str:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} CUDA device(s)")
        return "0" if num_gpus == 1 else ",".join(str(i) for i in range(num_gpus))
    logger.info("No CUDA devices found, using CPU")
    return "cpu"


def load_yolo_model(model_path: str) -> YOLO:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = YOLO(model_path)
    model.predictor = None
    logger.info(f"Model loaded from: {model_path}")
    return model


def fetch_panorama_views(region_name: str) -> List[Dict[str, Any]]:
    views_table = Table("panorama_candidate_views", meta, autoload_with=engine)

    stmt = select(
        views_table.c.building_id,
        views_table.c.region,
        views_table.c.edge_idx,
        views_table.c.pano_id,
        views_table.c.view_type,
        views_table.c.angle,
        views_table.c.distance,
        views_table.c.is_chosen,
    ).where(views_table.c.region == region_name)

    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(stmt)]


def crop_image_to_building(image_path: Path) -> Optional[Image.Image]:
    """TODO: Implement tight cropping using building footprint coordinates.
    This requires:
    1. Transform building footprint from geographic coordinates to panorama pixel coordinates
    2. Use panorama metadata (position, orientation, field of view) for projection
    3. Create mask to exclude adjacent buildings from view
    4. Apply mask to crop image to focal building only
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        logger.warning(f"Failed to load {image_path}: {e}")
        return None


def calculate_feature_score(
    detection: Dict[str, Any], image_width: int, image_height: int
) -> float:
    class_name = detection["class_name"]
    confidence = detection["confidence"]

    width = detection["bbox_x2"] - detection["bbox_x1"]
    height = detection["bbox_y2"] - detection["bbox_y1"]
    ratio = width / height if height > 0 else 0

    center_x = (detection["bbox_x1"] + detection["bbox_x2"]) / 2
    center_y = (detection["bbox_y1"] + detection["bbox_y2"]) / 2

    x_norm = center_x / image_width
    y_norm = center_y / image_height

    if class_name == "Front Door":
        x_score = abs(x_norm - 0.5)
        area_score = abs(ratio - FRONTDOOR_STANDARDS["ratio"])
        ratio_score = abs(ratio - FRONTDOOR_STANDARDS["ratio"])

        weighted_score = (
            WEIGHTS["x_location"] * x_score
            + WEIGHTS["area_m2"] * area_score
            + WEIGHTS["ratio"] * ratio_score
            + WEIGHTS["confidence"] * (1.0 - confidence)
        )

    else:
        y_score = 1.0 - y_norm
        area_score = abs(ratio - FRONTDOOR_STANDARDS["ratio"])
        ratio_score = abs(ratio - FRONTDOOR_STANDARDS["ratio"])

        weighted_score = (
            WEIGHTS["y_location"] * y_score
            + WEIGHTS["area_m2"] * area_score
            + WEIGHTS["ratio"] * ratio_score
            + WEIGHTS["confidence"] * (1.0 - confidence)
        )

    return float(max(0, 10.0 - weighted_score))


def calculate_view_score(
    detections: List[Dict[str, Any]], image_width: int, image_height: int
) -> float:
    if not detections:
        return 0.0

    priority_scores = []
    other_scores = []

    for detection in detections:
        score = calculate_feature_score(detection, image_width, image_height)

        if detection["class_name"] in FFH_PRIORITY_CLASSES:
            priority_scores.append(score)
        else:
            other_scores.append(score)

    total_score = sum(priority_scores) * 2.0 + sum(other_scores)

    unique_classes = len(set(det["class_name"] for det in detections))
    diversity_bonus = 1.0 + (0.1 * unique_classes)

    return total_score * diversity_bonus


def process_detections(
    detections: Any,
    image_path: Path,
    building_id: int,
    pano_id: str,
    view_info: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], float]:
    results = []
    image = Image.open(image_path)
    image_width, image_height = image.size

    for box in detections.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        detection = {
            "building_id": building_id,
            "pano_id": pano_id,
            "image_path": str(image_path),
            "class_name": CLASS_NAMES.get(cls_id, f"Unknown-{cls_id}"),
            "confidence": conf,
            "bbox_x1": float(x1),
            "bbox_y1": float(y1),
            "bbox_x2": float(x2),
            "bbox_y2": float(y2),
            "view_metadata": json.dumps(
                {
                    "edge_idx": view_info["edge_idx"],
                    "view_type": view_info["view_type"],
                    "angle": view_info["angle"],
                    "distance": view_info["distance"],
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "area": float((x2 - x1) * (y2 - y1)),
                    "center_x": float(x1 + (x2 - x1) / 2),
                    "center_y": float(y1 + (y2 - y1) / 2),
                }
            ),
            "model_version": "yolov8",
        }

        results.append(detection)

    view_score = calculate_view_score(results, image_width, image_height)
    return results, view_score


def select_best_views_per_building(
    view_scores: Dict[int, List[Tuple[str, float, Dict[str, Any]]]],
) -> Dict[int, str]:
    best_views = {}

    for building_id, scores in view_scores.items():
        if scores:
            scores.sort(key=lambda x: (-x[1], x[2]["distance"]))
            best_pano_id = scores[0][0]
            best_views[building_id] = best_pano_id
            logger.debug(
                f"Building {building_id}: best view {best_pano_id} (FFH score: {scores[0][1]:.2f})"
            )

    return best_views


def clear_view_flags(views_table: Table, region_name: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            update(views_table)
            .where(views_table.c.region == region_name)
            .values(is_chosen=False)
        )


def set_best_view_flags(
    views_table: Table, best_views: Dict[int, str], region_name: str
) -> None:
    with engine.begin() as conn:
        for building_id, pano_id in best_views.items():
            conn.execute(
                update(views_table)
                .where(
                    views_table.c.region == region_name,
                    views_table.c.building_id == building_id,
                    views_table.c.pano_id == pano_id,
                )
                .values(is_chosen=True)
            )


def update_best_view_flags(best_views: Dict[int, str], region_name: str) -> None:
    views_table = Table("panorama_candidate_views", meta, autoload_with=engine)
    clear_view_flags(views_table, region_name)
    set_best_view_flags(views_table, best_views, region_name)


def get_region_id(region_name: str) -> int:
    regions_table = Table("regions", meta, autoload_with=engine)
    with engine.connect() as conn:
        return int(
            conn.execute(
                select(regions_table.c.id).where(regions_table.c.name == region_name)
            ).scalar_one()
        )


def clear_existing_detections(table_name: str, region_id: int) -> None:
    if not inspect(engine).has_table(table_name):
        return

    with engine.begin() as conn:
        detections_table = Table(table_name, MetaData(), autoload_with=engine)
        conn.execute(
            detections_table.delete().where(detections_table.c.region_id == region_id)
        )


def create_detection_indexes(table_name: str) -> None:
    detections_table = Table(table_name, MetaData(), autoload_with=engine)
    Index(f"{table_name}_building_id_idx", detections_table.c.building_id).create(
        bind=engine, checkfirst=True
    )
    Index(f"{table_name}_region_id_idx", detections_table.c.region_id).create(
        bind=engine, checkfirst=True
    )


def write_detection_results(results: List[Dict[str, Any]], region_name: str) -> None:
    if not results:
        return

    table_name = "object_detections"
    table_exists = inspect(engine).has_table(table_name)
    region_id = get_region_id(region_name)

    clear_existing_detections(table_name, region_id)

    df = pd.DataFrame(results).assign(
        region_id=region_id, created_at=pd.Timestamp.now(tz="UTC")
    )

    df.to_sql(
        table_name,
        con=engine,
        if_exists="replace" if not table_exists else "append",
        index=False,
    )

    create_detection_indexes(table_name)


def execute_detection_processing(
    views: List[Dict[str, Any]],
    model: YOLO,
    clips_dir: Path,
    device: str,
    conf_threshold: float,
    region_name: str,
) -> Tuple[
    List[Dict[str, Any]], Dict[int, List[Tuple[str, float, Dict[str, Any]]]], int, int
]:
    all_results: List[Dict[str, Any]] = []
    view_scores: Dict[int, List[Tuple[str, float, Dict[str, Any]]]] = {}
    processed_count = detection_count = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(
            f"Detecting {region_name} (P: 0, D: 0)", total=len(views)
        )

        for view in views:
            building_id = view["building_id"]
            pano_id = view["pano_id"]

            image_files = list(clips_dir.glob(f"*{pano_id}*"))
            if not image_files:
                logger.debug(f"No clipped image found for panorama {pano_id}")
                progress.update(task_id, advance=1)
                continue

            image_path = image_files[0]

            cropped_image = crop_image_to_building(image_path)

            if cropped_image is None:
                progress.update(task_id, advance=1)
                continue

            detections = model.predict(
                source=str(image_path),
                conf=conf_threshold,
                device=device,
                verbose=False,
            )[0]

            if len(detections.boxes) > 0:
                view_info = {
                    "edge_idx": view["edge_idx"],
                    "view_type": view["view_type"],
                    "angle": view["angle"],
                    "distance": view["distance"],
                }

                detection_results, view_score = process_detections(
                    detections, image_path, building_id, pano_id, view_info
                )

                all_results.extend(detection_results)

                if building_id not in view_scores:
                    view_scores[building_id] = []
                view_scores[building_id].append((pano_id, view_score, view_info))

                detection_count += len(detection_results)

            processed_count += 1

            progress.update(task_id, advance=1)
            progress.update(
                task_id,
                description=f"Detecting {region_name} (P: {processed_count}, D: {detection_count})",
            )

    return all_results, view_scores, processed_count, detection_count


def process_region(region_name: str, conf_threshold: float) -> None:
    logger.info(f"Processing object detection for region: {region_name}")

    views = fetch_panorama_views(region_name)
    if not views:
        logger.warning(f"No panorama views found for {region_name}")
        return

    device = get_device()
    assert MODEL_PATH is not None
    model = load_yolo_model(MODEL_PATH)
    clips_dir = OUTPUT_ROOT / region_name / "clips"

    all_results, view_scores, processed_count, detection_count = (
        execute_detection_processing(
            views, model, clips_dir, device, conf_threshold, region_name
        )
    )

    best_views = select_best_views_per_building(view_scores)

    write_detection_results(all_results, region_name)
    update_best_view_flags(best_views, region_name)

    logger.success(
        f"{region_name}: {processed_count} views processed, {detection_count} detections, "
        f"{len(best_views)} best views selected"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 detection with FFH-aligned best view selection"
    )
    parser.add_argument("--region", choices=REGIONS, help="Process single region")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    regions = [args.region] if args.region else REGIONS
    for region in regions:
        process_region(region, args.conf)


if __name__ == "__main__":
    main()
