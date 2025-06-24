#!/usr/bin/env python
"""Stage-04b: Best view selection from object detections with SigLIP occlusion scoring.

Selects the best panorama view for each building based on:
- Object detection scores (doors, windows, stairs, foundation)
- SigLIP-based occlusion detection for better spatial localisation
- View diversity and angle preferences
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from floor_heights.config import CONFIG
from floor_heights.db.schemas import (
    BatchWriter,
    Stage04bBestViewRecord,
    initialize_all_stage_tables,
)
from floor_heights.utils.fh_io import (
    get_processed_stage_ids,
    read_table,
    validate_file_exists_and_valid,
)
from floor_heights.utils.progress import best_view_progress

_siglip_model = None
_siglip_processor = None
_siglip_tokenizer = None

CLEAR_PROMPTS = [
    "a clear unobstructed view of a building facade with visible ground level",
    "house front with clear view from ground to roofline",
    "building entrance and foundation clearly visible without obstructions",
]

BLOCKED_PROMPTS = [
    "building entrance blocked by trees or vegetation",
    "house obscured by parked vehicles",
    "facade heavily covered by shadows or foliage",
]


GROUND_INTERFACE_CLEAR_PROMPTS = [
    "clear view of where the building foundation meets the ground",
    "visible transition from ground level to building entrance",
    "unobstructed view of the building base and surrounding ground",
    "clear sight of the ground surface in front of the building entrance",
]

GROUND_INTERFACE_BLOCKED_PROMPTS = [
    "building base hidden by tall grass or vegetation",
    "ground level obscured by parked cars or bins",
    "foundation covered by bushes or garden beds",
    "shadows completely hiding the ground-building junction",
]


SINGLE_BUILDING_CLEAR_PROMPTS = [
    "single residential building clearly in focus",
    "one house prominently featured in the center",
    "clear view of a single building facade",
    "focused view of one residential property",
]

MULTIPLE_OR_UNCLEAR_PROMPTS = [
    "multiple buildings visible with unclear focus",
    "ambiguous view between neighboring houses",
    "view showing parts of multiple properties",
    "unclear which building is the target",
]

NON_RESIDENTIAL_PROMPTS = [
    "shed or outbuilding structure",
    "garage or carport without house visible",
    "commercial or industrial building",
    "non-residential structure or temporary building",
]


FEATURE_VERIFICATION_PROMPTS = {
    "Front Door": ["a residential front door with handle and frame", "entrance door to a house clearly visible"],
    "Foundation": ["concrete or brick building foundation", "base of building where it meets the ground"],
    "Stairs": ["outdoor stairs or steps leading to entrance", "concrete or wooden steps with visible risers"],
    "Garage Door": ["garage door or roller door clearly visible", "vehicle entrance to building"],
}

FFH_PRIORITY_CLASSES = ["Front Door", "Foundation", "Garage Door", "Stairs"]


def load_siglip_model() -> tuple[torch.nn.Module, Any, Any]:
    """Load SigLIP2 model with dual GPU support."""
    global _siglip_model, _siglip_processor, _siglip_tokenizer

    if _siglip_model is None or _siglip_processor is None or _siglip_tokenizer is None:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s) available")

            if gpu_count > 1:
                device = "cuda:0"
                logger.info(f"Loading SigLIP2 model with DataParallel on {gpu_count} GPUs...")
            else:
                device = "cuda:0"
                logger.info("Loading SigLIP2 model on single GPU...")
        else:
            device = "cpu"
            logger.info("Loading SigLIP2 model on CPU...")

        model_name = CONFIG.siglip.model_name

        max_retries = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading model components (attempt {attempt + 1}/{max_retries})")
                _siglip_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, use_auth_token=False)
                _siglip_tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False)
                _siglip_model = AutoModel.from_pretrained(model_name, use_auth_token=False)
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"Rate limited, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise e
        else:
            raise RuntimeError(f"Failed to load model after {max_retries} attempts")

        if device != "cpu":
            _siglip_model = _siglip_model.to(device)
            if torch.cuda.device_count() > 1:
                _siglip_model = torch.nn.DataParallel(_siglip_model)

        _siglip_model.eval()

    return _siglip_model, _siglip_processor, _siglip_tokenizer


def unload_siglip_model() -> None:
    """Explicitly unload SigLIP model and free memory."""
    global _siglip_model, _siglip_processor, _siglip_tokenizer

    if _siglip_model is not None:
        logger.info("Unloading SigLIP2 model...")
        del _siglip_model
        _siglip_model = None

    if _siglip_processor is not None:
        del _siglip_processor
        _siglip_processor = None

    if _siglip_tokenizer is not None:
        del _siglip_tokenizer
        _siglip_tokenizer = None

    cleanup_gpu_memory()


def cleanup_gpu_memory() -> None:
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_siglip_embeddings(texts: list[str], model: torch.nn.Module, tokenizer: Any, device: str) -> torch.Tensor:
    """Get normalized text embeddings for a list of texts using SigLIP2."""

    inputs = tokenizer(
        text=texts,
        return_tensors="pt",
        padding=CONFIG.siglip.tokenizer_padding,
        max_length=CONFIG.siglip.tokenizer_max_length,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if hasattr(model, "module"):
            text_features = model.module.get_text_features(**inputs)
        else:
            text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def compute_siglip_score(image_path: Path) -> float:
    """Compute basic SigLIP2-based occlusion score for an image."""
    try:
        model, processor, tokenizer = load_siglip_model()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(model, "module"):
                image_features = model.module.get_image_features(**inputs)
            else:
                image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        clear_embeds = get_siglip_embeddings(CLEAR_PROMPTS, model, tokenizer, device)
        blocked_embeds = get_siglip_embeddings(BLOCKED_PROMPTS, model, tokenizer, device)
        clear_sims = (image_features @ clear_embeds.T).cpu().numpy().flatten()
        blocked_sims = (image_features @ blocked_embeds.T).cpu().numpy().flatten()

        clear_mean = np.mean(clear_sims)
        blocked_mean = np.mean(blocked_sims)
        epsilon = 1e-6
        score = float((clear_mean - blocked_mean) / (clear_mean + blocked_mean + epsilon))

        if not hasattr(compute_siglip_score, "call_count"):
            compute_siglip_score.call_count = 0
        compute_siglip_score.call_count += 1

        if compute_siglip_score.call_count <= 5:
            logger.debug(f"SigLIP basic score debug #{compute_siglip_score.call_count}:")
            logger.debug(f"  Clear sims: {clear_sims} (mean: {clear_mean:.4f})")
            logger.debug(f"  Blocked sims: {blocked_sims} (mean: {blocked_mean:.4f})")
            logger.debug(
                f"  Normalized score: {score:.4f} (was {clear_mean - blocked_mean:.4f} with simple subtraction)"
            )

        del image, image_features, clear_embeds, blocked_embeds

        return score

    except Exception as e:
        logger.warning(f"SigLIP2 scoring failed for {image_path}: {e}")
        return 0.0


def compute_enhanced_siglip_scores(image_path: Path, detections_group: pd.DataFrame) -> dict[str, float]:
    """Compute comprehensive SigLIP2-based scores for view selection."""
    try:
        model, processor, tokenizer = load_siglip_model()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        pil_image = Image.open(image_path)
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        scores = {}

        with torch.no_grad():
            if hasattr(model, "module"):
                image_features = model.module.get_image_features(**inputs)
            else:
                image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            clear_embeds = get_siglip_embeddings(CLEAR_PROMPTS, model, tokenizer, device)
            blocked_embeds = get_siglip_embeddings(BLOCKED_PROMPTS, model, tokenizer, device)
            clear_sims = (image_features @ clear_embeds.T).cpu().numpy().flatten()
            blocked_sims = (image_features @ blocked_embeds.T).cpu().numpy().flatten()

            clear_mean = np.mean(clear_sims)
            blocked_mean = np.mean(blocked_sims)
            epsilon = 1e-6
            scores["basic_score"] = float((clear_mean - blocked_mean) / (clear_mean + blocked_mean + epsilon))

            ground_clear_embeds = get_siglip_embeddings(GROUND_INTERFACE_CLEAR_PROMPTS, model, tokenizer, device)
            ground_blocked_embeds = get_siglip_embeddings(GROUND_INTERFACE_BLOCKED_PROMPTS, model, tokenizer, device)
            ground_clear_sims = (image_features @ ground_clear_embeds.T).cpu().numpy().flatten()
            ground_blocked_sims = (image_features @ ground_blocked_embeds.T).cpu().numpy().flatten()

            ground_clear_mean = np.mean(ground_clear_sims)
            ground_blocked_mean = np.mean(ground_blocked_sims)
            scores["ground_visibility"] = float(
                (ground_clear_mean - ground_blocked_mean) / (ground_clear_mean + ground_blocked_mean + epsilon)
            )

            single_building_embeds = get_siglip_embeddings(SINGLE_BUILDING_CLEAR_PROMPTS, model, tokenizer, device)
            multiple_unclear_embeds = get_siglip_embeddings(MULTIPLE_OR_UNCLEAR_PROMPTS, model, tokenizer, device)
            non_residential_embeds = get_siglip_embeddings(NON_RESIDENTIAL_PROMPTS, model, tokenizer, device)

            single_sims = (image_features @ single_building_embeds.T).cpu().numpy().flatten()
            multiple_sims = (image_features @ multiple_unclear_embeds.T).cpu().numpy().flatten()
            non_res_sims = (image_features @ non_residential_embeds.T).cpu().numpy().flatten()

            single_mean = np.mean(single_sims)
            multiple_mean = np.mean(multiple_sims)
            non_res_mean = np.mean(non_res_sims)

            bad_mean = (multiple_mean + non_res_mean) / 2
            scores["building_focus_score"] = float((single_mean - bad_mean) / (single_mean + bad_mean + epsilon))

            h, w = pil_image.height, pil_image.width

            bottom_third = pil_image.crop((0, 2 * h // 3, w, h))
            bottom_inputs = processor(images=bottom_third, return_tensors="pt")
            bottom_inputs = {k: v.to(device) for k, v in bottom_inputs.items()}
            if hasattr(model, "module"):
                bottom_features = model.module.get_image_features(**bottom_inputs)
            else:
                bottom_features = model.get_image_features(**bottom_inputs)
            bottom_features /= bottom_features.norm(dim=-1, keepdim=True)

            bottom_ground_clear = (bottom_features @ ground_clear_embeds.T).cpu().numpy().flatten()
            bottom_ground_blocked = (bottom_features @ ground_blocked_embeds.T).cpu().numpy().flatten()
            scores["bottom_third_ground"] = float(np.mean(bottom_ground_clear) - np.mean(bottom_ground_blocked))

            door_detections = detections_group[detections_group["class_name"] == "Front Door"]
            if not door_detections.empty:
                best_door = door_detections.loc[door_detections["confidence"].idxmax()]

                x1, y1, x2, y2 = best_door["bbox_x1"], best_door["bbox_y1"], best_door["bbox_x2"], best_door["bbox_y2"]
                bbox_w, bbox_h = x2 - x1, y2 - y1
                x1 = max(0, x1 - bbox_w * 0.25)
                y1 = max(0, y1 - bbox_h * 0.25)
                x2 = min(w, x2 + bbox_w * 0.25)
                y2 = min(h, y2 + bbox_h * 0.25)

                door_region = pil_image.crop((int(x1), int(y1), int(x2), int(y2)))
                door_inputs = processor(images=door_region, return_tensors="pt")
                door_inputs = {k: v.to(device) for k, v in door_inputs.items()}

                if hasattr(model, "module"):
                    door_features = model.module.get_image_features(**door_inputs)
                else:
                    door_features = model.get_image_features(**door_inputs)
                door_features /= door_features.norm(dim=-1, keepdim=True)

                door_verif_embeds = get_siglip_embeddings(
                    FEATURE_VERIFICATION_PROMPTS["Front Door"], model, tokenizer, device
                )
                door_verif_sims = (door_features @ door_verif_embeds.T).cpu().numpy().flatten()
                scores["door_verification"] = float(np.mean(door_verif_sims))
            else:
                scores["door_verification"] = 0.0

        del inputs, image_features, pil_image
        if "bottom_inputs" in locals():
            del bottom_inputs, bottom_features
        if "door_inputs" in locals():
            del door_inputs, door_features
        if device == "cuda":
            torch.cuda.empty_cache()

        return scores

    except Exception as e:
        logger.warning(f"Enhanced SigLIP scoring failed for {image_path}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "basic_score": 0.0,
            "ground_visibility": 0.0,
            "building_focus_score": 0.0,
            "bottom_third_ground": 0.0,
            "door_verification": 0.0,
        }


def compute_enhanced_siglip_scores_batch(
    image_paths: list[Path], detections_groups: list[pd.DataFrame]
) -> list[dict[str, float]]:
    """Batch compute SigLIP2 scores for multiple images - optimized for GPU."""
    if not image_paths:
        return []

    try:
        model, processor, tokenizer = load_siglip_model()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        pil_images = [Image.open(path) for path in image_paths]

        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(model, "module"):
                image_features = model.module.get_image_features(**inputs)
            else:
                image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            clear_embeds = get_siglip_embeddings(CLEAR_PROMPTS, model, tokenizer, device)
            blocked_embeds = get_siglip_embeddings(BLOCKED_PROMPTS, model, tokenizer, device)
            ground_clear_embeds = get_siglip_embeddings(GROUND_INTERFACE_CLEAR_PROMPTS, model, tokenizer, device)
            ground_blocked_embeds = get_siglip_embeddings(GROUND_INTERFACE_BLOCKED_PROMPTS, model, tokenizer, device)

            clear_sims = (image_features @ clear_embeds.T).cpu().numpy()
            blocked_sims = (image_features @ blocked_embeds.T).cpu().numpy()
            ground_clear_sims = (image_features @ ground_clear_embeds.T).cpu().numpy()
            ground_blocked_sims = (image_features @ ground_blocked_embeds.T).cpu().numpy()

        results = []
        for i in range(len(image_paths)):
            scores = {}
            epsilon = 1e-6

            clear_mean = np.mean(clear_sims[i])
            blocked_mean = np.mean(blocked_sims[i])
            scores["basic_score"] = float((clear_mean - blocked_mean) / (clear_mean + blocked_mean + epsilon))

            ground_clear_mean = np.mean(ground_clear_sims[i])
            ground_blocked_mean = np.mean(ground_blocked_sims[i])
            scores["ground_visibility"] = float(
                (ground_clear_mean - ground_blocked_mean) / (ground_clear_mean + ground_blocked_mean + epsilon)
            )

            scores["building_focus_score"] = 0.0
            scores["bottom_third_ground"] = 0.0
            scores["door_verification"] = 0.0

            results.append(scores)

        return results

    except Exception as e:
        logger.warning(f"Batch SigLIP2 scoring failed: {e}")

        return [
            compute_enhanced_siglip_scores(path, dg) for path, dg in zip(image_paths, detections_groups, strict=False)
        ]


def calculate_detection_score_enhanced(detections: pd.DataFrame) -> dict[str, float]:
    """
    Calculate enhanced detection scores aligned with FFH priorities.

    Returns:
        Dictionary with total_score and individual class max confidences
    """
    if detections.empty:
        return {
            "total_score": 0.0,
            "door": 0.0,
            "garage": 0.0,
            "stairs": 0.0,
            "foundation": 0.0,
            "window": 0.0,
            "tot_boxes": 0,
        }

    class_confidences = {
        "door": detections[detections["class_name"] == "Front Door"]["confidence"].max()
        if not detections[detections["class_name"] == "Front Door"].empty
        else 0.0,
        "garage": detections[detections["class_name"] == "Garage Door"]["confidence"].max()
        if not detections[detections["class_name"] == "Garage Door"].empty
        else 0.0,
        "stairs": detections[detections["class_name"] == "Stairs"]["confidence"].max()
        if not detections[detections["class_name"] == "Stairs"].empty
        else 0.0,
        "foundation": detections[detections["class_name"] == "Foundation"]["confidence"].max()
        if not detections[detections["class_name"] == "Foundation"].empty
        else 0.0,
        "window": detections[detections["class_name"] == "Window"]["confidence"].max()
        if not detections[detections["class_name"] == "Window"].empty
        else 0.0,
    }

    ffh_detections = detections[detections["class_name"] != "Window"]
    tot_boxes = len(ffh_detections)

    total_score = (
        1.5 * class_confidences["door"]
        + 1.0 * class_confidences["stairs"]
        + 0.75 * class_confidences["foundation"]
        + 0.5 * class_confidences["garage"]
    )

    return {
        "total_score": total_score,
        "door": class_confidences["door"],
        "garage": class_confidences["garage"],
        "stairs": class_confidences["stairs"],
        "foundation": class_confidences["foundation"],
        "window": class_confidences["window"],
        "tot_boxes": tot_boxes,
    }


def calculate_adaptive_siglip_weight(building_views: dict[tuple[str, int, str], pd.DataFrame]) -> float:
    """Calculate adaptive SigLIP weight based on door detection variance across views."""
    door_confidences = []

    for detections in building_views.values():
        door_detections = detections[detections["class_name"] == "Front Door"]
        if not door_detections.empty:
            door_confidences.append(door_detections["confidence"].max())
        else:
            door_confidences.append(0.0)

    variance = np.var(door_confidences) if len(door_confidences) > 1 else 1.0
    return 12.0 if variance < CONFIG.siglip.variance_threshold else 10.0


def calculate_combined_score(
    detection_scores: dict[str, float],
    siglip_score: float,
    siglip_weight: float,
    view_type: str,
    distance: float,
    enhanced_siglip_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate combined score with diversity and view angle bonuses."""
    has_ground_feature = (
        detection_scores["foundation"] > CONFIG.siglip.detection_threshold
        or detection_scores["stairs"] > CONFIG.siglip.detection_threshold
    )
    has_entrance_feature = (
        detection_scores["door"] > CONFIG.siglip.detection_threshold
        or detection_scores["garage"] > CONFIG.siglip.detection_threshold
    )
    diversity_bonus = CONFIG.siglip.diversity_bonus if (has_ground_feature and has_entrance_feature) else 1.0
    if view_type == "direct":
        view_angle_bonus = CONFIG.siglip.direct_view_bonus
    elif view_type in ["oblique_left", "oblique_right"]:
        view_angle_bonus = CONFIG.siglip.oblique_view_penalty
    else:
        view_angle_bonus = 0.0

    base_penalty = (
        1.0
        if (detection_scores["tot_boxes"] < 3 and detection_scores["door"] < CONFIG.siglip.detection_threshold)
        else 0.0
    )

    if enhanced_siglip_scores:
        detection_component = detection_scores["total_score"]

        siglip_component = siglip_score + 0.5
        siglip_component = max(0, min(1, siglip_component))

        ground_score = max(
            enhanced_siglip_scores.get("ground_visibility", 0), enhanced_siglip_scores.get("bottom_third_ground", 0)
        )
        ground_component = ground_score + 0.5
        ground_component = max(0, min(1, ground_component))

        focus_score = enhanced_siglip_scores.get("building_focus_score", 0)
        focus_component = focus_score + 0.5
        focus_component = max(0, min(1, focus_component))

        distance_score = 1.0 - min(distance / 50.0, 1.0)
        view_quality = distance_score * 0.7 + (0.3 if view_type == "direct" else 0.15)

        weights = CONFIG.siglip.scoring_weights

        combined_score = (
            weights["detection"] * detection_component
            + weights["siglip"] * siglip_component
            + weights["ground"] * ground_component
            + weights["focus"] * focus_component
            + weights["view"] * view_quality
            + diversity_bonus
            - base_penalty
        )
    else:
        detection_component = detection_scores["total_score"]
        siglip_component = siglip_score + 0.5
        siglip_component = max(0, min(1, siglip_component))

        combined_score = 0.6 * detection_component + 0.4 * siglip_component + view_angle_bonus - base_penalty

    return {
        "combined_score": combined_score,
        "has_ground_feature": has_ground_feature,
        "has_entrance_feature": has_entrance_feature,
        "diversity_bonus": diversity_bonus,
        "view_angle_bonus": view_angle_bonus,
        "has_door_override": detection_scores["door"] >= CONFIG.siglip.detection_threshold,
        "ground_visibility_score": enhanced_siglip_scores.get("ground_visibility", 0) if enhanced_siglip_scores else 0,
    }


def process_building_views(
    row_id: str,
    building_id: str,
    gnaf_id: str,
    region: str,
    detections_df: pd.DataFrame,
    clips_df: pd.DataFrame,
    distance_lookup: dict[tuple[int, str, int, str], float],
    data_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Process views for a single building and return best view results."""
    results = []

    building_detections = detections_df[detections_df["id"] == row_id]
    if building_detections.empty:
        return results

    view_groups = building_detections.groupby(["pano_id", "edge_idx", "view_type"])

    door_confidences = []
    for _, group in view_groups:
        door_detections = group[group["class_name"] == "Front Door"]
        if not door_detections.empty:
            door_confidences.append(door_detections["confidence"].max())
        else:
            door_confidences.append(0.0)

    variance = np.var(door_confidences) if len(door_confidences) > 1 else 1.0

    siglip_weight = (
        CONFIG.siglip.low_variance_weight
        if variance < CONFIG.siglip.variance_threshold
        else CONFIG.siglip.high_variance_weight
    )

    clips_to_process = []
    clip_infos = []

    for (pano_id, edge_idx, view_type), detections_group in view_groups:
        detection_scores = calculate_detection_score_enhanced(detections_group)

        gnaf_str = gnaf_id if gnaf_id and gnaf_id != "NO_GNAF" else "NO_GNAF"
        clip_path = (
            data_dir
            / region.capitalize()
            / "clips"
            / f"{row_id}_{building_id}_{gnaf_str}"
            / f"{pano_id[:-4] if pano_id.lower().endswith('.jpg') else pano_id}_edge{edge_idx}_{view_type}.jpg"
        )

        if not validate_file_exists_and_valid(clip_path, file_type="image", min_size_bytes=1000):
            logger.debug(f"Clip image not found or invalid: {clip_path}")
            continue

        distance_key = (row_id, pano_id, edge_idx, view_type)
        distance = distance_lookup.get(distance_key, float("inf"))

        clips_to_process.append(clip_path)
        clip_infos.append(
            {
                "pano_id": pano_id,
                "edge_idx": int(edge_idx),
                "view_type": view_type,
                "detection_scores": detection_scores,
                "detections_group": detections_group,
                "distance": distance,
                "clip_path": str(clip_path.relative_to(data_dir)),
            }
        )

    if not clips_to_process:
        return results, 0

    all_enhanced_scores = compute_enhanced_siglip_scores_batch(
        clips_to_process, [info["detections_group"] for info in clip_infos]
    )

    view_results = []
    for _i, (clip_info, enhanced_scores) in enumerate(zip(clip_infos, all_enhanced_scores, strict=False)):
        siglip_score = enhanced_scores["basic_score"]

        score_info = calculate_combined_score(
            clip_info["detection_scores"],
            siglip_score,
            siglip_weight,
            clip_info["view_type"],
            clip_info["distance"],
            enhanced_scores,
        )

        view_results.append(
            {
                "pano_id": clip_info["pano_id"],
                "edge_idx": clip_info["edge_idx"],
                "view_type": clip_info["view_type"],
                "detection_scores": clip_info["detection_scores"],
                "siglip_score": siglip_score,
                "enhanced_siglip_scores": enhanced_scores,
                "combined_score": score_info["combined_score"],
                "has_door_override": score_info["has_door_override"],
                "distance": clip_info["distance"],
                "score_info": score_info,
                "clip_path": clip_info["clip_path"],
            }
        )

    if not view_results:
        return results, 0

    door_override_views = [r for r in view_results if r["has_door_override"]]
    if len(door_override_views) == 1:
        best_result = door_override_views[0]
    else:
        best_result = max(view_results, key=lambda x: (x["combined_score"], -x["distance"]))

    results.append(
        {
            "id": row_id,
            "building_id": building_id,
            "region_name": region,
            "gnaf_id": gnaf_id,
            "pano_id": best_result["pano_id"],
            "edge_idx": best_result["edge_idx"],
            "view_type": best_result["view_type"],
            "detection_score": float(best_result["detection_scores"]["total_score"]),
            "siglip_score": float(best_result["siglip_score"]),
            "combined_score": float(best_result["combined_score"]),
            "distance": float(best_result["distance"]),
            "siglip_weight": float(siglip_weight),
            "has_door_override": bool(best_result["score_info"]["has_door_override"]),
            "diversity_bonus": float(best_result["score_info"]["diversity_bonus"]),
            "view_angle_bonus": float(best_result["score_info"]["view_angle_bonus"]),
            "has_ground_feature": bool(best_result["score_info"]["has_ground_feature"]),
            "has_entrance_feature": bool(best_result["score_info"]["has_entrance_feature"]),
            "selection_type": "best",
            "status": "success",
            "error_message": "",
            "clip_image_path": best_result["clip_path"],
            "ground_visibility_score": float(best_result["score_info"].get("ground_visibility_score", 0)),
        }
    )

    direct_views = [r for r in view_results if r["view_type"] == "direct"]
    if direct_views:
        closest_direct = min(direct_views, key=lambda x: x["distance"])
        results.append(
            {
                "id": row_id,
                "building_id": building_id,
                "region_name": region,
                "gnaf_id": gnaf_id,
                "pano_id": closest_direct["pano_id"],
                "edge_idx": closest_direct["edge_idx"],
                "view_type": closest_direct["view_type"],
                "detection_score": float(closest_direct["detection_scores"]["total_score"]),
                "siglip_score": float(closest_direct["siglip_score"]),
                "combined_score": float(closest_direct["combined_score"]),
                "distance": float(closest_direct["distance"]),
                "siglip_weight": float(siglip_weight),
                "has_door_override": bool(closest_direct["score_info"]["has_door_override"]),
                "diversity_bonus": float(closest_direct["score_info"]["diversity_bonus"]),
                "view_angle_bonus": float(closest_direct["score_info"]["view_angle_bonus"]),
                "has_ground_feature": bool(closest_direct["score_info"]["has_ground_feature"]),
                "has_entrance_feature": bool(closest_direct["score_info"]["has_entrance_feature"]),
                "selection_type": "closest_direct",
                "status": "success",
                "error_message": None,
                "clip_image_path": closest_direct["clip_path"],
                "ground_visibility_score": float(closest_direct["score_info"].get("ground_visibility_score", 0)),
            }
        )

    return results, len(view_results)


def process_region(region: str, workers: int = -1, batch_size: int = 100) -> None:
    """Process best view selection for a region."""
    logger.info(f"Processing best view selection for region: {region}")

    detections_df = read_table("stage04a_detections", region=region)

    if detections_df.empty:
        logger.warning(f"No object detections found for {region}")
        return

    clips_df = read_table("stage03_clips", region=region)

    if clips_df.empty:
        logger.warning(f"No clips found for {region}")
        return

    candidate_views_df = read_table("stage02a_candidate_views", region=region)

    candidate_views_df = candidate_views_df[candidate_views_df["is_chosen"]]

    if candidate_views_df.empty:
        logger.warning(f"No candidate views found for {region}")
        return

    distance_lookup = {}
    for _, row in candidate_views_df.iterrows():
        key = (row["id"], row["pano_id"], row["edge_idx"], row["view_type"])
        distance_lookup[key] = row["distance"]

    processed_ids = get_processed_stage_ids("stage04b_best_views", region)

    building_combos = detections_df[["id", "building_id", "gnaf_id"]].drop_duplicates()
    building_combos["gnaf_id"] = building_combos["gnaf_id"].fillna("NO_GNAF")

    to_process = building_combos[~building_combos["id"].isin(processed_ids)]

    logger.info(f"Found {len(building_combos)} buildings with detections")
    logger.info(f"Already processed: {len(processed_ids)}")
    logger.info(f"To process: {len(to_process)}")

    if len(to_process) == 0:
        logger.info(f"No new buildings to process for {region}")
        return

    if _siglip_model is None:
        logger.info("Loading SigLIP model for occlusion scoring...")
        load_siglip_model()

    data_dir = CONFIG.output_root

    score_totals = {
        "detection_score": 0.0,
        "siglip_score": 0.0,
        "combined_score": 0.0,
        "ground_visibility_score": 0.0,
        "has_door": 0,
    }

    all_siglip_scores = []
    buildings_with_best_view = 0
    total_clips_evaluated = 0

    with (
        best_view_progress(f"Selecting best views for {region}", len(to_process)) as prog,
        BatchWriter("stage04b_best_views", batch_size=5000, progress_tracker=prog) as writer,
    ):
        for _, row in to_process.iterrows():
            try:
                building_results, clips_evaluated = process_building_views(
                    row_id=str(row["id"]),
                    building_id=str(row["building_id"]),
                    gnaf_id=row["gnaf_id"],
                    region=region,
                    detections_df=detections_df,
                    clips_df=clips_df,
                    distance_lookup=distance_lookup,
                    data_dir=data_dir,
                )

                if building_results:
                    total_clips_evaluated += clips_evaluated

                    for result in building_results:
                        writer.add(
                            Stage04bBestViewRecord(
                                id=result["id"],
                                building_id=result["building_id"],
                                region_name=result["region_name"],
                                gnaf_id=result["gnaf_id"],
                                pano_id=result.get("pano_id"),
                                edge_idx=result.get("edge_idx"),
                                view_type=result.get("view_type"),
                                detection_score=result["detection_score"],
                                siglip_score=result["siglip_score"],
                                combined_score=result["combined_score"],
                                distance=result["distance"],
                                siglip_weight=result["siglip_weight"],
                                has_door_override=result["has_door_override"],
                                diversity_bonus=result["diversity_bonus"],
                                view_angle_bonus=result["view_angle_bonus"],
                                has_ground_feature=result["has_ground_feature"],
                                has_entrance_feature=result["has_entrance_feature"],
                                selection_type=result["selection_type"],
                                status=result["status"],
                                error_message=result.get("error_message"),
                                clip_image_path=result["clip_image_path"],
                                ground_visibility_score=result["ground_visibility_score"],
                            )
                        )

                        if result["selection_type"] == "best":
                            score_totals["detection_score"] += result["detection_score"]
                            score_totals["siglip_score"] += result["siglip_score"]
                            score_totals["combined_score"] += result["combined_score"]
                            score_totals["ground_visibility_score"] += result["ground_visibility_score"]
                            if result["has_door_override"]:
                                score_totals["has_door"] += 1
                            buildings_with_best_view += 1

                            all_siglip_scores.append(result["siglip_score"])

                    if buildings_with_best_view > 0:
                        prog.fields["avg_det"] = round(score_totals["detection_score"] / buildings_with_best_view, 2)
                        prog.fields["avg_sig"] = round(score_totals["siglip_score"] / buildings_with_best_view, 2)
                        prog.fields["avg_comb"] = round(score_totals["combined_score"] / buildings_with_best_view, 2)
                        prog.fields["avg_gnd"] = round(
                            score_totals["ground_visibility_score"] / buildings_with_best_view, 2
                        )
                        prog.fields["pct_door"] = round(score_totals["has_door"] / buildings_with_best_view * 100, 1)

                        if len(all_siglip_scores) > 1:
                            prog.fields["sig_basic_var"] = float(np.var(all_siglip_scores))
                        else:
                            prog.fields["sig_basic_var"] = 0.0

                        prog.fields["clips"] = total_clips_evaluated
                        prog.fields["c/b"] = round(total_clips_evaluated / buildings_with_best_view, 1)

                        if prog.task_id is not None:
                            prog.progress.update(prog.task_id, **prog.fields)

                    prog.update("suc", len(building_results))
                else:
                    prog.update("fail", 1)

            except Exception as e:
                logger.error(f"Failed to process building {row['building_id']}: {e}")

                prog.update("fail", 1)

    cleanup_gpu_memory()

    logger.success(f"{region}: {prog.get_summary()}")


def run_stage(
    region: str | None = None,
    workers: int = -1,
    batch_size: int = 100,
) -> None:
    """Run stage 04b: best view selection with SigLIP scoring.

    Args:
        region: Specific region to process, or None for all regions
        workers: Number of workers (not used in this stage)
        batch_size: Batch size for database operations
    """

    initialize_all_stage_tables()

    regions = [region] if region else list(CONFIG.regions.keys())

    for region_name in regions:
        try:
            process_region(region_name, workers=workers, batch_size=batch_size)
            logger.success(f"Completed stage 04b for {region_name}")
        except Exception as e:
            logger.error(f"Failed stage 04b for {region_name}: {e}")
            raise

    unload_siglip_model()

    logger.info("Stage 04b complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 04b: Select best panorama views based on object detections with SigLIP occlusion scoring"
    )
    parser.add_argument("--region", type=str, help="Specific region to process")
    parser.add_argument("--workers", type=int, default=-1, help="Number of workers")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for database operations")
    parser.add_argument(
        "--no-siglip", action="store_true", help="Disable SigLIP occlusion scoring (use detection scores only)"
    )
    args = parser.parse_args()

    if args.no_siglip:
        logger.info("SigLIP occlusion scoring disabled")

        def _disabled_siglip_score(_: Path) -> float:
            return 0.0

        compute_siglip_score = _disabled_siglip_score

    run_stage(
        region=args.region,
        workers=args.workers,
        batch_size=args.batch_size,
    )
