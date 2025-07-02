"""FastAPI server for serving LiDAR point cloud data to the deck.gl viewer."""

import json
import os
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from PIL import Image

app = FastAPI(title="Floor Heights LiDAR Viewer API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CONFIG:
    output_root = Path(__file__).resolve().parents[3] / "output"


OUTPUT_BASE = CONFIG.output_root
REGIONS = ["wagga", "launceston", "tweed"]


@app.get("/api/lidar/file-exists/{region}/{clip_id}")
async def check_file_exists(region: str, clip_id: str) -> dict[str, Any]:
    """Check if a specific clip file exists in the region."""
    if region.lower() not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Region {region} not found")

    region_path = OUTPUT_BASE / region.capitalize() / "lidar" / "rev2-local" / "clipped"

    if not region_path.exists():
        logger.warning(f"Path does not exist: {region_path}")
        return {"exists": False, "filename": None}

    clip_parts = clip_id.split("_")
    main_id = clip_parts[0]
    gnaf_id = clip_parts[-1] if len(clip_parts) > 1 else None

    for las_file in list(region_path.glob("*.las")) + list(region_path.glob("*.laz")):
        filename = las_file.name

        has_main_id = main_id in filename
        has_gnaf_id = gnaf_id in filename if gnaf_id else True

        if has_main_id and has_gnaf_id:
            logger.info(f"Found matching file: {filename} for clip_id: {clip_id}")
            return {"exists": True, "filename": filename, "size_mb": f"{las_file.stat().st_size / (1024 * 1024):.2f}"}

    logger.warning(f"No file found for clip_id: {clip_id} in region: {region}")
    return {"exists": False, "filename": None}


@app.get("/api/lidar/files/{region}")
async def get_files_list(region: str) -> dict[str, Any]:
    """Get sorted list of all LAS files in the region for browsing."""
    if region.lower() not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Region {region} not found")

    region_path = OUTPUT_BASE / region.capitalize() / "lidar" / "rev2-local" / "clipped"

    if not region_path.exists():
        return {"files": [], "total": 0}

    las_files = sorted(list(region_path.glob("*.las")) + list(region_path.glob("*.laz")), key=lambda x: x.name)

    files_list = [f.name for f in las_files]

    return {"files": files_list, "total": len(files_list)}


@app.get("/api/lidar/data/{region}/{filename}")
async def get_point_cloud_data(region: str, filename: str) -> dict[str, Any]:
    """Load and return point cloud data for visualization."""
    if region.lower() not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Region {region} not found")

    file_path = OUTPUT_BASE / region.capitalize() / "lidar" / "rev2-local" / "clipped" / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    try:
        las = laspy.read(file_path)

        max_points = 100000
        total_points = len(las.points)

        if total_points > max_points:
            indices = np.random.choice(total_points, max_points, replace=False)
            x = np.array(las.x)[indices]
            y = np.array(las.y)[indices]
            z = np.array(las.z)[indices]
            classification = np.array(las.classification)[indices]
            intensity = np.array(las.intensity)[indices]
        else:
            x = np.array(las.x)
            y = np.array(las.y)
            z = np.array(las.z)
            classification = np.array(las.classification)
            intensity = np.array(las.intensity)

        unique_classes, counts = np.unique(classification, return_counts=True)
        classifications = {int(c): int(count) for c, count in zip(unique_classes, counts, strict=False)}

        intensity_min = intensity.min()
        intensity_max = intensity.max()
        if intensity_max > intensity_min:
            intensity_normalized = (intensity - intensity_min) / (intensity_max - intensity_min)
        else:
            intensity_normalized = np.ones_like(intensity) * 0.5

        points = [
            {
                "x": float(x[i]),
                "y": float(y[i]),
                "z": float(z[i]),
                "classification": int(classification[i]),
                "intensity": float(intensity_normalized[i]),
            }
            for i in range(len(x))
        ]

        return {
            "filename": filename,
            "total_points": total_points,
            "displayed_points": len(points),
            "points": points,
            "classifications": classifications,
        }

    except Exception as e:
        logger.error(f"Error loading LAS file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "floor-heights-lidar-viewer"}


@app.get("/api/initial-state")
async def get_initial_state() -> dict[str, Any]:
    """Get initial state from environment variables."""
    import time

    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if key.startswith("INITIAL_") or key == "PORT":
            logger.info(f"  {key}: {value}")

    clip_id = os.environ.get("INITIAL_CLIP_ID")
    region = os.environ.get("INITIAL_REGION", "wagga").lower()

    logger.info(f"Initial state - clip_id: {clip_id}, region: {region}")

    return {"clip_id": clip_id, "region": region, "timestamp": time.time()}


@app.get("/api/annotation/images")
async def get_annotation_images() -> JSONResponse:
    """Get list of all images in the dataset."""
    dataset_root = Path(__file__).resolve().parents[3] / "data" / "datasets"
    images_dir = dataset_root / "images"

    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return JSONResponse(
            content={"images": [], "total": 0},
            headers={"Cache-Control": "public, max-age=300"},
        )

    image_files = sorted([f.name for f in images_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    return JSONResponse(
        content={"images": image_files, "total": len(image_files)},
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/api/annotation/coco")
async def get_coco_annotations() -> dict[str, Any]:
    """Load and return COCO annotations with caching."""

    dataset_root = Path(__file__).resolve().parents[3] / "data" / "datasets"
    annotations_path = dataset_root / "annotations" / "annotations.json"

    if not annotations_path.exists():
        raise HTTPException(status_code=404, detail="Annotations file not found")

    try:
        with annotations_path.open() as f:
            coco_data = json.load(f)

        image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
        filename_to_image_id = {img["file_name"]: img["id"] for img in coco_data["images"]}

        annotations_by_image = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        response_data = {
            "categories": coco_data["categories"],
            "images": coco_data["images"],
            "annotations": coco_data["annotations"],
            "image_id_to_filename": image_id_to_filename,
            "filename_to_image_id": filename_to_image_id,
            "annotations_by_image": annotations_by_image,
        }

        return JSONResponse(
            content=response_data,
            headers={
                "Cache-Control": "public, max-age=300",
            },
        )

    except Exception as e:
        logger.error(f"Error loading COCO annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/annotation/image/{filename}")
async def get_annotation_image(filename: str):
    """Serve an image file from the dataset with caching headers."""
    dataset_root = Path(__file__).resolve().parents[3] / "data" / "datasets"
    image_path = dataset_root / "images" / filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {filename} not found")

    try:
        img = Image.open(image_path)
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file") from None

    media_type = "image/jpeg"
    if filename.lower().endswith(".png"):
        media_type = "image/png"
    elif filename.lower().endswith(".webp"):
        media_type = "image/webp"

    return FileResponse(
        image_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600, immutable", "X-Content-Type-Options": "nosniff"},
    )


if __name__ == "__main__":
    logger.info("Starting Floor Heights LiDAR Viewer API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104 - Development server
