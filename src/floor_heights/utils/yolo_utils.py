"""YOLO utilities for exporting clips and viewing detection results."""

import io
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

try:
    from sixel import SixelWriter

    SIXEL_AVAILABLE = True
except ImportError:
    SIXEL_AVAILABLE = False

from floor_heights.config import CONFIG
from floor_heights.utils.fh_io import read_table
from floor_heights.utils.visualization import (
    OBJECT_CLASS_COLORS,
    OBJECT_CLASS_NAMES,
)

console = Console()


def export_clips_by_class(
    region: str,
    class_names: list[str] | None = None,
    class_ids: list[int] | None = None,
    export_dir: Path | None = None,
    include_annotations: bool = True,
    include_visualizations: bool = False,
    confidence_threshold: float = 0.5,
    limit: int | None = None,
    random_sample: bool = False,
) -> dict[str, Any]:
    """Export clips containing specific object classes.

    Args:
        region: Region name to export from
        class_names: List of class names to filter (e.g., ["Front Door", "Window"])
        class_ids: List of class IDs to filter (alternative to class_names)
        export_dir: Directory to export to (default: CONFIG.output_root / 'exports' / 'clips_by_class')
        include_annotations: Whether to export detection annotations as JSON
        include_visualizations: Whether to create visualized images with bounding boxes
        confidence_threshold: Minimum confidence score for detections
        limit: Maximum number of clips to export
        random_sample: Whether to randomly sample clips (if limit is set)

    Returns:
        Dictionary with export statistics
    """
    if class_names:
        name_to_id = {v: k for k, v in OBJECT_CLASS_NAMES.items()}
        class_ids = [name_to_id[name] for name in class_names if name in name_to_id]
        if not class_ids:
            console.print(f"[red]No valid class names found. Available: {list(OBJECT_CLASS_NAMES.values())}")
            return {"error": "Invalid class names"}

    if not class_ids:
        class_ids = list(OBJECT_CLASS_NAMES.keys())

    if export_dir is None:
        export_dir = CONFIG.output_root / "exports" / "clips_by_class"
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    clips_dir = export_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    if include_annotations:
        annotations_dir = export_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)

    if include_visualizations:
        viz_dir = export_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

    console.print(f"[cyan]Querying detections for classes: {[OBJECT_CLASS_NAMES[id] for id in class_ids]}")

    detections = read_table("stage04a_detections", region=region)

    if not detections.empty:
        detections = detections[
            (detections["class_id"].isin(class_ids)) & (detections["confidence"] >= confidence_threshold)
        ]

    if detections.empty:
        console.print("[yellow]No detections found matching criteria")
        return {"clips_exported": 0}

    clip_groups = detections.groupby(["clip_path", "building_id", "gnaf_id", "pano_id", "edge_idx", "view_type"])

    clip_list = list(clip_groups)
    if limit and len(clip_list) > limit:
        if random_sample:
            import random

            clip_list = random.sample(clip_list, limit)
        else:
            clip_list = clip_list[:limit]

    stats = {
        "clips_exported": 0,
        "detections_exported": 0,
        "by_class": dict.fromkeys(OBJECT_CLASS_NAMES.values(), 0),
        "errors": [],
    }

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"Exporting {len(clip_list)} clips...", total=len(clip_list))

        for (clip_path, building_id, gnaf_id, pano_id, edge_idx, view_type), group_df in clip_list:
            try:
                src_path = CONFIG.output_root / clip_path
                clip_name = f"{building_id}_{gnaf_id}_{pano_id}_edge{edge_idx}_{view_type}.jpg"
                dst_path = clips_dir / clip_name

                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    stats["clips_exported"] += 1

                    if include_annotations:
                        annotations = []
                        for _, det in group_df.iterrows():
                            annotations.append(
                                {
                                    "class_id": int(det["class_id"]),
                                    "class_name": det["class_name"],
                                    "confidence": float(det["confidence"]),
                                    "bbox": {
                                        "x1": int(det["bbox_x1"]),
                                        "y1": int(det["bbox_y1"]),
                                        "x2": int(det["bbox_x2"]),
                                        "y2": int(det["bbox_y2"]),
                                        "width": int(det["bbox_width"]),
                                        "height": int(det["bbox_height"]),
                                        "area": int(det["bbox_area"]),
                                        "center_x": int(det["bbox_center_x"]),
                                        "center_y": int(det["bbox_center_y"]),
                                    },
                                }
                            )
                            stats["detections_exported"] += 1
                            stats["by_class"][det["class_name"]] += 1

                        ann_path = annotations_dir / f"{clip_name}.json"
                        with ann_path.open("w") as f:
                            json.dump(
                                {
                                    "clip_name": clip_name,
                                    "building_id": building_id,
                                    "gnaf_id": gnaf_id,
                                    "pano_id": pano_id,
                                    "edge_idx": int(edge_idx),
                                    "view_type": view_type,
                                    "image_width": int(group_df.iloc[0]["image_width"]),
                                    "image_height": int(group_df.iloc[0]["image_height"]),
                                    "detections": annotations,
                                },
                                f,
                                indent=2,
                            )

                    if include_visualizations:
                        image = cv2.imread(str(dst_path))
                        if image is not None:
                            for _, det in group_df.iterrows():
                                color = OBJECT_CLASS_COLORS[int(det["class_id"])]
                                cv2.rectangle(
                                    image,
                                    (int(det["bbox_x1"]), int(det["bbox_y1"])),
                                    (int(det["bbox_x2"]), int(det["bbox_y2"])),
                                    color,
                                    2,
                                )
                                label = f"{det['class_name']} {det['confidence']:.2f}"
                                cv2.putText(
                                    image,
                                    label,
                                    (int(det["bbox_x1"]), int(det["bbox_y1"] - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2,
                                )

                            viz_path = viz_dir / f"{clip_name}"
                            cv2.imwrite(str(viz_path), image)
                else:
                    stats["errors"].append(f"Clip not found: {src_path}")

            except Exception as e:
                stats["errors"].append(f"Error processing {clip_path}: {e!s}")

            progress.update(task, advance=1)

    console.print("\n[green]Export Complete!")

    table = Table(title="Export Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Clips Exported", str(stats["clips_exported"]))
    table.add_row("Total Detections", str(stats["detections_exported"]))

    for class_name, count in stats["by_class"].items():
        if count > 0:
            table.add_row(f"  - {class_name}", str(count))

    if stats["errors"]:
        table.add_row("Errors", str(len(stats["errors"])))

    console.print(table)
    console.print(f"\n[cyan]Files exported to: {export_dir}")

    return stats


def get_class_statistics(region: str, min_confidence: float = 0.5) -> pd.DataFrame:
    """Get statistics about object detection classes in a region.

    Args:
        region: Region name
        min_confidence: Minimum confidence threshold

    Returns:
        DataFrame with class statistics
    """
    detections = read_table("stage04a_detections", region=region)

    if not detections.empty:
        detections = detections[detections["confidence"] >= min_confidence]

    if detections.empty:
        return pd.DataFrame()

    stats = []
    for class_id, class_name in OBJECT_CLASS_NAMES.items():
        class_dets = detections[detections["class_id"] == class_id]
        if not class_dets.empty:
            stats.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "total_detections": len(class_dets),
                    "unique_clips": class_dets["clip_path"].nunique(),
                    "unique_buildings": class_dets["building_id"].nunique(),
                    "avg_confidence": class_dets["confidence"].mean(),
                    "min_confidence": class_dets["confidence"].min(),
                    "max_confidence": class_dets["confidence"].max(),
                    "avg_bbox_area": class_dets["bbox_area"].mean(),
                }
            )

    return pd.DataFrame(stats)


def find_rare_examples(
    region: str, class_names: list[str], max_examples: int = 10, min_confidence: float = 0.7
) -> pd.DataFrame:
    """Find high-confidence examples of rare classes.

    Args:
        region: Region name
        class_names: List of class names to find
        max_examples: Maximum examples per class
        min_confidence: Minimum confidence threshold

    Returns:
        DataFrame with rare examples
    """
    name_to_id = {v: k for k, v in OBJECT_CLASS_NAMES.items()}
    class_ids = [name_to_id[name] for name in class_names if name in name_to_id]

    if not class_ids:
        return pd.DataFrame()

    detections = read_table("stage04a_detections", region=region)

    if not detections.empty:
        detections = detections[(detections["class_id"].isin(class_ids)) & (detections["confidence"] >= min_confidence)]

    if detections.empty:
        return pd.DataFrame()

    examples = []
    for class_id in class_ids:
        class_dets = detections[detections["class_id"] == class_id]
        if not class_dets.empty:
            top_examples = class_dets.nlargest(max_examples, "confidence")
            examples.append(top_examples)

    if examples:
        return pd.concat(examples, ignore_index=True)
    return pd.DataFrame()


def display_image_sixel(image_path: str | Path, max_width: int = 800) -> bool:
    """Display an image in the terminal using sixel protocol.

    Args:
        image_path: Path to the image file
        max_width: Maximum width for display (pixels)

    Returns:
        True if successful, False otherwise
    """
    if not SIXEL_AVAILABLE:
        console.print("[yellow]Sixel library not installed. Install with: pip install sixel")
        console.print("[dim]Sixel is supported by terminals like VSCode's integrated terminal")
        return False

    image_path = Path(image_path)
    if not image_path.exists():
        console.print(f"[red]Image not found: {image_path}")
        return False

    try:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            console.print(f"[red]Failed to read image: {image_path}")
            return False

        height, width = img.shape[:2]

        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        from PIL import Image

        pil_img = Image.fromarray(img_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)

        import sys

        sys.stdout.write("\033[s")

        writer = SixelWriter()
        writer.draw(buffer)

        _, img_height = pil_img.size

        lines_estimate = (img_height // 20) + 2

        sys.stdout.write(f"\033[{lines_estimate}B")
        sys.stdout.write("\033[0G")
        sys.stdout.flush()

        return True

    except Exception as e:
        console.print(f"[red]Error displaying image: {e}")
        return False


def view_detection_results(
    region: str,
    building_id: str | None = None,
    class_names: list[str] | None = None,
    confidence_threshold: float = 0.5,
    max_images: int = 5,
    show_annotations: bool = True,
) -> None:
    """View detection results in the terminal.

    Args:
        region: Region name
        building_id: Optional specific building ID to view
        class_names: Optional list of class names to filter
        confidence_threshold: Minimum confidence score
        max_images: Maximum number of images to display
        show_annotations: Whether to draw bounding boxes
    """
    if not SIXEL_AVAILABLE:
        console.print("[yellow]Sixel library not installed. Install with: pip install sixel")
        console.print("[dim]Sixel is supported by terminals like VSCode's integrated terminal")
        return

    filters = {}

    if building_id:
        filters["building_id"] = building_id

    detections = read_table("stage04a_detections", region=region, filters=filters)

    if detections.empty:
        console.print("[yellow]No detections found for this region")
        return

    detections = detections[detections["confidence"] >= confidence_threshold]

    if class_names:
        name_to_id = {v: k for k, v in OBJECT_CLASS_NAMES.items()}
        class_ids = [name_to_id[name] for name in class_names if name in name_to_id]
        if class_ids:
            detections = detections[detections["class_id"].isin(class_ids)]

    if detections.empty:
        console.print("[yellow]No detections found matching criteria")
        return

    clip_groups = detections.groupby(["clip_path", "building_id", "gnaf_id"])

    displayed = 0
    for (clip_path, building_id, gnaf_id), group_df in clip_groups:
        if displayed >= max_images:
            break

        img_path = CONFIG.output_root / clip_path
        if not img_path.exists():
            continue

        console.print(f"\n[cyan]Building: {building_id} (GNAF: {gnaf_id})")
        console.print(f"[dim]Image: {clip_path}")

        detection_summary = group_df.groupby("class_name").size()
        console.print("[green]Detections:")
        for class_name, count in detection_summary.items():
            console.print(f"  - {class_name}: {count}")

        console.print()

        if show_annotations:
            img = cv2.imread(str(img_path))
            if img is not None:
                for _, det in group_df.iterrows():
                    color = OBJECT_CLASS_COLORS[int(det["class_id"])]
                    cv2.rectangle(
                        img,
                        (int(det["bbox_x1"]), int(det["bbox_y1"])),
                        (int(det["bbox_x2"]), int(det["bbox_y2"])),
                        color,
                        2,
                    )
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    cv2.putText(
                        img,
                        label,
                        (int(det["bbox_x1"]), int(det["bbox_y1"] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    cv2.imwrite(tmp.name, img)
                    display_image_sixel(tmp.name)
                    Path(tmp.name).unlink()
            else:
                display_image_sixel(img_path)
        else:
            display_image_sixel(img_path)

        displayed += 1

        if displayed < len(clip_groups) and displayed < max_images:
            console.print()
            if not Confirm.ask("Show next image?"):
                break

    console.print(f"\n[dim]Displayed {displayed} images")


def interactive_class_selector() -> list[str]:
    """Interactive menu to select object classes."""
    console.print("\n[cyan]Select object classes:")

    for class_id, class_name in OBJECT_CLASS_NAMES.items():
        console.print(f"  {class_id}: {class_name}")

    console.print("\n[dim]Enter class numbers separated by commas (e.g., 0,1,4)")
    console.print("[dim]Or press Enter to select all classes")

    selection = Prompt.ask("Selection", default="")

    if not selection:
        return list(OBJECT_CLASS_NAMES.values())

    selected_classes = []
    for part in selection.split(","):
        try:
            class_id = int(part.strip())
            if class_id in OBJECT_CLASS_NAMES:
                selected_classes.append(OBJECT_CLASS_NAMES[class_id])
        except ValueError:
            continue

    return selected_classes
