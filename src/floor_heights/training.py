import json
import os
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from ultralytics import YOLO, settings

from floor_heights.config import CONFIG

os.environ["YOLO_CONFIG_DIR"] = str(CONFIG.project_root / "weights" / ".cache")
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(CONFIG.project_root / "weights" / ".cache")
settings.update({"weights_dir": str(CONFIG.project_root / "weights" / ".cache")})

console = Console()


def detect_hardware() -> str:
    if not torch.cuda.is_available():
        console.print("[yellow]No GPU detected, using CPU for training[/yellow]")
        return "cpu"

    gpu_count = torch.cuda.device_count()
    if gpu_count == 1:
        console.print(f"[green]Using 1 GPU: {torch.cuda.get_device_name(0)}[/green]")
        return "0"
    else:
        devices = ",".join(str(i) for i in range(gpu_count))
        console.print(f"[green]Using {gpu_count} GPUs[/green]")
        for i in range(gpu_count):
            console.print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return devices


def coco_to_yolo_bbox(bbox: list[float], img_width: int, img_height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height


def coco_to_yolo_polygon(segmentation: list[list[float]], img_width: int, img_height: int) -> list[float]:
    if not segmentation or not segmentation[0]:
        return []
    polygon = segmentation[0]
    normalized = []
    for i in range(0, len(polygon), 2):
        if i + 1 < len(polygon):
            x_norm = polygon[i] / img_width
            y_norm = polygon[i + 1] / img_height
            normalized.extend([x_norm, y_norm])

    return normalized


def prepare_yolo_dataset(
    annotations_path: Path, splits_path: Path, images_dir: Path, output_dir: Path, task: str = "segment"
) -> Path:
    console.print(f"[cyan]Preparing YOLO dataset for {task} task...[/cyan]")
    with annotations_path.open() as f:
        coco_data = json.load(f)

    with splits_path.open() as f:
        splits = json.load(f)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    coco_cat_ids = [cat["id"] for cat in coco_data["categories"]]
    cat_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(coco_cat_ids)}
    image_id_to_info = {img["id"]: img for img in coco_data["images"]}
    filename_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}
    image_annotations = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    stats = {"train": 0, "val": 0, "test": 0}

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        for split_name, file_list in splits.items():
            if split_name not in ["train", "val", "test"]:
                continue

            task_id = progress.add_task(f"Processing {split_name} split...", total=len(file_list))

            for filename in file_list:
                if filename not in filename_to_id:
                    progress.advance(task_id)
                    continue

                img_id = filename_to_id[filename]
                img_info = image_id_to_info[img_id]
                src_img = images_dir / filename
                dst_img = output_dir / split_name / "images" / filename

                if src_img.exists() and not dst_img.exists():
                    dst_img.symlink_to(src_img.resolve())
                label_file = output_dir / split_name / "labels" / f"{Path(filename).stem}.txt"

                if img_id in image_annotations:
                    with label_file.open("w") as f:
                        for ann in image_annotations[img_id]:
                            class_id = cat_id_to_yolo[ann["category_id"]]

                            if task == "segment" and "segmentation" in ann:
                                polygon = coco_to_yolo_polygon(
                                    ann["segmentation"], img_info["width"], img_info["height"]
                                )
                                if polygon:
                                    coords = " ".join(f"{v:.6f}" for v in polygon)
                                    f.write(f"{class_id} {coords}\n")
                            else:
                                x_center, y_center, width, height = coco_to_yolo_bbox(
                                    ann["bbox"], img_info["width"], img_info["height"]
                                )
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                else:
                    label_file.touch()

                stats[split_name] += 1
                progress.advance(task_id)

    console.print("\n[green]Dataset prepared:[/green]")
    for split, count in stats.items():
        console.print(f"  {split}: {count} images")
    data_yaml_path = create_data_yaml(output_dir, coco_data["categories"], task)
    return data_yaml_path


def create_data_yaml(output_dir: Path, categories: list[dict], task: str) -> Path:
    id2name = {}
    for idx, cat in enumerate(categories):
        id2name[idx] = cat["name"]
    names = [id2name[i] for i in range(len(categories))]

    data_config = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
        "nc": len(names),
    }

    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(data_config, f, sort_keys=False)

    console.print(f"[green]Created data.yaml with {len(names)} classes[/green]")
    return yaml_path


def train_model(
    data_yaml: Path,
    epochs: int = 100,
    batch: int = -1,
    task: str = "segment",
    resume: bool = False,
    name: str = "facade_seg",
    device: str | None = None,
    pretrained: Path | None = None,
) -> Path:
    if device is None:
        device = detect_hardware()
    if batch == -1 and "," in str(device):
        gpu_count = len(str(device).split(","))
        batch = 32 * gpu_count
        console.print(f"[yellow]Multi-GPU detected. Setting batch size to {batch} ({32} per GPU)[/yellow]")
    base_model = (
        str(pretrained)
        if pretrained and pretrained.exists() and pretrained.suffix == ".pt"
        else CONFIG.object_detection.training_model
        if task == "segment"
        else CONFIG.object_detection.training_model.replace("-seg", "")
    )

    if pretrained and pretrained.exists():
        if pretrained.suffix == ".ckpt":
            raise ValueError("Lightning checkpoint conversion not supported. Convert to .pt format first.")
        console.print(f"[green]Using pretrained model: {pretrained}[/green]")

    console.print(f"\n[cyan]Training YOLO {task} model[/cyan]")
    console.print(f"Base model: {base_model}")
    console.print(f"Device: {device}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch size: {'auto' if batch == -1 else batch}")
    model_cache_dir = CONFIG.project_root / "weights" / ".cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    if not Path(base_model).exists():
        cached_model = model_cache_dir / base_model
        if cached_model.exists():
            base_model = str(cached_model)
        else:
            import ultralytics.utils

            ultralytics.utils.SETTINGS["weights_dir"] = str(model_cache_dir)

    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=CONFIG.object_detection.training_imgsz,
        patience=CONFIG.object_detection.training_patience,
        device=device,
        project=str(CONFIG.output_root / "training" / "runs"),
        name=name,
        resume=resume,
        exist_ok=resume,
        hsv_h=CONFIG.object_detection.hsv_h,
        hsv_s=CONFIG.object_detection.hsv_s,
        hsv_v=CONFIG.object_detection.hsv_v,
        translate=CONFIG.object_detection.translate,
        scale=CONFIG.object_detection.scale,
        fliplr=CONFIG.object_detection.fliplr,
        degrees=CONFIG.object_detection.degrees,
        flipud=CONFIG.object_detection.flipud,
        mosaic=CONFIG.object_detection.mosaic,
        copy_paste=CONFIG.object_detection.copy_paste,
        save=True,
        save_period=10,
        val=True,
        amp=True,
        verbose=True,
    )
    best_weights = CONFIG.output_root / "training" / "runs" / name / "weights" / "best.pt"
    if best_weights.exists():
        console.print("\n[green]Training complete! Best model saved to:[/green]")
        console.print(f"[cyan]{best_weights}[/cyan]")
        console.print("\n[yellow]To use this model, copy it to your desired location or reference it directly[/yellow]")
        return best_weights
    else:
        raise FileNotFoundError(f"Training completed but best.pt not found at {best_weights}")


def masks_to_bboxes(masks: torch.Tensor) -> list[list[float]]:
    bboxes = []

    for mask in masks:
        points = torch.nonzero(mask)

        if len(points) == 0:
            bboxes.append([0, 0, 0, 0])
            continue

        y_min = points[:, 0].min().item()
        y_max = points[:, 0].max().item()
        x_min = points[:, 1].min().item()
        x_max = points[:, 1].max().item()

        bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes
