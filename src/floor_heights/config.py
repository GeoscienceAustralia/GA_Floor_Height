from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


REGION_CONFIGS: dict[str, Any] = _load_yaml(Path(__file__).with_name("region_configs.yaml"))


@dataclass(frozen=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class RegionConfig:
    name: str
    id: str
    crs_projected: int
    bbox: BoundingBox
    crs_source: int = 7844
    tile_index_path: str | None = None
    lidar_prefix: str | None = None
    trajectory_path: str | None = None

    @property
    def crs_epsg(self) -> str:
        return f"EPSG:{self.crs_projected}"


@dataclass(frozen=True)
class StageConstants:
    buffer_m: float = 5.0
    lidar_data_root_name: str | None = None
    clip_upper_prop: float = 0.25
    clip_lower_prop: float = 0.60
    angle_extend: float = 40.0
    max_image_pixels: int = 300_000_000
    default_workers: int = -1
    chunk_size: int = 1000
    timeout_seconds: int = 300
    s3_bucket: str = "frontiersi-p127-floor-height-woolpert"
    s3_region: str = "ap-southeast-2"
    default_lidar_source: str = "s3"


@dataclass(frozen=True)
class CRSConfig:
    geographic: str = "EPSG:7844"
    wgs84: str = "EPSG:4326"


@dataclass(frozen=True)
class PDALConfig:
    scale_x: float = 0.001
    scale_y: float = 0.001
    scale_z: float = 0.001
    offset_x: str = "auto"
    offset_y: str = "auto"
    offset_z: str = "auto"
    extra_dims: str = "all"
    forward: str = "all"
    simplification_tolerance: float = 0.1
    buffer_join_style: int = 2


@dataclass(frozen=True)
class PanoramaConfig:
    max_distance_m: float = 40.0
    direct_tolerance_deg: float = 15.0
    min_semi_oblique_deg: float = 20.0
    max_semi_oblique_deg: float = 40.0
    min_oblique_deg: float = 45.0
    max_oblique_deg: float = 50.0
    eps: float = 1e-8
    eps_len: float = 0.5
    eps_area: float = 1e-4
    sector_polygon_points: int = 16
    edge_length_threshold: float = 1e-6
    ray_shortening_factor: float = 0.999
    visibility_distance_threshold: float = 0.001


@dataclass(frozen=True)
class ClippingConfig:
    span_buffer_percent: float = 0.05
    min_buffer_pixels: int = 20
    jpeg_quality: int = 95
    jpeg_optimize: bool = True


@dataclass(frozen=True)
class ObjectDetectionConfig:
    model_path: str = "weights/best.pt"
    confidence_threshold: float = 0.25
    batch_size: int = 1000
    items_per_batch: int = 50
    workers_per_gpu: int = 8
    visualization_quality: int = 100

    # Training configuration
    training_model: str = "yolov8x-seg.pt"  # Segmentation base model
    training_epochs: int = 100
    training_batch: int = -1  # Auto batch size
    training_imgsz: int = 1280  # Higher res for better facade detail
    training_patience: int = 50  # Early stopping

    # Augmentations suitable for street view facades
    hsv_h: float = 0.015  # Slight hue variation (lighting conditions)
    hsv_s: float = 0.7  # Saturation (weather conditions)
    hsv_v: float = 0.4  # Brightness (time of day)
    translate: float = 0.1  # Small translations (camera position variance)
    scale: float = 0.5  # Increased scale for better distance variation
    fliplr: float = 0.5  # Horizontal flip (buildings on either side)
    degrees: float = 0.0  # No rotation (street view is level)
    flipud: float = 0.0  # No vertical flip
    mosaic: float = 0.0  # No mosaic (we want clear building views)
    copy_paste: float = 0.1  # Copy-paste augmentation for small dataset


@dataclass(frozen=True)
class ProjectionConfig:
    panorama_width: int = 11000
    panorama_height: int = 5500
    downscale_factor: int = 8
    max_hole_size: int = 10
    nodata_float: float = 9999.0
    nodata_int: int = 255
    elevation_resample_order: int = 1
    intensity_resample_order: int = 1
    depth_resample_order: int = 1
    classification_resample_order: int = 0


@dataclass(frozen=True)
class GroundElevationConfig:
    dtm_resolution: float = 0.1
    generate_dtm: bool = True
    csf_resolution: float = 1.0
    csf_hdiff: float = 0.5
    csf_smooth: bool = False
    ground_classification: int = 2
    noise_classification: int = 7
    min_points_for_dtm: int = 10
    min_extent_for_dtm: float = 0.3
    dtm_nodata: float = -9999.0
    batch_size: int = 1000


@dataclass(frozen=True)
class FrontDoorStandards:
    width_m: float = 0.82
    height_m: float = 2.04
    area_m2: float = 1.67
    ratio: float = 0.40


@dataclass(frozen=True)
class FeatureWeights:
    area_m2: float = 1.0
    ratio: float = 1.0
    confidence: float = 1.0
    x_location: float = 1.0
    y_location: float = 1.0


@dataclass(frozen=True)
class FFHEstimationConfig:
    min_ffh: float = 0.0
    max_ffh: float = 2.0
    nodata_depth: float = 9999.0
    nodata_elevation: float = 9999.0
    min_ground_area_pixels: int = 640
    batch_size: int = 100
    target_classes: list[str] = field(default_factory=lambda: ["Foundation", "Front Door", "Garage Door", "Stairs"])
    frontdoor_standards: FrontDoorStandards = field(default_factory=FrontDoorStandards)
    feature_weights: FeatureWeights = field(default_factory=FeatureWeights)


@dataclass(frozen=True)
class S3PanoramaConfig:
    wagga: str = "01_WaggaWagga/01_StreetViewImagery/"
    tweed: str = "02_TweedHeads/01_StreetViewImagery/"
    launceston: str = "03_Launceston/01_StreetViewImagery/"


@dataclass(frozen=True)
class SigLIPConfig:
    model_name: str = "google/siglip2-so400m-patch14-384"
    tokenizer_max_length: int = 64
    tokenizer_padding: str = "max_length"
    variance_threshold: float = 0.05
    high_variance_weight: float = 2.5
    low_variance_weight: float = 3.0
    detection_threshold: float = 0.15
    diversity_bonus: float = 1.5
    direct_view_bonus: float = 0.15
    oblique_view_penalty: float = -0.1
    scoring_weights: dict[str, float] = field(
        default_factory=lambda: {"detection": 0.35, "siglip": 0.20, "ground": 0.20, "focus": 0.15, "view": 0.10}
    )


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    output_root: Path
    db_path: Path
    lidar_data_root: Path | None
    regions: dict[str, RegionConfig]
    constants: StageConstants = field(default_factory=StageConstants)
    crs: CRSConfig = field(default_factory=CRSConfig)
    pdal: PDALConfig = field(default_factory=PDALConfig)
    panorama: PanoramaConfig = field(default_factory=PanoramaConfig)
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    object_detection: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    ground_elevation: GroundElevationConfig = field(default_factory=GroundElevationConfig)
    ffh_estimation: FFHEstimationConfig = field(default_factory=FFHEstimationConfig)
    s3_panorama: S3PanoramaConfig = field(default_factory=S3PanoramaConfig)
    siglip: SigLIPConfig = field(default_factory=SigLIPConfig)

    @property
    def yolo_model_path(self) -> Path:
        return self.project_root / self.object_detection.model_path

    @classmethod
    def load(cls) -> PipelineConfig:
        project_root = Path(__file__).resolve().parents[2]

        output_root_env = os.getenv("FH_OUTPUT_ROOT")
        if output_root_env is None or output_root_env.startswith("/path/to/"):
            raise ValueError(
                "FH_OUTPUT_ROOT not configured. Please update .env file with a valid output directory path.\n"
                "Example: FH_OUTPUT_ROOT=./output or FH_OUTPUT_ROOT=/absolute/path/to/output"
            )
        output_root = Path(output_root_env)

        db_path_env = os.getenv("FH_DB_PATH")
        if db_path_env is None or db_path_env.startswith("/path/to/"):
            raise ValueError(
                "FH_DB_PATH not configured. Please update .env file with a valid database path.\n"
                "Example: FH_DB_PATH=./data/floor_heights.duckdb"
            )
        db_path = Path(db_path_env)

        lidar_root_env = os.getenv("FH_LIDAR_DATA_ROOT")
        lidar_data_root = Path(lidar_root_env) if lidar_root_env else None

        constants = StageConstants(lidar_data_root_name=str(lidar_data_root) if lidar_data_root else None)

        available_regions = list(REGION_CONFIGS.keys())
        selected_regions = os.getenv("FH_REGIONS", ",".join(available_regions)).split(",")
        selected_regions = [r.strip() for r in selected_regions if r.strip()]

        regions = {}

        for region_name in selected_regions:
            if region_name in REGION_CONFIGS:
                reg_cfg = REGION_CONFIGS[region_name]

                regions[region_name] = RegionConfig(
                    name=region_name.capitalize(),
                    id=reg_cfg["id"],
                    crs_projected=reg_cfg["crs"],
                    crs_source=reg_cfg.get("crs_source", 7844),
                    bbox=BoundingBox(**reg_cfg["bbox"]),
                    tile_index_path=reg_cfg.get("tile_index"),
                    lidar_prefix=reg_cfg.get("lidar_prefix"),
                    trajectory_path=reg_cfg.get("trajectory"),
                )

        return cls(
            project_root=project_root,
            output_root=output_root,
            db_path=db_path,
            lidar_data_root=lidar_data_root,
            regions=regions,
            constants=constants,
        )

    def get_region(self, region_name: str) -> RegionConfig:
        if region_name not in self.regions:
            raise KeyError(f"Region '{region_name}' not found. Available: {list(self.regions.keys())}")
        return self.regions[region_name]

    def region_folder(self, region_name: str) -> Path:
        return self.output_root / self.get_region(region_name).name

    @property
    def region_names(self) -> list[str]:
        return list(self.regions.keys())

    def get_region_crs_map(self) -> dict[str, int]:
        return {name: region.crs_projected for name, region in self.regions.items()}


CONFIG = PipelineConfig.load()

REGIONS = CONFIG.region_names
OUTPUT_ROOT = CONFIG.output_root
DB_PATH = CONFIG.db_path
REGION_CRS = CONFIG.get_region_crs_map()
