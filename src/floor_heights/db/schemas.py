from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StageRecordBase(BaseModel):
    id: int
    building_id: str
    region_name: str
    gnaf_id: str | None = None
    processed_at: datetime = Field(default_factory=datetime.now)


class Stage01ClipRecord(StageRecordBase):
    clip_path: str
    tile_count: int


class Stage02aBuildingRecord(StageRecordBase):
    candidate_count: int
    chosen_count: int


class Stage02aBuildingGeomRecord(StageRecordBase):
    geom_type: str
    view_type: str
    edge_idx: int
    geom_wkt: str


class Stage02aDerivedGeomRecord(StageRecordBase):
    geom_type: str
    view_type: str
    edge_idx: int
    geom_wkt: str


class Stage02aPanoramaRecord(StageRecordBase):
    pano_id: str
    edge_idx: int
    view_type: str
    distance: float
    heading_diff: float
    coverage_score: float
    quality_score: float
    capture_date: datetime


class Stage02aCandidateViewRecord(StageRecordBase):
    pano_id: str
    edge_idx: int
    view_type: str
    distance: float
    angle: float
    is_chosen: bool
    ray_wkt: str
    pano_lat: float
    pano_lon: float
    pano_heading: float
    footprint_geom_mga: str


class Stage02bDownloadRecord(StageRecordBase):
    pano_id: str
    edge_idx: int
    view_type: str
    download_path: str | None = None


class Stage03ClipRecord(StageRecordBase):
    pano_id: str
    edge_idx: int
    view_type: str
    clip_path: str
    clip_left: float
    clip_right: float
    clip_top: float
    clip_bottom: float


class Stage04aClipRecord(StageRecordBase):
    pano_id: str
    edge_idx: int
    view_type: str
    clip_path: str
    detection_count: int = 0


class Stage04aDetectionRecord(StageRecordBase):
    pano_id: str
    edge_idx: int
    view_type: str
    clip_path: str
    class_id: int
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    bbox_center_x: float
    bbox_center_y: float
    image_width: int
    image_height: int


class Stage04bBestViewRecord(StageRecordBase):
    pano_id: str | None = None
    edge_idx: int | None = None
    view_type: str | None = None
    detection_score: float = 0.0
    siglip_score: float = 0.0
    combined_score: float = 0.0
    distance: float = 0.0
    siglip_weight: float = 0.0
    has_door_override: bool = False
    diversity_bonus: float = 0.0
    view_angle_bonus: float = 0.0
    has_ground_feature: bool = False
    has_entrance_feature: bool = False
    selection_type: str | None = None
    status: str = "success"
    error_message: str | None = None
    clip_image_path: str | None = None
    ground_visibility_score: float = 0.0


class Stage05ProjectionRecord(StageRecordBase):
    pano_id: str
    edge_idx: int
    view_type: str
    projection_path: str | None = None
    point_count: int = 0
    coverage_percent: float = 0.0


class Stage06GroundElevationRecord(StageRecordBase):
    ground_elevation_m: float | None = None
    ground_points_count: int = 0
    confidence_score: float | None = None
    method: str | None = None


class Stage07FloorHeightRecord(StageRecordBase):
    ffh1: float | None = None
    ffh2: float | None = None
    ffh3: float | None = None
    method: str | None = None


class Stage08ValidationRecord(StageRecordBase):
    ffh_method: str
    predicted_ffh: float
    ground_truth_ffh: float
    error: float
    absolute_error: float
    squared_error: float


class Stage09aLidarStatsRecord(StageRecordBase):
    las_path: str
    point_count: int
    footprint_area: float
    point_density: float
    building_height: float
    convex_hull_area: float
    coverage_ratio: float
    spatial_coverage: float
    z_min: float
    z_max: float
    z_range: float
    z_mean: float
    z_std: float
    z_p10: float
    z_p25: float
    z_p50: float
    z_p75: float
    z_p90: float
    pts_0_3m: int
    pts_3_6m: int
    pts_6_9m: int
    pts_9_12m: int
    pts_above_12m: int
    returns_single: int
    returns_multiple: int
    intensity_mean: float
    intensity_std: float
    ground_point_count: int
    ground_z_mean: float
    ground_z_std: float
    noise_point_count: int
    roof_z_variance: float
    vegetation_proximity_count: int
    verticality_score: float
    planarity_score: float
    building_height_peaks: int
    building_height_regularity: float
    facade_alignment_score: float
    building_density_0_3m: float
    building_density_3_6m: float
    building_density_6_9m: float
    building_density_9_12m: float
    building_density_12_15m: float
    building_density_15_20m: float
    building_density_20_30m: float
    building_density_30_50m: float
    multi_return_ratio_0_10m: float
    multi_return_ratio_10_20m: float
    multi_return_ratio_20_30m: float
    multi_return_ratio_30_50m: float
    ground_height_variance: float
    ground_height_iqr: float
    building_height_variance: float
    building_height_iqr: float
    class_0_never: int
    class_1_unassigned: int
    class_2_ground: int
    class_3_low_veg: int
    class_4_med_veg: int
    class_5_high_veg: int
    class_6_building: int
    class_7_noise: int


STAGE_SCHEMAS = {
    "stage01_clips": {
        "model": Stage01ClipRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "clip_path": "str",
            "tile_count": "int32",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage02a_buildings": {
        "model": Stage02aBuildingRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "candidate_count": "int32",
            "chosen_count": "int32",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage02a_building_geoms": {
        "model": Stage02aBuildingGeomRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "geom_type": "str",
            "view_type": "str",
            "edge_idx": "int32",
            "geom_wkt": "str",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage02a_derived_geoms": {
        "model": Stage02aDerivedGeomRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "geom_type": "str",
            "view_type": "str",
            "edge_idx": "int32",
            "geom_wkt": "str",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage02a_panoramas": {
        "model": Stage02aPanoramaRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "pano_id": "str",
            "edge_idx": "int32",
            "view_type": "str",
            "distance": "float32",
            "heading_diff": "float32",
            "coverage_score": "float32",
            "quality_score": "float32",
            "capture_date": "datetime64[ns]",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage02a_candidate_views": {
        "model": Stage02aCandidateViewRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "pano_id": "str",
            "edge_idx": "int32",
            "view_type": "str",
            "distance": "float64",
            "angle": "float64",
            "is_chosen": "bool",
            "ray_wkt": "str",
            "pano_lat": "float64",
            "pano_lon": "float64",
            "pano_heading": "float64",
            "footprint_geom_mga": "str",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage02b_downloads": {
        "model": Stage02bDownloadRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "pano_id": "str",
            "edge_idx": "int32",
            "view_type": "str",
            "download_path": "str",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage03_clips": {
        "model": Stage03ClipRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "pano_id": "str",
            "edge_idx": "int32",
            "view_type": "str",
            "clip_path": "str",
            "clip_left": "float32",
            "clip_right": "float32",
            "clip_top": "float32",
            "clip_bottom": "float32",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage04a_clips": {
        "model": Stage04aClipRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "pano_id": "str",
            "edge_idx": "int32",
            "view_type": "str",
            "clip_path": "str",
            "detection_count": "int32",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage04a_detections": {
        "model": Stage04aDetectionRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "pano_id": "str",
            "edge_idx": "int32",
            "view_type": "str",
            "clip_path": "str",
            "class_id": "int32",
            "class_name": "str",
            "confidence": "float32",
            "bbox_x1": "float32",
            "bbox_y1": "float32",
            "bbox_x2": "float32",
            "bbox_y2": "float32",
            "bbox_width": "float32",
            "bbox_height": "float32",
            "bbox_area": "float32",
            "bbox_center_x": "float32",
            "bbox_center_y": "float32",
            "image_width": "int32",
            "image_height": "int32",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage04b_best_views": {
        "model": Stage04bBestViewRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "pano_id": "str",
            "edge_idx": "int32",
            "view_type": "str",
            "detection_score": "float32",
            "siglip_score": "float32",
            "combined_score": "float32",
            "distance": "float32",
            "siglip_weight": "float32",
            "has_door_override": "bool",
            "diversity_bonus": "float32",
            "view_angle_bonus": "float32",
            "has_ground_feature": "bool",
            "has_entrance_feature": "bool",
            "selection_type": "str",
            "status": "str",
            "error_message": "str",
            "clip_image_path": "str",
            "ground_visibility_score": "float32",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage05_projections": {
        "model": Stage05ProjectionRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "pano_id": "str",
            "edge_idx": "int32",
            "view_type": "str",
            "projection_path": "str",
            "point_count": "int32",
            "coverage_percent": "float32",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage06_ground_elevations": {
        "model": Stage06GroundElevationRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "ground_elevation_m": "float32",
            "ground_points_count": "int32",
            "confidence_score": "float32",
            "method": "str",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage07_floor_heights": {
        "model": Stage07FloorHeightRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "ffh1": "float32",
            "ffh2": "float32",
            "ffh3": "float32",
            "method": "str",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage08_validation": {
        "model": Stage08ValidationRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "ffh_method": "str",
            "predicted_ffh": "float32",
            "ground_truth_ffh": "float32",
            "error": "float32",
            "absolute_error": "float32",
            "squared_error": "float32",
            "processed_at": "datetime64[ns]",
        },
    },
    "stage09a_lidar_stats": {
        "model": Stage09aLidarStatsRecord,
        "columns": {
            "id": "int64",
            "building_id": "str",
            "region_name": "str",
            "gnaf_id": "str",
            "las_path": "str",
            "point_count": "int32",
            "footprint_area": "float32",
            "point_density": "float32",
            "building_height": "float32",
            "convex_hull_area": "float32",
            "coverage_ratio": "float32",
            "spatial_coverage": "float32",
            "z_min": "float32",
            "z_max": "float32",
            "z_range": "float32",
            "z_mean": "float32",
            "z_std": "float32",
            "z_p10": "float32",
            "z_p25": "float32",
            "z_p50": "float32",
            "z_p75": "float32",
            "z_p90": "float32",
            "pts_0_3m": "int32",
            "pts_3_6m": "int32",
            "pts_6_9m": "int32",
            "pts_9_12m": "int32",
            "pts_above_12m": "int32",
            "returns_single": "int32",
            "returns_multiple": "int32",
            "intensity_mean": "float32",
            "intensity_std": "float32",
            "ground_point_count": "int32",
            "ground_z_mean": "float32",
            "ground_z_std": "float32",
            "noise_point_count": "int32",
            "roof_z_variance": "float32",
            "vegetation_proximity_count": "int32",
            "verticality_score": "float32",
            "planarity_score": "float32",
            "building_height_peaks": "int32",
            "building_height_regularity": "float32",
            "facade_alignment_score": "float32",
            "building_density_0_3m": "float32",
            "building_density_3_6m": "float32",
            "building_density_6_9m": "float32",
            "building_density_9_12m": "float32",
            "building_density_12_15m": "float32",
            "building_density_15_20m": "float32",
            "building_density_20_30m": "float32",
            "building_density_30_50m": "float32",
            "multi_return_ratio_0_10m": "float32",
            "multi_return_ratio_10_20m": "float32",
            "multi_return_ratio_20_30m": "float32",
            "multi_return_ratio_30_50m": "float32",
            "ground_height_variance": "float32",
            "ground_height_iqr": "float32",
            "building_height_variance": "float32",
            "building_height_iqr": "float32",
            "class_0_never": "int32",
            "class_1_unassigned": "int32",
            "class_2_ground": "int32",
            "class_3_low_veg": "int32",
            "class_4_med_veg": "int32",
            "class_5_high_veg": "int32",
            "class_6_building": "int32",
            "class_7_noise": "int32",
            "processed_at": "datetime64[ns]",
        },
    },
}


def get_stage_schema(stage_name: str) -> dict[str, Any]:
    if stage_name not in STAGE_SCHEMAS:
        raise ValueError(f"Unknown stage: {stage_name}")
    return STAGE_SCHEMAS[stage_name]["columns"]


def get_stage_model(stage_name: str) -> type[BaseModel]:
    if stage_name not in STAGE_SCHEMAS:
        raise ValueError(f"Unknown stage: {stage_name}")
    return STAGE_SCHEMAS[stage_name]["model"]


class BatchWriter:
    def __init__(self, table_name: str, batch_size: int = 1000, progress_tracker=None):
        self.table_name = table_name
        self.batch_size = batch_size
        self.buffer: list[BaseModel] = []
        self.model_class = get_stage_model(table_name)
        self.progress_tracker = progress_tracker
        self.total_written = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.buffer:
            self._flush()

    def add(self, record: BaseModel) -> None:
        """Adds a Pydantic model record to the buffer."""
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self._flush()

    def _flush(self):
        if self.buffer:
            from floor_heights.utils.fh_io import save_stage_result_batch

            dict_buffer = [r.model_dump() for r in self.buffer]
            save_stage_result_batch(self.table_name, dict_buffer)

            write_count = len(self.buffer)
            self.total_written += write_count
            if self.progress_tracker and hasattr(self.progress_tracker, "fields"):
                self.progress_tracker.fields["writes"] += write_count
                if hasattr(self.progress_tracker, "progress") and self.progress_tracker.task_id is not None:
                    self.progress_tracker.progress.update(self.progress_tracker.task_id, **self.progress_tracker.fields)

            self.buffer.clear()


def initialize_all_stage_tables():
    """Initialize all stage tables in the database."""
    from floor_heights.utils.fh_io import ensure_stage_table

    for table_name, schema_def in STAGE_SCHEMAS.items():
        ensure_stage_table(table_name, schema_def["columns"])
