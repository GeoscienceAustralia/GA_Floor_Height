#!/usr/bin/env python
"""Visualization utilities for panorama clipping and analysis."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Point
from shapely.wkt import loads as wkt_loads

matplotlib.use("Agg")


COLOR_DSM_LINE = (139, 69, 19, 255)
COLOR_FIXED_LINE = (255, 140, 0, 255)
COLOR_VISIBILITY_RECT = (255, 0, 0, 255)
COLOR_CENTER_LINE = (255, 255, 0, 255)
COLOR_VISIBLE_VERTEX = (0, 0, 255, 255)
COLOR_HIDDEN_VERTEX = (128, 128, 128, 255)
COLOR_SELECTED_VERTEX = (0, 255, 0, 255)


def get_visualization_path(
    output_root: Path, region: str, building_id: int, gnaf_id: str, stage: str, filename: str
) -> Path:
    """Get standardized visualization path for any pipeline stage.

    Args:
        output_root: Base output directory
        region: Region name
        building_id: Building ID
        gnaf_id: GNAF ID (or 'NO_GNAF')
        stage: Pipeline stage name (e.g., 'stage03_clip')
        filename: Output filename

    Returns:
        Path to visualization file
    """
    viz_dir = output_root / region.capitalize() / "visualizations" / f"{building_id}_{gnaf_id}" / stage
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir / filename


def create_panorama_clip_visualization(
    img: Image.Image,
    pano_lat: float,
    pano_lon: float,
    building_polygon_wkt: str,
    heading: float,
    distance: float,
    h_range: tuple[float, float],
    v_range: tuple[int, int] | None,
    elevations: tuple[float, float] | None,
    geo_module,
    clip_upper_prop: float = 0.25,
    clip_lower_prop: float = 0.60,
) -> Image.Image:
    """Create visualization showing clipping bounds and building vertices.

    Returns composite image with panorama view and spatial map.
    """
    viz_img = img.copy()
    draw = ImageDraw.Draw(viz_img, "RGBA")

    adjusted_heading = -heading
    if adjusted_heading < 0:
        adjusted_heading += 360

    building_polygon = wkt_loads(building_polygon_wkt)
    pano_point = Point(pano_lon, pano_lat)

    if v_range is not None:
        top_y, bottom_y = v_range
        fixed_top = int(img.height * clip_upper_prop)
        fixed_bottom = int(img.height * clip_lower_prop)

        draw.line([(0, top_y), (img.width, top_y)], fill=COLOR_DSM_LINE, width=4)
        draw.line([(0, bottom_y), (img.width, bottom_y)], fill=COLOR_DSM_LINE, width=4)

        dash_length = 20
        gap_length = 10
        for x in range(0, img.width, dash_length + gap_length):
            draw.line([(x, fixed_top), (min(x + dash_length, img.width), fixed_top)], fill=COLOR_FIXED_LINE, width=3)
            draw.line(
                [(x, fixed_bottom), (min(x + dash_length, img.width), fixed_bottom)], fill=COLOR_FIXED_LINE, width=3
            )
    else:
        top_y = int(img.height * clip_upper_prop)
        bottom_y = int(img.height * clip_lower_prop)
        draw.line([(0, top_y), (img.width, top_y)], fill=COLOR_FIXED_LINE, width=3)
        draw.line([(0, bottom_y), (img.width, bottom_y)], fill=COLOR_FIXED_LINE, width=3)

    left_x, right_x = h_range
    draw.rectangle([(left_x - 2, top_y - 2), (right_x + 2, bottom_y + 2)], outline=(0, 0, 0, 128), width=2)
    draw.rectangle([(left_x, top_y), (right_x, bottom_y)], outline=COLOR_VISIBILITY_RECT, width=5)

    centroid = building_polygon.centroid
    centroid_result = geo_module.localize_house_in_panorama(
        lat_c=pano_lat,
        lon_c=pano_lon,
        lat_house=centroid.y,
        lon_house=centroid.x,
        beta_yaw_deg=adjusted_heading,
        wim=img.width,
        angle_extend=0,
    )
    center_x = centroid_result["horizontal_pixel_house"]

    for y in range(0, img.height, 15):
        draw.line([(center_x, y), (center_x, min(y + 8, img.height))], fill=COLOR_CENTER_LINE, width=3)

    vertices_info = _process_vertices_visibility(
        building_polygon, pano_point, pano_lat, pano_lon, adjusted_heading, img.width, geo_module
    )

    visible_vertices = [v for v in vertices_info if v["is_visible"]]
    if visible_vertices:
        leftmost = min(visible_vertices, key=lambda v: v["vertex_x"])
        rightmost = max(visible_vertices, key=lambda v: v["vertex_x"])
    else:
        leftmost = rightmost = None

    for v_info in vertices_info:
        vertex_x = v_info["vertex_x"]

        if not v_info["is_visible"]:
            for y in range(0, img.height, 20):
                draw.line([(vertex_x, y), (vertex_x, min(y + 8, img.height))], fill=COLOR_HIDDEN_VERTEX, width=2)
        elif v_info in (leftmost, rightmost):
            draw.line([(vertex_x, 0), (vertex_x, img.height)], fill=COLOR_SELECTED_VERTEX, width=5)
        else:
            draw.line([(vertex_x, 0), (vertex_x, img.height)], fill=COLOR_VISIBLE_VERTEX, width=3)

    _add_text_annotations(draw, distance, heading, adjusted_heading, elevations)

    _add_legend(draw, img, v_range is not None)

    map_img = _create_spatial_map(
        pano_lat,
        pano_lon,
        building_polygon,
        vertices_info,
        leftmost,
        rightmost,
        heading,
        distance,
        adjusted_heading,
        geo_module,
    )

    return _create_composite_image(viz_img, map_img)


def _process_vertices_visibility(
    building_polygon, pano_point, pano_lat, pano_lon, adjusted_heading, img_width, geo_module
) -> list:
    """Process building vertices and determine visibility from panorama."""
    vertices = list(building_polygon.exterior.coords[:-1])
    vertices_info = []

    for vertex in vertices:
        vertex_point = Point(vertex)
        ray = LineString([pano_point, vertex_point])

        ray_coords = list(ray.coords)
        shortened_ray = LineString(
            [
                ray_coords[0],
                (
                    ray_coords[0][0] + 0.999 * (ray_coords[1][0] - ray_coords[0][0]),
                    ray_coords[0][1] + 0.999 * (ray_coords[1][1] - ray_coords[0][1]),
                ),
            ]
        )

        intersection = shortened_ray.intersection(building_polygon)

        is_visible = intersection.is_empty or (
            intersection.geom_type == "Point" and vertex_point.distance(intersection) > 0.001
        )

        vert_result = geo_module.localize_house_in_panorama(
            lat_c=pano_lat,
            lon_c=pano_lon,
            lat_house=vertex[1],
            lon_house=vertex[0],
            beta_yaw_deg=adjusted_heading,
            wim=img_width,
            angle_extend=0,
        )

        vertex_x = vert_result["horizontal_pixel_house"]
        vertices_info.append({"vertex": vertex, "vertex_x": vertex_x, "is_visible": is_visible})

    return vertices_info


def _add_text_annotations(draw, distance, heading, adjusted_heading, elevations):
    """Add text annotations to the visualization."""
    font = ImageFont.load_default()

    info_lines = []
    info_lines.append(f"Distance: {distance:.1f}m  |  Heading: {heading:.1f}° → {adjusted_heading:.1f}°")

    if elevations:
        ground_elev, roof_elev = elevations
        height = roof_elev - ground_elev

        camera_elev = ground_elev + 2.5
        pitch_to_roof = np.degrees(np.arctan2(roof_elev - camera_elev, distance))
        pitch_to_ground = np.degrees(np.arctan2(ground_elev - camera_elev, distance))

        info_lines.append(f"Ground: {ground_elev:.1f}m  |  Roof: {roof_elev:.1f}m  |  Height: {height:.1f}m")
        info_lines.append(f"Pitch to roof: {pitch_to_roof:.1f}°  |  Pitch to ground: {pitch_to_ground:.1f}°")

    text_x = 15
    text_y = 15
    for line in info_lines:
        bbox = draw.textbbox((text_x, text_y), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle(
            [(text_x - 5, text_y - 3), (text_x + text_width + 5, text_y + text_height + 3)],
            fill=(255, 255, 255, 255),
            outline=(0, 0, 0, 255),
            width=1,
        )

        draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font)
        text_y += text_height + 8


def _add_legend(draw, img, has_dsm_lines):
    """Add legend to the visualization."""
    font = ImageFont.load_default()

    legend_items = []
    if has_dsm_lines:
        legend_items.append(("DSM roof/ground lines", COLOR_DSM_LINE, "solid", 4))
        legend_items.append(("Fixed 25%/60% lines", COLOR_FIXED_LINE, "dashed", 3))
    else:
        legend_items.append(("Fixed 25%/60% lines", COLOR_FIXED_LINE, "solid", 3))
    legend_items.append(("Clipping box", COLOR_VISIBILITY_RECT, "rect", 4))
    legend_items.append(("Building center", COLOR_CENTER_LINE, "dashed", 3))
    legend_items.append(("Selected vertices (min/max)", COLOR_SELECTED_VERTEX, "solid", 5))
    legend_items.append(("Visible vertices", COLOR_VISIBLE_VERTEX, "solid", 3))
    legend_items.append(("Hidden vertices", COLOR_HIDDEN_VERTEX, "dashed", 2))

    legend_padding = 10
    legend_item_height = 20
    legend_width = 200
    legend_height = len(legend_items) * legend_item_height + 2 * legend_padding

    legend_x = img.width - legend_width - 20
    legend_y = img.height - legend_height - 20

    draw.rectangle(
        [(legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height)],
        fill=(255, 255, 255, 255),
        outline=(0, 0, 0, 255),
        width=2,
    )

    item_y = legend_y + legend_padding
    for label, color, style, width in legend_items:
        line_x1 = legend_x + legend_padding
        line_x2 = line_x1 + 30
        line_y = item_y + legend_item_height // 2

        if style == "rect":
            draw.rectangle([(line_x1, line_y - 5), (line_x2, line_y + 5)], outline=color, width=width)
        elif style == "dashed":
            dash_len = 8 if width > 2 else 6
            gap_len = 4 if width > 2 else 3
            for x in range(line_x1, line_x2, dash_len + gap_len):
                draw.line([(x, line_y), (min(x + dash_len, line_x2), line_y)], fill=color, width=width)
        else:
            draw.line([(line_x1, line_y), (line_x2, line_y)], fill=color, width=width)

        draw.text((line_x2 + 10, item_y), label, fill=(0, 0, 0), font=font)
        item_y += legend_item_height


def _create_spatial_map(
    pano_lat,
    pano_lon,
    building_polygon,
    vertices_info,
    leftmost,
    rightmost,
    heading,
    distance,
    adjusted_heading,
    geo_module,
) -> Image.Image:
    """Create spatial map showing building and panorama location."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    Point(pano_lon, pano_lat)

    building_coords = list(building_polygon.exterior.coords)
    building_patch = MplPolygon(
        [(c[0], c[1]) for c in building_coords], facecolor="red", alpha=0.2, edgecolor="red", linewidth=2
    )
    ax.add_patch(building_patch)

    ax.plot(
        pano_lon,
        pano_lat,
        "ko",
        markersize=10,
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=5,
    )

    for v_info in vertices_info:
        vertex = v_info["vertex"]

        if not v_info["is_visible"]:
            ax.plot(vertex[0], vertex[1], "o", color="gray", markersize=6, alpha=0.5, zorder=3)
        elif v_info in (leftmost, rightmost):
            ax.plot(
                vertex[0],
                vertex[1],
                "o",
                color="lime",
                markersize=10,
                markeredgecolor="darkgreen",
                markeredgewidth=2,
                zorder=5,
            )
            ax.plot([pano_lon, vertex[0]], [pano_lat, vertex[1]], "g-", alpha=0.5, linewidth=2, zorder=2)
        else:
            ax.plot(
                vertex[0],
                vertex[1],
                "o",
                color="blue",
                markersize=8,
                markeredgecolor="darkblue",
                markeredgewidth=1,
                zorder=4,
            )
            ax.plot([pano_lon, vertex[0]], [pano_lat, vertex[1]], "b--", alpha=0.3, linewidth=1, zorder=2)

    arrow_length = distance * 0.3 / 111000
    arrow_end_lon = pano_lon + arrow_length * np.cos(np.radians(90 - heading))
    arrow_end_lat = pano_lat + arrow_length * np.sin(np.radians(90 - heading))
    ax.annotate(
        "",
        xy=(arrow_end_lon, arrow_end_lat),
        xytext=(pano_lon, pano_lat),
        arrowprops={"arrowstyle": "->", "color": "blue", "lw": 2},
    )

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(f"Spatial View - Distance: {distance:.1f}m", fontsize=12)

    vertices = list(building_polygon.exterior.coords)
    all_lons = [pano_lon] + [v[0] for v in vertices]
    all_lats = [pano_lat] + [v[1] for v in vertices]
    lon_margin = (max(all_lons) - min(all_lons)) * 0.2
    lat_margin = (max(all_lats) - min(all_lats)) * 0.2
    ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
    ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)

    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    map_img = Image.open(buf)
    plt.close(fig)

    return map_img


def _create_composite_image(viz_img: Image.Image, map_img: Image.Image) -> Image.Image:
    """Create composite image with panorama and map side by side."""
    pano_width = viz_img.width
    pano_height = viz_img.height
    map_width = map_img.width
    map_height = map_img.height

    if map_height != pano_height:
        scale_factor = pano_height / map_height
        new_map_width = int(map_width * scale_factor)
        map_img = map_img.resize((new_map_width, pano_height), Image.Resampling.LANCZOS)
        map_width = new_map_width

    composite_width = pano_width + map_width + 20
    composite_height = pano_height
    composite_img = Image.new("RGB", (composite_width, composite_height), "white")

    composite_img.paste(viz_img, (0, 0))
    composite_img.paste(map_img, (pano_width + 20, 0))

    separator_draw = ImageDraw.Draw(composite_img)
    separator_x = pano_width + 10
    separator_draw.line([(separator_x, 0), (separator_x, composite_height)], fill=(200, 200, 200), width=2)

    return composite_img


OBJECT_CLASS_NAMES = {
    0: "Front Door",
    1: "Foundation",
    2: "Garage Door",
    3: "Stairs",
    4: "Window",
}

OBJECT_CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 255),
}


def create_object_detection_visualization(
    image_path: Path,
    detections: list[dict[str, Any]],
) -> Image.Image:
    """Create visualization with detection bounding boxes overlaid.

    Args:
        image_path: Path to the image file
        detections: List of detection dictionaries with bbox coordinates

    Returns:
        PIL Image with detection overlays
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    for detection in detections:
        x1 = int(detection["bbox_x1"])
        y1 = int(detection["bbox_y1"])
        x2 = int(detection["bbox_x2"])
        y2 = int(detection["bbox_y2"])
        cls_id = detection["class_id"]
        conf = detection["confidence"]

        class_name = OBJECT_CLASS_NAMES.get(cls_id, f"Unknown-{cls_id}")
        color = OBJECT_CLASS_COLORS.get(cls_id, (128, 128, 128))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {conf:.2f}"

        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(
            img,
            (x1, y1 - label_height - baseline - 2),
            (x1 + label_width + 2, y1),
            color,
            -1,
        )

        cv2.putText(
            img,
            label,
            (x1 + 1, y1 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
