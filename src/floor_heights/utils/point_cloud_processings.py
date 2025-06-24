import json

import numpy as np
import pandas as pd
import pdal
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, label, map_coordinates
from skimage.transform import resize
from sklearn.cluster import DBSCAN


def calculate_gapfill_depth(depth_arr, classification_arr, nodata_depth=9999):
    full_mask = depth_arr != nodata_depth
    if not np.any(full_mask):
        return nodata_depth
    building_mask = (classification_arr == 6) & (depth_arr != nodata_depth)
    gapfill_depth = np.mean(depth_arr[building_mask])
    return gapfill_depth


def extract_interpolate(x, y, arr, nodata=9999, search_radius=20):
    """Inverse-distance weighted interpolation using valid pixels within a radius.
    If no valid pixels within radius, use the closest valid pixel value."""
    h, w = arr.shape
    xi, yi = round(x), round(y)

    x0, x1 = max(xi - search_radius, 0), min(xi + search_radius + 1, w)
    y0, y1 = max(yi - search_radius, 0), min(yi + search_radius + 1, h)
    window = arr[y0:y1, x0:x1]

    yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    mask = window != nodata
    if not np.any(mask):
        full_mask = arr != nodata
        if not np.any(full_mask):
            return nodata

        valid_y, valid_x = np.where(full_mask)
        valid_values = arr[valid_y, valid_x]

        dists = np.sqrt((valid_x - x) ** 2 + (valid_y - y) ** 2)

        closest_idx = np.argmin(dists)
        return valid_values[closest_idx]
    dists = np.sqrt((xx[mask] - x) ** 2 + (yy[mask] - y) ** 2)
    values = window[mask]
    weights = 1 / (dists + 1e-6)
    return np.sum(weights * values) / np.sum(weights)


def calculate_width_difference(left_pixel, right_pixel, depth_map, width_pano=11000, search_radius=20, nodata=9999):
    """Calculate real-world horizontal distance between two pixels using depth map"""

    (xl, yl), (xr, yr) = left_pixel, right_pixel
    depth_l, depth_r = depth_map[yl, xl], depth_map[yr, xr]

    depth_l = (
        extract_interpolate(x=xl, y=yl, arr=depth_map, nodata=nodata, search_radius=search_radius)
        if depth_l == nodata
        else depth_l
    )
    depth_r = (
        extract_interpolate(x=xr, y=yr, arr=depth_map, nodata=nodata, search_radius=search_radius)
        if depth_r == nodata
        else depth_r
    )
    if (depth_l is None) or (depth_r is None):
        return None

    w_img = depth_map.shape[1]
    angle_extend = w_img * 180.0 / width_pano
    phi = lambda x: np.radians((2 * x / w_img - 1) * (angle_extend / 2))  # noqa: E731
    x1, z1 = depth_l * np.sin(phi(xl)), depth_l * np.cos(phi(xl))
    x2, z2 = depth_r * np.sin(phi(xr)), depth_r * np.cos(phi(xr))

    return np.hypot(x2 - x1, z2 - z1)


def calculate_height_difference(
    top_pixels,
    bottom_pixels,
    elevation_map,
    gapfill_depth,
    height_pano=5500,
    upper_crop=0.25,
    nodata=9999,
):
    """Calculate real-world horizontal distance between two pixels using elevation and depth maps"""
    (xt, yt), (xb, yb) = top_pixels, bottom_pixels
    elevation_t, elevation_b = elevation_map[yt, xt], elevation_map[yb, xb]
    if (elevation_t == nodata) or (elevation_b == nodata):
        yt_origin = yt + height_pano * upper_crop
        theta_t = np.radians((height_pano / 2.0 - yt_origin) * (180.0 / height_pano))
        yb_origin = yb + height_pano * upper_crop
        theta_b = np.radians((height_pano / 2.0 - yb_origin) * (180.0 / height_pano))
        return gapfill_depth * (np.sin(theta_t) - np.sin(theta_b))
    height_difference = elevation_t - elevation_b
    return height_difference


def compute_feature_properties(row, elevation_arr, depth_arr, gapfill_depth, nodata=9999):
    """Calculate dimension metrics of detected features"""

    img_height = elevation_arr.shape[0]
    img_width = elevation_arr.shape[1]
    mean_top = (int(np.mean([row["x1"], row["x2"]])), int(row["y1"]))
    mean_bottom = (
        int(np.mean([row["x1"], row["x2"]])),
        int(min(row["y2"], img_height - 1)),
    )
    mean_left = (int(row["x1"]), int(np.mean([row["y1"], row["y2"]])))
    mean_right = (
        int(min(row["x2"], img_width - 1)),
        int(np.mean([row["y1"], row["y2"]])),
    )

    feature_top = elevation_arr[mean_top[1], mean_top[0]]
    feature_bottom = elevation_arr[mean_bottom[1], mean_bottom[0]]
    feature_top = None if feature_top == nodata else feature_top
    feature_bottom = None if feature_bottom == nodata else feature_bottom

    feature_width = calculate_width_difference(
        mean_left, mean_right, depth_map=depth_arr, width_pano=11000, nodata=nodata
    )

    feature_height = calculate_height_difference(
        mean_top,
        mean_bottom,
        elevation_map=elevation_arr,
        gapfill_depth=gapfill_depth,
        height_pano=5500,
        nodata=nodata,
    )

    feature_area, dimension_ratio = None, None
    if (feature_height is not None) and (feature_width is not None):
        feature_area = feature_height * feature_width
        dimension_ratio = feature_width / feature_height

    return (
        feature_top,
        feature_bottom,
        feature_width,
        feature_height,
        feature_area,
        dimension_ratio,
    )


def select_best_feature(df, weights, classes, img_width, img_height, frontdoor_standards):
    """
    Select the best feature if multiples are detected
    """
    selected_rows = []
    for feature_class in classes:
        subset = df[df["class"] == feature_class]
        if subset.empty:
            continue
        if feature_class in ["Foundation", "Stairs", "Garage Door"]:
            weighted_diff = (
                weights["area_m2"]
                * abs(subset["area_m2"] - frontdoor_standards["area_m2"])
                / frontdoor_standards["area_m2"]
                + weights["ratio"] * abs(subset["ratio"] - frontdoor_standards["ratio"]) / frontdoor_standards["ratio"]
                + weights["confidence"] * subset["confidence"]
                + weights["y_location"] * subset["y2"] / img_height
            )
            best_row = subset.iloc[np.argmin(weighted_diff)]
        elif feature_class == "Front Door":
            weighted_diff = (
                weights["area_m2"]
                * abs(subset["area_m2"] - frontdoor_standards["area_m2"])
                / frontdoor_standards["area_m2"]
                + weights["ratio"] * abs(subset["ratio"] - frontdoor_standards["ratio"]) / frontdoor_standards["ratio"]
                + weights["confidence"] * subset["confidence"]
                + weights["x_location"] * abs(img_width - (subset["x2"] - subset["x1"]) / 2.0) / img_width
            )
            best_row = subset.iloc[np.argmin(weighted_diff)]
        selected_rows.append(best_row)

    return pd.DataFrame(selected_rows)


def get_closest_ground_to_feature(row, classification_arr, elevation_arr, min_area=5):
    """
    Find the elevation of closest ground area to the average position of two points from the same feature.

    Args:
        classification_arr: 2D array of classification values
        elevation_arr: 2D array of elevation values
        min_area: Minimum area threshold for ground regions

    Returns:
        Dictionary with information about the closest ground area to the feature's average position,
        or None if none found
    """
    x1, y1 = int(row["x1"]), int(row["y2"])
    x2, y2 = int(row["x2"]), int(row["y2"])

    avg_y = round((y1 + y2) / 2)
    avg_x = round((x1 + x2) / 2)

    struct = generate_binary_structure(2, 2)
    labeled_ground, _ = label((classification_arr == 2), structure=struct)

    nearest_ground_elevation = None
    min_distance = float("inf")

    ground_labels = np.unique(labeled_ground)
    ground_labels = ground_labels[ground_labels != 0]

    for label_id in ground_labels:
        ground_mask = labeled_ground == label_id
        area_size = np.sum(ground_mask)

        if area_size >= min_area:
            yy, xx = np.where(ground_mask)
            distances = np.sqrt((yy - avg_y) ** 2 + (xx - avg_x) ** 2)
            idx = np.argmin(distances)
            current_dist = distances[idx]

            if current_dist < min_distance:
                elev_values = elevation_arr[ground_mask]
                nearest_ground_elevation = np.median(elev_values)
                min_distance = current_dist

    return nearest_ground_elevation


def estimate_ffh(df_features, ground_elevation_gapfill, min_ffh=0, max_ffh=1.5):
    """
    Calculate FFHs using elevations of features and ground elevation values.
    """
    elev_ground = None
    df_features_filtered = df_features.dropna(subset=["bottom_elevation"])
    filtered_classes = df_features_filtered["class"].values
    if "Garage Door" in filtered_classes:
        elev_ground = df_features_filtered[df_features_filtered["class"] == "Garage Door"]["bottom_elevation"].values[0]
    elif "Stairs" in filtered_classes:
        elev_ground = df_features_filtered[df_features_filtered["class"] == "Stairs"]["bottom_elevation"].values[0]
    elif "Foundation" in filtered_classes:
        elev_ground = df_features_filtered[df_features_filtered["class"] == "Foundation"]["bottom_elevation"].values[0]
    if elev_ground is None:
        ffh_1 = None

    elev_floor = None
    df_features_filtered = df_features.dropna(subset=["top_elevation"])
    nearest_ground_elev = None
    all_classes = df_features["class"].values
    if "Front Door" in all_classes:
        frontdoor_bottom = df_features[df_features["class"] == "Front Door"]["bottom_elevation"].values[0]
        if frontdoor_bottom is not None:
            elev_floor = frontdoor_bottom
            nearest_ground_elev = df_features[df_features["class"] == "Front Door"]["nearest_ground_elev"].values[0]
    else:
        df_features_filtered = df_features.dropna(subset=["top_elevation"])
        if "Stairs" in filtered_classes:
            elev_floor = df_features_filtered[df_features_filtered["class"] == "Stairs"]["top_elevation"].values[0]
            nearest_ground_elev = df_features[df_features["class"] == "Stairs"]["nearest_ground_elev"].values[0]
        elif "Foundation" in filtered_classes:
            elev_floor = df_features_filtered[df_features_filtered["class"] == "Foundation"]["top_elevation"].values[0]
            nearest_ground_elev = df_features[df_features["class"] == "Foundation"]["nearest_ground_elev"].values[0]
    if elev_floor is None:
        ffh_1 = None

    if (elev_floor is not None) and (elev_ground is not None):
        ffh_1 = elev_floor - elev_ground
        if min_ffh > ffh_1 or max_ffh < ffh_1:
            ffh_1 = None

    ffh_2 = None
    if elev_floor is not None and nearest_ground_elev is not None:
        ffh_2 = elev_floor - nearest_ground_elev
    if min_ffh > ffh_2 or max_ffh < ffh_2:
        ffh_2 = None

    ffh_3 = None
    if (elev_floor is not None) and (ground_elevation_gapfill is not None):
        ffh_3 = elev_floor - ground_elevation_gapfill
    if min_ffh > ffh_3 or max_ffh < ffh_3:
        ffh_3 = None
    return ffh_1, ffh_2, ffh_3


def project_las_to_equirectangular(
    input_las,
    camera_pos=None,
    camera_angles=None,
    width=2048,
    height=1024,
    nodata_float=9999,
    nodata_int=255,
):
    """
    Projects LAS to equirectangular maps with intrinsic XYZ rotation.
    Returns:
        rgb_raster (np.uint8): (H,W,3) RGB image
        z_raster (np.float32): (H,W) elevation map
        depth_raster (np.float32): (H,W) depth map
        class_raster (np.float32): (H,W) classification map
    """

    if camera_angles is None:
        camera_angles = [0, 0, 0]
    if camera_pos is None:
        camera_pos = [0, 0, 0]
    pipeline = pdal.Reader.las(filename=input_las).pipeline()
    pipeline.execute()
    points = pipeline.arrays[0]
    x, y, z = points["X"], points["Y"], points["Z"]
    rgb = np.vstack([points["Red"], points["Green"], points["Blue"]]).T / 256
    classification = points["Classification"].astype(np.uint8)
    intensity = points["Intensity"].astype(np.uint8)

    yaw_rad = np.radians(camera_angles[0])
    pitch_rad = np.radians(camera_angles[1])
    roll_rad = np.radians(camera_angles[2])

    x -= camera_pos[0]
    y -= camera_pos[1]
    z -= camera_pos[2]

    r_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )
    r_pitch = np.array(
        [
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )
    r_heading = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1],
        ]
    )

    r_total = r_heading @ r_roll @ r_pitch

    coords = np.vstack([x, y, z])
    coords_local = r_total @ coords

    x_cam = coords_local[0]
    y_cam = -coords_local[2]
    z_cam = coords_local[1]

    r = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    theta = np.arctan2(x_cam, z_cam)
    phi = np.arccos(-y_cam / r)

    u_norm = 0.5 * (theta / np.pi + 1)
    v_norm = phi / np.pi

    u_idx = np.floor(u_norm * (width - 1)).astype(np.int32)
    v_idx = np.floor(v_norm * (height - 1)).astype(np.int32)

    u_idx = np.clip(u_idx, 0, width - 1)
    v_idx = np.clip(v_idx, 0, height - 1)

    rgb_raster = np.full((height, width, 3), nodata_int, dtype=np.uint8)
    z_raster = np.full((height, width), nodata_float, dtype=np.float32)
    depth_raster = np.full((height, width), nodata_float, dtype=np.float32)
    class_raster = np.full((height, width), nodata_int, dtype=np.uint8)
    intensity_raster = np.full((height, width), nodata_int, dtype=np.uint8)

    sort_idx = np.argsort(r)
    u_idx = u_idx[sort_idx]
    v_idx = v_idx[sort_idx]
    r = r[sort_idx]
    rgb = rgb[sort_idx]
    z = z[sort_idx]
    classification = classification[sort_idx]
    intensity = intensity[sort_idx]

    valid = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height)
    u_valid = u_idx[valid]
    v_valid = v_idx[valid]

    mask = r[valid] < depth_raster[v_valid, u_valid]
    depth_raster[v_valid[mask], u_valid[mask]] = r[valid][mask]
    rgb_raster[v_valid[mask], u_valid[mask]] = (rgb[valid][mask]).astype(np.uint8)
    z_raster[v_valid[mask], u_valid[mask]] = z[valid][mask] + camera_pos[2]
    class_raster[v_valid[mask], u_valid[mask]] = classification[valid][mask]
    intensity_raster[v_valid[mask], u_valid[mask]] = intensity[valid][mask]

    return rgb_raster, z_raster, depth_raster, class_raster, intensity_raster


def project_las_to_equirectangular_classification_aware(
    input_las,
    camera_pos=None,
    camera_angles=None,
    width=2048,
    height=1024,
    nodata_float=9999,
    nodata_int=255,
):
    """
    Classification-aware projection that preserves edges between ground and building.
    Uses nearest-neighbor for ground (class 2) and building (class 6) to maintain sharp boundaries,
    while using bilinear interpolation for vegetation and other classes.

    Returns:
        rgb_raster (np.uint8): (H,W,3) RGB image
        z_raster (np.float32): (H,W) elevation map
        depth_raster (np.float32): (H,W) depth map
        class_raster (np.float32): (H,W) classification map
        intensity_raster (np.uint8): (H,W) intensity map
    """

    if camera_angles is None:
        camera_angles = [0, 0, 0]
    if camera_pos is None:
        camera_pos = [0, 0, 0]
    rgb_base, z_base, depth_base, class_base, intensity_base = project_las_to_equirectangular(
        input_las, camera_pos, camera_angles, width, height, nodata_float, nodata_int
    )

    pipeline = pdal.Reader.las(filename=input_las).pipeline()
    pipeline.execute()
    points = pipeline.arrays[0]
    x, y, z = points["X"], points["Y"], points["Z"]
    classification = points["Classification"].astype(np.uint8)
    intensity = points["Intensity"].astype(np.uint8)

    yaw_rad = np.radians(camera_angles[0])
    pitch_rad = np.radians(camera_angles[1])
    roll_rad = np.radians(camera_angles[2])

    x -= camera_pos[0]
    y -= camera_pos[1]
    z -= camera_pos[2]

    r_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )
    r_pitch = np.array(
        [
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )
    r_heading = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1],
        ]
    )
    r_total = r_heading @ r_roll @ r_pitch

    coords = np.vstack([x, y, z])
    coords_local = r_total @ coords

    x_cam = coords_local[0]
    y_cam = -coords_local[2]
    z_cam = coords_local[1]

    r = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    theta = np.arctan2(x_cam, z_cam)
    phi = np.arccos(-y_cam / r)

    u_norm = 0.5 * (theta / np.pi + 1)
    v_norm = phi / np.pi

    u_coords = u_norm * (width - 1)
    v_coords = v_norm * (height - 1)

    ground_mask = classification == 2
    building_mask = classification == 6
    critical_mask = ground_mask | building_mask
    other_mask = ~critical_mask

    z_raster = np.full((height, width), nodata_float, dtype=np.float32)
    depth_raster = np.full((height, width), nodata_float, dtype=np.float32)
    class_raster = np.full((height, width), nodata_int, dtype=np.uint8)
    intensity_raster = np.full((height, width), nodata_int, dtype=np.uint8)

    if np.any(critical_mask):
        critical_u = u_coords[critical_mask]
        critical_v = v_coords[critical_mask]
        critical_z = z[critical_mask] + camera_pos[2]
        critical_depth = r[critical_mask]
        critical_class = classification[critical_mask]
        critical_intensity = intensity[critical_mask]

        u_idx = np.clip(np.round(critical_u).astype(np.int32), 0, width - 1)
        v_idx = np.clip(np.round(critical_v).astype(np.int32), 0, height - 1)

        sort_idx = np.argsort(critical_depth)
        u_idx = u_idx[sort_idx]
        v_idx = v_idx[sort_idx]
        critical_depth = critical_depth[sort_idx]
        critical_z = critical_z[sort_idx]
        critical_class = critical_class[sort_idx]
        critical_intensity = critical_intensity[sort_idx]

        for i in range(len(u_idx)):
            if critical_depth[i] < depth_raster[v_idx[i], u_idx[i]]:
                depth_raster[v_idx[i], u_idx[i]] = critical_depth[i]
                z_raster[v_idx[i], u_idx[i]] = critical_z[i]
                class_raster[v_idx[i], u_idx[i]] = critical_class[i]
                intensity_raster[v_idx[i], u_idx[i]] = critical_intensity[i]

    if np.any(other_mask):
        other_valid = (class_base != 2) & (class_base != 6) & (class_base != nodata_int)
        z_raster[other_valid] = z_base[other_valid]
        depth_raster[other_valid] = depth_base[other_valid]
        class_raster[other_valid] = class_base[other_valid]
        intensity_raster[other_valid] = intensity_base[other_valid]

    rgb_raster = rgb_base

    return rgb_raster, z_raster, depth_raster, class_raster, intensity_raster


def fill_small_nans(arr, max_hole_size=10, nodata_value=9999):
    """
    Fills small nodata regions using local interpolation from surrounding valid pixels.

    Parameters:
        arr: 2D numpy array with nodata values
        max_hole_size: Maximum size (in pixels) of nodata regions to fill
        nodata_value: The value representing nodata (default: 9999)

    Returns:
        Array with small nodata regions filled, large ones preserved
    """

    nodata_mask = arr == nodata_value

    labeled, num_features = ndimage.label(nodata_mask)

    sizes = ndimage.sum(nodata_mask, labeled, range(num_features + 1))  # noqa: F841

    filled = arr.copy()

    distances, indices = ndimage.distance_transform_edt(
        nodata_mask,
        return_indices=True,
    )

    for i in range(1, num_features + 1):
        region_mask = labeled == i
        region_size = np.sum(region_mask)

        if region_size <= max_hole_size:
            filled[region_mask] = arr[indices[0][region_mask], indices[1][region_mask]]
    return filled


def resize_preserve_nans(arr, target_height, target_width, order=1, nodata_value=9999):
    """
    Resizes an array while preserving NoData regions, preventing artifacts at edges.
    """

    valid_mask = arr != nodata_value

    arr.shape[0] / target_height
    arr.shape[1] / target_width

    y_idx, x_idx = np.meshgrid(
        np.linspace(0.5, arr.shape[0] - 0.5, target_height),
        np.linspace(0.5, arr.shape[1] - 0.5, target_width),
        indexing="ij",
    )
    coords = np.array([y_idx.ravel(), x_idx.ravel()])

    resized_mask = (
        resize(
            valid_mask.astype(float),
            (target_height, target_width),
            order=0,
            anti_aliasing=False,
        )
        > 0.5
    )

    from scipy.ndimage import distance_transform_edt

    dist_to_nodata = distance_transform_edt(valid_mask)
    edge_zone = dist_to_nodata <= 1

    resized_data = map_coordinates(arr, coords, order=order, cval=nodata_value)
    resized_data = resized_data.reshape((target_height, target_width))

    if np.any(edge_zone):
        edge_data = map_coordinates(arr, coords, order=0, cval=nodata_value)
        edge_data = edge_data.reshape((target_height, target_width))
        edge_coverage = map_coordinates(edge_zone.astype(float), coords, order=order)
        edge_coverage = edge_coverage.reshape((target_height, target_width)) > 0.1

        resized_data = np.where(edge_coverage, edge_data, resized_data)

    resized_data[~resized_mask] = nodata_value

    return resized_data


def process_extract_ground_elevations(las_file_path, resolution, crs, output_tiff=None):
    """
    Apply filtering to remove outliers of ground points and extract elevation stats.
    """
    pipeline_steps = [
        {"type": "readers.las", "filename": las_file_path},
        {
            "type": "filters.range",
            "limits": "Classification[2:2]",
        },
        {"type": "filters.outlier"},
        {
            "type": "filters.csf",
            "ignore": "Classification[7:7]",  # Ignore noise class if present
            "resolution": 1,
            "hdiff": 0.5,
            "smooth": False,
        },
        {"type": "filters.range", "limits": "Classification[2:2]"},
    ]

    if output_tiff is not None:
        pipeline_steps.append(
            {
                "type": "writers.gdal",
                "filename": output_tiff,
                "dimension": "Z",
                "output_type": "idw",
                "resolution": resolution,
                "gdaldriver": "GTiff",
                "data_type": "float32",
                "nodata": -9999,
                "override_srs": crs,
            }
        )

    pipeline_json = {"pipeline": pipeline_steps}
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()

    ground_points = pipeline.arrays[0]

    elevations = ground_points["Z"]

    stats = {
        "lidar_elev_mean": np.mean(elevations),
        "lidar_elev_med": np.median(elevations),
        "lidar_elev_min": np.min(elevations),
        "lidar_elev_max": np.max(elevations),
        "lidar_elev_std": np.std(elevations),
        "lidar_elev_25pct": np.percentile(elevations, 25),
        "lidar_elev_75pct": np.percentile(elevations, 75),
    }
    return stats


def remove_noise(points, eps, min_samples):
    """
    Remove noise points by filtering out isolated clusters using DBSCAN.

    Parameters:
        points (numpy.ndarray): LiDAR points as an (N, 3) array of [x, y, z].
        eps (float): Maximum distance between points to be considered in the same neighborhood.
        min_samples (int): Minimum number of points to form a dense cluster.

    Returns:
        numpy.ndarray: Filtered points with noise removed.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
    labels = clustering.labels_

    return points[labels != -1]


def create_bev_density_map(points, x_limits, y_limits, resolution=0.1):
    """
    Project LiDAR point cloud to BEV plane as a density map.

    Parameters:
        points (numpy.ndarray): LiDAR points as an (N, 3) array of [x, y, z].
        x_limits (tuple): Min and max range for x-axis (e.g., (-50, 50)).
        y_limits (tuple): Min and max range for y-axis (e.g., (-50, 50)).
        resolution (float): Grid resolution in meters (e.g., 0.1).

    Returns:
        density_map (numpy.ndarray): Gridded density map as a 2D array.
    """

    x_min, x_max = x_limits
    y_min, y_max = y_limits

    mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & (points[:, 1] >= y_min) & (points[:, 1] < y_max)
    filtered_points = points[mask]

    grid_x_size = int((x_max - x_min) / resolution)
    grid_y_size = int((y_max - y_min) / resolution)

    grid_x = ((filtered_points[:, 0] - x_min) / resolution).astype(np.int32)
    grid_y = ((filtered_points[:, 1] - y_min) / resolution).astype(np.int32)

    grid_x = np.clip(grid_x, 0, grid_x_size - 1)
    grid_y = np.clip(grid_y, 0, grid_y_size - 1)

    density_map = np.zeros((grid_y_size, grid_x_size), dtype=np.float32)

    for gx, gy in zip(grid_x, grid_y, strict=False):
        density_map[gy, gx] += 1

    density_map /= density_map.max() if density_map.max() > 0 else 1

    return density_map


def project_lidar_perspective(point_cloud, position, orientation, resolution, fov, no_data_value=-9999):
    """
    Project LiDAR point cloud onto a 2D perspective image and calculate an elevation map.

    Parameters:
        point_cloud (numpy.ndarray): Nx3 array of LiDAR points (x, y, z).
        position (tuple): Viewpoint position as a 3-tuple (x, y, z).
        orientation (tuple): Viewpoint orientation as a 3-tuple (yaw, pitch, roll in radians).
        resolution (tuple): Image resolution (width, height) in pixels.
        fov (float): Horizontal field of view in radians.

    Returns:
        numpy.ndarray: 2D elevation map.
        numpy.ndarray: 2D depth map.
    """

    px, py, pz = point_cloud.T
    vx, vy, vz = position
    yaw, pitch, roll = orientation
    img_width, img_height = resolution

    points = np.array([px - vx, py - vy, pz - vz]).T

    rotation_matrix = get_rotation_matrix(yaw, pitch, roll)
    points = points @ rotation_matrix.T

    print("total number of points: ", len(points))
    points = np.asarray(points)

    focal_length = 0.5 * img_width / np.tan(0.5 * fov)

    valid = points[:, 2] > 0

    x, y, z = points.T
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]
    pz = pz[valid]

    u = (focal_length * x / z + img_width / 2.0).astype(int)
    v = (focal_length * y / z + img_height / 2.0).astype(int)

    v = img_height - v

    valid_pixels = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    print("number of points within view: ", np.sum(valid_pixels))
    u, v, z = u[valid_pixels], v[valid_pixels], z[valid_pixels]
    pz = pz[valid_pixels]

    elevation_map = np.full((img_height, img_width), no_data_value, dtype=np.float32)
    depth_map = np.full((img_height, img_width), np.inf, dtype=np.float32)
    for px, py, z_val, depth in zip(u, v, pz, z, strict=False):
        if depth < depth_map[py, px]:
            elevation_map[py, px] = z_val
            depth_map[py, px] = depth

    return elevation_map, depth_map


def project_lidar_equirectangular(point_cloud, position, orientation, hfov, vfov, resolution, no_data_value=-9999):
    """
    Calculate an elevation image from a point cloud projected within the field of view.

    Parameters:
        point_cloud (numpy.ndarray): Array of shape (N, 3) with 3D points (x, y, z).
        position (tuple): Viewpoint position as a 3-tuple (x, y, z).
        orientation (tuple): Viewpoint orientation as a 3-tuple (yaw, pitch, roll in radians).
        hfov (float): Horizontal field of view in radians.
        vfov (float): Vertical field of view in radians.
        resolution (tuple): Image resolution (width, height).

    Returns:
        numpy.ndarray: 2D array representing the elevation image.
        numpy.ndarray: 2D depth map.
    """

    px, py, pz = point_cloud.T
    vx, vy, vz = position
    yaw, pitch, roll = orientation
    img_width, img_height = resolution

    points = np.array([px - vx, py - vy, pz - vz]).T

    rotation_matrix = get_rotation_matrix(yaw, pitch, roll)
    points = points @ rotation_matrix.T

    r = np.linalg.norm(points, axis=1)

    theta = np.arctan2(points[:, 0], r)

    phi = np.arctan2((-1.0) * points[:, 1], r)

    mask = (theta >= -hfov / 2) & (theta <= hfov / 2) & (phi >= -vfov / 2) & (phi <= vfov / 2)
    print("number of points within view: ", np.sum(mask))
    points = points[mask]
    theta = theta[mask]
    phi = phi[mask]
    r = r[mask]
    pz = pz[mask]

    u = ((theta + hfov / 2) / hfov * img_width).astype(int)
    v = ((phi + vfov / 2) / vfov * img_height).astype(int)

    u = np.clip(u, 0, img_width - 1)
    v = np.clip(v, 0, img_height - 1)

    elevation_map = np.full((img_height, img_width), no_data_value, dtype=np.float32)
    depth_map = np.full((img_height, img_width), np.inf, dtype=np.float32)
    for px, py, z_val, depth in zip(u, v, pz, r, strict=False):
        if depth < depth_map[py, px]:
            elevation_map[py, px] = z_val
            depth_map[py, px] = depth

    return elevation_map, depth_map


def get_rotation_matrix(yaw, pitch, roll):
    """
    Compute the rotation matrix from yaw, pitch, and roll angles.

    Parameters:
        yaw (float): Yaw angle in radians (rotation around z-axis).
        pitch (float): Pitch angle in radians (rotation around y-axis).
        roll (float): Roll angle in radians (rotation around x-axis).

    Returns:
        R (numpy.ndarray): 3x3 rotation matrix.
    """

    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    r = np.dot(rz, np.dot(ry, rx))

    r_align = np.array(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ]
    )

    r_total = np.dot(r_align, r)
    return r_total
