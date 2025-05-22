import json

import numpy as np
import pandas as pd
import pdal
from scipy import ndimage
from scipy.ndimage import map_coordinates, label, generate_binary_structure
from skimage.transform import resize
from sklearn.cluster import DBSCAN


def calculate_gapfill_depth(depth_arr, classification_arr, nodata_depth=9999):
    # Create mask of all valid pixels in entire array
    full_mask = depth_arr != nodata_depth
    if not np.any(full_mask):
        return nodata_depth  # no valid pixels in entire image
    # create mask of building pixels
    building_mask = (classification_arr == 6) & (depth_arr != nodata_depth)
    gapfill_depth = np.mean(depth_arr[building_mask])
    return gapfill_depth


def extract_interpolate(x, y, arr, nodata=9999, search_radius=20):
    """Inverse-distance weighted interpolation using valid pixels within a radius.
    If no valid pixels within radius, use the closest valid pixel value."""
    h, w = arr.shape
    xi, yi = int(round(x)), int(round(y))

    # Define window bounds for initial search
    x0, x1 = max(xi - search_radius, 0), min(xi + search_radius + 1, w)
    y0, y1 = max(yi - search_radius, 0), min(yi + search_radius + 1, h)
    window = arr[y0:y1, x0:x1]

    # Create coordinate grids
    yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    mask = window != nodata
    if not np.any(mask):
        # Create mask of all valid pixels in entire array
        full_mask = arr != nodata
        if not np.any(full_mask):
            return nodata  # no valid pixels in entire image

        # Find all valid pixel coordinates and values
        valid_y, valid_x = np.where(full_mask)
        valid_values = arr[valid_y, valid_x]

        # Calculate distances to all valid pixels
        dists = np.sqrt((valid_x - x) ** 2 + (valid_y - y) ** 2)

        # Return the value of the closest pixel
        closest_idx = np.argmin(dists)
        return valid_values[closest_idx]
    # Perform IDW interpolation with valid pixels in window
    dists = np.sqrt((xx[mask] - x) ** 2 + (yy[mask] - y) ** 2)
    values = window[mask]
    weights = 1 / (dists + 1e-6)  # avoid divide-by-zero
    return np.sum(weights * values) / np.sum(weights)


def calculate_width_difference(
    left_pixel, right_pixel, depth_map, width_pano=11000, search_radius=20, nodata=9999
):
    """Calculate real-world horizontal distance between two pixels using depth map"""

    # extract depth
    (xL, yL), (xR, yR) = left_pixel, right_pixel
    depthL, depthR = depth_map[yL, xL], depth_map[yR, xR]
    # print('depths before depth interpolation ',depthL,depthR)

    # interpolate depth values if invalid
    depthL = (
        extract_interpolate(
            x=xL, y=yL, arr=depth_map, nodata=nodata, search_radius=search_radius
        )
        if depthL == nodata
        else depthL
    )
    depthR = (
        extract_interpolate(
            x=xR, y=yR, arr=depth_map, nodata=nodata, search_radius=search_radius
        )
        if depthR == nodata
        else depthR
    )
    # print('depths after depth interpolation ',depthL,depthR)
    if (depthL is None) or (depthR is None):
        return None

    # calculate real world width
    W_img = depth_map.shape[1]
    angle_extend = W_img * 180.0 / width_pano
    phi = lambda x: np.radians((2 * x / W_img - 1) * (angle_extend / 2))  # noqa: E731
    # print('phi ',phi)
    x1, z1 = depthL * np.sin(phi(xL)), depthL * np.cos(phi(xL))
    x2, z2 = depthR * np.sin(phi(xR)), depthR * np.cos(phi(xR))

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
    # extract elevations
    (xT, yT), (xB, yB) = top_pixels, bottom_pixels
    elevationT, elevationB = elevation_map[yT, xT], elevation_map[yB, xB]
    # if invalid, use gapfilling depth and approximate
    if (elevationT == nodata) or (elevationB == nodata):
        # print('using gapfilling depth')
        # original y coordinate before cropping
        yT_origin = yT + height_pano * upper_crop
        theta_T = np.radians((height_pano / 2.0 - yT_origin) * (180.0 / height_pano))
        yB_origin = yB + height_pano * upper_crop
        theta_B = np.radians((height_pano / 2.0 - yB_origin) * (180.0 / height_pano))
        return gapfill_depth * (np.sin(theta_T) - np.sin(theta_B))
    height_difference = elevationT - elevationB
    return height_difference


def compute_feature_properties(
    row, elevation_arr, depth_arr, gapfill_depth, nodata=9999
):
    """Calculate dimension metrics of detected features"""

    # calculate average pixels of each boundary
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

    # caculate feature top and bottom elevations
    feature_top = elevation_arr[mean_top[1], mean_top[0]]
    feature_bottom = elevation_arr[mean_bottom[1], mean_bottom[0]]
    feature_top = None if feature_top == nodata else feature_top
    feature_bottom = None if feature_bottom == nodata else feature_bottom
    # print('feature top elevation:',feature_top)
    # print('feature bottom elevation:', feature_bottom)

    # calculate feature width
    feature_width = calculate_width_difference(
        mean_left, mean_right, depth_map=depth_arr, width_pano=11000, nodata=nodata
    )
    # print('feature width',feature_width)

    # calculate feature height
    feature_height = calculate_height_difference(
        mean_top,
        mean_bottom,
        elevation_map=elevation_arr,
        gapfill_depth=gapfill_depth,
        height_pano=5500,
        nodata=nodata,
    )
    # print('feature height',feature_height)

    # calculate area and ratio
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


def select_best_feature(
    df, weights, classes, img_width, img_height, frontdoor_standards
):
    """
    Select the best feature if multiples are detected
    """
    selected_rows = []
    for feature_class in classes:
        subset = df[df["class"] == feature_class]
        if subset.empty:
            continue
        if feature_class in ["Foundation", "Stairs", "Garage Door"]:
            # Select bottom-most and with highest confidence
            # best_row = subset.sort_values(by=['y2', 'confidence'], ascending=[False, False]).iloc[0]
            weighted_diff = (
                weights["area_m2"]
                * abs(subset["area_m2"] - frontdoor_standards["area_m2"])
                / frontdoor_standards["area_m2"]
                + weights["ratio"]
                * abs(subset["ratio"] - frontdoor_standards["ratio"])
                / frontdoor_standards["ratio"]
                + weights["confidence"] * subset["confidence"]
                + weights["y_location"] * subset["y2"] / img_height
            )
            best_row = subset.iloc[np.argmin(weighted_diff)]
        elif feature_class == "Front Door":
            # For Front Door: select closet to standard metrics, with highest confidence and horizontally closest to image centre
            weighted_diff = (
                weights["area_m2"]
                * abs(subset["area_m2"] - frontdoor_standards["area_m2"])
                / frontdoor_standards["area_m2"]
                + weights["ratio"]
                * abs(subset["ratio"] - frontdoor_standards["ratio"])
                / frontdoor_standards["ratio"]
                + weights["confidence"] * subset["confidence"]
                + weights["x_location"]
                * abs(img_width - (subset["x2"] - subset["x1"]) / 2.0)
                / img_width
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

    # Calculate average position of the two points
    avg_y = int(round((y1 + y2) / 2))
    avg_x = int(round((x1 + x2) / 2))

    # Label all ground areas (search everywhere)
    struct = generate_binary_structure(2, 2)
    labeled_ground, _ = label((classification_arr == 2), structure=struct)

    nearest_ground_elevation = None
    min_distance = float("inf")

    # Get unique labels in the array (excluding 0)
    ground_labels = np.unique(labeled_ground)
    ground_labels = ground_labels[ground_labels != 0]

    for label_id in ground_labels:
        ground_mask = labeled_ground == label_id
        area_size = np.sum(ground_mask)

        if area_size >= min_area:
            # Find closest point in this area to the average position
            yy, xx = np.where(ground_mask)
            distances = np.sqrt((yy - avg_y) ** 2 + (xx - avg_x) ** 2)
            idx = np.argmin(distances)
            current_dist = distances[idx]

            if current_dist < min_distance:
                elev_values = elevation_arr[ground_mask]
                nearest_ground_elevation = np.median(elev_values)
                min_distance = current_dist

    return nearest_ground_elevation


def estimate_FFH(df_features, ground_elevation_gapfill, min_ffh=0, max_ffh=1.5):
    """
    Calculate FFHs using elevations of features and ground elevation values.
    """
    # determine ground elevation: prioritisation garage door>stairs>foundation
    elev_ground = None
    df_features_filtered = df_features.dropna(subset=["bottom_elevation"])
    filtered_classes = df_features_filtered["class"].values
    # print(filtered_classes)
    if "Garage Door" in filtered_classes:
        elev_ground = df_features_filtered[
            df_features_filtered["class"] == "Garage Door"
        ]["bottom_elevation"].values[0]
    elif "Stairs" in filtered_classes:
        elev_ground = df_features_filtered[df_features_filtered["class"] == "Stairs"][
            "bottom_elevation"
        ].values[0]
    elif "Foundation" in filtered_classes:
        elev_ground = df_features_filtered[
            df_features_filtered["class"] == "Foundation"
        ]["bottom_elevation"].values[0]
    if elev_ground is None:
        FFH_1 = None
    # print('elev_ground',elev_ground)

    # determine floor elevation: prioritisation Front Door>stairs>foundation
    elev_floor = None
    df_features_filtered = df_features.dropna(subset=["top_elevation"])
    nearest_ground_elev = None
    all_classes = df_features["class"].values
    if "Front Door" in all_classes:
        frontdoor_bottom = df_features[df_features["class"] == "Front Door"][
            "bottom_elevation"
        ].values[0]
        if frontdoor_bottom is not None:
            elev_floor = frontdoor_bottom
            nearest_ground_elev = df_features[df_features["class"] == "Front Door"][
                "nearest_ground_elev"
            ].values[0]
    else:
        df_features_filtered = df_features.dropna(subset=["top_elevation"])
        if "Stairs" in filtered_classes:
            elev_floor = df_features_filtered[
                df_features_filtered["class"] == "Stairs"
            ]["top_elevation"].values[0]
            nearest_ground_elev = df_features[df_features["class"] == "Stairs"][
                "nearest_ground_elev"
            ].values[0]
        elif "Foundation" in filtered_classes:
            elev_floor = df_features_filtered[
                df_features_filtered["class"] == "Foundation"
            ]["top_elevation"].values[0]
            nearest_ground_elev = df_features[df_features["class"] == "Foundation"][
                "nearest_ground_elev"
            ].values[0]
    if elev_floor is None:
        FFH_1 = None
    # print('elev_floor',elev_floor)

    # 1. FFH calculated from floor feature(Front Door/stair/foundation) and ground feature (stairs/foundation) - not always available
    if (elev_floor is not None) and (elev_ground is not None):
        FFH_1 = elev_floor - elev_ground
        if FFH_1 < min_ffh or FFH_1 > max_ffh:
            FFH_1 = None

    # 2. FFH calculated from floor feature (Front Door/stair/foundation) and ground elevation derived
    # from closet ground area (likely available whenever a ground feature is detected)
    FFH_2 = None
    if elev_floor is not None:
        if nearest_ground_elev is not None:
            FFH_2 = elev_floor - nearest_ground_elev
    if FFH_2 < min_ffh or FFH_2 > max_ffh:
        FFH_2 = None
    # 3. FFH calculated from floor feature (Front Door/stair/foundation) and ground elevation derived
    # from DTM (available whenever a ground feature is detected)
    FFH_3 = None
    if (elev_floor is not None) and (ground_elevation_gapfill is not None):
        FFH_3 = elev_floor - ground_elevation_gapfill
    if FFH_3 < min_ffh or FFH_3 > max_ffh:
        FFH_3 = None
    return FFH_1, FFH_2, FFH_3


def project_las_to_equirectangular(
    input_las,
    camera_pos=[0, 0, 0],
    camera_angles=[0, 0, 0],
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
    # --- Data Loading ---
    pipeline = pdal.Reader.las(filename=input_las).pipeline()
    pipeline.execute()
    points = pipeline.arrays[0]
    x, y, z = points["X"], points["Y"], points["Z"]
    rgb = np.vstack([points["Red"], points["Green"], points["Blue"]]).T / 256  # 16 bits
    classification = points["Classification"].astype(np.uint8)
    intensity = points["Intensity"].astype(np.uint8)
    print("RGB min/max:", rgb.min(axis=0), rgb.max(axis=0))

    # --- Coordinate Transformation ---
    # Convert angles to radians
    yaw_rad = np.radians(camera_angles[0])
    pitch_rad = np.radians(camera_angles[1])
    roll_rad = np.radians(camera_angles[2])

    # Translate to camera origin
    x -= camera_pos[0]
    y -= camera_pos[1]
    z -= camera_pos[2]

    # Intrinsic XYZ rotation matrices
    R_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )
    R_pitch = np.array(
        [
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )
    R_heading = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1],
        ]
    )
    # R_total = R_heading @ R_pitch @ R_roll  # Intrinsic XYZ order
    R_total = R_heading @ R_roll @ R_pitch  # Intrinsic XYZ order

    # Apply rotation
    coords = np.vstack([x, y, z])
    coords_local = R_total @ coords

    # Transform to camera coordinate convention:
    #    LiDAR's +Z (up) should become camera's +Y (down)
    #    LiDAR's +Y (north) should become camera's -Z (forward)
    x_cam = coords_local[0]
    y_cam = -coords_local[2]  # LiDAR Z (up) -> Camera Y (down)
    z_cam = coords_local[1]  # LiDAR Y (north) -> Camera Z (forward)
    print("Camera-relative Z:", z_cam.min(), z_cam.max())

    # --- Equirectangular Projection ---
    r = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)  # Depth
    theta = np.arctan2(x_cam, z_cam)  # Azimuth
    phi = np.arccos(-y_cam / r)  # flip Zenith

    # Normalized coordinates [0,1] range
    u_norm = 0.5 * (theta / np.pi + 1)
    v_norm = phi / np.pi

    # Convert to pixel coordinates using precise scaling
    u_idx = np.floor(u_norm * (width - 1)).astype(np.int32)
    v_idx = np.floor(v_norm * (height - 1)).astype(np.int32)

    # Ensure indices are within bounds
    u_idx = np.clip(u_idx, 0, width - 1)
    v_idx = np.clip(v_idx, 0, height - 1)

    # --- Rasterization ---
    rgb_raster = np.full((height, width, 3), nodata_int, dtype=np.uint8)
    z_raster = np.full((height, width), nodata_float, dtype=np.float32)
    depth_raster = np.full((height, width), nodata_float, dtype=np.float32)
    class_raster = np.full((height, width), nodata_int, dtype=np.uint8)
    intensity_raster = np.full((height, width), nodata_int, dtype=np.uint8)

    # Sort points by depth (closest first)
    sort_idx = np.argsort(r)
    u_idx = u_idx[sort_idx]
    v_idx = v_idx[sort_idx]
    r = r[sort_idx]
    rgb = rgb[sort_idx]
    z = z[sort_idx]
    classification = classification[sort_idx]
    intensity = intensity[sort_idx]

    # Vectorized depth test
    valid = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height)
    u_valid = u_idx[valid]
    v_valid = v_idx[valid]

    # Update only if closer than existing depth
    mask = r[valid] < depth_raster[v_valid, u_valid]
    depth_raster[v_valid[mask], u_valid[mask]] = r[valid][mask]
    rgb_raster[v_valid[mask], u_valid[mask]] = (rgb[valid][mask]).astype(np.uint8)
    z_raster[v_valid[mask], u_valid[mask]] = z[valid][mask] + camera_pos[2]
    class_raster[v_valid[mask], u_valid[mask]] = classification[valid][mask]
    intensity_raster[v_valid[mask], u_valid[mask]] = intensity[valid][mask]

    # # for debugging purpose
    # print('number of valid depth points: ',np.sum(depth_raster != nodata_float))
    # print('number of valid elevation points: ',np.sum(z_raster!=nodata_float))
    # print('number of classification points: ',np.sum(class_raster != nodata_int))
    # print('number of valid rgb points: ',np.sum(np.any(rgb_raster!= nodata_int, axis=-1)))
    # print('number of valid intensity points: ',np.sum(intensity_raster != nodata_int))

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
    print(f"Initial nodata count: {np.sum(arr == nodata_value)}")

    # Create mask of nodata regions
    nodata_mask = arr == nodata_value

    # Label connected nodata regions
    labeled, num_features = ndimage.label(nodata_mask)

    # Measure size of each nodata region
    sizes = ndimage.sum(nodata_mask, labeled, range(num_features + 1))  # noqa: F841

    # Create output array
    filled = arr.copy()

    # Compute global distance transform once (from all valid pixels)
    distances, indices = ndimage.distance_transform_edt(
        nodata_mask,  # Important: input is the nodata mask
        return_indices=True,
    )

    # Process each nodata region
    for i in range(1, num_features + 1):
        region_mask = labeled == i
        region_size = np.sum(region_mask)

        if region_size <= max_hole_size:
            # Fill using precomputed nearest valid pixels
            filled[region_mask] = arr[indices[0][region_mask], indices[1][region_mask]]
    print(f"Final nodata count: {np.sum(filled == nodata_value)}")
    return filled


def resize_preserve_nans(arr, target_height, target_width, order=1, nodata_value=9999):
    """
    Resizes an array while preserving NoData regions, preventing artifacts at edges.
    """
    # Create valid mask (1=valid, 0=nodata)
    valid_mask = arr != nodata_value

    # For interpolation, replace nodata with 0 but we'll mask later
    # arr_filled = np.where(valid_mask, arr, 0)

    # Compute scale factors
    scale_y = arr.shape[0] / target_height  # noqa: F841
    scale_x = arr.shape[1] / target_width  # noqa: F841

    # Create coordinate grids for interpolation
    y_idx, x_idx = np.meshgrid(
        np.linspace(0.5, arr.shape[0] - 0.5, target_height),
        np.linspace(0.5, arr.shape[1] - 0.5, target_width),
        indexing="ij",
    )
    coords = np.array([y_idx.ravel(), x_idx.ravel()])

    # Resize mask using nearest-neighbor to keep sharp edges
    resized_mask = (
        resize(
            valid_mask.astype(float),
            (target_height, target_width),
            order=0,  # Nearest-neighbor
            anti_aliasing=False,
        )
        > 0.5
    )

    # Create distance-to-edge map to identify border regions
    from scipy.ndimage import distance_transform_edt

    dist_to_nodata = distance_transform_edt(valid_mask)
    edge_zone = dist_to_nodata <= 1  # Pixels adjacent to nodata

    # Interpolate main data
    resized_data = map_coordinates(arr, coords, order=order, cval=nodata_value)
    resized_data = resized_data.reshape((target_height, target_width))

    # For edge pixels, use nearest-neighbor to prevent bleeding
    if np.any(edge_zone):
        edge_data = map_coordinates(arr, coords, order=0, cval=nodata_value)
        edge_data = edge_data.reshape((target_height, target_width))

        # Find where original edge pixels map to in output
        edge_coverage = map_coordinates(edge_zone.astype(float), coords, order=order)
        edge_coverage = edge_coverage.reshape((target_height, target_width)) > 0.1

        # Use nearest-neighbor result for edge-affected areas
        resized_data = np.where(edge_coverage, edge_data, resized_data)

    # Restore NoData values using the resized mask
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
            "limits": "Classification[2:2]",  # Ground points only
        },
        # default outliers removal (statistical)
        {"type": "filters.outlier"},
        # Cloth Simulation Filter (CSF) filtering: removing flying objects
        {
            "type": "filters.csf",
            "ignore": "Classification[7:7]",  # Ignore noise class if present
            "resolution": 1,
            "hdiff": 0.5,
            "smooth": False,
        },
        # Kee only ground points
        {"type": "filters.range", "limits": "Classification[2:2]"},
    ]

    if output_tiff is not None:
        pipeline_steps.append(
            {
                "type": "writers.gdal",
                "filename": output_tiff,
                "dimension": "Z",  # Export elevation values
                "output_type": "idw",  # Inverse Distance Weighting
                "resolution": resolution,
                "gdaldriver": "GTiff",
                "data_type": "float32",
                "nodata": -9999,
                "override_srs": crs,  # Explicitly set output CRS
            }
        )
    # Main processing pipeline
    pipeline_json = {"pipeline": pipeline_steps}
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()

    # Get the processed ground points as a numpy array
    ground_points = pipeline.arrays[0]

    # Extract Z values (elevations)
    elevations = ground_points["Z"]

    # Calculate statistics
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

    # Keep only points belonging to clusters (label != -1)
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
    # Filter points within the specified x and y limits
    x_min, x_max = x_limits
    y_min, y_max = y_limits

    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    )
    filtered_points = points[mask]

    # Calculate grid size
    grid_x_size = int((x_max - x_min) / resolution)
    grid_y_size = int((y_max - y_min) / resolution)

    # Convert points to grid coordinates
    grid_x = ((filtered_points[:, 0] - x_min) / resolution).astype(np.int32)
    grid_y = ((filtered_points[:, 1] - y_min) / resolution).astype(np.int32)

    # Clip indices to ensure they are within valid range
    grid_x = np.clip(grid_x, 0, grid_x_size - 1)
    grid_y = np.clip(grid_y, 0, grid_y_size - 1)

    # Initialize density map
    density_map = np.zeros((grid_y_size, grid_x_size), dtype=np.float32)

    # Populate density map
    for gx, gy in zip(grid_x, grid_y):
        density_map[gy, gx] += 1

    # Normalize the density map (optional, scale to 0-1)
    density_map /= density_map.max() if density_map.max() > 0 else 1

    return density_map


def project_lidar_perspective(
    point_cloud, position, orientation, resolution, fov, no_data_value=-9999
):
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
    # Unpack inputs
    px, py, pz = point_cloud.T  # Point cloud coordinates
    vx, vy, vz = position  # Viewpoint position
    yaw, pitch, roll = orientation  # Viewpoint orientation
    img_width, img_height = resolution  # Image resolution

    # Step 1: Translate points to the viewpoint's position
    points = np.array([px - vx, py - vy, pz - vz]).T
    # Step 2: Apply rotation to align with the viewpoint's orientation
    rotation_matrix = get_rotation_matrix(yaw, pitch, roll)
    points = points @ rotation_matrix.T
    # print('Points transformed: ',points)
    # print("Transformed X min/max:", np.min(points[:, 0]), np.max(points[:, 0]))
    # print("Transformed Y min/max:", np.min(points[:, 1]), np.max(points[:, 1]))
    # print("Transformed Z min/max:", np.min(points[:, 2]), np.max(points[:, 2]))

    print("total number of points: ", len(points))
    points = np.asarray(points)

    # Perspective projection parameters
    focal_length = 0.5 * img_width / np.tan(0.5 * fov)  # focal length in pixels
    # print("Min/max transformed Z:", np.min(points[:, 2]), np.max(points[:, 2]))
    # Filter points in front of the camera (z > 0)
    valid = points[:, 2] > 0
    # print("Number of points in front of the camera:", np.sum(valid))
    x, y, z = points.T
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]
    pz = pz[valid]

    # Project to image plane
    u = (focal_length * x / z + img_width / 2.0).astype(int)
    v = (focal_length * y / z + img_height / 2.0).astype(int)
    # Flip v-axis to match image coordinates (in many cases, v increases downward)
    v = img_height - v

    # Debugging: Check projected values
    # print("Projected u min/max:", np.min(u), np.max(u))
    # print("Projected v min/max:", np.min(v), np.max(v))

    # Clip points to be within image bounds
    valid_pixels = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    print("number of points within view: ", np.sum(valid_pixels))
    u, v, z = u[valid_pixels], v[valid_pixels], z[valid_pixels]
    pz = pz[valid_pixels]

    # calculate elevation map
    elevation_map = np.full((img_height, img_width), no_data_value, dtype=np.float32)
    depth_map = np.full((img_height, img_width), np.inf, dtype=np.float32)
    for px, py, z_val, depth in zip(u, v, pz, z):
        if depth < depth_map[py, px]:  # Update only if this point is closer
            elevation_map[py, px] = z_val
            depth_map[py, px] = depth  # Update depth to track closest point

    return elevation_map, depth_map


def project_lidar_equirectangular(
    point_cloud, position, orientation, hfov, vfov, resolution, no_data_value=-9999
):
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
    # Unpack inputs
    px, py, pz = point_cloud.T  # Point cloud coordinates
    vx, vy, vz = position  # Viewpoint position
    yaw, pitch, roll = orientation  # Viewpoint orientation
    img_width, img_height = resolution  # Image resolution

    # Step 1: Translate points to the viewpoint's position
    points = np.array([px - vx, py - vy, pz - vz]).T

    # Step 2: Apply rotation to align with the viewpoint's orientation
    # rotation_matrix = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    rotation_matrix = get_rotation_matrix(yaw, pitch, roll)
    points = points @ rotation_matrix.T

    # Step 3: Convert to spherical coordinates
    r = np.linalg.norm(points, axis=1)  # Radial distance
    # theta = np.arctan2(points[:, 2], points[:, 0])  # Azimuth angle
    theta = np.arctan2(points[:, 0], r)  # Azimuth angle
    # phi = np.arcsin(points[:, 1] / r)  # Elevation angle
    phi = np.arctan2((-1.0) * points[:, 1], r)  # Elevation angle

    # Step 4: Filter points within the field of view
    mask = (
        (theta >= -hfov / 2)
        & (theta <= hfov / 2)
        & (phi >= -vfov / 2)
        & (phi <= vfov / 2)
    )
    print("number of points within view: ", np.sum(mask))
    points = points[mask]
    theta = theta[mask]
    phi = phi[mask]
    r = r[mask]
    pz = pz[mask]

    # Step 5: Map to image plane
    u = ((theta + hfov / 2) / hfov * img_width).astype(int)
    v = ((phi + vfov / 2) / vfov * img_height).astype(int)

    # print('min/max u: ',np.min(u),np.max(u))
    # print('min/max v: ',np.min(v),np.max(v))
    u = np.clip(u, 0, img_width - 1)
    v = np.clip(v, 0, img_height - 1)

    # Step 6: Create the elevation image
    elevation_map = np.full((img_height, img_width), no_data_value, dtype=np.float32)
    depth_map = np.full((img_height, img_width), np.inf, dtype=np.float32)
    for px, py, z_val, depth in zip(u, v, pz, r):
        if depth < depth_map[py, px]:  # Update only if this point is closer
            elevation_map[py, px] = z_val
            depth_map[py, px] = depth  # Update depth to track closest point

    return elevation_map, depth_map


# superseded
# def project_point_cloud_vertical(points, angle, pixel_size):
#     # Step 1: Define rotation matrix for the vertical plane
#     theta = np.radians(angle)
#     R = np.array([
#         [np.cos(theta), -np.sin(theta), 0],
#         [np.sin(theta), np.cos(theta), 0],
#         [0, 0, 1]
#     ])

#     # Step 2: Rotate points
#     rotated_points = points @ R.T

#     # Step 3: Use rotated x' and z for the vertical plane
#     x_prime = rotated_points[:, 0]  # Horizontal axis of the vertical plane
#     z_prime = rotated_points[:, 2]  # Elevation

#     print("x' range:", x_prime.min(), x_prime.max())
#     print("z range:", z_prime.min(), z_prime.max())
#     print("Elevation range (original z):", rotated_points[:, 2].min(), rotated_points[:, 2].max())


#     # Discretize x' and z' for a 2D grid
#     x_min, x_max = x_prime.min(), x_prime.max()
#     z_min, z_max = z_prime.min(), z_prime.max()
#     grid_x = np.arange(x_min, x_max, pixel_size)
#     grid_z = np.arange(z_min, z_max, pixel_size)

#     x_idx = np.floor((x_prime - x_min) / pixel_size).astype(int)
#     z_idx = np.floor((z_prime - z_min) / pixel_size).astype(int)

#     # Step 4: Aggregate mean elevation for each grid cell
#     elevation_map = np.full((len(grid_z), len(grid_x)), np.nan)
#     count_map = np.zeros_like(elevation_map, dtype=int)

#     for xi, zi, yi in zip(x_idx, z_idx, rotated_points[:, 2]):
#         if 0 <= xi < elevation_map.shape[1] and 0 <= zi < elevation_map.shape[0]:
#             if np.isnan(elevation_map[zi, xi]):
#                 elevation_map[zi, xi] = yi
#                 count_map[zi, xi] = 1
#             else:
#                 elevation_map[zi, xi] += yi
#                 count_map[zi, xi] += 1

#     # Compute mean elevation
#     elevation_map = elevation_map / count_map

#     return elevation_map


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
    # Rotation matrix for yaw (z-axis)
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    # Rotation matrix for pitch (y-axis)
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    # Rotation matrix for roll (x-axis)
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    # Combined rotation matrix: R = Rz * Ry * Rx
    R = np.dot(Rz, np.dot(Ry, Rx))
    # Alignment rotation matrix (real-world to camera coordinates)
    R_align = np.array(
        [
            [1, 0, 0],  # X -> X
            [0, 0, -1],  # Z -> -Y
            [0, 1, 0],  # Y -> Z
        ]
    )
    # Total rotation matrix: R_total = R_align * R
    R_total = np.dot(R_align, R)
    return R_total
