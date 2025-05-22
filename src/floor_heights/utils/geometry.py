# functions to localise house and restore geometry
import math

import numpy as np
from scipy.ndimage import label
from shapely.geometry import LineString


def calculate_bearing(lat_c, lon_c, lat_house, lon_house):
    """
    Calculate the bearing angle β_house from the camera to the house

    Parameters:
        lat_c (float): Latitude of the camera (in radians).
        lon_c (float): Longitude of the camera (in radians).
        lat_house (float): Latitude of the house (in radians).
        lon_house (float): Longitude of the house (in radians).

    Returns:
        float: Bearing angle β_house in radians.
    """
    X = math.sin(lon_house - lon_c) * math.cos(lat_house)  # Equation (2)
    Y = math.cos(lat_c) * math.sin(lat_house) - math.sin(lat_c) * math.cos(
        lat_house
    ) * math.cos(lon_house - lon_c)  # Equation (3)

    # Calculate bearing angle β_house in radians
    beta_house = math.atan2(X, Y)

    # Normalize to [0, 2π) before returning
    if beta_house < 0:
        beta_house += 2 * math.pi
    return beta_house


def calculate_horizontal_pixel(beta_house_deg, beta_yaw_deg, Wim):
    """
    Correctly maps bearing to pixels where:
    - px=0 = left edge
    - px=Wim-1 = right edge
    - Camera heading points to center (Wim/2)
    """
    # Normalize angles to [0°, 360°)
    beta_house_deg = beta_house_deg % 360
    beta_yaw_deg = beta_yaw_deg % 360

    # Calculate angle difference [-180°, 180°]
    delta_beta = (beta_house_deg - beta_yaw_deg + 180) % 360 - 180

    # Convert to pixel position (0° at center, -180° at left edge, +180° at right edge)
    px = (delta_beta / 360) * Wim + (Wim / 2)

    # Wrap around and clamp
    px = px % Wim
    px = min(max(0, px), Wim - 1)

    return int(round(px))


def localize_house_in_panorama(
    lat_c, lon_c, lat_house, lon_house, beta_yaw_deg, Wim, angle_extend=30
):
    """
    Localize the house in the panorama by calculating the horizontal pixel position and range.

    Parameters:
        lat_c (float): Latitude of the camera (in degrees).
        lon_c (float): Longitude of the camera (in degrees).
        lat_house (float): Latitude of the house (in degrees).
        lon_house (float): Longitude of the house (in degrees).
        beta_yaw_deg (float): Yaw angle of the camera (in degrees).
        Wim (int): Width of the panorama in pixels.
        angle_extend: an angle (in degrees) limit to identify the house

    Returns:
        dict: A dictionary containing the bearing angle, horizontal pixel position and location range.
    """
    # Convert latitudes and longitudes to radians
    lat_c_rad = math.radians(lat_c)
    lon_c_rad = math.radians(lon_c)
    lat_house_rad = math.radians(lat_house)
    lon_house_rad = math.radians(lon_house)

    # Calculate the bearing angle β_house in degrees
    beta_house_rad = calculate_bearing(
        lat_c_rad, lon_c_rad, lat_house_rad, lon_house_rad
    )
    beta_house_deg = math.degrees(beta_house_rad)

    # Calculate the horizontal pixel position px
    px_house = calculate_horizontal_pixel(beta_house_deg, beta_yaw_deg, Wim)

    # Determine the possible front door location range ± angle_extend degrees from β_house
    # Calculate bearing range (normalized to [0°, 360°))
    front_door_range = (
        (beta_house_deg - angle_extend) % 360,
        (beta_house_deg + angle_extend) % 360,
    )
    px_house_range = (
        calculate_horizontal_pixel(front_door_range[0], beta_yaw_deg, Wim),
        calculate_horizontal_pixel(front_door_range[1], beta_yaw_deg, Wim),
    )

    return {
        "camera_house_bearing": beta_house_deg,
        "camera_house_bearing_range": front_door_range,
        "horizontal_pixel_house": px_house,
        "horizontal_pixel_range_house": px_house_range,
    }


# # functions to extract segmentation features and calculate FFH
# def extract_feature_pixels(feature_mask, extract="bottom"):
#     """
#     Extracts the bottom or top pixels of a building feature (e.g., a door) from the feature mask.

#     Parameters:
#     - feature_mask: 2D numpy array where pixels belonging to the feature are labeled (e.g., with 1),
#                     and other pixels are labeled with 0.
#     - extract: str, specifies whether to extract "bottom" or "top" pixels. Defaults to "bottom".

#     Returns:
#     - feature_pixels: Set of (px, py) tuples representing the bottom or top pixels of the feature.
#     """
#     if extract not in {"bottom", "top"}:
#         raise ValueError("Invalid value for 'extract'. Must be 'bottom' or 'top'.")

#     feature_pixels = set()  # Set to store the extracted pixels of the feature
#     # Get horizontal pixel indices (Px_feature) where there are feature pixels
#     Px_feature = np.where(np.any(feature_mask == 1, axis=0))[0]
#     for px in Px_feature:
#         # Get vertical pixel indices (Py_feature(px)) for the given px where feature pixels are present
#         Py_feature_px = np.where(feature_mask[:, px] == 1)[0]
#         if Py_feature_px.size > 0:
#             if extract == "bottom":
#                 py = Py_feature_px.max()  # Get the maximum py (bottom-most pixel)
#             else:  # extract == "top"
#                 py = Py_feature_px.min()  # Get the minimum py (top-most pixel)
#             feature_pixels.add((px, py))  # Add the pixel (px, py) to the set

#     return feature_pixels


def extract_feature_pixels_lowest_region(feature_mask, extract="bottom"):
    """
    Extracts the bottom or top pixels of the lowest connected region (by bottom-most extent)
    from the feature mask.

    Parameters:
    - feature_mask: 2D numpy array where pixels belonging to the feature are labeled (e.g., with 1),
                    and other pixels are labeled with 0.
    - extract: str, specifies whether to extract "bottom" or "top" pixels. Defaults to "bottom".

    Returns:
    - feature_pixels: Set of (px, py) tuples representing the bottom or top pixels
                      of the lowest connected region of the feature.
    """
    if extract not in {"bottom", "top"}:
        raise ValueError("Invalid value for 'extract'. Must be 'bottom' or 'top'.")

    # Label connected regions in the feature mask
    labeled_mask, num_features = label(feature_mask)

    lowest_region_id = None
    lowest_bottom = None  # Track the bottom-most extent of the lowest region

    # Identify the lowest region based on bottom-most extent
    for feature_id in range(1, num_features + 1):
        region_mask = labeled_mask == feature_id
        vertical_indices = np.where(region_mask)[
            0
        ]  # Get vertical positions (row indices)

        if vertical_indices.size > 0:
            bottom_position = vertical_indices.max()  # Bottom-most pixel in this region

            # Update the lowest region if necessary
            if lowest_bottom is None or bottom_position > lowest_bottom:
                lowest_bottom = bottom_position
                lowest_region_id = feature_id

    # If no regions are found, return an empty set
    if lowest_region_id is None:
        return set()

    # Extract pixels from the lowest region
    feature_pixels = set()
    region_mask = labeled_mask == lowest_region_id
    Px_feature = np.where(np.any(region_mask, axis=0))[0]
    for px in Px_feature:
        Py_feature_px = np.where(region_mask[:, px] == 1)[0]
        if Py_feature_px.size > 0:
            if extract == "bottom":
                py = Py_feature_px.max()  # Bottom-most pixel
            else:  # extract == "top"
                py = Py_feature_px.min()  # Top-most pixel
            feature_pixels.add((px, py))

    return feature_pixels


def calculate_height_difference(
    bottom_pixels, depth_map, H_img, upper_crop=0.25, lower_crop=0.6, depth_gapfill=None
):
    """
    Calculate the height difference between feature bottom and camera and based on equirectangular projection of GSV pano images

    Parameters:
    - bottom_pixels: Set of (px, py) tuples representing the bottom pixels of the feature
    - depth_map: 2D numpy array where each value represents the depth at each pixel
    - H_img: Height (number of rows) of the panorama image
    - upper_crop: upper cropping proportions range
    - lower_crop: lower cropping proportions range
    - depth_gapfill: gapfilling depth value

    Returns:
    - height_diff: height difference between the camera and the feature bottom
    """
    if bottom_pixels is None:
        return None
    height_diff = []
    # original height or image before cropping
    H_img_orign = H_img / (lower_crop - upper_crop)
    # gap filling depth value
    # positive_depths=depth_map[depth_map > 0]
    # depth_gapfill=positive_depths.mean() if positive_depths.size > 0 else None

    for px, py in bottom_pixels:
        # deal with boundary detection results that somehow could happen to object detection results
        px = min(px, depth_map.shape[1] - 1)
        py = min(py, depth_map.shape[0] - 1)
        # original y coordinate before cropping
        py_orign = py + H_img_orign * upper_crop
        # Depth from the camera to the door bottom (ddb,c) from the depth map
        ddb_c = depth_map[py, px]
        # deal with abnormal depth values
        ddb_c = depth_gapfill
        # if ddb_c<5:
        #     print('using gap filling depth')
        #     ddb_c=depth_gapfill
        if ddb_c is not None:
            # Pitch angle from the camera to the door bottom (Δθdb,c)
            delta_theta_db_c = (H_img_orign / 2.0 - py_orign) * (180.0 / H_img_orign)
            # delta_theta_db_c = ((H_img/(lower_crop-upper_crop)) *(1.0/2-upper_crop) - py) * (180 / (H_img/(lower_crop-upper_crop)))
            # Height difference between door bottom and camera (Δhdb,c)
            delta_hdb_c = ddb_c * np.sin(np.radians(delta_theta_db_c))
            height_diff.append(delta_hdb_c)

    # Returning the average height difference if there are multiple feature bottom pixels
    return np.mean(height_diff) if height_diff else None


def calculate_width_difference(
    left_pixels, right_pixels, depth_map, W_img, angle_extend=40, depth_gapfill=None
):
    """
    Calculate the width difference on equirectangular projection of GSV pano images

    Parameters:
    - left_pixels: Set of (px, py) tuples representing the left side pixels of the feature
    - right_pixels: Set of (px, py) tuples representing the right side pixels of the feature
    - depth_map: 2D numpy array where each value represents the depth at each pixel
    - W_img: Width (number of columns) of the panorama image
    - angle_extend: the horizontal expand angle (in degrees) of the full pano image
    - depth_gapfill: gapfilling depth value

    Returns:
    - width_diff: width difference between the camera and the feature bottom
    """
    if (left_pixels is None) or (right_pixels is None):
        return None

    # mean of left and right pixels
    x_left, y_left = (
        np.mean([px[0] for px in left_pixels]),
        np.mean([px[1] for px in left_pixels]),
    )
    x_right, y_right = (
        np.mean([px[0] for px in right_pixels]),
        np.mean([px[1] for px in right_pixels]),
    )

    # Compute bounding box center
    x_c, y_c = np.mean([x_left, x_right]), np.mean([y_left, y_right])

    # depth of centre
    # deal with boundary detection results that somehow could happen to object detection results
    x_c = min(x_c, depth_map.shape[1] - 1)
    y_c = min(y_c, depth_map.shape[0] - 1)

    # D_c=depth_map[y_c, x_c] # using depth map
    D_c = depth_gapfill

    # calculate real-world width
    delta_phi = ((x_right - x_left) / W_img) * angle_extend  # Horizontal FOV
    width_diff = 2 * D_c * np.tan(np.radians(delta_phi / 2))
    return width_diff


# functions to restore geometry from RICS images
def focal_length_to_pixels(focal_length_mm, sensor_width_mm, image_width_px):
    """
    Convert focal length from millimeters to pixels.
    """
    return (focal_length_mm * image_width_px) / sensor_width_mm


# def calculate_height_difference_perspective(y1, y2, Z1, Z2, focal_length_px, image_height_px):
#     """
#     Calculate the height difference between two points in an image with perspective projection.

#     Parameters:
#     - y1, y2: Vertical pixel coordinates of the two points.
#     - Z1, Z2: Distances to the building points in real-world units (e.g., meters).
#     - focal_length_px: Focal length in pixels.
#     - image_height_px: Height of the image in pixels.

#     Returns:
#     - Height difference in real-world units.
#     """
#     # Calculate the vertical distances from the principal point (assumed to be the center of the image)
#     y_center = image_height_px / 2
#     y1_prime = y1 - y_center
#     y2_prime = y2 - y_center

#     # Apply the pinhole camera model for each point
#     Y1 = (y1_prime * Z1) / focal_length_px
#     Y2 = (y2_prime * Z2) / focal_length_px

#     # Calculate the height difference
#     delta_Y = abs(Y2 - Y1)
#     return delta_Y


def calculate_height_difference_perspective(
    y_coords, distances, focal_length_px, image_height_px
):
    """
    Calculate the height difference between points and centre of image.

    Parameters:
    - y1, y2: Vertical pixel coordinates of the two points.
    - Z1, Z2: Distances to the building points in real-world units (e.g., meters).
    - focal_length_px: Focal length in pixels.
    - image_height_px: Height of the image in pixels.

    Returns:
    - Height difference in real-world units.
    """
    # Calculate the vertical distances from the principal point (assumed to be the center of the image)
    y_center = image_height_px / 2
    height_diffs = []
    for y, distance in zip(y_coords, distances):
        y_prime = y - y_center
        # Apply the pinhole camera model for each point
        height_diffs.append((y_prime * distance) / focal_length_px)
    if len(height_diffs) < 1:
        return None
    else:
        return np.mean(height_diffs)
    # return height_diffs


def estimate_FFH(
    delta_foundation_top=None,
    delta_foundation_bottom=None,
    delta_stairs_top=None,
    delta_stairs_bottom=None,
    delta_frontdoor_bottom=None,
    delta_garagedoor_bottom=None,
    max_ffh=2,
):
    """
    Calculate FFH using elevation difference between available features and camera.
    """
    # determine ground elevation
    delta_elev_ground = None
    if delta_garagedoor_bottom is not None:
        delta_elev_ground = delta_garagedoor_bottom
    elif delta_stairs_bottom is not None:
        delta_elev_ground = delta_stairs_bottom
    elif delta_foundation_bottom is not None:
        delta_elev_ground = delta_foundation_bottom
    if delta_elev_ground is None:
        return None

    # determine floor elevation
    delta_elev_floor = None
    if delta_frontdoor_bottom is not None:
        delta_elev_floor = delta_frontdoor_bottom
    elif delta_stairs_top is not None:
        delta_elev_floor = delta_stairs_top
    elif delta_foundation_top is not None:
        delta_elev_floor = delta_foundation_top
    if delta_elev_floor is None:
        return None

    # calculate difference as FFH
    FFH = delta_elev_floor - delta_elev_ground
    # handle unreasonable height due to partial front door
    if FFH < 0 or FFH >= max_ffh:
        if delta_elev_floor == delta_frontdoor_bottom and not all(
            v is None for v in [delta_stairs_top, delta_foundation_top]
        ):
            delta_elev_floor = max(
                (v for v in [delta_stairs_top, delta_foundation_top] if v is not None),
                default=None,
            )
            FFH_new = delta_elev_floor - delta_elev_ground
            if FFH >= 0 and FFH_new < FFH:
                FFH = FFH_new
    if FFH < 0 or FFH >= max_ffh:
        return None
    return FFH


def estimate_FFE(
    delta_foundation_top=None,
    delta_stairs_top=None,
    delta_frontdoor_bottom=None,
    delta_garagedoor_bottom=None,
    elev_camera=None,
):
    """
    Calculate FFE using elevation of features and GSV camera elevations
    """
    # determine floor elevation
    delta_elev_floor = None
    if delta_frontdoor_bottom is not None:
        delta_elev_floor = delta_frontdoor_bottom
    elif delta_garagedoor_bottom is not None:
        delta_elev_floor = delta_garagedoor_bottom
    elif delta_stairs_top is not None:
        delta_elev_floor = delta_stairs_top
    elif delta_foundation_top is not None:
        delta_elev_floor = delta_foundation_top
    if delta_elev_floor is None:
        return None

    # calculate FFE
    FFE = delta_elev_floor + elev_camera
    return FFE


def estimate_Z_for_points(
    building_outline,
    building_center,
    camera_position,
    image_points,
    focal_length_px,
    sensor_width_mm,
    image_width_px,
):
    """
    Estimate the distance (Z) to specific points on the building using building outline geometry and image coordinates.

    Parameters:
    - building_outline: Polygon representing the building outline.
    - building_center: Point representing the building center.
    - camera_position: Tuple (x, y, z) representing the camera's real-world position.
    - image_points: List of (x, y) tuples representing the points in image coordinates.
    - focal_length_px: Focal length in pixels.
    - sensor_width_mm: Sensor width in mm.
    - image_width_px: Image width in pixels.

    Returns:
    - List of estimated Z distances and y coordinates for each point.
    """
    # Calculate the field of view (FOV) of the camera in radians
    fov = 2 * math.atan(
        (sensor_width_mm / 2) / (focal_length_px / (image_width_px / sensor_width_mm))
    )
    # Calculate distances from the camera to each point in the image
    Z_distances = []
    y_coords = []
    for point in image_points:
        # Calculate the angle of the point relative to the principal point (center of the image)
        x = point[0] - (image_width_px / 2)
        angle = (x / focal_length_px) * (fov / image_width_px)
        # Create a line from the camera position in the direction of the point
        direction_line = LineString(
            [
                camera_position,
                (
                    camera_position.x + math.cos(angle),
                    camera_position.y + math.sin(angle),
                ),
            ]
        )
        # Calculate intersection with the building outline
        intersection = building_outline.intersection(direction_line)
        # If there is an intersection, calculate the distance
        if not intersection.is_empty:
            Z = camera_position.distance(intersection)
        else:
            # If no intersection, estimate Z as the distance to the building center
            Z = camera_position.distance(building_center)
        Z_distances.append(Z)
        y_coords.append(point[1])
    # if len(Z_distances)>0:
    #     z_mean=np.mean(Z_distances)
    # else:
    #     z_mean=None
    return Z_distances, y_coords
