# functions to localise house and restore geometry
import math
import numpy as np
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
    Y = math.cos(lat_c) * math.sin(lat_house) - math.sin(lat_c) * math.cos(lat_house) * math.cos(lon_house - lon_c)  # Equation (3)
    
    # Calculate bearing angle β_house in radians
    beta_house = math.atan2(X, Y)
    
    return beta_house

def calculate_horizontal_pixel(beta_house_deg, beta_yaw_deg, Wim):
    """
    Calculate the horizontal pixel px of the house
    
    Parameters:
        beta_house_deg (float): Bearing angle from the camera to the house in degrees.
        beta_yaw_deg (float): Yaw/bearing angle of the camera in degrees.
        Wim (int): Width of the panorama in pixels.
    
    Returns:
        float: Horizontal pixel position px.
    """
    # Normalize angles to [0, 360)
    beta_house_deg = beta_house_deg % 360
    beta_yaw_deg = beta_yaw_deg % 360
    
    # Calculate the shortest angle difference (mod 360 ensures wrapping)
    delta_beta = (beta_house_deg - beta_yaw_deg + 180) % 360 - 180
    
    # Calculate the horizontal pixel px based on the bearing angle
    px = (Wim / 2) + (delta_beta / 180) * (Wim / 2)
    
    return px

def localize_house_in_panorama(lat_c, lon_c, lat_house, lon_house, beta_yaw_deg, Wim, angle_extend=30):
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
    beta_house_rad = calculate_bearing(lat_c_rad, lon_c_rad, lat_house_rad, lon_house_rad)
    beta_house_deg = math.degrees(beta_house_rad)
    
    # Calculate the horizontal pixel position px
    px_house = calculate_horizontal_pixel(beta_house_deg, beta_yaw_deg, Wim)
    
    # Determine the possible front door location range ± angle_extend degrees from β_house
    front_door_range = (beta_house_deg - angle_extend, beta_house_deg + angle_extend)
    px_house_range = (calculate_horizontal_pixel(front_door_range[0],beta_yaw_deg,Wim),
                      calculate_horizontal_pixel(front_door_range[1],beta_yaw_deg,Wim))
    
    return {
        'camera_house_bearing': beta_house_deg,
        'camera_house_bearing_range': front_door_range,
        'horizontal_pixel_house': px_house,
        'horizontal_pixel_range_house':px_house_range,
    }

# functions to extract segmentation features and calculate FFH
def extract_feature_pixels(feature_mask, extract="bottom"):
    """
    Extracts the bottom or top pixels of a building feature (e.g., a door) from the feature mask.

    Parameters:
    - feature_mask: 2D numpy array where pixels belonging to the feature are labeled (e.g., with 1),
                    and other pixels are labeled with 0.
    - extract: str, specifies whether to extract "bottom" or "top" pixels. Defaults to "bottom".

    Returns:
    - feature_pixels: Set of (px, py) tuples representing the bottom or top pixels of the feature.
    """
    if extract not in {"bottom", "top"}:
        raise ValueError("Invalid value for 'extract'. Must be 'bottom' or 'top'.")

    feature_pixels = set()  # Set to store the extracted pixels of the feature
    # Get horizontal pixel indices (Px_feature) where there are feature pixels
    Px_feature = np.where(np.any(feature_mask == 1, axis=0))[0]
    for px in Px_feature:
        # Get vertical pixel indices (Py_feature(px)) for the given px where feature pixels are present
        Py_feature_px = np.where(feature_mask[:, px] == 1)[0]
        if Py_feature_px.size > 0:
            if extract == "bottom":
                py = Py_feature_px.max()  # Get the maximum py (bottom-most pixel)
            else:  # extract == "top"
                py = Py_feature_px.min()  # Get the minimum py (top-most pixel)
            feature_pixels.add((px, py))  # Add the pixel (px, py) to the set

    return feature_pixels


def calculate_height_difference(bottom_pixels, depth_map, H_img, upper_crop=0.25,lower_crop=0.6):
    """
    Calculate the height difference between feature bottom and camera and based on equirectangular projection of GSV pano images
    
    Parameters:
    - bottom_pixels: Set of (px, py) tuples representing the bottom pixels of the feature
    - depth_map: 2D numpy array where each value represents the depth at each pixel
    - H_img: Height (number of rows) of the panorama image
    - upper_crop: upper cropping proportions range
    - lower_crop: lower cropping proportions range
    
    Returns:
    - height_diff: height difference between the camera and the feature bottom
    """
    height_diff = []
    # original height or image before cropping
    H_img_orign=H_img/(lower_crop-upper_crop)
    # gap filling depth value
    positive_depths=depth_map[depth_map > 0]
    depth_gapfill=positive_depths.mean() if positive_depths.size > 0 else None

    for (px, py) in bottom_pixels:
        # original y coordinate before cropping
        py_orign=py+H_img_orign*upper_crop
        # Depth from the camera to the door bottom (ddb,c) from the depth map
        ddb_c = depth_map[py, px]
        # deal with abnormal depth value
        if ddb_c<1:
            ddb_c=depth_gapfill
        if ddb_c is not None:
            # Pitch angle from the camera to the door bottom (Δθdb,c)
            delta_theta_db_c=(H_img_orign/2.0-py_orign)*(180.0/H_img_orign)
            # delta_theta_db_c = ((H_img/(lower_crop-upper_crop)) *(1.0/2-upper_crop) - py) * (180 / (H_img/(lower_crop-upper_crop)))
            # Height difference between door bottom and camera (Δhdb,c)
            delta_hdb_c = ddb_c * np.sin(np.radians(delta_theta_db_c))
            height_diff.append(delta_hdb_c)

    # Returning the average height difference if there are multiple feature bottom pixels
    return np.mean(height_diff) if height_diff else None

# functions to restore geometry from RICS images
def focal_length_to_pixels(focal_length_mm, sensor_width_mm, image_width_px):
    """
    Convert focal length from millimeters to pixels.
    """
    return (focal_length_mm * image_width_px) / sensor_width_mm

def calculate_height_difference_perspective(y1, y2, Z1, Z2, focal_length_px, image_height_px):
    """
    Calculate the height difference between two points in an image with perspective projection.
    
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
    y1_prime = y1 - y_center
    y2_prime = y2 - y_center
    
    # Apply the pinhole camera model for each point
    Y1 = (y1_prime * Z1) / focal_length_px
    Y2 = (y2_prime * Z2) / focal_length_px
    
    # Calculate the height difference
    delta_Y = abs(Y2 - Y1)
    return delta_Y

def estimate_FFH(elev_foundation_top=None, elev_foundation_bottom=None,elev_stairs_top=None, elev_stairs_bottom=None, 
                 elev_frontdoor_bottom=None,elev_garagedoor_bottom=None, max_ffh=1.5, elev_camera=None):
    '''
    Calculate FFH using elevations of available features
    '''
    # determine ground elevation
    elev_ground=None
    if elev_garagedoor_bottom is not None:
        elev_ground=elev_garagedoor_bottom
    elif elev_stairs_bottom is not None:
        elev_ground=elev_stairs_bottom
    elif elev_foundation_bottom is not None:
        elev_ground=elev_foundation_bottom
    if elev_ground is None:
        return None
    
    elev_floor=None
    if elev_frontdoor_bottom is not None:
        elev_floor=elev_frontdoor_bottom
    elif elev_stairs_top is not None:
        elev_floor=elev_stairs_top
    elif elev_foundation_top is not None:
        elev_floor=elev_foundation_top
    if elev_floor is None:
        return None
    
    # calculate FFH
    FFH=elev_floor-elev_ground
    # handle unreasonable height due to partial front door
    if FFH<0 or FFH>=max_ffh:
        if elev_floor==elev_frontdoor_bottom and not all(v is None for v in [elev_stairs_top,elev_foundation_top]):
            elev_floor=max((v for v in [elev_stairs_top,elev_foundation_top] if v is not None), default=None)
            FFH_new=elev_floor-elev_ground
            if FFH>=0 and FFH_new<FFH:
                FFH=FFH_new
    if FFH<0:
        FFH=0
    if FFH>=max_ffh:
        return None
    return FFH