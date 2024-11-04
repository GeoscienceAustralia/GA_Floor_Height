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
def extract_feature_bottom(feature_mask):
    """
    Extracts the bottom pixels of a building feature (e.g. the door) from the feature mask.
    
    Parameters:
    - feature_mask: 2D numpy array where pixels belonging to the feature are labeled (e.g., with 1), 
                 and other pixels are labeled with 0.
                 
    Returns:
    - bottom_pixels: Set of (px, py) tuples representing the bottom pixels of the feature.
    """
    bottom_pixels = set()  # Set to store bottom pixels of the feature
    # Get horizontal pixel indices (Px_feature) where there are feature pixels
    Px_feature = np.where(np.any(feature_mask == 1, axis=0))[0]
    for px in Px_feature:
        # Get vertical pixel indices (Py_feature(px)) for the given px where feature pixels are present
        Py_door_px = np.where(feature_mask[:, px] == 1)[0]
        # Get the maximum py (bottom-most pixel) for the current px
        if Py_door_px.size > 0:
            py = Py_door_px.max()
            bottom_pixels.add((px, py))  # Add the bottom pixel (px, py) to the set
    return bottom_pixels

def calculate_height_difference(bottom_pixels, depth_map, H_img, upper_crop=0.25,lower_crop=0.6):
    """
    Calculate the height difference between the camera and the feature bottom
    
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

    for (px, py) in bottom_pixels:
        # original y coordinate before cropping
        py_orign=py+H_img_orign*upper_crop
        # Depth from the camera to the door bottom (ddb,c) from the depth map
        ddb_c = depth_map[py, px]
        # Pitch angle from the camera to the door bottom (Δθdb,c)
        delta_theta_db_c=(H_img_orign/2.0-py_orign)*(180.0/H_img_orign)
        # delta_theta_db_c = ((H_img/(lower_crop-upper_crop)) *(1.0/2-upper_crop) - py) * (180 / (H_img/(lower_crop-upper_crop)))
        # Height difference between the camera and the door bottom (Δhdb,c)
        delta_hdb_c = ddb_c * np.sin(np.radians(delta_theta_db_c))
        height_diff.append(delta_hdb_c)

    # Returning the average height difference if there are multiple feature bottom pixels
    return np.mean(height_diff) if height_diff else None