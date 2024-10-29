# functions to localise house and restore geometry
import math
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