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
    x = math.sin(lon_house - lon_c) * math.cos(lat_house)
    y = math.cos(lat_c) * math.sin(lat_house) - math.sin(lat_c) * math.cos(lat_house) * math.cos(lon_house - lon_c)

    beta_house = math.atan2(x, y)

    if beta_house < 0:
        beta_house += 2 * math.pi
    return beta_house


def calculate_horizontal_pixel(beta_house_deg, beta_yaw_deg, wim):
    """
    Correctly maps bearing to pixels where:
    - px=0 = left edge
    - px=wim-1 = right edge
    - Camera heading points to center (wim/2)
    """
    beta_house_deg = beta_house_deg % 360
    beta_yaw_deg = beta_yaw_deg % 360

    delta_beta = (beta_house_deg - beta_yaw_deg + 180) % 360 - 180

    px = (delta_beta / 360) * wim + (wim / 2)

    px = px % wim
    px = min(max(0, px), wim - 1)

    return round(px)


def localize_house_in_panorama(lat_c, lon_c, lat_house, lon_house, beta_yaw_deg, wim, angle_extend=30):
    """
    Localize the house in the panorama by calculating the horizontal pixel position and range.

    Parameters:
        lat_c (float): Latitude of the camera (in degrees).
        lon_c (float): Longitude of the camera (in degrees).
        lat_house (float): Latitude of the house (in degrees).
        lon_house (float): Longitude of the house (in degrees).
        beta_yaw_deg (float): Yaw angle of the camera (in degrees).
        wim (int): Width of the panorama in pixels.
        angle_extend: an angle (in degrees) limit to identify the house

    Returns:
        dict: A dictionary containing the bearing angle, horizontal pixel position and location range.
    """
    lat_c_rad = math.radians(lat_c)
    lon_c_rad = math.radians(lon_c)
    lat_house_rad = math.radians(lat_house)
    lon_house_rad = math.radians(lon_house)

    beta_house_rad = calculate_bearing(lat_c_rad, lon_c_rad, lat_house_rad, lon_house_rad)
    beta_house_deg = math.degrees(beta_house_rad)

    px_house = calculate_horizontal_pixel(beta_house_deg, beta_yaw_deg, wim)

    front_door_range = (
        (beta_house_deg - angle_extend) % 360,
        (beta_house_deg + angle_extend) % 360,
    )
    px_house_range = (
        calculate_horizontal_pixel(front_door_range[0], beta_yaw_deg, wim),
        calculate_horizontal_pixel(front_door_range[1], beta_yaw_deg, wim),
    )

    return {
        "camera_house_bearing": beta_house_deg,
        "camera_house_bearing_range": front_door_range,
        "horizontal_pixel_house": px_house,
        "horizontal_pixel_range_house": px_house_range,
    }
