from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import numpy as np

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
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max)
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

def project_lidar_perspective(point_cloud, position, orientation,resolution, fov, no_data_value=-9999):
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
    rotation_matrix=get_rotation_matrix(yaw, pitch,roll)
    points = points @ rotation_matrix.T
    # print('Points transformed: ',points)
    # print("Transformed X min/max:", np.min(points[:, 0]), np.max(points[:, 0]))
    # print("Transformed Y min/max:", np.min(points[:, 1]), np.max(points[:, 1]))
    # print("Transformed Z min/max:", np.min(points[:, 2]), np.max(points[:, 2]))

    print('total number of points: ',len(points))
    points = np.asarray(points)

    # Perspective projection parameters
    focal_length = 0.5 * img_width / np.tan(0.5 * fov) # focal length in pixels
    # print("Min/max transformed Z:", np.min(points[:, 2]), np.max(points[:, 2]))
    # Filter points in front of the camera (z > 0)
    valid = points[:, 2] > 0
    # print("Number of points in front of the camera:", np.sum(valid))
    x, y, z = points.T
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]
    pz=pz[valid]

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
    print('number of points within view: ',np.sum(valid_pixels))
    u, v, z = u[valid_pixels], v[valid_pixels], z[valid_pixels]
    pz=pz[valid_pixels]

    # calculate elevation map
    elevation_map = np.full((img_height, img_width), no_data_value, dtype=np.float32)
    depth_map = np.full((img_height, img_width), np.inf, dtype=np.float32)
    for px, py, z_val, depth in zip(u, v, pz, z):
        if depth < depth_map[py, px]:  # Update only if this point is closer
            elevation_map[py, px] = z_val
            depth_map[py, px] = depth  # Update depth to track closest point

    return elevation_map, depth_map

def project_lidar_equirectangular(point_cloud, position, orientation, hfov, vfov, resolution,no_data_value=-9999):
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
    rotation_matrix=get_rotation_matrix(yaw, pitch,roll)
    points = points @ rotation_matrix.T

    # Step 3: Convert to spherical coordinates
    r = np.linalg.norm(points, axis=1)  # Radial distance
    # theta = np.arctan2(points[:, 2], points[:, 0])  # Azimuth angle
    theta = np.arctan2(points[:, 0], r)  # Azimuth angle
    # phi = np.arcsin(points[:, 1] / r)  # Elevation angle
    phi = np.arctan2((-1.0)*points[:, 1], r)  # Elevation angle

    # Step 4: Filter points within the field of view
    mask = (
        (theta >= -hfov / 2) & (theta <= hfov / 2) &
        (phi >= -vfov / 2) & (phi <= vfov / 2)
    )
    print('number of points within view: ',np.sum(mask))
    points = points[mask]
    theta = theta[mask]
    phi = phi[mask]
    r = r[mask]
    pz=pz[mask]

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
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # Rotation matrix for pitch (y-axis)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    # Rotation matrix for roll (x-axis)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    # Combined rotation matrix: R = Rz * Ry * Rx
    R = np.dot(Rz, np.dot(Ry, Rx))
    # Alignment rotation matrix (real-world to camera coordinates)
    R_align = np.array([
        [1, 0, 0], # X -> X
        [0, 0, -1], # Z -> -Y
        [0, 1, 0] # Y -> Z
    ])
    # Total rotation matrix: R_total = R_align * R
    R_total = np.dot(R_align, R)
    return R_total