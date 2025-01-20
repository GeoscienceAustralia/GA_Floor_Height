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

def project_point_cloud_vertical(points, angle, pixel_size):
    # Step 1: Define rotation matrix for the vertical plane
    theta = np.radians(angle)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Step 2: Rotate points
    rotated_points = points @ R.T
    
    # Step 3: Use rotated x' and z for the vertical plane
    x_prime = rotated_points[:, 0]  # Horizontal axis of the vertical plane
    z_prime = rotated_points[:, 2]  # Elevation

    print("x' range:", x_prime.min(), x_prime.max())
    print("z range:", z_prime.min(), z_prime.max())
    print("Elevation range (original z):", rotated_points[:, 2].min(), rotated_points[:, 2].max())

    
    # Discretize x' and z' for a 2D grid
    x_min, x_max = x_prime.min(), x_prime.max()
    z_min, z_max = z_prime.min(), z_prime.max()
    grid_x = np.arange(x_min, x_max, pixel_size)
    grid_z = np.arange(z_min, z_max, pixel_size)
    
    x_idx = np.floor((x_prime - x_min) / pixel_size).astype(int)
    z_idx = np.floor((z_prime - z_min) / pixel_size).astype(int)
    
    # Step 4: Aggregate mean elevation for each grid cell
    elevation_map = np.full((len(grid_z), len(grid_x)), np.nan)
    count_map = np.zeros_like(elevation_map, dtype=int)
    
    for xi, zi, yi in zip(x_idx, z_idx, rotated_points[:, 2]):
        if 0 <= xi < elevation_map.shape[1] and 0 <= zi < elevation_map.shape[0]:
            if np.isnan(elevation_map[zi, xi]):
                elevation_map[zi, xi] = yi
                count_map[zi, xi] = 1
            else:
                elevation_map[zi, xi] += yi
                count_map[zi, xi] += 1
    
    # Compute mean elevation
    elevation_map = elevation_map / count_map
    
    return elevation_map