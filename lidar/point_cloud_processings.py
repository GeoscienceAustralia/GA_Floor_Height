from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import numpy as np
import pdal
from scipy import ndimage
from skimage.transform import resize
from scipy.ndimage import map_coordinates, generic_filter

def project_las_to_equirectangular( input_las, camera_pos=[0, 0, 0], camera_angles=[0, 0, 0], 
                                   width=2048, height=1024, nodata_float=9999, nodata_int=255):
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
    rgb = np.vstack([points["Red"], points["Green"], points["Blue"]]).T/256 # 16 bits
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
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    R_pitch = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    R_heading = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    R_total = R_heading @ R_pitch @ R_roll  # Intrinsic XYZ order

    # Apply rotation
    coords = np.vstack([x, y, z])
    coords_local = R_total @ coords

    # Transform to camera coordinate convention:
    #    LiDAR's +Z (up) should become camera's +Y (down)
    #    LiDAR's +Y (north) should become camera's -Z (forward)
    x_cam = coords_local[0]
    y_cam = -coords_local[2]  # LiDAR Z (up) -> Camera Y (down)
    z_cam = coords_local[1]   # LiDAR Y (north) -> Camera Z (forward)
    print("Camera-relative Z:", z_cam.min(), z_cam.max())

    # --- Equirectangular Projection ---
    r = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)  # Depth
    theta = np.arctan2(x_cam, z_cam)             # Azimuth
    phi = np.arccos(-y_cam / r)                  # flip Zenith
    
    # Normalized coordinates [0,1] range
    u_norm = 0.5 * (theta/np.pi + 1)
    v_norm = phi/np.pi

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
    nodata_mask = (arr == nodata_value)
    
    # Label connected nodata regions
    labeled, num_features = ndimage.label(nodata_mask)
    
    # Measure size of each nodata region
    sizes = ndimage.sum(nodata_mask, labeled, range(num_features + 1))
    
    # Create output array
    filled = arr.copy()
    
    # Compute global distance transform once (from all valid pixels)
    distances, indices = ndimage.distance_transform_edt(
        nodata_mask,  # Important: input is the nodata mask
        return_indices=True
    )
    
    # Process each nodata region
    for i in range(1, num_features + 1):
        region_mask = (labeled == i)
        region_size = np.sum(region_mask)
        
        if region_size <= max_hole_size:
            # Fill using precomputed nearest valid pixels
            filled[region_mask] = arr[
                indices[0][region_mask], 
                indices[1][region_mask]
            ]
    print(f"Final nodata count: {np.sum(filled == nodata_value)}")
    return filled

def resize_preserve_nans(arr, target_height, target_width, nodata_value=9999):
    """
    Resizes an array while preserving NoData regions, preventing artifacts at edges.
    """
    # Create valid mask (1=valid, 0=nodata)
    valid_mask = (arr != nodata_value)
    
    # For interpolation, replace nodata with 0 but we'll mask later
    # arr_filled = np.where(valid_mask, arr, 0)
    
    # Compute scale factors
    scale_y = arr.shape[0] / target_height
    scale_x = arr.shape[1] / target_width

    # Create coordinate grids for interpolation
    y_idx, x_idx = np.meshgrid(np.linspace(0.5, arr.shape[0]-0.5, target_height),
                               np.linspace(0.5, arr.shape[1]-0.5, target_width),
                               indexing='ij')
    coords = np.array([y_idx.ravel(), x_idx.ravel()])

    # Resize mask using nearest-neighbor to keep sharp edges
    resized_mask = resize(valid_mask.astype(float),
                         (target_height, target_width),
                         order=0,  # Nearest-neighbor
                         anti_aliasing=False) > 0.5

    # Create distance-to-edge map to identify border regions
    from scipy.ndimage import distance_transform_edt
    dist_to_nodata = distance_transform_edt(valid_mask)
    edge_zone = dist_to_nodata <= 1  # Pixels adjacent to nodata

    # Interpolate main data
    # resized_data = map_coordinates(arr_filled, coords, order=1, cval=0)
    resized_data = map_coordinates(arr, coords, order=1, cval=nodata_value)
    resized_data = resized_data.reshape((target_height, target_width))

    # For edge pixels, use nearest-neighbor to prevent bleeding
    if np.any(edge_zone):
        # edge_data = map_coordinates(arr_filled, coords, order=0, cval=0)
        edge_data = map_coordinates(arr, coords, order=0, cval=nodata_value)
        edge_data = edge_data.reshape((target_height, target_width))
        
        # Find where original edge pixels map to in output
        edge_coverage = map_coordinates(edge_zone.astype(float), coords, order=1)
        edge_coverage = edge_coverage.reshape((target_height, target_width)) > 0.1
        
        # Use nearest-neighbor result for edge-affected areas
        resized_data = np.where(edge_coverage, edge_data, resized_data)

    # Restore NoData values using the resized mask
    resized_data[~resized_mask] = nodata_value
    
    return resized_data

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