# GA Floor Height Pipeline

A pipeline for extracting building floor heights from LiDAR and street-view imagery data.

## Setup

### 1. Initial Setup

Run the setup script to install dependencies:

```bash
./setup.sh
```

This will:
- Create a conda environment with required packages
- Install the floor_heights package in development mode
- Set up pre-commit hooks

### 2. Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# AWS Configuration (required for data download and S3 access)
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=ap-southeast-2

# Optional: Local LiDAR data path (if not using S3)
FH_LIDAR_DATA_ROOT=/path/to/local/lidar/data

# Optional: Database path (defaults to data/floor_heights.duckdb)
FH_DB_PATH=/path/to/database.duckdb
```

### 3. Download Required Data

Before running the pipeline, download the required trajectory and tileset files:

```bash
# Download data for all regions
fh download-data

# Download data for a specific region
fh download-data -r wagga

# Preview what would be downloaded
fh download-data --dry-run
```

This downloads:
- Trajectory files (FramePosOptimised CSV files)
- Tile index shapefiles (.shp, .shx, .dbf, .prj, .cpg files)

Files are saved to:
- `data/raw/{region}/FramePosOptimised-{region}-rev2.csv`
- `data/raw/{region}/tileset/*.shp` (and related files)

## Running the Pipeline

### Run Individual Stages

```bash
# Stage 1: Clip LiDAR tiles to building footprints
fh 1 --region wagga

# Stage 2a: Harvest candidate panoramas
fh 2a --region wagga

# Stage 2b: Download panorama images
fh 2b --region wagga

# Continue with other stages...
```

### Run Multiple Stages

```bash
# Run stages 1 through 4b for all regions
fh run 1 2a 2b 3 4a 4b

# Run stages for a specific region
fh run 1 2a 2b 3 4a 4b -r wagga

# Run in background using screen
fh run 1 2a 2b 3 4a 4b --screen
```

### Available Stages

- **Stage 0**: `fh download-data` - Download required AWS data files
- **Stage 1**: `fh 1` - Clip LiDAR tiles to residential footprints
- **Stage 2a**: `fh 2a` - Harvest candidate panoramas from Street View
- **Stage 2b**: `fh 2b` - Download panorama images
- **Stage 3**: `fh 3` - Clip panoramas to building views
- **Stage 4a**: `fh 4a` - Run object detection on clipped panoramas
- **Stage 4b**: `fh 4b` - Select best view with SigLIP occlusion scoring
- **Stage 5**: `fh 5` - Project point clouds to facade rasters
- **Stage 6**: `fh 6` - Extract ground elevation from clipped LiDAR
- **Stage 7**: `fh 7` - Estimate First Floor Heights (FFH)
- **Stage 8**: `fh 8` - Validate results against ground truth

### Other Commands

```bash
# List all pipeline stages
fh stages

# List available regions
fh regions

# Check pipeline configuration
fh check

# Show pipeline info
fh info

# Database utilities
fh db status
fh db init
fh db validate

# YOLO model utilities
fh yolo check
fh yolo download
```

## Regions

The pipeline supports three regions:
- `wagga` - Wagga Wagga
- `tweed` - Tweed Heads
- `launceston` - Launceston

## Requirements
-  appropriate credentials for AWS
- ~4TB disk space for data and outputs
