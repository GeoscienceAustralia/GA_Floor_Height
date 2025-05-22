-- Schema tables for floor heights
CREATE TABLE regions (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    geom geometry(MultiPolygon, 4326)
);

CREATE TABLE building_footprints (
    id SERIAL PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id),
    external_id TEXT,
    gnaf_id TEXT,
    address TEXT[],
    geocode_type TEXT,
    land_use_zone TEXT,
    property_type TEXT,
    geom geometry(Polygon, 4326),
    is_residential BOOLEAN DEFAULT FALSE
);

CREATE TABLE building_points (
    id SERIAL PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id),
    footprint_id INTEGER REFERENCES building_footprints(id),
    source_point_id TEXT,
    floor_level_m DOUBLE PRECISION,
    ground_level_m DOUBLE PRECISION,
    floor_height_m DOUBLE PRECISION,
    property_type TEXT,
    wall_material TEXT,
    geom geometry(Point, 4326)
);

CREATE TABLE panoramas (
    id SERIAL PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id),
    ucid TEXT,
    system_time BIGINT,
    frame_index INTEGER,
    longitude_deg DOUBLE PRECISION,
    latitude_deg DOUBLE PRECISION,
    altitude_m DOUBLE PRECISION,
    ltp_x_m DOUBLE PRECISION,
    ltp_y_m DOUBLE PRECISION,
    ltp_z_m DOUBLE PRECISION,
    roll_deg DOUBLE PRECISION,
    pitch_deg DOUBLE PRECISION,
    heading_deg DOUBLE PRECISION,
    imgid TEXT UNIQUE,
    geom geometry(PointZ, 4326)
);

-- Tilesets: Metadata about LiDAR collections
CREATE TABLE tilesets (
    id SERIAL PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    format VARCHAR(50) NOT NULL,
    crs INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (region_id, name)
);

CREATE TABLE tileset_indexes (
    id SERIAL PRIMARY KEY,
    tileset_id INTEGER REFERENCES tilesets(id) ON DELETE CASCADE,
    file_name TEXT,
    geom geometry(Polygon, 4326)
);

CREATE TABLE building_tileset_associations (
    building_id INTEGER REFERENCES building_footprints(id) ON DELETE CASCADE,
    tileset_index_id INTEGER REFERENCES tileset_indexes(id) ON DELETE CASCADE,
    intersection_percent DOUBLE PRECISION,
    PRIMARY KEY (building_id, tileset_index_id)
);
