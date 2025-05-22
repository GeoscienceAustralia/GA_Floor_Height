-- Indexes for floor heights schema
CREATE INDEX IF NOT EXISTS idx_building_footprints_geom ON building_footprints USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_building_points_geom ON building_points USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_panoramas_geom ON panoramas USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_tileset_indexes_geom ON tileset_indexes USING GIST (geom);
CREATE INDEX IF NOT EXISTS building_footprints_is_residential_idx ON building_footprints(is_residential);

-- Tilesets indexes
CREATE INDEX IF NOT EXISTS tilesets_region_id_idx ON tilesets(region_id);
CREATE INDEX IF NOT EXISTS tilesets_name_idx ON tilesets(name);
CREATE INDEX IF NOT EXISTS tileset_indexes_tileset_id_idx ON tileset_indexes(tileset_id);
CREATE INDEX IF NOT EXISTS bta_building_id_idx ON building_tileset_associations(building_id);
CREATE INDEX IF NOT EXISTS bta_tileset_index_id_idx ON building_tileset_associations(tileset_index_id);
