from pathlib import Path

import ibis

from floor_heights.config import CONFIG, REGION_CONFIGS

from .ibis_client import connect
from .mappings import RESIDENTIAL_ZONES


def load_from_parquet(db_path: Path = Path("data/floor_heights.duckdb")) -> None:
    conn = connect(db_path, read_only=False)

    core_tables = [
        "regions",
        "buildings",
        "panoramas",
        "tilesets",
        "building_features",
        "building_features_unique",
        "validation_labels",
        "frontiersi_validation_labels",
    ]
    existing_tables = conn.list_tables()

    for table in core_tables:
        if table in existing_tables:
            conn.raw_sql(f"DROP TABLE IF EXISTS {table}")
            print(f"  Dropped existing table: {table}")

    base_dir = Path("data/processed")
    create_unified_tables(conn, base_dir)


def create_unified_tables(conn, base_dir: Path) -> None:
    conn.raw_sql("""
        CREATE TABLE regions (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            geom GEOMETRY
        )
    """)

    for region_name, config in REGION_CONFIGS.items():
        bbox = config.get("bbox", {})
        if bbox:
            conn.raw_sql(f"""
                INSERT INTO regions (id, name, geom)
                VALUES ({config["id"]}, '{region_name}', ST_MakeEnvelope({bbox["xmin"]}, {bbox["ymin"]}, {bbox["xmax"]}, {bbox["ymax"]}))
            """)

    print("✓ Created regions table with bounding boxes")

    tables = []
    for region in ["wagga", "tweed", "launceston"]:
        parquet_file = base_dir / region / "panoramas.parquet"
        if parquet_file.exists():
            config = REGION_CONFIGS[region]
            t = conn.read_parquet(str(parquet_file))
            t = t.mutate(region_id=ibis.literal(config["id"]), region_name=ibis.literal(region))
            tables.append(t)

    if tables:
        panoramas = ibis.union(*tables)
        conn.create_table("panoramas", panoramas, overwrite=True)
        print("✓ Created panoramas table")

    tables = []
    for region in ["wagga", "tweed", "launceston"]:
        parquet_file = base_dir / region / "tileset.parquet"
        if parquet_file.exists():
            config = REGION_CONFIGS[region]
            t = conn.read_parquet(str(parquet_file))

            special_mappings = {
                "classifica": "classification",
                "roadname": "road_name",
                "tileindexn": "tile_index",
                "geom": "geometry",
            }

            def standardise_cols(expr, mappings):
                return expr.select(*[expr[c].name(mappings.get(c.lower(), c.lower())) for c in expr.columns])

            t = standardise_cols(t, special_mappings)

            t = t.mutate(region_id=ibis.literal(config["id"]), region_name=ibis.literal(region))

            if "bounds" not in t.columns:
                t = t.mutate(bounds=ibis.literal(None, type="string"))

            tables.append(t)

    if tables:
        tilesets = ibis.union(*tables)
        conn.create_table("tilesets", tilesets, overwrite=True)
        print("✓ Created tilesets table")

    buildings_sql = f"""
    CREATE TABLE buildings AS
    WITH all_footprints AS (
        SELECT * FROM '{base_dir}/buildings.parquet'
    ),
    footprints_filtered AS (
        SELECT *
        FROM all_footprints
        WHERE land_use_zone IN ({", ".join([f"'{zone}'" for zone in RESIDENTIAL_ZONES])})
    ),
    tile_extents AS (
        SELECT
            region_id,
            ANY_VALUE(region_name) as region_name,
            ST_Union_Agg(geometry) as tile_extent
        FROM tilesets
        GROUP BY region_id
    ),
    footprints_with_region AS (
        SELECT
            f.*,
            COALESCE(
                (SELECT r.id FROM regions r WHERE ST_Contains(r.geom, ST_Centroid(f.geom)) LIMIT 1),
                (SELECT r.id FROM regions r ORDER BY ST_Distance(ST_Centroid(f.geom), ST_Centroid(r.geom)) LIMIT 1)
            ) as region_id
        FROM footprints_filtered f
    )
    SELECT
        f.id,
        f.building_id,
        f.region_id,
        r.name as region_name,
        f.gnaf_id,
        f.gnaf_address,
        f.geocode_type,
        f.land_use_zone,
        f.geom as footprint_geom,

        COALESCE(f.primary_secondary, 'primary') as primary_secondary,
        'Residential' as property_type,

        ST_Centroid(f.geom) as point_geom,

        CURRENT_TIMESTAMP as processed_at

    FROM footprints_with_region f
    JOIN regions r ON f.region_id = r.id
    WHERE EXISTS (
        SELECT 1
        FROM tile_extents te
        WHERE te.region_id = f.region_id
          AND ST_Intersects(f.geom, te.tile_extent)
    )
    """

    conn.raw_sql(buildings_sql)

    row_count = conn.table("buildings").count().execute()
    print(f"✓ Created buildings table with {row_count:,} rows")

    buildings = conn.table("buildings")
    breakdown = buildings.group_by("region_name").agg(count=buildings.count()).order_by("region_name").execute()

    for _, row in breakdown.iterrows():
        print(f"  - {row['region_name']}: {row['count']:,} buildings")

    if (base_dir / "building_features_all.parquet").exists():
        building_features_sql = f"""
        CREATE TABLE building_features AS
        WITH all_features AS (
            SELECT * FROM '{base_dir}/building_features_all.parquet'
        ),
        features_filtered AS (
            SELECT *
            FROM all_features
            WHERE land_use_zone IN ({", ".join([f"'{zone}'" for zone in RESIDENTIAL_ZONES])})
        ),
        features_with_region AS (
            SELECT
                f.*,
                COALESCE(
                    (SELECT r.id FROM regions r WHERE ST_Contains(r.geom, ST_Centroid(f.geom)) LIMIT 1),
                    (SELECT r.id FROM regions r ORDER BY ST_Distance(ST_Centroid(f.geom), ST_Centroid(r.geom)) LIMIT 1)
                ) as region_id,
                COALESCE(
                    (SELECT r.name FROM regions r WHERE ST_Contains(r.geom, ST_Centroid(f.geom)) LIMIT 1),
                    (SELECT r.name FROM regions r ORDER BY ST_Distance(ST_Centroid(f.geom), ST_Centroid(r.geom)) LIMIT 1)
                ) as region_name
            FROM features_filtered f
        )
        SELECT * FROM features_with_region
        """
        conn.raw_sql(building_features_sql)

        features_count = conn.table("building_features").count().execute()
        features_cols = len(conn.table("building_features").columns)
        print(f"\n✓ Created building_features table: {features_count:,} rows, {features_cols} columns")

        conn.raw_sql("""
            CREATE TABLE building_features_unique AS
            SELECT DISTINCT ON (building_id) *
            FROM building_features
            ORDER BY building_id, gnaf_id
        """)
        unique_count = conn.table("building_features_unique").count().execute()
        print(f"✓ Created building_features_unique table: {unique_count:,} rows (deduplicated by building_id)")

    if (base_dir / "validation_labels.parquet").exists():
        conn.raw_sql(f"""
            CREATE TABLE validation_labels AS
            SELECT * FROM '{base_dir}/validation_labels.parquet'
        """)
        validation_count = conn.table("validation_labels").count().execute()
        print(f"\n✓ Created validation_labels table: {validation_count:,} rows")

    if (base_dir / "validation_frontiersi.parquet").exists():
        conn.raw_sql(f"""
            CREATE TABLE frontiersi_validation_labels AS
            SELECT * FROM '{base_dir}/validation_frontiersi.parquet'
        """)
        frontiersi_count = conn.table("frontiersi_validation_labels").count().execute()
        print(
            f"✓ Created frontiersi_validation_labels table: {frontiersi_count:,} rows (from validation_frontiersi.parquet)"
        )


def convert_to_parquet() -> None:
    conn = connect(":memory:", read_only=False)

    data_dir = Path("data")
    processed = data_dir / "processed"

    for region in ["wagga", "tweed", "launceston"]:
        (processed / region).mkdir(parents=True, exist_ok=True)

    buildings_path = CONFIG.project_root / "data" / "all_aoi_ffh_v5_3a2a2ee6e864.gpkg"
    if not buildings_path.exists():
        raise FileNotFoundError(f"Required file not found: {buildings_path}")

    region_values = []
    for region_name, config in REGION_CONFIGS.items():
        bbox = config.get("bbox", {})
        if bbox:
            region_values.append(f"""
                SELECT '{region_name}' as name,
                       ST_MakeEnvelope({bbox["xmin"]}, {bbox["ymin"]}, {bbox["xmax"]}, {bbox["ymax"]}) as geom
            """)

    if region_values:
        regions_union = " UNION ALL ".join(region_values)
        conn.raw_sql(f"""
            CREATE OR REPLACE TEMPORARY TABLE temp_regions AS
            {regions_union}
        """)

    conn.raw_sql(f"""
        COPY (
            WITH all_labels AS (
                SELECT
                    ROW_NUMBER() OVER (ORDER BY building_id, gnaf_id) AS id,
                    building_id,
                    gnaf_id,
                    CASE WHEN dataset = 'FrontierSI Validation' THEN floor_height_m ELSE NULL END as frontiersi_floor_height_m,
                    CASE WHEN dataset = 'NEXIS' THEN floor_height_m ELSE NULL END as nexis_floor_height_m,
                    CASE WHEN dataset = 'Council Validation' AND method = 'Surveyed' THEN floor_height_m ELSE NULL END as council_surveyed_floor_height_m,
                    CASE WHEN dataset = 'Council Validation' AND method = 'Step counting' THEN floor_height_m ELSE NULL END as council_step_floor_height_m,
                    CASE WHEN dataset = 'FrontierSI Validation' THEN floor_absolute_height_m ELSE NULL END as frontiersi_floor_absolute_height_m,
                    min_building_height_ahd,
                    max_building_height_ahd,
                    CASE WHEN dataset = 'Council Validation' THEN floor_leve ELSE NULL END as council_floor_leve,
                    CASE WHEN dataset = 'Council Validation' THEN ground_level ELSE NULL END as council_ground_level,
                    dataset,
                    method
                FROM ST_Read('{buildings_path}')
                WHERE floor_height_m IS NOT NULL
            )
            SELECT * FROM all_labels
        )
        TO '{processed}/validation_labels.parquet' (FORMAT 'PARQUET')
    """)
    print("✓ Extracted all validation labels with source-specific columns to validation_labels.parquet")

    conn.raw_sql(f"""
        COPY (
            SELECT DISTINCT ON (building_id)
                ROW_NUMBER() OVER (ORDER BY building_id) AS id,
                building_id,
                gnaf_id,
                floor_height_m,
                floor_absolute_height_m,
                min_building_height_ahd,
                max_building_height_ahd
            FROM ST_Read('{buildings_path}')
            WHERE dataset = 'FrontierSI Validation'
              AND floor_height_m IS NOT NULL
            ORDER BY building_id, gnaf_id
        )
        TO '{processed}/validation_frontiersi.parquet' (FORMAT 'PARQUET')
    """)
    print("✓ Extracted FrontierSI validation data to validation_frontiersi.parquet (deduplicated by building_id)")

    conn.raw_sql(f"""
        COPY (
            SELECT
                ROW_NUMBER() OVER (ORDER BY building_id, gnaf_id) AS id,
                *
            FROM ST_Read('{buildings_path}')
        )
        TO '{processed}/building_features_all.parquet' (FORMAT 'PARQUET')
    """)
    print(
        "✓ Converted all_aoi_ffh_v5_3a2a2ee6e864.gpkg to building_features_all.parquet with deterministic IDs (including ALL records)"
    )

    conn.raw_sql(f"""
        COPY (
            SELECT DISTINCT ON (building_id)
                ROW_NUMBER() OVER (ORDER BY building_id) AS id,
                building_id,
                gnaf_id,
                gnaf_address,
                geocode_type,
                primary_secondary,
                land_use_zone,
                geom
            FROM ST_Read('{buildings_path}')
            ORDER BY building_id, gnaf_id
        )
        TO '{processed}/buildings.parquet' (FORMAT 'PARQUET')
    """)
    print("✓ Created minimal buildings.parquet with essential columns (deduplicated by building_id)")

    regions = {
        "wagga": {
            "panoramas": "FramePosOptimised-wagga-wagga-rev2.csv",
            "tileset": "tileset/48068_Wagga_Wagga_TileSet.shp",
        },
        "tweed": {
            "panoramas": "FramePosOptimised-tweed-heads-rev2.csv",
            "tileset": "tileset/48068_Tweed_Heads_TileSet.shp",
        },
        "launceston": {
            "panoramas": "FramePosOptimised-launceston-rev2.csv",
            "tileset": "tileset/48068_Launceston_TileSet.shp",
        },
    }

    for region, files in regions.items():
        raw_dir = data_dir / "raw" / region
        out_dir = processed / region
        config = REGION_CONFIGS[region]

        csv_path = raw_dir / files["panoramas"]
        if csv_path.exists():
            conn.raw_sql(f"""
                COPY (
                    SELECT *, ST_Point(longitude_deg, latitude_deg) as geometry
                    FROM read_csv_auto('{csv_path}', header=true)
                ) TO '{out_dir}/panoramas.parquet' (FORMAT 'PARQUET')
            """)
            print(f"✓ Converted {region} panoramas (WGS84)")

        tile_path = raw_dir / files["tileset"]
        if tile_path.exists():
            conn.raw_sql(f"""
                COPY (
                    SELECT
                        * EXCLUDE (geom),
                        ST_FlipCoordinates(ST_Transform(geom, 'EPSG:{config["crs"]}', 'EPSG:7844')) as geom
                    FROM ST_Read('{tile_path}')
                )
                TO '{out_dir}/tileset.parquet' (FORMAT 'PARQUET')
            """)
            print(f"✓ Converted {region} tileset (GDA2020 EPSG:{config['crs']} -> GDA2020 Geographic EPSG:7844)")

    if (processed / "validation_labels.parquet").exists():
        validation_summary = conn.raw_sql(f"""
            SELECT
                dataset,
                method,
                COUNT(*) as total_records,
                COUNT(frontiersi_floor_height_m) as frontiersi_labels,
                COUNT(nexis_floor_height_m) as nexis_labels,
                COUNT(council_surveyed_floor_height_m) as council_surveyed_labels,
                COUNT(council_step_floor_height_m) as council_step_labels
            FROM read_parquet('{processed}/validation_labels.parquet')
            GROUP BY dataset, method
            ORDER BY total_records DESC
        """).fetchall()

        print("\n✓ Validation Labels Summary:")
        for row in validation_summary:
            dataset, method = row[0], row[1]
            total = row[2]
            print(f"  - {dataset} ({method}): {total:,} records")

    if (processed / "validation_frontiersi.parquet").exists():
        frontiersi_count = conn.raw_sql(f"""
            SELECT COUNT(DISTINCT building_id) as unique_buildings
            FROM read_parquet('{processed}/validation_frontiersi.parquet')
        """).fetchone()[0]
        print(f"\n✓ FrontierSI validation buildings: {frontiersi_count} unique buildings")


if __name__ == "__main__":
    import sys

    def usage():
        print("""
Usage: python -m src.floor_heights.duck.loader [command]

Commands:
    convert  - Convert raw data to GeoParquet
    load     - Load GeoParquet into DuckDB
    all      - Run full pipeline (convert + load)

Example:
    python -m src.floor_heights.duck.loader all
""")

    command = sys.argv[1] if len(sys.argv) > 1 else "all"

    if command == "--help" or command == "-h":
        usage()
        sys.exit(0)

    if command not in ["convert", "load", "all"]:
        print(f"Error: Unknown command '{command}'")
        usage()
        sys.exit(1)

    if command in ["convert", "all"]:
        print("Converting raw data to GeoParquet...")
        convert_to_parquet()

    if command in ["load", "all"]:
        print("Loading GeoParquet into DuckDB...")
        db_path = Path("data/floor_heights.duckdb")
        load_from_parquet(db_path)

        conn = connect(db_path, read_only=True)
        tables_to_check = [
            "regions",
            "buildings",
            "panoramas",
            "tilesets",
            "building_features",
            "building_features_unique",
            "validation_labels",
            "frontiersi_validation_labels",
        ]
        existing_tables = [t for t in tables_to_check if t in conn.list_tables()]
        print(f"\nCreated {len(existing_tables)} tables:")
        for table_name in existing_tables:
            try:
                count = conn.table(table_name).count().execute()
                if table_name == "building_features":
                    cols = len(conn.table(table_name).columns)
                    print(f"  - {table_name}: {count:,} rows, {cols} columns")
                else:
                    print(f"  - {table_name}: {count:,} rows")
            except Exception:
                pass

        try:
            print("\nBuildings by region:")
            buildings = conn.table("buildings")
            regions = buildings.group_by("region_name").agg(count=buildings.count()).order_by("region_name").execute()
            for _, row in regions.iterrows():
                print(f"  - {row['region_name']}: {row['count']:,} buildings")
        except Exception:
            pass

        print("\nDatabase ready at: data/floor_heights.duckdb")
