"""
Delhi OSM Road Segment Pipeline
================================
1. Downloads the full Delhi road network from OpenStreetMap via osmnx
2. Classifies each segment into a Delhi zone (Central_NDMC, Dense_OldDelhi,
   Outer_Periphery, General_Urban, Industrial, University_Campus) based on coordinates
3. Assigns all 17 model-compatible features intelligently based on
   OSM highway type + zone + time of day
4. Saves to delhi_road_segments.csv + delhi_road_segments.gpkg

Author : Fear-Free Night Navigator Team
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# =============================================================================
#  CONFIG
# =============================================================================
DELHI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DELHI_DIR, "data")
CACHE_FILE = os.path.join(OUTPUT_DIR, "delhi_osm_raw.gpkg")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =============================================================================
#  STEP 1 — Download Delhi road network (with caching)
# =============================================================================
def download_delhi_network():
    """Download Delhi road network from OSM, or load from cache."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached network from {CACHE_FILE}...")
        edges = gpd.read_file(CACHE_FILE, layer="edges")
        return edges

    import osmnx as ox

    print("Downloading Delhi road network from OpenStreetMap...")
    print("(This may take 3-10 minutes depending on connection speed)")

    # Download drivable + walkable network for Delhi NCT
    G = ox.graph_from_place(
        "Delhi, India",
        network_type="all",          # all road types
        simplify=True,
        retain_all=True,
    )

    # Convert to GeoDataFrame
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    print(f"Downloaded {len(edges)} road segments, {len(nodes)} nodes")

    # Cache raw data for re-runs
    edges.to_file(CACHE_FILE, layer="edges", driver="GPKG")
    print(f"Cached to {CACHE_FILE}")

    return edges


# =============================================================================
#  STEP 2 — Delhi Zone Classification
# =============================================================================

# Approximate zone polygons (WGS84 lat/lon)
# Central/NDMC: Lutyens' Delhi, Connaught Place, India Gate, Rajpath
CENTRAL_POLYGON = Polygon([
    (77.185, 28.575), (77.245, 28.575), (77.245, 28.645),
    (77.185, 28.645), (77.185, 28.575)
])

# Dense Old Delhi: Chandni Chowk, Red Fort, Jama Masjid, Daryaganj, Sadar Bazaar,
# Paharganj, Karol Bagh old areas, Civil Lines south
OLD_DELHI_POLYGON = Polygon([
    (77.190, 28.630), (77.270, 28.630), (77.270, 28.690),
    (77.190, 28.690), (77.190, 28.630)
])

# Delhi center for distance-based classification
DELHI_CENTER = Point(77.2167, 28.6139)  # Connaught Place
OUTER_RADIUS_DEG = 0.12   # ~13 km — beyond this is "Outer"

# Industrial zone polygons
INDUSTRIAL_POLYGONS = [
    # Okhla Industrial Area
    Polygon([(77.26, 28.52), (77.30, 28.52), (77.30, 28.55), (77.26, 28.55), (77.26, 28.52)]),
    # Wazirpur Industrial Area
    Polygon([(77.15, 28.68), (77.18, 28.68), (77.18, 28.70), (77.15, 28.70), (77.15, 28.68)]),
    # Bawana Industrial Area
    Polygon([(77.03, 28.77), (77.08, 28.77), (77.08, 28.80), (77.03, 28.80), (77.03, 28.77)]),
]

# University/Campus zone polygons
UNIVERSITY_POLYGONS = [
    # JNU campus
    Polygon([(77.16, 28.53), (77.18, 28.53), (77.18, 28.56), (77.16, 28.56), (77.16, 28.53)]),
    # DU North Campus
    Polygon([(77.19, 28.68), (77.22, 28.68), (77.22, 28.70), (77.19, 28.70), (77.19, 28.68)]),
    # Jamia Millia Islamia
    Polygon([(77.27, 28.55), (77.29, 28.55), (77.29, 28.57), (77.27, 28.57), (77.27, 28.55)]),
]


def classify_zone(lon, lat):
    """Classify a coordinate into a Delhi zone."""
    pt = Point(lon, lat)

    if OLD_DELHI_POLYGON.contains(pt):
        return "Dense_OldDelhi"
    if CENTRAL_POLYGON.contains(pt):
        return "Central_NDMC"

    # Check Industrial polygons
    for poly in INDUSTRIAL_POLYGONS:
        if poly.contains(pt):
            return "Industrial"

    # Check University polygons
    for poly in UNIVERSITY_POLYGONS:
        if poly.contains(pt):
            return "University_Campus"

    # Distance from center
    dist = pt.distance(DELHI_CENTER)
    if dist > OUTER_RADIUS_DEG:
        return "Outer_Periphery"

    return "General_Urban"


# =============================================================================
#  STEP 3 — Highway type → road characteristics mapping
# =============================================================================

# Map OSM highway types to our model's expected characteristics
HIGHWAY_PROFILES = {
    # (speed_limit, infra_probs, luminosity_factor, cctv_factor, commercial_factor)
    "motorway":       {"speed": (80, 100), "infra_p": [0.02, 0.18, 0.80], "lum": 1.2, "cctv": 0.8, "comm": 0.1},
    "motorway_link":  {"speed": (60, 80),  "infra_p": [0.05, 0.25, 0.70], "lum": 1.1, "cctv": 0.7, "comm": 0.1},
    "trunk":          {"speed": (60, 80),  "infra_p": [0.05, 0.25, 0.70], "lum": 1.0, "cctv": 0.8, "comm": 0.3},
    "trunk_link":     {"speed": (40, 60),  "infra_p": [0.10, 0.30, 0.60], "lum": 0.9, "cctv": 0.6, "comm": 0.3},
    "primary":        {"speed": (50, 70),  "infra_p": [0.05, 0.25, 0.70], "lum": 1.0, "cctv": 1.0, "comm": 0.8},
    "primary_link":   {"speed": (40, 60),  "infra_p": [0.08, 0.30, 0.62], "lum": 0.9, "cctv": 0.8, "comm": 0.6},
    "secondary":      {"speed": (40, 60),  "infra_p": [0.10, 0.35, 0.55], "lum": 0.8, "cctv": 0.7, "comm": 0.7},
    "secondary_link": {"speed": (30, 50),  "infra_p": [0.15, 0.40, 0.45], "lum": 0.7, "cctv": 0.5, "comm": 0.5},
    "tertiary":       {"speed": (30, 50),  "infra_p": [0.15, 0.40, 0.45], "lum": 0.7, "cctv": 0.5, "comm": 0.6},
    "tertiary_link":  {"speed": (25, 40),  "infra_p": [0.20, 0.45, 0.35], "lum": 0.6, "cctv": 0.4, "comm": 0.4},
    "residential":    {"speed": (20, 40),  "infra_p": [0.15, 0.40, 0.45], "lum": 0.6, "cctv": 0.4, "comm": 0.3},
    "living_street":  {"speed": (15, 25),  "infra_p": [0.25, 0.45, 0.30], "lum": 0.5, "cctv": 0.3, "comm": 0.2},
    "unclassified":   {"speed": (20, 40),  "infra_p": [0.30, 0.40, 0.30], "lum": 0.4, "cctv": 0.2, "comm": 0.2},
    "service":        {"speed": (15, 30),  "infra_p": [0.30, 0.45, 0.25], "lum": 0.4, "cctv": 0.3, "comm": 0.1},
    "pedestrian":     {"speed": (10, 20),  "infra_p": [0.20, 0.50, 0.30], "lum": 0.5, "cctv": 0.3, "comm": 0.5},
    "footway":        {"speed": (5, 15),   "infra_p": [0.40, 0.40, 0.20], "lum": 0.3, "cctv": 0.2, "comm": 0.1},
    "path":           {"speed": (5, 15),   "infra_p": [0.60, 0.30, 0.10], "lum": 0.2, "cctv": 0.1, "comm": 0.0},
    "track":          {"speed": (10, 30),  "infra_p": [0.60, 0.30, 0.10], "lum": 0.2, "cctv": 0.1, "comm": 0.0},
    "cycleway":       {"speed": (10, 20),  "infra_p": [0.30, 0.50, 0.20], "lum": 0.4, "cctv": 0.2, "comm": 0.0},
}

DEFAULT_PROFILE = {"speed": (25, 40), "infra_p": [0.25, 0.40, 0.35], "lum": 0.5, "cctv": 0.3, "comm": 0.3}

# Zone-specific multipliers
ZONE_PARAMS = {
    "Central_NDMC": {
        "lum_base": (6, 5),    # gamma(shape, scale)
        "sl_base": (8, 1.5),   # normal(mean, std)
        "shadow_beta": (2, 7), # beta(a, b) — low shadow
        "footfall_log": (3.2, 0.7),
        "vehicle_log": (4.5, 0.6),
        "transit_exp": 150,
        "safezone_exp": 300,
        "incident_exp": 4,
        "cctv_lam": 4.0,
        "zone_type": "Urban",
    },
    "Dense_OldDelhi": {
        "lum_base": (2, 4),
        "sl_base": (3, 2.0),
        "shadow_beta": (5, 3),  # high shadow — narrow lanes
        "footfall_log": (3.8, 0.9),
        "vehicle_log": (2.5, 0.7),
        "transit_exp": 350,
        "safezone_exp": 500,
        "incident_exp": 7,
        "cctv_lam": 1.5,
        "zone_type": "Urban",
    },
    "Outer_Periphery": {
        "lum_base": (1.5, 2.5),
        "sl_base": (1.5, 1.0),
        "shadow_beta": (4, 3),  # moderate-high
        "footfall_log": (1.5, 0.8),
        "vehicle_log": (3.5, 0.8),
        "transit_exp": 2000,
        "safezone_exp": 2500,
        "incident_exp": 3,
        "cctv_lam": 0.2,
        "zone_type": "Rural",
    },
    "General_Urban": {
        "lum_base": (4, 3.5),
        "sl_base": (5, 2.0),
        "shadow_beta": (3, 5),
        "footfall_log": (2.8, 0.8),
        "vehicle_log": (4.0, 0.7),
        "transit_exp": 400,
        "safezone_exp": 600,
        "incident_exp": 5,
        "cctv_lam": 2.0,
        "zone_type": "Urban",
    },
    "Industrial": {
        "lum_base": (1.5, 3),     # low luminosity — sparse lighting
        "sl_base": (2, 1.5),
        "shadow_beta": (5, 3),    # high shadow — warehouses, walls
        "footfall_log": (1.2, 0.6),  # very low footfall at night
        "vehicle_log": (3.0, 0.8),
        "transit_exp": 1500,
        "safezone_exp": 2000,
        "incident_exp": 6,        # moderate-high incidents
        "cctv_lam": 0.5,          # low cctv
        "zone_type": "Urban",
    },
    "University_Campus": {
        "lum_base": (5, 3),       # moderate-good lighting
        "sl_base": (6, 1.5),
        "shadow_beta": (3, 6),    # low shadow — open grounds
        "footfall_log": (3.5, 0.7),  # moderate footfall
        "vehicle_log": (2.5, 0.6),
        "transit_exp": 300,
        "safezone_exp": 400,
        "incident_exp": 2,        # low incidents
        "cctv_lam": 3.0,          # good cctv coverage
        "zone_type": "Urban",
    },
}


# =============================================================================
#  STEP 4 — Assign model features to each segment
# =============================================================================
def assign_features(edges_gdf):
    """
    Assign all 17 model features to each OSM road segment.

    Uses the segment's zone, highway type, road length, and
    coordinate-based heuristics to produce realistic parameters.
    """
    N = len(edges_gdf)
    print(f"\nAssigning features to {N} segments...")

    # --- Extract basic properties from OSM ---
    # Compute midpoint of each segment
    midpoints = edges_gdf.geometry.interpolate(0.5, normalized=True)
    lons = midpoints.x.values
    lats = midpoints.y.values

    # Classify zones
    print("  Classifying zones...")
    delhi_zones = np.array([classify_zone(lon, lat) for lon, lat in
                            tqdm(zip(lons, lats), total=N, desc="  Zones")])

    # Get highway types (OSM tags)
    highway_tags = edges_gdf.get("highway", pd.Series(["unclassified"] * N))
    # Some entries are lists (e.g., ['residential', 'tertiary']); take the first
    highway_tags = highway_tags.apply(
        lambda x: x[0] if isinstance(x, list) else (x if isinstance(x, str) else "unclassified")
    )

    # Road length from OSM geometry (in meters)
    # Project to UTM zone 43N (Delhi) for accurate meter distances
    edges_utm = edges_gdf.to_crs(epsg=32643)
    road_length = np.round(edges_utm.geometry.length.values, 1)
    road_length = np.clip(road_length, 5, 10000)

    # --- Pre-compute zone masks ---
    zone_masks = {z: delhi_zones == z for z in ZONE_PARAMS}

    # --- 1. segment_id ---
    segment_ids = [f"DEL-{i:07d}" for i in range(N)]

    # --- 2. road_length_m ---
    # Already computed from geometry

    # --- 3. zone_type ---
    zone_type = np.array([ZONE_PARAMS[z]["zone_type"] for z in delhi_zones])

    # --- 4. speed_limit_kmh ---
    speed_limit = np.zeros(N, dtype=int)
    for i in range(N):
        hw = highway_tags.iloc[i]
        profile = HIGHWAY_PROFILES.get(hw, DEFAULT_PROFILE)
        lo, hi = profile["speed"]
        # Adjust by zone
        zone_adj = {"Central_NDMC": -5, "Dense_OldDelhi": -10,
                    "Outer_Periphery": 10, "General_Urban": 0,
                    "Industrial": 5, "University_Campus": -5}
        speed_limit[i] = np.clip(
            np.random.randint(lo, hi + 1) + zone_adj.get(delhi_zones[i], 0),
            5, 100
        )

    # --- 5. luminosity_lux ---
    luminosity = np.zeros(N)
    for zone, mask in zone_masks.items():
        n = mask.sum()
        if n == 0:
            continue
        shape, scale = ZONE_PARAMS[zone]["lum_base"]
        # Modulate by highway type luminosity factor
        hw_factor = np.array([
            HIGHWAY_PROFILES.get(hw, DEFAULT_PROFILE)["lum"]
            for hw in highway_tags[mask]
        ])
        luminosity[mask] = np.clip(
            np.random.gamma(shape, scale, size=n) * hw_factor, 0.1, 50
        )
    luminosity = np.round(luminosity, 2)

    # --- 6. streetlight_density (per 100m) ---
    streetlight_density = np.zeros(N)
    for zone, mask in zone_masks.items():
        n = mask.sum()
        if n == 0:
            continue
        mean, std = ZONE_PARAMS[zone]["sl_base"]
        hw_factor = np.array([
            HIGHWAY_PROFILES.get(hw, DEFAULT_PROFILE)["lum"]
            for hw in highway_tags[mask]
        ])
        streetlight_density[mask] = np.clip(
            np.random.normal(mean, std, size=n) * hw_factor, 0, 15
        )
    streetlight_density = np.round(streetlight_density, 1)

    # --- 7. shadow_index (0–1) ---
    shadow_index = np.zeros(N)
    for zone, mask in zone_masks.items():
        n = mask.sum()
        if n == 0:
            continue
        a, b = ZONE_PARAMS[zone]["shadow_beta"]
        shadow_index[mask] = np.clip(np.random.beta(a, b, size=n), 0, 1)
    shadow_index = np.round(shadow_index, 3)

    # --- 8. infrastructure_quality ---
    infra_options = ["None", "One_Side", "Both_Sides"]
    infrastructure_quality = np.empty(N, dtype=object)
    for i in range(N):
        hw = highway_tags.iloc[i]
        profile = HIGHWAY_PROFILES.get(hw, DEFAULT_PROFILE)
        # Blend highway profile with zone adjustment
        base_p = np.array(profile["infra_p"])
        if delhi_zones[i] == "Central_NDMC":
            # Shift toward Both_Sides
            base_p = base_p * np.array([0.5, 0.8, 1.3])
        elif delhi_zones[i] == "Dense_OldDelhi":
            # Shift toward None (narrow lanes)
            base_p = base_p * np.array([1.5, 1.2, 0.6])
        elif delhi_zones[i] == "Outer_Periphery":
            base_p = base_p * np.array([1.4, 1.1, 0.7])
        base_p = base_p / base_p.sum()  # re-normalize
        infrastructure_quality[i] = np.random.choice(infra_options, p=base_p)

    # --- Helper: length factor for density features ---
    length_factor = road_length / 100.0

    # --- 9. commercial_activity ---
    commercial_activity = np.zeros(N, dtype=int)
    for i in range(N):
        hw = highway_tags.iloc[i]
        profile = HIGHWAY_PROFILES.get(hw, DEFAULT_PROFILE)
        zone = delhi_zones[i]
        lam = max(0.1, profile["comm"] * 3.0 * length_factor[i])
        commercial_activity[i] = np.clip(
            np.random.poisson(lam=lam), 0,
            int(10 * length_factor[i])
        )

    # --- 10. footfall_density (base — time-independent, scaled later) ---
    footfall_density_base = np.zeros(N)
    for zone, mask in zone_masks.items():
        n = mask.sum()
        if n == 0:
            continue
        mean, sigma = ZONE_PARAMS[zone]["footfall_log"]
        footfall_density_base[mask] = np.clip(
            np.random.lognormal(mean, sigma, size=n), 0, 600
        )

    # --- 11. vehicular_volume (base) ---
    vehicular_volume_base = np.zeros(N)
    for zone, mask in zone_masks.items():
        n = mask.sum()
        if n == 0:
            continue
        mean, sigma = ZONE_PARAMS[zone]["vehicle_log"]
        vehicular_volume_base[mask] = np.clip(
            np.random.lognormal(mean, sigma, size=n), 0, 1500
        )

    # --- 12. transit_distance_m ---
    transit_distance = np.zeros(N)
    for zone, mask in zone_masks.items():
        n = mask.sum()
        if n == 0:
            continue
        transit_distance[mask] = np.clip(
            np.random.exponential(
                scale=ZONE_PARAMS[zone]["transit_exp"], size=n
            ), 10, 10000
        )
    transit_distance = np.round(transit_distance, 1)

    # --- 13. safe_zone_distance_m ---
    safe_zone_distance = np.zeros(N)
    for zone, mask in zone_masks.items():
        n = mask.sum()
        if n == 0:
            continue
        safe_zone_distance[mask] = np.clip(
            np.random.exponential(
                scale=ZONE_PARAMS[zone]["safezone_exp"], size=n
            ), 20, 15000
        )
    safe_zone_distance = np.round(safe_zone_distance, 1)

    # --- 14. incident_density ---
    incident_density = np.zeros(N)
    for zone, mask in zone_masks.items():
        n = mask.sum()
        if n == 0:
            continue
        incident_density[mask] = np.clip(
            np.random.exponential(
                scale=ZONE_PARAMS[zone]["incident_exp"], size=n
            ), 0, 60
        )
    incident_density = np.round(incident_density, 1)

    # --- 15. cctv_density ---
    cctv_density = np.zeros(N, dtype=int)
    for zone, mask in zone_masks.items():
        n = mask.sum()
        if n == 0:
            continue
        base_lam = ZONE_PARAMS[zone]["cctv_lam"]
        # Scale by highway CCTV factor
        hw_factor = np.array([
            HIGHWAY_PROFILES.get(hw, DEFAULT_PROFILE)["cctv"]
            for hw in highway_tags[mask]
        ])
        cctv_density[mask] = np.clip(
            np.random.poisson(lam=base_lam * hw_factor * length_factor[mask]),
            0, (8 * length_factor[mask]).astype(int)
        )

    print(f"  Feature assignment complete.")
    print(f"  Zone distribution: { {z: int(m.sum()) for z, m in zone_masks.items()} }")

    # --- Build output DataFrame ---
    result = gpd.GeoDataFrame({
        "segment_id":             segment_ids,
        "osm_highway":            highway_tags.values,
        "osm_name":               edges_gdf.get("name", pd.Series([None]*N)).values,
        "road_length_m":          road_length,
        "luminosity_lux":         luminosity,
        "streetlight_density":    streetlight_density,
        "shadow_index":           shadow_index,
        "footfall_density":       footfall_density_base.astype(int),
        "commercial_activity":    commercial_activity,
        "vehicular_volume":       vehicular_volume_base.astype(int),
        "transit_distance_m":     transit_distance,
        "incident_density":       incident_density,
        "infrastructure_quality": infrastructure_quality,
        "cctv_density":           cctv_density,
        "safe_zone_distance_m":   safe_zone_distance,
        "speed_limit_kmh":        speed_limit,
        "zone_type":              zone_type,
        "delhi_zone":             delhi_zones,
        "mid_lon":                lons,
        "mid_lat":                lats,
    }, geometry=edges_gdf.geometry.values, crs=edges_gdf.crs)

    return result


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    # 1. Download
    edges = download_delhi_network()

    # 2. Assign features (time-independent spatial parameters)
    segments = assign_features(edges)

    # 3. Save
    csv_path = os.path.join(OUTPUT_DIR, "delhi_road_segments.csv")
    gpkg_path = os.path.join(OUTPUT_DIR, "delhi_road_segments.gpkg")

    # CSV (drop geometry for flat file)
    segments.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}  ({len(segments)} segments)")

    # GeoPackage (with geometry for GIS/map visualization)
    segments.to_file(gpkg_path, driver="GPKG")
    print(f"GPKG saved: {gpkg_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"DELHI ROAD SEGMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total segments: {len(segments)}")
    print(f"\nZone distribution:")
    print(segments["delhi_zone"].value_counts().to_string())
    print(f"\nHighway type distribution (top 10):")
    print(segments["osm_highway"].value_counts().head(10).to_string())
    print(f"\nSample segments:")
    print(segments[["segment_id", "osm_highway", "osm_name", "road_length_m",
                     "delhi_zone", "luminosity_lux", "cctv_density"]].head(8).to_string())
