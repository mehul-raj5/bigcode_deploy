"""
Delhi-Specific Synthetic Dataset Generator for Fear-Free Night Navigator
=========================================================================
Generates 100K road-segment rows with 3 distinct Delhi zone profiles:
  1. Central / NDMC  (Connaught Place, Lutyens, Rajpath)  ~20%
  2. Dense Old Delhi  (Chandni Chowk, Jama Masjid area)   ~25%
  3. Outer / Periphery (Narela, Najafgarh, Dwarka edges)  ~25%
  4. General Urban Delhi (Rohini, Saket, Lajpat Nagar etc) ~30%

All 17 features + target_safety_score, same schema as original dataset.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
N = 100_000

# ========== Zone assignment ==========
ZONES = {
    "Central_NDMC":    0.20,
    "Dense_OldDelhi":  0.25,
    "Outer_Periphery": 0.25,
    "General_Urban":   0.30,
}

zone_labels = np.random.choice(
    list(ZONES.keys()), size=N, p=list(ZONES.values())
)

is_central  = zone_labels == "Central_NDMC"
is_olddel   = zone_labels == "Dense_OldDelhi"
is_outer    = zone_labels == "Outer_Periphery"
is_general  = zone_labels == "General_Urban"

# Map to Urban / Rural for compatibility with downstream pipeline
zone_type = np.where(is_outer, "Rural", "Urban")

# ========== 1. segment_id ==========
segment_ids = [f"DEL-{i:06d}" for i in range(N)]

# ========== 2. road_length_m ==========
road_length = np.zeros(N)
road_length[is_central] = np.clip(
    np.random.lognormal(mean=5.2, sigma=0.5, size=is_central.sum()), 50, 2000
)
road_length[is_olddel] = np.clip(
    np.random.lognormal(mean=4.2, sigma=0.6, size=is_olddel.sum()), 20, 500  # narrow galis
)
road_length[is_outer] = np.clip(
    np.random.lognormal(mean=5.8, sigma=0.6, size=is_outer.sum()), 100, 5000
)
road_length[is_general] = np.clip(
    np.random.lognormal(mean=5.0, sigma=0.5, size=is_general.sum()), 50, 2500
)
road_length = np.round(road_length, 1)

# ========== 3. time_of_night (18:00 – 05:00) ==========
hours_pool = list(range(18, 24)) + list(range(0, 6))  # 18-23, 0-5
hour_weights = [0.15, 0.14, 0.13, 0.11, 0.10, 0.08,
                0.07, 0.06, 0.05, 0.04, 0.04, 0.03]
hour_weights = np.array(hour_weights) / sum(hour_weights)
time_of_night = np.random.choice(hours_pool, size=N, p=hour_weights)

# Activity factor: peaks in early evening, lowest at 2-4 AM
activity_map = {
    18: 1.0, 19: 0.95, 20: 0.85, 21: 0.70, 22: 0.55, 23: 0.40,
    0: 0.25, 1: 0.15, 2: 0.10, 3: 0.08, 4: 0.07, 5: 0.12,
}
time_factor = np.array([activity_map[h] for h in time_of_night])

# ========== 4. day_of_week ==========
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_weights = [0.12, 0.12, 0.13, 0.13, 0.16, 0.18, 0.16]
day_of_week = np.random.choice(days, size=N, p=day_weights)
is_weekend = np.isin(day_of_week, ["Friday", "Saturday", "Sunday"])
weekend_boost = np.where(is_weekend, 1.3, 1.0)

# ========== 5. speed_limit_kmh ==========
speed_limit = np.zeros(N, dtype=int)
speed_limit[is_central] = np.random.choice([30, 40, 50], size=is_central.sum(), p=[0.20, 0.55, 0.25])
speed_limit[is_olddel]  = np.random.choice([15, 20, 25, 30], size=is_olddel.sum(), p=[0.30, 0.40, 0.20, 0.10])
speed_limit[is_outer]   = np.random.choice([50, 60, 70, 80], size=is_outer.sum(), p=[0.15, 0.35, 0.30, 0.20])
speed_limit[is_general] = np.random.choice([30, 40, 50, 60], size=is_general.sum(), p=[0.10, 0.35, 0.40, 0.15])

# ========== 6. luminosity_lux ==========
luminosity = np.zeros(N)
# Central/NDMC: well-lit government areas, ornamental lighting
luminosity[is_central] = np.clip(
    np.random.gamma(shape=6, scale=5, size=is_central.sum()), 5, 50
)
# Old Delhi: patchy, some market areas bright, many dark galis
luminosity[is_olddel] = np.clip(
    np.random.gamma(shape=2, scale=4, size=is_olddel.sum()), 0.3, 40
)
# Outer: sparse streetlights, poor maintenance
luminosity[is_outer] = np.clip(
    np.random.gamma(shape=1.5, scale=2.5, size=is_outer.sum()), 0.1, 25
)
# General Urban: moderate
luminosity[is_general] = np.clip(
    np.random.gamma(shape=4, scale=3.5, size=is_general.sum()), 1, 45
)
luminosity = np.round(luminosity, 2)

# ========== 7. streetlight_density (per 100m) ==========
streetlight_density = np.zeros(N)
streetlight_density[is_central] = np.clip(np.random.normal(8, 1.5, size=is_central.sum()), 3, 15)
streetlight_density[is_olddel]  = np.clip(np.random.normal(3, 2.0, size=is_olddel.sum()), 0, 10)
streetlight_density[is_outer]   = np.clip(np.random.normal(1.5, 1.0, size=is_outer.sum()), 0, 6)
streetlight_density[is_general] = np.clip(np.random.normal(5, 2.0, size=is_general.sum()), 1, 12)
streetlight_density = np.round(streetlight_density, 1)

# ========== 8. shadow_index (0-1) ==========
shadow_index = np.zeros(N)
# Central: wide open roads, less shadow
shadow_index[is_central] = np.clip(np.random.beta(2, 7, size=is_central.sum()), 0, 1)
# Old Delhi: narrow lanes, overhanging balconies → heavy shadows
shadow_index[is_olddel]  = np.clip(np.random.beta(5, 3, size=is_olddel.sum()), 0, 1)
# Outer: open but unlit fields → moderate-high shadow perception
shadow_index[is_outer]   = np.clip(np.random.beta(4, 3, size=is_outer.sum()), 0, 1)
# General Urban
shadow_index[is_general] = np.clip(np.random.beta(3, 5, size=is_general.sum()), 0, 1)
shadow_index = np.round(shadow_index, 3)

# ========== 9. infrastructure_quality ==========
infra_options = ["None", "One_Side", "Both_Sides"]
infrastructure_quality = np.empty(N, dtype=object)
infrastructure_quality[is_central] = np.random.choice(infra_options, size=is_central.sum(), p=[0.02, 0.18, 0.80])
infrastructure_quality[is_olddel]  = np.random.choice(infra_options, size=is_olddel.sum(), p=[0.35, 0.45, 0.20])
infrastructure_quality[is_outer]   = np.random.choice(infra_options, size=is_outer.sum(), p=[0.45, 0.35, 0.20])
infrastructure_quality[is_general] = np.random.choice(infra_options, size=is_general.sum(), p=[0.08, 0.32, 0.60])

# ========== 10. commercial_activity ==========
length_factor = road_length / 100.0
commercial_activity = np.zeros(N, dtype=int)

# Central: shops close early, some restaurants/hotels
lam_central = 2.5 * length_factor[is_central] * time_factor[is_central] * weekend_boost[is_central]
commercial_activity[is_central] = np.clip(
    np.random.poisson(lam=lam_central), 0, (6 * length_factor[is_central]).astype(int)
)

# Old Delhi: extremely high commercial (bazaars), but most shut by 22:00
olddel_time_decay = np.where(time_of_night[is_olddel] < 22, 1.0,
                    np.where(time_of_night[is_olddel] < 24, 0.3, 0.05))
lam_olddel = 5.0 * length_factor[is_olddel] * olddel_time_decay * weekend_boost[is_olddel]
commercial_activity[is_olddel] = np.clip(
    np.random.poisson(lam=lam_olddel), 0, (10 * length_factor[is_olddel]).astype(int)
)

# Outer: very sparse commercial
lam_outer = 0.3 * length_factor[is_outer] * time_factor[is_outer]
commercial_activity[is_outer] = np.clip(
    np.random.poisson(lam=lam_outer), 0, (1.5 * length_factor[is_outer]).astype(int)
)

# General Urban
lam_general = 3.0 * length_factor[is_general] * time_factor[is_general] * weekend_boost[is_general]
commercial_activity[is_general] = np.clip(
    np.random.poisson(lam=lam_general), 0, (7 * length_factor[is_general]).astype(int)
)

# ========== 11. footfall_density (per hour) ==========
footfall_density = np.zeros(N, dtype=int)

footfall_density[is_central] = np.clip(
    (np.random.lognormal(3.2, 0.7, size=is_central.sum()) * time_factor[is_central] * weekend_boost[is_central]),
    0, 400
).astype(int)

# Old Delhi: massive footfall in evening, dies off sharply post-midnight
footfall_density[is_olddel] = np.clip(
    (np.random.lognormal(3.8, 0.9, size=is_olddel.sum()) * olddel_time_decay * weekend_boost[is_olddel]),
    0, 600
).astype(int)

footfall_density[is_outer] = np.clip(
    (np.random.lognormal(1.5, 0.8, size=is_outer.sum()) * time_factor[is_outer]),
    0, 100
).astype(int)

footfall_density[is_general] = np.clip(
    (np.random.lognormal(2.8, 0.8, size=is_general.sum()) * time_factor[is_general] * weekend_boost[is_general]),
    0, 350
).astype(int)

# ========== 12. vehicular_volume (per hour) ==========
vehicular_volume = np.zeros(N, dtype=int)

vehicular_volume[is_central] = np.clip(
    (np.random.lognormal(4.5, 0.6, size=is_central.sum()) * time_factor[is_central] * 0.7),
    0, 1200
).astype(int)

# Old Delhi: very low vehicular — narrow streets, mostly pedestrian/rickshaw
vehicular_volume[is_olddel] = np.clip(
    (np.random.lognormal(2.5, 0.7, size=is_olddel.sum()) * time_factor[is_olddel] * 0.5),
    0, 150
).astype(int)

# Outer: trucks, sparse traffic but high-speed
vehicular_volume[is_outer] = np.clip(
    (np.random.lognormal(3.5, 0.8, size=is_outer.sum()) * time_factor[is_outer] * 0.9),
    0, 800
).astype(int)

vehicular_volume[is_general] = np.clip(
    (np.random.lognormal(4.0, 0.7, size=is_general.sum()) * time_factor[is_general] * 0.8),
    0, 1000
).astype(int)

# ========== 13. transit_distance_m (nearest metro/bus stop) ==========
transit_distance = np.zeros(N)
# Central: Dense metro network (Blue/Yellow/Violet lines)
transit_distance[is_central] = np.clip(
    np.random.exponential(scale=150, size=is_central.sum()), 10, 1500
)
# Old Delhi: Chandni Chowk metro, but narrow access
transit_distance[is_olddel] = np.clip(
    np.random.exponential(scale=350, size=is_olddel.sum()), 20, 2500
)
# Outer: Sparse metro coverage
transit_distance[is_outer] = np.clip(
    np.random.exponential(scale=2000, size=is_outer.sum()), 100, 10000
)
# General Urban: Moderate
transit_distance[is_general] = np.clip(
    np.random.exponential(scale=400, size=is_general.sum()), 20, 4000
)
transit_distance = np.round(transit_distance, 1)

# ========== 14. safe_zone_distance_m (police station / hospital / fire stn) ==========
safe_zone_distance = np.zeros(N)
safe_zone_distance[is_central] = np.clip(
    np.random.exponential(scale=300, size=is_central.sum()), 20, 3000
)
safe_zone_distance[is_olddel] = np.clip(
    np.random.exponential(scale=500, size=is_olddel.sum()), 30, 4000
)
safe_zone_distance[is_outer] = np.clip(
    np.random.exponential(scale=2500, size=is_outer.sum()), 200, 15000
)
safe_zone_distance[is_general] = np.clip(
    np.random.exponential(scale=600, size=is_general.sum()), 30, 6000
)
safe_zone_distance = np.round(safe_zone_distance, 1)

# ========== 15. incident_density (incidents per year per km) ==========
incident_density = np.zeros(N)
# Central: moderate crime (snatching, traffic incidents)
incident_density[is_central] = np.clip(
    np.random.exponential(scale=4, size=is_central.sum()), 0, 40
)
# Old Delhi: higher petty crime density due to crowding
incident_density[is_olddel] = np.clip(
    np.random.exponential(scale=7, size=is_olddel.sum()), 0, 50
)
# Outer: lower density but more severe (vehicular, isolated crime)
incident_density[is_outer] = np.clip(
    np.random.exponential(scale=3, size=is_outer.sum()), 0, 30
)
# General Urban
incident_density[is_general] = np.clip(
    np.random.exponential(scale=5, size=is_general.sum()), 0, 45
)
incident_density = np.round(incident_density, 1)

# ========== 16. cctv_density ==========
cctv_density = np.zeros(N, dtype=int)
# Central/NDMC: Delhi Police + NDMC cameras, extremely high density
cctv_density[is_central] = np.clip(
    np.random.poisson(lam=4.0 * length_factor[is_central]), 0, (8 * length_factor[is_central]).astype(int)
)
# Old Delhi: some market cameras, but gaps in galis
cctv_density[is_olddel] = np.clip(
    np.random.poisson(lam=1.5 * length_factor[is_olddel]), 0, (4 * length_factor[is_olddel]).astype(int)
)
# Outer: very sparse
cctv_density[is_outer] = np.clip(
    np.random.poisson(lam=0.2 * length_factor[is_outer]), 0, (1.5 * length_factor[is_outer]).astype(int)
)
# General Urban
cctv_density[is_general] = np.clip(
    np.random.poisson(lam=2.0 * length_factor[is_general]), 0, (5 * length_factor[is_general]).astype(int)
)

# ========== 17. target_safety_score (composite weighted) ==========

def norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

# Positive contributors
s_lum  = norm(luminosity, 0.1, 50)
s_sl   = norm(streetlight_density, 0, 15)
s_ff   = norm(np.minimum(footfall_density, 200), 0, 200)
s_ca   = norm(np.minimum(commercial_activity, 40), 0, 40)
s_cctv = norm(cctv_density, 0, 25)

infra_score = np.where(infrastructure_quality == "Both_Sides", 1.0,
              np.where(infrastructure_quality == "One_Side", 0.5, 0.0))

# Negative contributors (inverted)
s_shadow   = 1.0 - shadow_index
s_incident = 1.0 - norm(incident_density, 0, 50)
s_transit  = 1.0 - norm(transit_distance, 10, 10000)
s_safezone = 1.0 - norm(safe_zone_distance, 20, 15000)

# Time safety
s_time = time_factor

# Vehicular: moderate is good, extremes are bad
veh_norm  = norm(vehicular_volume, 0, 1200)
s_vehicle = 1.0 - np.abs(veh_norm - 0.20) * 1.5
s_vehicle = np.clip(s_vehicle, 0, 1)

# Zone bonus (Delhi-specific: Central safest, Outer least safe)
zone_bonus = np.where(is_central, 0.80,
             np.where(is_general, 0.55,
             np.where(is_olddel, 0.40, 0.25)))

# Weights (summing to 1.0)
weights = {
    'luminosity':   0.14,
    'streetlights': 0.08,
    'shadow':       0.10,
    'footfall':     0.09,
    'commercial':   0.07,
    'cctv':         0.08,
    'infra':        0.07,
    'incident':     0.12,
    'transit':      0.05,
    'safezone':     0.06,
    'time':         0.06,
    'vehicle':      0.04,
    'zone':         0.04,
}

raw_score = (
    weights['luminosity']   * s_lum +
    weights['streetlights'] * s_sl +
    weights['shadow']       * s_shadow +
    weights['footfall']     * s_ff +
    weights['commercial']   * s_ca +
    weights['cctv']         * s_cctv +
    weights['infra']        * infra_score +
    weights['incident']     * s_incident +
    weights['transit']      * s_transit +
    weights['safezone']     * s_safezone +
    weights['time']         * s_time +
    weights['vehicle']      * s_vehicle +
    weights['zone']         * zone_bonus
)

# Scale to 0-100 with noise
raw_score_scaled = raw_score * 100
noise = np.random.normal(0, 2.5, size=N)
target_safety_score = np.clip(raw_score_scaled + noise, 0, 100)
target_safety_score = np.round(target_safety_score, 2)

# Format time
time_str = [f"{h:02d}:00" for h in time_of_night]

# ========== Build DataFrame ==========
df = pd.DataFrame({
    'segment_id':             segment_ids,
    'road_length_m':          road_length,
    'luminosity_lux':         luminosity,
    'streetlight_density':    streetlight_density,
    'shadow_index':           shadow_index,
    'footfall_density':       footfall_density,
    'commercial_activity':    commercial_activity,
    'vehicular_volume':       vehicular_volume,
    'transit_distance_m':     transit_distance,
    'incident_density':       incident_density,
    'infrastructure_quality': infrastructure_quality,
    'cctv_density':           cctv_density,
    'safe_zone_distance_m':   safe_zone_distance,
    'time_of_night':          time_str,
    'day_of_week':            day_of_week,
    'zone_type':              zone_type,
    'speed_limit_kmh':        speed_limit,
    'target_safety_score':    target_safety_score,
    'delhi_zone':             zone_labels,  # extra column for Delhi analysis
})

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fear_free_night_navigator.csv')
df.to_csv(output_path, index=False)

print(f"Dataset saved: {output_path}")
print(f"Shape: {df.shape}")
print(f"\nTarget safety score stats:\n{df['target_safety_score'].describe()}")
print(f"\nDelhi zone distribution:\n{df['delhi_zone'].value_counts()}")
print(f"\nZone type distribution:\n{df['zone_type'].value_counts()}")
print(f"\nSafety score by Delhi zone:")
print(df.groupby('delhi_zone')['target_safety_score'].describe().round(2))
print(f"\nSample rows:\n{df.head(5).to_string()}")
