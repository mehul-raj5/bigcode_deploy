"""
Delhi Synthetic Road-Network Dataset Generator v2
===================================================
Generates 67,077 unique road segments × 24 hours = ~1.61 M rows.

Key design constraints:
  - NO Python for-loops over 1.6 M rows; fully vectorised (numpy + pandas).
  - 6 Delhi zone profiles with CPTED-correlated incident_density.
  - Temporal dynamics: luminosity, vehicular volume, footfall, commercial
    activity all vary realistically across a 24-hour cycle.

Output: training/data/fear_free_night_navigator.csv  (same schema as v1,
        compatible with the existing preprocessing pipeline).

Author : Fear-Free Night Navigator Team — Synthetic Data Engine v2
"""

import os, sys, time
import numpy as np
import pandas as pd

# =====================================================================
# 0.  CONFIG
# =====================================================================
SEED = 42
N_SEGMENTS = 67_077
HOURS = np.arange(24)  # 0 – 23
N_HOURS = len(HOURS)
N_TOTAL = N_SEGMENTS * N_HOURS  # 1,609,848

np.random.seed(SEED)

print(f"[1/6]  Initialising {N_SEGMENTS:,} segments × {N_HOURS} hours "
      f"= {N_TOTAL:,} rows …")

# =====================================================================
# 1.  GEOGRAPHIC BASE — Static attributes for 67,077 segments
# =====================================================================

# --- Zone distribution ---------------------------------------------------
ZONE_CFG = {
    #                     pct    speed_choices      speed_probs
    "Central_NDMC":      (0.15, [40, 50],           [0.55, 0.45]),
    "Dense_OldDelhi":    (0.20, [20, 30],           [0.55, 0.45]),
    "General_Urban":     (0.25, [30, 40, 50],       [0.30, 0.40, 0.30]),
    "Outer_Periphery":   (0.20, [50, 60, 70],       [0.30, 0.40, 0.30]),
    "Industrial":        (0.10, [30, 40],           [0.50, 0.50]),
    "University_Campus": (0.10, [20, 30],           [0.50, 0.50]),
}

zone_names = list(ZONE_CFG.keys())
zone_probs = [v[0] for v in ZONE_CFG.values()]

zone_labels = np.random.choice(zone_names, size=N_SEGMENTS, p=zone_probs)

# Boolean masks (segment-level, length N_SEGMENTS)
masks = {z: (zone_labels == z) for z in zone_names}

# Map to Urban / Rural for compatibility
zone_type = np.where(masks["Outer_Periphery"] | masks["Industrial"],
                     "Rural", "Urban")

# --- Segment IDs ---------------------------------------------------------
segment_ids = np.array([f"DEL-{i:06d}" for i in range(N_SEGMENTS)])

# --- road_length_m  (log-normal, 20 – 2500 m for all zones) -------------
road_length = np.clip(
    np.random.lognormal(mean=5.0, sigma=0.65, size=N_SEGMENTS), 20, 2500
).round(1)

# --- speed_limit_kmh ----------------------------------------------------
speed_limit = np.zeros(N_SEGMENTS, dtype=int)
for z, (_, choices, probs) in ZONE_CFG.items():
    m = masks[z]
    speed_limit[m] = np.random.choice(choices, size=m.sum(), p=probs)

# --- highway_type  (categorical, affects vehicular scaling) ---------------
HW_TYPES = ["residential", "tertiary", "secondary", "primary", "trunk"]
hw_probs_by_zone = {
    "Central_NDMC":      [0.10, 0.20, 0.30, 0.30, 0.10],
    "Dense_OldDelhi":    [0.40, 0.35, 0.15, 0.08, 0.02],
    "General_Urban":     [0.25, 0.30, 0.25, 0.15, 0.05],
    "Outer_Periphery":   [0.10, 0.15, 0.20, 0.30, 0.25],
    "Industrial":        [0.15, 0.25, 0.30, 0.20, 0.10],
    "University_Campus": [0.45, 0.30, 0.15, 0.08, 0.02],
}
highway_type = np.empty(N_SEGMENTS, dtype=object)
for z in zone_names:
    m = masks[z]
    highway_type[m] = np.random.choice(HW_TYPES, size=m.sum(),
                                       p=hw_probs_by_zone[z])

# Numeric scale for vehicular volume (residential=1 … trunk=5)
hw_scale = np.zeros(N_SEGMENTS)
for i, hw in enumerate(HW_TYPES, start=1):
    hw_scale[highway_type == hw] = i

# --- infrastructure_quality -----------------------------------------------
infra_options = ["None", "One_Side", "Both_Sides"]
infra_probs = {
    "Central_NDMC":      [0.02, 0.18, 0.80],
    "Dense_OldDelhi":    [0.35, 0.45, 0.20],
    "General_Urban":     [0.08, 0.32, 0.60],
    "Outer_Periphery":   [0.45, 0.35, 0.20],
    "Industrial":        [0.30, 0.40, 0.30],
    "University_Campus": [0.05, 0.25, 0.70],
}
infrastructure_quality = np.empty(N_SEGMENTS, dtype=object)
for z in zone_names:
    m = masks[z]
    infrastructure_quality[m] = np.random.choice(infra_options, size=m.sum(),
                                                  p=infra_probs[z])

# --- Per-zone continuous static features ----------------------------------
#  cctv_density, safe_zone_distance_m, shadow_index, streetlight_density

cctv_density     = np.zeros(N_SEGMENTS)
safe_zone_dist   = np.zeros(N_SEGMENTS)
shadow_index     = np.zeros(N_SEGMENTS)
streetlight_dens = np.zeros(N_SEGMENTS)

def _uniform(lo, hi, n):
    return np.random.uniform(lo, hi, size=n)

# Central_NDMC
m = masks["Central_NDMC"]
cctv_density[m]     = _uniform(10, 30, m.sum())
safe_zone_dist[m]   = _uniform(50, 800, m.sum())
shadow_index[m]     = _uniform(0.1, 0.3, m.sum())
streetlight_dens[m] = _uniform(30, 50, m.sum())

# Dense_OldDelhi
m = masks["Dense_OldDelhi"]
cctv_density[m]     = _uniform(5, 15, m.sum())
safe_zone_dist[m]   = _uniform(300, 1500, m.sum())
shadow_index[m]     = _uniform(0.7, 0.95, m.sum())
streetlight_dens[m] = _uniform(10, 25, m.sum())

# General_Urban
m = masks["General_Urban"]
cctv_density[m]     = _uniform(5, 20, m.sum())
safe_zone_dist[m]   = _uniform(200, 3000, m.sum())
shadow_index[m]     = _uniform(0.2, 0.5, m.sum())
streetlight_dens[m] = _uniform(15, 35, m.sum())

# Outer_Periphery
m = masks["Outer_Periphery"]
cctv_density[m]     = _uniform(0, 3, m.sum())
safe_zone_dist[m]   = _uniform(2000, 8000, m.sum())
shadow_index[m]     = _uniform(0.0, 0.2, m.sum())
streetlight_dens[m] = _uniform(0, 10, m.sum())

# Industrial
m = masks["Industrial"]
cctv_density[m]     = _uniform(1, 5, m.sum())
safe_zone_dist[m]   = _uniform(1500, 5000, m.sum())
shadow_index[m]     = _uniform(0.6, 0.8, m.sum())
streetlight_dens[m] = _uniform(5, 15, m.sum())

# University_Campus
m = masks["University_Campus"]
cctv_density[m]     = _uniform(10, 20, m.sum())
safe_zone_dist[m]   = _uniform(200, 1000, m.sum())
shadow_index[m]     = _uniform(0.4, 0.6, m.sum())
streetlight_dens[m] = _uniform(20, 40, m.sum())

# Round for readability
cctv_density     = np.round(cctv_density, 1)
safe_zone_dist   = np.round(safe_zone_dist, 1)
shadow_index     = np.round(shadow_index, 3)
streetlight_dens = np.round(streetlight_dens, 1)

# --- transit_distance_m  (zone-aware exponential) -------------------------
transit_distance = np.zeros(N_SEGMENTS)
transit_cfg = {
    "Central_NDMC":      (150, 10, 1500),
    "Dense_OldDelhi":    (350, 20, 2500),
    "General_Urban":     (400, 20, 4000),
    "Outer_Periphery":   (2000, 100, 10000),
    "Industrial":        (1200, 80, 6000),
    "University_Campus": (300, 30, 2000),
}
for z, (scale, lo, hi) in transit_cfg.items():
    m = masks[z]
    transit_distance[m] = np.clip(
        np.random.exponential(scale=scale, size=m.sum()), lo, hi
    )
transit_distance = np.round(transit_distance, 1)

# --- Base commercial_activity (static per segment, 0–1 scale) -----------
#     Will be modulated hourly in step 2.
base_commercial = np.zeros(N_SEGMENTS)
base_commercial[masks["Central_NDMC"]]      = _uniform(0.5, 1.0, masks["Central_NDMC"].sum())
base_commercial[masks["Dense_OldDelhi"]]    = _uniform(0.6, 1.0, masks["Dense_OldDelhi"].sum())
base_commercial[masks["General_Urban"]]     = _uniform(0.3, 0.8, masks["General_Urban"].sum())
base_commercial[masks["Outer_Periphery"]]   = _uniform(0.0, 0.2, masks["Outer_Periphery"].sum())
base_commercial[masks["Industrial"]]        = _uniform(0.05, 0.25, masks["Industrial"].sum())
base_commercial[masks["University_Campus"]] = _uniform(0.3, 0.7, masks["University_Campus"].sum())
base_commercial = np.round(base_commercial, 3)

# Flag 5 % of Central + General_Urban as 24/7 hubs
is_24h_hub = np.zeros(N_SEGMENTS, dtype=bool)
for z in ["Central_NDMC", "General_Urban"]:
    m = masks[z]
    idx = np.where(m)[0]
    hub_idx = np.random.choice(idx, size=int(0.05 * len(idx)), replace=False)
    is_24h_hub[hub_idx] = True

print(f"[2/6]  Static segment attributes ready.  "
      f"24/7 hubs: {is_24h_hub.sum():,}")

# =====================================================================
# 2.  HOURLY TEMPORAL MATRIX  — Cross-join + dynamic features
# =====================================================================

# --- Broadcast: repeat each segment-level array 24 times ----------------
#     seg_* = segment-level arrays tiled for N_TOTAL rows
#     hr    = hour array repeated for N_TOTAL rows

seg_idx = np.arange(N_SEGMENTS)

# Tile: each segment appears 24 consecutive times
# np.repeat(arr, 24) => [s0,s0,...(24 times),s1,s1,...]
hr = np.tile(HOURS, N_SEGMENTS)  # [0..23, 0..23, ...]  length N_TOTAL

seg_zone_labels      = np.repeat(zone_labels, N_HOURS)
seg_zone_type        = np.repeat(zone_type, N_HOURS)
seg_segment_ids      = np.repeat(segment_ids, N_HOURS)
seg_road_length      = np.repeat(road_length, N_HOURS)
seg_speed_limit      = np.repeat(speed_limit, N_HOURS)
seg_hw_type          = np.repeat(highway_type, N_HOURS)
seg_hw_scale         = np.repeat(hw_scale, N_HOURS)
seg_infra            = np.repeat(infrastructure_quality, N_HOURS)
seg_cctv             = np.repeat(cctv_density, N_HOURS)
seg_safe_zone        = np.repeat(safe_zone_dist, N_HOURS)
seg_shadow           = np.repeat(shadow_index, N_HOURS)
seg_streetlight      = np.repeat(streetlight_dens, N_HOURS)
seg_transit          = np.repeat(transit_distance, N_HOURS)
seg_base_commercial  = np.repeat(base_commercial, N_HOURS)
seg_is_24h           = np.repeat(is_24h_hub, N_HOURS)

# Zone masks at N_TOTAL scale
t_masks = {z: (seg_zone_labels == z) for z in zone_names}

print(f"[3/6]  Cross-join complete: {N_TOTAL:,} rows.  "
      f"Computing dynamic features …")

# --- 2a.  commercial_activity  -------------------------------------------
#     Static per segment but drops to 0.1 between 23:00-06:00
#     EXCEPT 24/7 hubs keep their base value.
is_night_commercial = (hr >= 23) | (hr <= 5)  # 23,0,1,2,3,4,5
commercial_activity = seg_base_commercial.copy()
drop_mask = is_night_commercial & ~seg_is_24h
commercial_activity[drop_mask] = 0.1

# --- 2b.  luminosity_lux  ------------------------------------------------
luminosity = np.zeros(N_TOTAL)

# Daylight hours: 07-17  → base 10000 × (1 − shadow_index)
is_day = (hr >= 7) & (hr <= 17)
luminosity[is_day] = 10000.0 * (1.0 - seg_shadow[is_day])

# Nighttime: 00-05, 19-23  → streetlight + commercial spillover
is_night = (hr >= 19) | (hr <= 5)
luminosity[is_night] = (seg_streetlight[is_night] * 2.5
                        + commercial_activity[is_night] * 15.0)

# Twilight: hours 6 and 18  → blend daylight and nighttime formulas
is_twilight = (hr == 6) | (hr == 18)
day_component = 10000.0 * (1.0 - seg_shadow[is_twilight])
night_component = (seg_streetlight[is_twilight] * 2.5
                   + commercial_activity[is_twilight] * 15.0)
luminosity[is_twilight] = 0.5 * day_component + 0.5 * night_component

# Add small Gaussian noise (never negative)
luminosity += np.abs(np.random.normal(0, 0.5, size=N_TOTAL))
luminosity = np.round(luminosity, 2)

# --- 2c.  vehicular_volume  ----------------------------------------------
#     Bimodal Gaussian:  Peak 1 ~ 09:45 (σ=1.5), Peak 2 ~ 19:15 (σ=2.0)
#     Scaled by highway_type and zone_type.
#     Near-zero 01-04 (except Outer_Periphery → truck traffic).

hour_float = hr.astype(float)
peak1 = np.exp(-0.5 * ((hour_float - 9.75) / 1.5) ** 2)
peak2 = np.exp(-0.5 * ((hour_float - 19.25) / 2.0) ** 2)
veh_temporal = peak1 + peak2  # 0-1 range bimodal curve

# Scale base volume:  highway_scale × zone factor × base magnitude
zone_veh_factor = np.ones(N_TOTAL)
zone_veh_factor[t_masks["Central_NDMC"]]      = 1.0
zone_veh_factor[t_masks["Dense_OldDelhi"]]    = 0.35   # narrow lanes
zone_veh_factor[t_masks["General_Urban"]]     = 0.85
zone_veh_factor[t_masks["Outer_Periphery"]]   = 0.70
zone_veh_factor[t_masks["Industrial"]]        = 0.50
zone_veh_factor[t_masks["University_Campus"]] = 0.30

base_vol = 300.0  # max vehicles/hr on a residential road at peak
vehicular_volume = (base_vol * veh_temporal * seg_hw_scale * zone_veh_factor)

# Dead-of-night suppression (01-04): multiply by 0.05 …
dead_night = (hr >= 1) & (hr <= 4)
vehicular_volume[dead_night] *= 0.05
# … except Outer_Periphery keeps moderate truck traffic
truck_mask = dead_night & t_masks["Outer_Periphery"]
vehicular_volume[truck_mask] *= 8.0  # undo most of the 0.05 → ~0.40

# Add Poisson noise proportional to mean
vehicular_volume = np.clip(
    vehicular_volume + np.random.normal(0, vehicular_volume * 0.08 + 1),
    0, None
).round(0).astype(int)

# --- 2d.  footfall_density  -----------------------------------------------
#     Follows vehicular traffic profile, but:
#       - Industrial / Outer_Periphery → 0 after 20:00
#       - Dense_OldDelhi stays high until 23:00

footfall_base = veh_temporal * 0.6  # pedestrians track vehicles loosely

zone_ff_factor = np.ones(N_TOTAL)
zone_ff_factor[t_masks["Central_NDMC"]]      = 120.0
zone_ff_factor[t_masks["Dense_OldDelhi"]]    = 200.0
zone_ff_factor[t_masks["General_Urban"]]     = 80.0
zone_ff_factor[t_masks["Outer_Periphery"]]   = 15.0
zone_ff_factor[t_masks["Industrial"]]        = 10.0
zone_ff_factor[t_masks["University_Campus"]] = 60.0

footfall_density = footfall_base * zone_ff_factor

# Industrial / Outer_Periphery → drop to 0 after 20:00
post20 = hr >= 20
footfall_density[post20 & (t_masks["Industrial"] | t_masks["Outer_Periphery"])] = 0.0

# Dense_OldDelhi: sustain high footfall through 23:00
late_old_delhi = (hr >= 20) & (hr <= 23) & t_masks["Dense_OldDelhi"]
footfall_density[late_old_delhi] *= 1.8  # boost evening bazaar crowds

# Add noise, floor at 0
footfall_density = np.clip(
    footfall_density + np.random.normal(0, footfall_density * 0.1 + 0.5),
    0, None
).round(0).astype(int)

print(f"[4/6]  Dynamic features computed.")

# =====================================================================
# 3.  CPTED-CORRELATED GROUND TRUTH — incident_density
# =====================================================================
#   Risk ∝  exp(α × safe_zone_distance_norm + β × shadow_index)
#         / log(1 + cctv_density) / log(1 + streetlight_density)

# Normalise safe_zone_distance to 0-1 range
szd_norm = seg_safe_zone / 8000.0  # max is 8000 m

alpha = 1.8
beta  = 2.0

numerator = np.exp(alpha * szd_norm + beta * seg_shadow)
denominator = (np.log1p(seg_cctv) + 0.5) * (np.log1p(seg_streetlight) + 0.5)

base_risk = numerator / denominator

# Scale to a reasonable incidents-per-year-per-km range (0 – ~60)
base_risk = base_risk / np.percentile(base_risk, 99) * 40.0

# Add Gaussian noise
noise = np.random.normal(0, base_risk * 0.15 + 0.3, size=N_TOTAL)
incident_density = np.clip(base_risk + noise, 0, 80).round(1)

print(f"[5/6]  Incident density (CPTED-correlated) computed.")

# =====================================================================
# 4.  DAY-OF-WEEK  (random assignment per row)
# =====================================================================
days = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]
day_weights = [0.12, 0.12, 0.13, 0.13, 0.16, 0.18, 0.16]
day_of_week = np.random.choice(days, size=N_TOTAL, p=day_weights)

# =====================================================================
# 5.  TARGET SAFETY SCORE  (composite, same weighting scheme as v1)
# =====================================================================

def norm(x, lo, hi):
    """Min-max normalise to [0, 1]."""
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

# Positive contributors (higher = safer)
s_lum  = norm(luminosity, 0, 10000)
s_sl   = norm(seg_streetlight, 0, 50)
s_ff   = norm(np.minimum(footfall_density, 300), 0, 300)
s_ca   = norm(commercial_activity, 0, 1)
s_cctv = norm(seg_cctv, 0, 30)

infra_score = np.where(seg_infra == "Both_Sides", 1.0,
              np.where(seg_infra == "One_Side", 0.5, 0.0))

# Negative contributors (inverted — lower raw = safer)
s_shadow   = 1.0 - seg_shadow
s_incident = 1.0 - norm(incident_density, 0, 80)
s_transit  = 1.0 - norm(seg_transit, 10, 10000)
s_safezone = 1.0 - norm(seg_safe_zone, 20, 8000)

# Time-of-day factor (activity-based proxy)
activity_map = np.array([
    0.25,  # 0
    0.15,  # 1
    0.10,  # 2
    0.08,  # 3
    0.07,  # 4
    0.12,  # 5
    0.30,  # 6
    0.55,  # 7
    0.80,  # 8
    0.95,  # 9
    1.00,  # 10
    0.95,  # 11
    0.90,  # 12
    0.85,  # 13
    0.85,  # 14
    0.80,  # 15
    0.75,  # 16
    0.80,  # 17
    0.90,  # 18
    0.85,  # 19
    0.70,  # 20
    0.55,  # 21
    0.40,  # 22
    0.30,  # 23
])
s_time = activity_map[hr]

# Vehicular: moderate is good, extremes are bad
veh_norm  = norm(vehicular_volume, 0, 2000)
s_vehicle = 1.0 - np.abs(veh_norm - 0.25) * 1.5
s_vehicle = np.clip(s_vehicle, 0, 1)

# Zone bonus
zone_bonus = np.zeros(N_TOTAL)
zone_bonus[t_masks["Central_NDMC"]]      = 0.80
zone_bonus[t_masks["Dense_OldDelhi"]]    = 0.40
zone_bonus[t_masks["General_Urban"]]     = 0.55
zone_bonus[t_masks["Outer_Periphery"]]   = 0.25
zone_bonus[t_masks["Industrial"]]        = 0.30
zone_bonus[t_masks["University_Campus"]] = 0.60

# Weights (sum ≈ 1.0)
W = {
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
    W['luminosity']   * s_lum
    + W['streetlights'] * s_sl
    + W['shadow']       * s_shadow
    + W['footfall']     * s_ff
    + W['commercial']   * s_ca
    + W['cctv']         * s_cctv
    + W['infra']        * infra_score
    + W['incident']     * s_incident
    + W['transit']      * s_transit
    + W['safezone']     * s_safezone
    + W['time']         * s_time
    + W['vehicle']      * s_vehicle
    + W['zone']         * zone_bonus
)

target_safety_score = np.clip(
    raw_score * 100 + np.random.normal(0, 2.5, size=N_TOTAL), 0, 100
).round(2)

# =====================================================================
# 6.  ASSEMBLE DATAFRAME + SAVE
# =====================================================================

# Format time_of_night as "HH:00"
time_str = np.array([f"{h:02d}:00" for h in range(24)])
seg_time_str = time_str[hr]

print(f"[6/6]  Assembling DataFrame …")

df = pd.DataFrame({
    "segment_id":             seg_segment_ids,
    "road_length_m":          seg_road_length,
    "luminosity_lux":         luminosity,
    "streetlight_density":    seg_streetlight,
    "shadow_index":           seg_shadow,
    "footfall_density":       footfall_density,
    "commercial_activity":    commercial_activity,
    "vehicular_volume":       vehicular_volume,
    "transit_distance_m":     seg_transit,
    "incident_density":       incident_density,
    "infrastructure_quality": seg_infra,
    "cctv_density":           seg_cctv,
    "safe_zone_distance_m":   seg_safe_zone,
    "time_of_night":          seg_time_str,
    "day_of_week":            day_of_week,
    "zone_type":              seg_zone_type,
    "speed_limit_kmh":        seg_speed_limit,
    "target_safety_score":    target_safety_score,
    "delhi_zone":             seg_zone_labels,
})

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data",
    "fear_free_night_navigator.csv",
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

t0 = time.time()
df.to_csv(output_path, index=False)
elapsed = time.time() - t0

print(f"\n{'=' * 60}")
print(f"  Dataset saved: {output_path}")
print(f"  Shape: {df.shape}   ({elapsed:.1f}s to write)")
print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
print(f"{'=' * 60}")
print(f"\nTarget safety score stats:\n{df['target_safety_score'].describe()}")
print(f"\nDelhi zone distribution:\n{df['delhi_zone'].value_counts()}")
print(f"\nZone type distribution:\n{df['zone_type'].value_counts()}")
print(f"\nSafety score by Delhi zone:")
print(df.groupby("delhi_zone")["target_safety_score"].describe().round(2))
print(f"\nSample rows:\n{df.head(5).to_string()}")
