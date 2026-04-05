"""
FT-Transformer + GAT — Self-Contained Colab Training Script
=============================================================
Runs end-to-end inside Google Colab:
  Cell 0: Install dependencies
  Cell 1: Generate 100K synthetic Delhi dataset (6 zones, wider distribution)
  Cell 2: Preprocessing pipeline (inline)
  Cell 3: Graph construction (OSM adjacency or synthetic fallback)
  Cell 4: Model architecture (FTTransformer + GAT)
  Cell 5: Asymmetric Huber loss
  Cell 6: Training loop
  Cell 7: Score real 67K OSM segments
  Cell 8: Run everything

Author : Fear-Free Night Navigator Team
"""

# ══════════════════════════════════════════════════════════════════════
#  Cell 0 — Dependencies (uncomment in Colab)
# ══════════════════════════════════════════════════════════════════════
# !pip install torch-geometric geopandas joblib tqdm

import os
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    PowerTransformer,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

try:
    from torch_geometric.nn import GATv2Conv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("torch_geometric not available — using GATFallback (pure PyTorch)")


# ══════════════════════════════════════════════════════════════════════
#  Cell 1 — Dataset Generation (100K, 6 zones, wider distribution)
# ══════════════════════════════════════════════════════════════════════

ZONE_CONFIGS = {
    "Central_NDMC": {
        "share": 0.15,
        "lum": (6, 5), "sl": (8, 1.5), "shadow_beta": (2, 7),
        "footfall_log": (3.2, 0.7), "vehicle_log": (4.5, 0.6),
        "transit_exp": 150, "safezone_exp": 300, "incident_exp": 4,
        "cctv_lam": 4.0, "zone_type": "Urban",
    },
    "Dense_OldDelhi": {
        "share": 0.20,
        "lum": (2, 4), "sl": (3, 2.0), "shadow_beta": (5, 3),
        "footfall_log": (3.8, 0.9), "vehicle_log": (2.5, 0.7),
        "transit_exp": 350, "safezone_exp": 500, "incident_exp": 7,
        "cctv_lam": 1.5, "zone_type": "Urban",
    },
    "Outer_Periphery": {
        "share": 0.20,
        "lum": (1.5, 2.5), "sl": (1.5, 1.0), "shadow_beta": (4, 3),
        "footfall_log": (1.5, 0.8), "vehicle_log": (3.5, 0.8),
        "transit_exp": 2000, "safezone_exp": 2500, "incident_exp": 3,
        "cctv_lam": 0.2, "zone_type": "Rural",
    },
    "General_Urban": {
        "share": 0.25,
        "lum": (4, 3.5), "sl": (5, 2.0), "shadow_beta": (3, 5),
        "footfall_log": (2.8, 0.8), "vehicle_log": (4.0, 0.7),
        "transit_exp": 400, "safezone_exp": 600, "incident_exp": 5,
        "cctv_lam": 2.0, "zone_type": "Urban",
    },
    "Industrial": {
        "share": 0.10,
        "lum": (1.5, 3), "sl": (2, 1.5), "shadow_beta": (5, 3),
        "footfall_log": (1.2, 0.6), "vehicle_log": (3.0, 0.8),
        "transit_exp": 1500, "safezone_exp": 2000, "incident_exp": 6,
        "cctv_lam": 0.5, "zone_type": "Urban",
    },
    "University_Campus": {
        "share": 0.10,
        "lum": (5, 3), "sl": (6, 1.5), "shadow_beta": (3, 6),
        "footfall_log": (3.5, 0.7), "vehicle_log": (2.5, 0.6),
        "transit_exp": 300, "safezone_exp": 400, "incident_exp": 2,
        "cctv_lam": 3.0, "zone_type": "Urban",
    },
}

# Score weights (13 factors)
SCORE_WEIGHTS = {
    "luminosity":   0.14, "incident":     0.12, "shadow":      0.10,
    "footfall":     0.09, "cctv":         0.08, "streetlights": 0.08,
    "commercial":   0.07, "infra":        0.07, "safezone":    0.06,
    "time":         0.06, "transit":      0.05, "vehicle":     0.04,
    "zone":         0.04,
}

HOURS = list(range(18, 24)) + list(range(0, 6))  # 18–05
DAYS = ["Wednesday", "Saturday"]

TIME_SLOTS = [(h, d) for d in DAYS for h in HOURS]


def generate_realistic_delhi_dataset(n_samples=100_000, seed=42):
    """Generate 100K synthetic road segments with 6 zones and wide score distribution."""
    rng = np.random.RandomState(seed)

    # Assign zones
    zones_list = list(ZONE_CONFIGS.keys())
    shares = [ZONE_CONFIGS[z]["share"] for z in zones_list]
    zone_assignments = rng.choice(zones_list, size=n_samples, p=shares)

    # Generate base spatial features per zone
    road_length = rng.lognormal(5.0, 0.8, n_samples).clip(10, 5000).round(1)

    luminosity = np.zeros(n_samples)
    streetlight_density = np.zeros(n_samples)
    shadow_index = np.zeros(n_samples)
    footfall_base = np.zeros(n_samples)
    vehicular_base = np.zeros(n_samples)
    transit_distance = np.zeros(n_samples)
    safe_zone_distance = np.zeros(n_samples)
    incident_density = np.zeros(n_samples)
    cctv_density = np.zeros(n_samples, dtype=int)
    commercial_activity = np.zeros(n_samples, dtype=int)
    zone_type = np.empty(n_samples, dtype=object)
    speed_limit = np.zeros(n_samples, dtype=int)

    infra_options = ["None", "One_Side", "Both_Sides"]
    infra_probs = {
        "Central_NDMC": [0.05, 0.20, 0.75],
        "Dense_OldDelhi": [0.35, 0.40, 0.25],
        "Outer_Periphery": [0.40, 0.35, 0.25],
        "General_Urban": [0.15, 0.35, 0.50],
        "Industrial": [0.30, 0.40, 0.30],
        "University_Campus": [0.10, 0.30, 0.60],
    }
    infrastructure_quality = np.empty(n_samples, dtype=object)

    for z in zones_list:
        mask = zone_assignments == z
        n = mask.sum()
        cfg = ZONE_CONFIGS[z]

        luminosity[mask] = rng.gamma(cfg["lum"][0], cfg["lum"][1], n).clip(0.1, 50)
        streetlight_density[mask] = rng.normal(cfg["sl"][0], cfg["sl"][1], n).clip(0, 15)
        shadow_index[mask] = rng.beta(cfg["shadow_beta"][0], cfg["shadow_beta"][1], n).clip(0, 1)
        footfall_base[mask] = rng.lognormal(cfg["footfall_log"][0], cfg["footfall_log"][1], n).clip(0, 600)
        vehicular_base[mask] = rng.lognormal(cfg["vehicle_log"][0], cfg["vehicle_log"][1], n).clip(0, 1500)
        transit_distance[mask] = rng.exponential(cfg["transit_exp"], n).clip(10, 10000)
        safe_zone_distance[mask] = rng.exponential(cfg["safezone_exp"], n).clip(20, 15000)
        incident_density[mask] = rng.exponential(cfg["incident_exp"], n).clip(0, 60)
        cctv_density[mask] = rng.poisson(cfg["cctv_lam"], n).clip(0, 15)
        length_factor = road_length[mask] / 100.0
        commercial_activity[mask] = rng.poisson(2.0 * length_factor).clip(0, int(10 * length_factor.max()))
        zone_type[mask] = cfg["zone_type"]
        speed_limit[mask] = rng.randint(20, 61, n)
        infrastructure_quality[mask] = rng.choice(infra_options, n, p=infra_probs[z])

    # Crime hotspot clustering: 10% of segments get amplified incidents
    hotspot_mask = rng.rand(n_samples) < 0.10
    incident_density[hotspot_mask] *= rng.uniform(1.5, 3.0, hotspot_mask.sum())
    incident_density = incident_density.clip(0, 60).round(1)

    # Build DataFrame with base features
    segment_ids = [f"SYN-{i:07d}" for i in range(n_samples)]

    df = pd.DataFrame({
        "segment_id": segment_ids,
        "road_length_m": road_length,
        "luminosity_lux": luminosity.round(2),
        "streetlight_density": streetlight_density.round(1),
        "shadow_index": shadow_index.round(3),
        "footfall_density": footfall_base.astype(int),
        "commercial_activity": commercial_activity,
        "vehicular_volume": vehicular_base.astype(int),
        "transit_distance_m": transit_distance.round(1),
        "incident_density": incident_density,
        "infrastructure_quality": infrastructure_quality,
        "cctv_density": cctv_density,
        "safe_zone_distance_m": safe_zone_distance.round(1),
        "speed_limit_kmh": speed_limit,
        "zone_type": zone_type,
        "delhi_zone": zone_assignments,
    })

    # Generate per-time-slot scores
    all_scores = []
    for hour, day in TIME_SLOTS:
        time_of_night = hour
        day_of_week = day

        # Time-dependent activity multipliers
        if 0 <= hour <= 3:
            time_activity = 0.15 + rng.uniform(-0.05, 0.05, n_samples)
        elif 4 <= hour <= 5:
            time_activity = 0.3 + rng.uniform(-0.05, 0.05, n_samples)
        elif 18 <= hour <= 20:
            time_activity = 0.85 + rng.uniform(-0.1, 0.1, n_samples)
        else:  # 21-23
            time_activity = 0.45 + rng.uniform(-0.1, 0.1, n_samples)

        weekend_boost = 1.25 if day == "Saturday" else 1.0

        ff = footfall_base * time_activity * weekend_boost
        vv = vehicular_base * time_activity * weekend_boost
        ca = commercial_activity * time_activity * weekend_boost

        # ── Score Formula ──
        # Normalize each factor to [0, 1] range
        lum_norm = luminosity.clip(0, 50) / 50.0
        sl_norm = streetlight_density.clip(0, 15) / 15.0
        shd_norm = 1.0 - shadow_index  # inverted: low shadow = good
        ff_norm = np.log1p(ff) / np.log1p(600)
        vv_norm = np.log1p(vv) / np.log1p(1500)
        transit_norm = 1.0 - np.log1p(transit_distance) / np.log1p(10000)
        safezone_norm = 1.0 - np.log1p(safe_zone_distance) / np.log1p(15000)
        incident_norm = 1.0 - np.log1p(incident_density) / np.log1p(60)
        cctv_norm = cctv_density.clip(0, 15) / 15.0
        ca_norm = np.log1p(ca) / np.log1p(100)
        infra_map = {"None": 0.0, "One_Side": 0.5, "Both_Sides": 1.0}
        infra_norm = np.array([infra_map.get(q, 0.5) for q in infrastructure_quality])

        # Time factor: late night = worse
        hour_adj = hour if hour >= 18 else hour + 24
        time_norm = 1.0 - (hour_adj - 18) / 11.0  # 18:00 → 1.0, 05:00 → 0.0

        # Zone factor
        zone_factor_map = {
            "Central_NDMC": 0.85, "Dense_OldDelhi": 0.35, "Outer_Periphery": 0.25,
            "General_Urban": 0.6, "Industrial": 0.2, "University_Campus": 0.7,
        }
        zone_norm = np.array([zone_factor_map[z] for z in zone_assignments])

        # Weighted sum
        w = SCORE_WEIGHTS
        base_score = (
            w["luminosity"]   * lum_norm +
            w["incident"]     * incident_norm +
            w["shadow"]       * shd_norm +
            w["footfall"]     * ff_norm +
            w["cctv"]         * cctv_norm +
            w["streetlights"] * sl_norm +
            w["commercial"]   * ca_norm +
            w["infra"]        * infra_norm +
            w["safezone"]     * safezone_norm +
            w["time"]         * time_norm +
            w["transit"]      * transit_norm +
            w["vehicle"]      * vv_norm +
            w["zone"]         * zone_norm
        ) * 100.0

        # ── Compound danger penalties ──
        # Dark + isolated + no CCTV → extra 8–18% penalty
        dark = luminosity < 5
        isolated = footfall_base < 15
        no_cctv = cctv_density == 0
        compound = dark & isolated & no_cctv
        penalty = rng.uniform(8, 18, n_samples) * compound
        base_score -= penalty

        # Far from safety + high incidents → extra penalty
        far_unsafe = (safe_zone_distance > 3000) & (incident_density > 10)
        base_score -= rng.uniform(5, 12, n_samples) * far_unsafe

        # ── Maintenance / upkeep bonus ──
        well_maintained = (infra_norm > 0.7) & (sl_norm > 0.5) & (cctv_norm > 0.3)
        base_score += rng.uniform(3, 8, n_samples) * well_maintained

        # Add noise
        noise = rng.normal(0, 3.5, n_samples)
        score = (base_score + noise).clip(0, 100).round(1)

        col_day = "Wed" if day == "Wednesday" else "Sat"
        col_name = f"score_{hour:02d}_{col_day}"
        df[col_name] = score
        all_scores.append(score)

    # Aggregate scores
    score_cols = [c for c in df.columns if c.startswith("score_")]
    weekday_cols = [c for c in score_cols if c.endswith("_Wed")]
    weekend_cols = [c for c in score_cols if c.endswith("_Sat")]

    df["worst_score_weekday"] = df[weekday_cols].min(axis=1)
    df["worst_score_weekend"] = df[weekend_cols].min(axis=1)
    df["worst_score_overall"] = df[score_cols].min(axis=1)
    df["best_score_overall"] = df[score_cols].max(axis=1)
    df["avg_score"] = df[score_cols].mean(axis=1).round(1)
    df["worst_time_col"] = df[score_cols].idxmin(axis=1)

    print(f"Dataset generated: {len(df)} samples, {len(score_cols)} time slots")
    all_flat = np.concatenate(all_scores)
    print(f"Score distribution: mean={all_flat.mean():.1f}, std={all_flat.std():.1f}, "
          f"min={all_flat.min():.1f}, max={all_flat.max():.1f}")
    print(f"Zone distribution:\n{df['delhi_zone'].value_counts().to_string()}")

    return df


# ══════════════════════════════════════════════════════════════════════
#  Cell 2 — Preprocessing Pipeline (inline)
# ══════════════════════════════════════════════════════════════════════

TEMPORAL_COLS = ["time_of_night", "day_of_week"]
CATEGORICAL_COLS = ["infrastructure_quality", "zone_type"]
SKEWED_COLS = [
    "vehicular_volume", "transit_distance_m", "safe_zone_distance_m",
    "incident_density", "footfall_density", "commercial_activity",
]
PASSTHROUGH_CONTINUOUS_COLS = [
    "road_length_m", "luminosity_lux", "streetlight_density",
    "shadow_index", "cctv_density", "speed_limit_kmh",
]


class TemporalCyclicalEncoder(BaseEstimator, TransformerMixin):
    _DAY_MAP = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        hour = X["time_of_night"]
        if pd.api.types.is_string_dtype(hour):
            hour = hour.str.split(":").str[0].astype(float)
        else:
            hour = hour.astype(float)
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        day_ord = X["day_of_week"].map(self._DAY_MAP).astype(float)
        out["day_sin"] = np.sin(2 * np.pi * day_ord / 7.0)
        out["day_cos"] = np.cos(2 * np.pi * day_ord / 7.0)
        out["is_weekend_night"] = X["day_of_week"].isin(
            {"Friday", "Saturday", "Sunday"}
        ).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend_night"]


class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        lum = X["luminosity_lux"].values
        sl = X["streetlight_density"].values
        shd = X["shadow_index"].values
        ff = X["footfall_density"].values
        ca = X["commercial_activity"].values
        cctv = X["cctv_density"].values
        td = X["transit_distance_m"].values
        road_len = X["road_length_m"].values

        out["effective_visibility"] = np.log1p(lum * sl * (1 - shd))
        out["surveillance_index"] = np.log1p(ff * ca * (1 + cctv))
        out["isolation_risk"] = np.log1p(shd * td * (1 / (1 + cctv)))
        out["lit_crowd_safety"] = np.log1p(lum * ff)
        out["commercial_per_100m"] = ca / road_len * 100
        out["footfall_per_100m"] = ff / road_len * 100
        out["vehicular_per_100m"] = X["vehicular_volume"].values / road_len * 100
        out["cctv_per_100m"] = cctv / road_len * 100
        return out

    def get_feature_names_out(self, input_features=None):
        return [
            "effective_visibility", "surveillance_index", "isolation_risk",
            "lit_crowd_safety", "commercial_per_100m", "footfall_per_100m",
            "vehicular_per_100m", "cctv_per_100m",
        ]


class ProximityDecayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or ["safe_zone_distance_m", "transit_distance_m"]

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for col in self.columns:
            name = col.replace("_distance_m", "_proximity")
            out[name] = 1.0 / np.log2(1.0 + X[col].values)
        return out

    def get_feature_names_out(self, input_features=None):
        return [c.replace("_distance_m", "_proximity") for c in self.columns]


def build_preprocessor():
    """Build the full preprocessing ColumnTransformer."""
    temporal_pipe = Pipeline([("cyclical", TemporalCyclicalEncoder())])
    skewed_pipe = Pipeline([
        ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("robust_scale", RobustScaler()),
    ])
    continuous_pipe = Pipeline([("robust_scale", RobustScaler())])
    categorical_pipe = Pipeline([
        ("onehot", OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="infrequent_if_exist",
        )),
    ])
    interaction_cols = [
        "luminosity_lux", "streetlight_density", "shadow_index",
        "footfall_density", "commercial_activity", "cctv_density",
        "transit_distance_m", "road_length_m", "vehicular_volume",
    ]
    interaction_pipe = Pipeline([
        ("interactions", InteractionFeatureGenerator()),
        ("robust_scale", RobustScaler()),
    ])
    proximity_cols = ["safe_zone_distance_m", "transit_distance_m"]
    proximity_pipe = Pipeline([
        ("decay", ProximityDecayTransformer(columns=proximity_cols)),
        ("robust_scale", RobustScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("temporal", temporal_pipe, TEMPORAL_COLS),
            ("skewed", skewed_pipe, SKEWED_COLS),
            ("continuous", continuous_pipe, PASSTHROUGH_CONTINUOUS_COLS),
            ("categorical", categorical_pipe, CATEGORICAL_COLS),
            ("interaction", interaction_pipe, interaction_cols),
            ("proximity", proximity_pipe, proximity_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


# ══════════════════════════════════════════════════════════════════════
#  Cell 3 — Graph Construction
# ══════════════════════════════════════════════════════════════════════

def build_adjacency_from_osm(osm_path, n_segments):
    """Build adjacency from OSM GeoPackage via shared u/v nodes."""
    try:
        import geopandas as gpd
        raw = gpd.read_file(osm_path, layer="edges")
        # Build node → segments mapping
        node_to_segs = {}
        for idx in range(min(len(raw), n_segments)):
            u = int(raw.iloc[idx]["u"])
            v = int(raw.iloc[idx]["v"])
            node_to_segs.setdefault(u, []).append(idx)
            node_to_segs.setdefault(v, []).append(idx)

        # Build edge pairs
        src, dst = [], []
        for node, segs in node_to_segs.items():
            for i in range(len(segs)):
                for j in range(i + 1, len(segs)):
                    src.extend([segs[i], segs[j]])
                    dst.extend([segs[j], segs[i]])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        print(f"OSM adjacency: {edge_index.shape[1]} edges for {n_segments} segments")
        return edge_index
    except Exception as e:
        print(f"OSM adjacency failed ({e}), using synthetic fallback")
        return build_synthetic_adjacency(n_segments)


def build_synthetic_adjacency(n_segments):
    """Fallback: chain + random shortcuts (~3 neighbors per node)."""
    rng = np.random.RandomState(42)
    src, dst = [], []
    # Chain: each segment connects to the next
    for i in range(n_segments - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    # Random shortcuts
    n_shortcuts = n_segments * 2
    for _ in range(n_shortcuts):
        a = rng.randint(0, n_segments)
        b = rng.randint(0, n_segments)
        if a != b:
            src.extend([a, b])
            dst.extend([b, a])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    print(f"Synthetic adjacency: {edge_index.shape[1]} edges for {n_segments} segments")
    return edge_index


def remap_edge_index(edge_index, index_map):
    """Remap edge_index to a subset (train/val/test split)."""
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    new_src, new_dst = [], []
    for s, d in zip(src, dst):
        if s in index_map and d in index_map:
            new_src.append(index_map[s])
            new_dst.append(index_map[d])
    if len(new_src) == 0:
        return None
    return torch.tensor([new_src, new_dst], dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════
#  Cell 4 — Model Architecture (FTTransformer + GAT)
# ══════════════════════════════════════════════════════════════════════

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class PreNormResidual(nn.Module):
    def __init__(self, d_model, sublayer, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        return x + self.dropout(self.sublayer(self.norm(x), **kwargs))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self._attn_weights = None

    def forward(self, x, store_attn=False):
        B, S, _ = x.shape
        qkv = self.W_qkv(x).reshape(B, S, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        if store_attn:
            self._attn_weights = attn.mean(dim=1)[:, 0, 1:]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, S, -1)
        return self.W_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.attn_block = PreNormResidual(
            d_model, MultiHeadSelfAttention(d_model, n_heads, dropout), dropout,
        )
        ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn * 2), GEGLU(),
            nn.Dropout(dropout), nn.Linear(d_ffn, d_model),
        )
        self.ffn_block = PreNormResidual(d_model, ffn, dropout)
        self.drop_path_prob = drop_path

    def forward(self, x, store_attn=False):
        if self.training and self.drop_path_prob > 0:
            if torch.rand(1).item() < self.drop_path_prob:
                return x
        x = self.attn_block(x, store_attn=store_attn)
        x = self.ffn_block(x)
        return x


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.d_model
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class GATFallback(nn.Module):
    """Pure-PyTorch GAT for environments without torch_geometric."""
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.layers_ = nn.ModuleList()
        for i in range(layers):
            d_in = in_dim if i == 0 else hidden_dim * heads
            d_out = hidden_dim if i < layers - 1 else out_dim
            self.layers_.append(nn.Linear(d_in, d_out * heads))
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers_):
            x = layer(x)
            if i < len(self.layers_) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        # Average head outputs
        B, D = x.shape
        x = x.view(B, self.heads, -1).mean(dim=1)
        return x


class FTTransformerGAT(nn.Module):
    """FT-Transformer with optional GAT spatial message-passing."""

    def __init__(
        self,
        n_features,
        d_model=192,
        n_heads=8,
        n_layers=4,
        d_ffn=512,
        dropout=0.15,
        drop_path=0.1,
        gat_heads=4,
        gat_layers=2,
        gat_hidden=128,
    ):
        super().__init__()
        self.n_features = n_features

        # FT-Transformer backbone
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        dp_rates = [drop_path * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ffn, dropout, dp_rates[i])
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        if HAS_PYG:
            for i in range(gat_layers):
                d_in = d_model if i == 0 else gat_hidden * gat_heads
                self.gat_layers.append(
                    GATv2Conv(d_in, gat_hidden, heads=gat_heads, dropout=dropout, concat=True)
                )
                self.gat_norms.append(nn.LayerNorm(gat_hidden * gat_heads))
            gat_out_dim = gat_hidden * gat_heads
        else:
            self.gat_fallback = GATFallback(d_model, gat_hidden, gat_hidden, gat_heads, gat_layers, dropout)
            gat_out_dim = gat_hidden

        self.gat_proj = nn.Linear(gat_out_dim, d_model)  # project back to d_model
        self.gat_gate = nn.Linear(d_model * 2, d_model)  # gating: combine transformer + GAT

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _transformer_cls(self, x, store_attn=False):
        """Run FT-Transformer backbone, return CLS representation."""
        tokens = self.tokenizer(x)
        for block in self.transformer_blocks:
            tokens = block(tokens, store_attn=store_attn)
        return self.final_norm(tokens[:, 0])  # (B, d_model)

    def _transformer_cls_chunked(self, x, chunk_size=2048):
        """Run FT-Transformer in memory-efficient chunks with gradient checkpointing.
        Recomputes activations during backward → saves ~10x VRAM."""
        cls_parts = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i + chunk_size]
            if self.training:
                cls_parts.append(grad_checkpoint(self._transformer_cls, chunk, use_reentrant=False))
            else:
                cls_parts.append(self._transformer_cls(chunk))
        return torch.cat(cls_parts, dim=0)

    def _gat_and_head(self, cls_repr, edge_index=None):
        """Run GAT message-passing + regression head on CLS representations."""
        if edge_index is not None and edge_index.numel() > 0:
            gat_in = cls_repr
            if HAS_PYG:
                for gat_layer, gat_norm in zip(self.gat_layers, self.gat_norms):
                    gat_out = gat_layer(gat_in, edge_index)
                    gat_out = F.elu(gat_norm(gat_out))
                    if gat_out.shape == gat_in.shape:
                        gat_out = gat_out + gat_in  # residual
                    gat_in = gat_out
            else:
                gat_in = self.gat_fallback(gat_in, edge_index)

            gat_proj = self.gat_proj(gat_in)
            gate = torch.sigmoid(self.gat_gate(torch.cat([cls_repr, gat_proj], dim=-1)))
            cls_repr = gate * cls_repr + (1 - gate) * gat_proj

        raw = self.head(cls_repr)
        return torch.sigmoid(raw) * 100.0

    def forward(self, x, edge_index=None, return_attention=False, chunk_size=0):
        # 1. FT-Transformer (optionally chunked for memory efficiency)
        if chunk_size > 0:
            cls_repr = self._transformer_cls_chunked(x, chunk_size)
        else:
            cls_repr = self._transformer_cls(x, store_attn=return_attention)

        # 2. GAT + regression head
        score = self._gat_and_head(cls_repr, edge_index)

        if return_attention:
            last_block = self.transformer_blocks[-1]
            attn = last_block.attn_block.sublayer._attn_weights
            return score, attn

        return score

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(n_features, config="base"):
    """Build FTTransformerGAT with preset configurations."""
    presets = {
        "small": dict(d_model=128, n_heads=4, n_layers=3, d_ffn=256,
                       dropout=0.1, drop_path=0.05, gat_heads=2, gat_layers=1, gat_hidden=64),
        "base":  dict(d_model=192, n_heads=8, n_layers=4, d_ffn=512,
                       dropout=0.15, drop_path=0.1, gat_heads=4, gat_layers=2, gat_hidden=128),
        "large": dict(d_model=256, n_heads=8, n_layers=6, d_ffn=768,
                       dropout=0.2, drop_path=0.15, gat_heads=4, gat_layers=3, gat_hidden=128),
    }
    if config not in presets:
        raise ValueError(f"Unknown config '{config}'. Choose from {list(presets)}")
    model = FTTransformerGAT(n_features=n_features, **presets[config])
    print(f"FTTransformerGAT ({config}): {model.count_parameters():,} trainable params")
    return model


# ══════════════════════════════════════════════════════════════════════
#  Cell 5 — Asymmetric Huber Loss
# ══════════════════════════════════════════════════════════════════════

class AsymmetricHuberLoss(nn.Module):
    """Huber loss with heavier penalty for overestimation (predicting safer than actual)."""
    def __init__(self, delta=5.0, alpha=3.0):
        super().__init__()
        self.delta = delta
        self.alpha = alpha  # overestimation penalty multiplier

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = diff.abs()
        # Huber
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        # Asymmetry: penalize overestimation (pred > target) more
        overestimate = (diff > 0).float()
        weight = 1.0 + (self.alpha - 1.0) * overestimate
        return (loss * weight).mean()


# ══════════════════════════════════════════════════════════════════════
#  Cell 6 — Training Loop
# ══════════════════════════════════════════════════════════════════════

def train_full_pipeline(
    output_dir="output",
    osm_path=None,
    config="base",
    n_samples=100_000,
    epochs=120,
    batch_size=512,
    lr=3e-3,
    patience=25,
    seed=42,
):
    """Full training pipeline: generate data → preprocess → train → save."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Generate dataset
    print("\n=== Generating dataset ===")
    df = generate_realistic_delhi_dataset(n_samples, seed)

    # 2. Expand to per-time-slot training rows
    print("\n=== Expanding to per-time-slot rows ===")
    rows = []
    for hour, day in TIME_SLOTS:
        col_day = "Wed" if day == "Wednesday" else "Sat"
        col_name = f"score_{hour:02d}_{col_day}"

        slot_df = df[["segment_id", "road_length_m", "luminosity_lux", "streetlight_density",
                       "shadow_index", "footfall_density", "commercial_activity",
                       "vehicular_volume", "transit_distance_m", "incident_density",
                       "infrastructure_quality", "cctv_density", "safe_zone_distance_m",
                       "speed_limit_kmh", "zone_type", "delhi_zone"]].copy()
        slot_df["time_of_night"] = hour
        slot_df["day_of_week"] = day
        slot_df["target_safety_score"] = df[col_name]

        # Time-dependent feature scaling
        if 0 <= hour <= 3:
            time_mult = 0.15
        elif 4 <= hour <= 5:
            time_mult = 0.3
        elif 18 <= hour <= 20:
            time_mult = 0.85
        else:
            time_mult = 0.45
        weekend_mult = 1.25 if day == "Saturday" else 1.0

        slot_df["footfall_density"] = (slot_df["footfall_density"] * time_mult * weekend_mult).astype(int)
        slot_df["vehicular_volume"] = (slot_df["vehicular_volume"] * time_mult * weekend_mult).astype(int)
        slot_df["commercial_activity"] = (slot_df["commercial_activity"] * time_mult * weekend_mult).astype(int)

        rows.append(slot_df)

    train_data = pd.concat(rows, ignore_index=True)
    print(f"Training data: {len(train_data)} rows ({n_samples} segments × {len(TIME_SLOTS)} time slots)")

    # Sample one time slot per segment for training (avoid massive dataset)
    # Use stratified sampling to keep diversity
    train_data = train_data.groupby("segment_id").apply(
        lambda g: g.sample(1, random_state=seed)
    ).reset_index(drop=True)
    print(f"After sampling 1 slot/segment: {len(train_data)} rows")

    # 3. Train/val/test split (70/15/15, stratified by score deciles)
    targets = train_data["target_safety_score"]
    score_bins = pd.qcut(targets, q=10, labels=False, duplicates="drop")

    train_idx, temp_idx = train_test_split(
        np.arange(len(train_data)), test_size=0.3, stratify=score_bins, random_state=seed
    )
    temp_bins = score_bins.iloc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_bins, random_state=seed
    )

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 4. Preprocess
    print("\n=== Preprocessing ===")
    preprocessor = build_preprocessor()

    feature_cols = [c for c in train_data.columns if c not in ["segment_id", "target_safety_score", "delhi_zone"]]
    X_train_raw = train_data.iloc[train_idx][feature_cols]
    X_val_raw = train_data.iloc[val_idx][feature_cols]
    X_test_raw = train_data.iloc[test_idx][feature_cols]

    y_train = targets.iloc[train_idx].values
    y_val = targets.iloc[val_idx].values
    y_test = targets.iloc[test_idx].values

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    n_features = X_train.shape[1]
    print(f"Features after preprocessing: {n_features}")

    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    # 5. Build graph adjacency
    print("\n=== Building graph adjacency ===")
    if osm_path and os.path.exists(osm_path):
        edge_index = build_adjacency_from_osm(osm_path, n_samples)
    else:
        edge_index = build_synthetic_adjacency(n_samples)

    # Remap edge indices for train/val/test
    train_map = {orig: new for new, orig in enumerate(train_idx)}
    val_map = {orig: new for new, orig in enumerate(val_idx)}
    test_map = {orig: new for new, orig in enumerate(test_idx)}

    train_ei = remap_edge_index(edge_index, train_map)
    val_ei = remap_edge_index(edge_index, val_map)
    test_ei = remap_edge_index(edge_index, test_map)

    # Move to device
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    if train_ei is not None:
        train_ei = train_ei.to(device)
    if val_ei is not None:
        val_ei = val_ei.to(device)
    if test_ei is not None:
        test_ei = test_ei.to(device)

    # 6. Build model
    print("\n=== Building model ===")
    model = build_model(n_features, config).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = AsymmetricHuberLoss(delta=5.0, alpha=3.0)

    # 7. Training loop (gradient checkpointing + AMP + full-batch graph training)
    print(f"\n=== Training for {epochs} epochs ===")
    best_val_loss = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    chunk_size = 2048  # transformer forward chunk size (grad checkpointed)

    for epoch in range(1, epochs + 1):
        # Full-batch training: one forward+backward per epoch
        # (GAT needs all nodes anyway, so mini-batching wastes compute)
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(X_train_t, edge_index=train_ei, chunk_size=chunk_size)
            loss = criterion(pred, y_train_t)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        avg_train_loss = loss.item()
        del pred, loss

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            val_pred = model(X_val_t, edge_index=val_ei, chunk_size=chunk_size)
            val_loss = criterion(val_pred, y_val_t).item()
            val_mae = (val_pred - y_val_t).abs().mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")

        # Early stopping / best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "n_features": n_features,
                "config": config,
                "architecture": "FTTransformerGAT",
            }, os.path.join(output_dir, "best_model.pt"))
        elif epoch - best_epoch >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
            break

    # 8. Test evaluation
    print(f"\n=== Test Evaluation (best epoch: {best_epoch}) ===")
    ckpt = torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
        test_pred = model(X_test_t, edge_index=test_ei, chunk_size=chunk_size)
        test_mae = (test_pred - y_test_t).abs().mean().item()
        test_mse = ((test_pred - y_test_t) ** 2).mean().item()
        ss_res = ((y_test_t - test_pred) ** 2).sum().item()
        ss_tot = ((y_test_t - y_test_t.mean()) ** 2).sum().item()
        test_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test RMSE: {test_mse**0.5:.2f}")
    print(f"Test R²: {test_r2:.4f}")

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "config": config,
        "architecture": "FTTransformerGAT",
        "test_mae": test_mae,
        "test_r2": test_r2,
    }, os.path.join(output_dir, "final_model.pt"))

    # Save history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nAll outputs saved to {output_dir}/")
    return {
        "model": model, 
        "preprocessor": preprocessor, 
        "history": history,
        "X_test_t": X_test_t, 
        "y_test_t": y_test_t, 
        "test_ei": test_ei, 
        "df": df
    }


# ══════════════════════════════════════════════════════════════════════
#  Cell 7 — Score Real OSM Segments
# ══════════════════════════════════════════════════════════════════════

def score_real_segments(model, preprocessor, segments_csv, output_csv, osm_path=None, device=None):
    """
    Score all 67K real OSM segments across 24 time slots.
    Produces the same format as delhi/score_delhi_roads.py output.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # Load real segments
    segments = pd.read_csv(segments_csv)
    n_segments = len(segments)
    print(f"\nScoring {n_segments} real OSM segments across {len(TIME_SLOTS)} time slots...")

    score_results = {}
    batch_size = 2048

    for hour, day in tqdm(TIME_SLOTS, desc="Time slots"):
        col_day = "Wed" if day == "Wednesday" else "Sat"
        col_name = f"score_{hour:02d}_{col_day}"

        # Prepare batch with temporal features
        batch_df = segments.copy()
        batch_df["time_of_night"] = hour
        batch_df["day_of_week"] = day

        # Time-dependent activity scaling
        if 0 <= hour <= 3:
            time_mult = 0.15
        elif 4 <= hour <= 5:
            time_mult = 0.3
        elif 18 <= hour <= 20:
            time_mult = 0.85
        else:
            time_mult = 0.45
        weekend_mult = 1.25 if day == "Saturday" else 1.0

        batch_df["footfall_density"] = (batch_df["footfall_density"] * time_mult * weekend_mult).astype(int)
        batch_df["vehicular_volume"] = (batch_df["vehicular_volume"] * time_mult * weekend_mult).astype(int)
        batch_df["commercial_activity"] = (batch_df["commercial_activity"] * time_mult * weekend_mult).astype(int)

        # Preprocess
        feature_cols = [c for c in batch_df.columns
                        if c not in ["segment_id", "delhi_zone", "osm_name", "osm_highway",
                                     "mid_lon", "mid_lat"]]
        X = preprocessor.transform(batch_df[feature_cols])
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        # Inference (no GAT edge_index — single-segment residual path)
        scores = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                batch = X_t[i:i + batch_size]
                pred = model(batch, edge_index=None)
                scores.append(pred.cpu().numpy().flatten())

        score_results[col_name] = np.concatenate(scores).round(1)

    # Build output DataFrame
    out = segments[["segment_id", "osm_highway", "osm_name", "road_length_m",
                     "luminosity_lux", "streetlight_density", "shadow_index",
                     "footfall_density", "commercial_activity", "vehicular_volume",
                     "transit_distance_m", "incident_density", "infrastructure_quality",
                     "cctv_density", "safe_zone_distance_m", "speed_limit_kmh",
                     "zone_type", "delhi_zone", "mid_lon", "mid_lat"]].copy()

    # Add time-slot scores
    for col_name, scores in score_results.items():
        out[col_name] = scores

    # Aggregates
    score_cols = [c for c in out.columns if c.startswith("score_")]
    weekday_cols = [c for c in score_cols if c.endswith("_Wed")]
    weekend_cols = [c for c in score_cols if c.endswith("_Sat")]

    out["worst_score_weekday"] = out[weekday_cols].min(axis=1).round(1)
    out["worst_score_weekend"] = out[weekend_cols].min(axis=1).round(1)
    out["worst_score_overall"] = out[score_cols].min(axis=1).round(1)
    out["best_score_overall"] = out[score_cols].max(axis=1).round(1)
    out["avg_score"] = out[score_cols].mean(axis=1).round(1)
    out["worst_time_col"] = out[score_cols].idxmin(axis=1)

    out.to_csv(output_csv, index=False)
    print(f"\nScored CSV saved: {output_csv}")

    # Print summary
    all_scores = out[score_cols].values.flatten()
    print(f"\n{'='*60}")
    print(f"SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"Segments scored: {n_segments}")
    print(f"Score distribution: mean={all_scores.mean():.1f}, std={all_scores.std():.1f}, "
          f"min={all_scores.min():.1f}, max={all_scores.max():.1f}")
    print(f"\nBy zone:")
    for zone in out["delhi_zone"].unique():
        z_scores = out.loc[out["delhi_zone"] == zone, "avg_score"]
        print(f"  {zone:25s}: mean={z_scores.mean():.1f}, min={z_scores.min():.1f}, max={z_scores.max():.1f}")

    return out


# ══════════════════════════════════════════════════════════════════════
#  Cell 8 — Run Everything
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUT_DIR = "output"
    OSM_PATH = "data/delhi_osm_raw.gpkg"        # Upload to Colab if available
    SEGMENTS_CSV = "data/delhi_road_segments.csv"  # Upload to Colab
    SCORED_CSV = "data/delhi_scored_roads.csv"

    # 1. Train model
    outputs = train_full_pipeline(
        output_dir=OUTPUT_DIR,
        osm_path=OSM_PATH if os.path.exists(OSM_PATH) else None,
        config="base",
        n_samples=100_000,
        epochs=120,
        batch_size=512,
        lr=3e-3,
        patience=25,
    )
    model = outputs["model"]
    preprocessor = outputs["preprocessor"]
    history = outputs["history"]
    X_test_t = outputs["X_test_t"]
    y_test_t = outputs["y_test_t"]
    test_ei = outputs["test_ei"]
    df = outputs["df"]

    # 2. Score real segments (if available)
    if os.path.exists(SEGMENTS_CSV):
        score_real_segments(
            model, preprocessor,
            SEGMENTS_CSV, SCORED_CSV,
            osm_path=OSM_PATH if os.path.exists(OSM_PATH) else None,
        )
    else:
        print(f"\n{SEGMENTS_CSV} not found — skipping real segment scoring.")
        print("Upload delhi_road_segments.csv to Colab to score real segments.")
