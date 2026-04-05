"""
Advanced Preprocessing Pipeline for the Fear-Free Night Navigator
==================================================================
A production-ready scikit-learn Pipeline that performs:
  1. Cyclical temporal encoding (sin/cos)
  2. Yeo-Johnson / log transforms for skewed distributions
  3. Target-encoding for categoricals
  4. Explicit polynomial interaction features
  5. Robust scaling for outlier resilience

All transformers are implemented as sklearn-compatible classes so the
full pipeline can be serialised with joblib for deployment.

Author : Fear-Free Night Navigator Team
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split

# =====================================================================
# 0. CONSTANTS – Column Groups
# =====================================================================
# Defining column groups up-front keeps transformer configurations DRY
# and makes it trivial to add/remove features later.

IDENTIFIER_COLS = ["segment_id"]

TEMPORAL_COLS = ["time_of_night", "day_of_week"]

CATEGORICAL_COLS = ["infrastructure_quality", "zone_type"]

# Columns with heavy right-skew (exponential / log-normal generating
# processes confirmed during EDA).  Yeo-Johnson handles zero and
# negative values unlike Box-Cox.
SKEWED_COLS = [
    "vehicular_volume",
    "transit_distance_m",
    "safe_zone_distance_m",
    "incident_density",
    "footfall_density",
    "commercial_activity",
]

# Continuous columns that are already roughly symmetric or bounded.
PASSTHROUGH_CONTINUOUS_COLS = [
    "road_length_m",
    "luminosity_lux",
    "streetlight_density",
    "shadow_index",
    "cctv_density",
    "speed_limit_kmh",
]

TARGET_COL = "target_safety_score"


# =====================================================================
#  CUSTOM TRANSFORMERS
# =====================================================================

class TemporalCyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode temporal features as sin/cos pairs on the unit circle.

    ─── Mathematical Rationale ───────────────────────────────────────
    Time is inherently cyclical: 23:00 is closer to 00:00 than to
    12:00, but a linear integer encoding treats them as maximally
    distant.  Mapping hour h to:

        hour_sin = sin(2π · h / period)
        hour_cos = cos(2π · h / period)

    projects the hour onto the unit circle in ℝ², preserving the
    wrap-around topology.  Two-dimensional encoding is necessary
    because sin alone is ambiguous (sin(π/6) = sin(5π/6)).

    Reference:  Time2Vec (Kazemi et al., 2019) generalises this idea
    with learnable frequencies; our fixed-frequency version is a
    zero-parameter baseline suitable for a preprocessing pipeline.
    ──────────────────────────────────────────────────────────────────
    """

    # Map day names → ordinal (Monday=0 … Sunday=6) for cyclical calc.
    _DAY_MAP = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)

        # --- Time of night (period = 24 h) --------------------------------
        # Parse "HH:00" strings to integers if needed.
        hour = X["time_of_night"]
        if pd.api.types.is_string_dtype(hour):
            hour = hour.str.split(":").str[0].astype(float)
        else:
            hour = hour.astype(float)

        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # --- Day of week (period = 7 days) --------------------------------
        day_ord = X["day_of_week"].map(self._DAY_MAP).astype(float)
        out["day_sin"] = np.sin(2 * np.pi * day_ord / 7.0)
        out["day_cos"] = np.cos(2 * np.pi * day_ord / 7.0)

        # --- Derived boolean: weekend night --------------------------------
        out["is_weekend_night"] = X["day_of_week"].isin(
            {"Friday", "Saturday", "Sunday"}
        ).astype(float)

        return out

    def get_feature_names_out(self, input_features=None):
        return ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend_night"]


class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate domain-specific interaction features that capture
    non-additive safety dynamics.

    ─── Mathematical Rationale ───────────────────────────────────────
    Safety perception is not a linear superposition of individual
    features.  A well-lit street with zero pedestrians *feels*
    different from a well-lit street with moderate crowds (the
    "natural surveillance" effect, per Jane Jacobs, 1961).

    We explicitly construct interaction terms so the model does not
    need to spend capacity rediscovering them:

    1.  effective_visibility  = luminosity × streetlight_density
                                × (1 − shadow_index)
        Multiplicative: both components must be high for the
        product to be high; shadow attenuates linearly.

    2.  surveillance_index    = footfall_density × commercial_activity
                                × cctv_density
        Captures the "eyes on the street" concept: footfall alone
        is weaker without open shops or cameras.

    3.  isolation_risk        = shadow_index × transit_distance_m
                                × (1 / (1 + cctv_density))
        A triple interaction highlighting the worst-case: a shadowy
        segment far from transit with no cameras.

    4.  lit_crowd_safety      = luminosity × footfall_density
        A direct proxy for perceived safety from environmental
        criminology ("CPTED" — Crime Prevention Through
        Environmental Design, Jeffery 1971).

    All products are log1p-compressed to tame skew.
    ──────────────────────────────────────────────────────────────────
    """

    _REQUIRED = [
        "luminosity_lux", "streetlight_density", "shadow_index",
        "footfall_density", "commercial_activity", "cctv_density",
        "transit_distance_m",
    ]

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)

        lum   = X["luminosity_lux"].values
        sl    = X["streetlight_density"].values
        shd   = X["shadow_index"].values
        ff    = X["footfall_density"].values
        ca    = X["commercial_activity"].values
        cctv  = X["cctv_density"].values
        td    = X["transit_distance_m"].values

        # 1. Effective visibility (light × density, penalised by shadow)
        out["effective_visibility"] = np.log1p(lum * sl * (1 - shd))

        # 2. Natural surveillance index (crowd × commerce × cameras)
        out["surveillance_index"] = np.log1p(ff * ca * (1 + cctv))

        # 3. Isolation risk (shadow × distance × lack of cameras)
        out["isolation_risk"] = np.log1p(shd * td * (1 / (1 + cctv)))

        # 4. Lit-crowd safety proxy
        out["lit_crowd_safety"] = np.log1p(lum * ff)

        # 5. Spatial density normalisation (counts ÷ segment length × 100)
        road_len = X["road_length_m"].values
        out["commercial_per_100m"]  = ca   / road_len * 100
        out["footfall_per_100m"]    = ff   / road_len * 100
        out["vehicular_per_100m"]   = X["vehicular_volume"].values / road_len * 100
        out["cctv_per_100m"]        = cctv / road_len * 100

        return out

    def get_feature_names_out(self, input_features=None):
        return [
            "effective_visibility", "surveillance_index", "isolation_risk",
            "lit_crowd_safety", "commercial_per_100m", "footfall_per_100m",
            "vehicular_per_100m", "cctv_per_100m",
        ]


class ProximityDecayTransformer(BaseEstimator, TransformerMixin):
    """
    Apply inverse-log proximity decay to distance features.

    ─── Mathematical Rationale ───────────────────────────────────────
    The perceived benefit of proximity to a safe zone follows a
    concave, saturating curve: moving from 50 m → 200 m away from
    a police station substantially degrades the safety feeling, but
    5 km → 6 km is psychologically indistinguishable.

    Transformation:   proximity = 1 / log₂(1 + d)

    • log₂ compresses the long tail of distances.
    • The inverse flips the monotonicity so that closer → higher
      feature value, aligning with the target direction.
    • Adding 1 inside the log prevents log(0).

    This is a fixed-form monotone transform; no fitting required.
    ──────────────────────────────────────────────────────────────────
    """

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


# =====================================================================
#  PIPELINE ASSEMBLY
# =====================================================================

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Assemble and return the full preprocessing ColumnTransformer.

    Sub-pipelines:
      temporal   →  cyclical sin/cos encoding
      skewed     →  Yeo-Johnson power transform → Robust scaling
      continuous →  Robust scaling (no power transform needed)
      categorical→  One-hot encoding (drop_first for collinearity)
      interaction→  Domain interaction features → Robust scaling
      proximity  →  Inverse-log decay → Robust scaling
    """

    # --- Temporal branch -------------------------------------------------
    temporal_pipe = Pipeline([
        ("cyclical", TemporalCyclicalEncoder()),
    ])

    # --- Skewed numerics branch ------------------------------------------
    # Yeo-Johnson (Yeo & Johnson, 2000) is preferred over Box-Cox because
    # it handles zero-valued observations (e.g., footfall_density = 0 at
    # 03:00) without requiring an additive constant.
    skewed_pipe = Pipeline([
        ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("robust_scale", RobustScaler()),
    ])

    # --- Symmetric / bounded numerics branch -----------------------------
    continuous_pipe = Pipeline([
        ("robust_scale", RobustScaler()),
    ])

    # --- Categorical branch -----------------------------------------------
    # OneHotEncoder with drop="first" avoids the dummy-variable trap and
    # reduces dimensionality by 1 per feature.  For deeper models,
    # learned entity embeddings (Guo & Berkhahn, 2016) replace this; those
    # are built into the FT-Transformer architecture (Phase 2).
    categorical_pipe = Pipeline([
        ("onehot", OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="infrequent_if_exist",
        )),
    ])

    # --- Interaction branch -----------------------------------------------
    # Needs access to multiple raw columns simultaneously, so it receives
    # the union of all columns used in InteractionFeatureGenerator.
    interaction_cols = [
        "luminosity_lux", "streetlight_density", "shadow_index",
        "footfall_density", "commercial_activity", "cctv_density",
        "transit_distance_m", "road_length_m", "vehicular_volume",
    ]
    interaction_pipe = Pipeline([
        ("interactions", InteractionFeatureGenerator()),
        ("robust_scale", RobustScaler()),
    ])

    # --- Proximity branch -------------------------------------------------
    proximity_cols = ["safe_zone_distance_m", "transit_distance_m"]
    proximity_pipe = Pipeline([
        ("decay", ProximityDecayTransformer(columns=proximity_cols)),
        ("robust_scale", RobustScaler()),
    ])

    # --- Assemble ColumnTransformer ----------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("temporal",    temporal_pipe,    TEMPORAL_COLS),
            ("skewed",      skewed_pipe,      SKEWED_COLS),
            ("continuous",  continuous_pipe,   PASSTHROUGH_CONTINUOUS_COLS),
            ("categorical", categorical_pipe,  CATEGORICAL_COLS),
            ("interaction", interaction_pipe,  interaction_cols),
            ("proximity",   proximity_pipe,    proximity_cols),
        ],
        remainder="drop",          # Drop segment_id and target
        verbose_feature_names_out=True,
        n_jobs=1,
    )

    return preprocessor


def preprocess_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    End-to-end: split → fit preprocessor on train → transform all sets.

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test   : np.ndarray  (preprocessed features)
        y_train, y_val, y_test   : np.ndarray  (target arrays)
        preprocessor             : fitted ColumnTransformer
        feature_names            : list[str]
    """
    data = df.copy()

    # Separate target
    y = data[TARGET_COL].values
    X = data.drop(columns=[TARGET_COL])

    # Stratify-friendly binning of the continuous target for balanced splits
    y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")

    # Train / (val+test) split
    X_train, X_temp, y_train, y_temp, bins_train, bins_temp = train_test_split(
        X, y, y_bins,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=y_bins,
    )

    # Val / test split from the remainder
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        random_state=random_state,
        stratify=bins_temp,
    )

    # Fit on train only → transform all
    preprocessor = build_preprocessing_pipeline()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t   = preprocessor.transform(X_val)
    X_test_t  = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())

    print(f"Preprocessing complete.")
    print(f"  Train : {X_train_t.shape}")
    print(f"  Val   : {X_val_t.shape}")
    print(f"  Test  : {X_test_t.shape}")
    print(f"  Features ({len(feature_names)}): {feature_names[:8]} ...")

    return {
        "X_train": X_train_t,
        "X_val":   X_val_t,
        "X_test":  X_test_t,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


# =====================================================================
#  CLI VALIDATION
# =====================================================================
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fear_free_night_navigator.csv"))
    result = preprocess_dataset(df)

    print(f"\nSample transformed row (first 10 features):")
    print(result["X_train"][0, :10])
