"""
Delhi Road Segment Scoring Pipeline
=====================================
Loads the OSM-derived road segments, runs them through the trained
FT-Transformer model at multiple time slots, and stores the results.

Output:
  - delhi_scored_roads.csv       (flat file with scores at each time)
  - delhi_scored_roads.geojson   (map-ready with worst-case score)

Author : Fear-Free Night Navigator Team
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import joblib

# Add project root and training module to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))

from preprocessing_pipeline import (
    preprocess_dataset, build_preprocessing_pipeline,
    TemporalCyclicalEncoder, InteractionFeatureGenerator, ProximityDecayTransformer,
)
from ft_transformer import build_ft_transformer

# Register custom transformers under __main__ so joblib can unpickle
# preprocessors that were saved from a Colab notebook's __main__ scope.
import __main__
__main__.TemporalCyclicalEncoder = TemporalCyclicalEncoder
__main__.InteractionFeatureGenerator = InteractionFeatureGenerator
__main__.ProximityDecayTransformer = ProximityDecayTransformer

# =============================================================================
#  CONFIG
# =============================================================================
DELHI_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DELHI_DIR, "data")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")

SEGMENTS_CSV = os.path.join(DATA_DIR, "delhi_road_segments.csv")
CHECKPOINT = os.path.join(TRAINING_DIR, "checkpoints", "final_model.pt")
COLAB_PREPROCESSOR = os.path.join(TRAINING_DIR, "checkpoints", "preprocessor.joblib")
TRAINING_DATA = os.path.join(TRAINING_DIR, "data", "fear_free_night_navigator.csv")

# Time slots to score (18:00 – 05:00)
TIME_SLOTS = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
# Representative days
DAYS = ["Wednesday", "Saturday"]  # weekday + weekend

BATCH_SIZE = 2048
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
#  STEP 1 — Fit preprocessor on training data
# =============================================================================
def get_fitted_preprocessor():
    """
    Load the preprocessor that was fitted during Colab training.
    Falls back to fitting locally on training data if not available.
    Also returns n_features for model instantiation.
    """
    # Prefer the Colab-trained preprocessor for consistency
    if os.path.exists(COLAB_PREPROCESSOR):
        try:
            print(f"Loading Colab preprocessor from {COLAB_PREPROCESSOR}...")
            preprocessor = joblib.load(COLAB_PREPROCESSOR)
            # Get feature count from a dummy transform
            df_sample = pd.read_csv(TRAINING_DATA, nrows=5)
            df_sample = df_sample.drop(columns=["target_safety_score"], errors="ignore")
            n_features = preprocessor.transform(df_sample).shape[1]
            print(f"  Output features: {n_features}")
            return preprocessor, n_features
        except Exception as e:
            print(f"  Could not load Colab preprocessor: {e}")
            print("  Falling back to local preprocessor...")

    # Fallback: locally cached preprocessor
    preprocessor_path = os.path.join(TRAINING_DIR, "data", "fitted_preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        print("Loading cached fitted preprocessor...")
        preprocessor = joblib.load(preprocessor_path)
        df_sample = pd.read_csv(TRAINING_DATA, nrows=5)
        df_sample = df_sample.drop(columns=["target_safety_score"], errors="ignore")
        n_features = preprocessor.transform(df_sample).shape[1]
        return preprocessor, n_features

    # Last resort: fit from scratch
    print("Fitting preprocessor on training data...")
    df_train = pd.read_csv(TRAINING_DATA)
    data = preprocess_dataset(df_train)
    preprocessor = data["preprocessor"]
    n_features = data["X_train"].shape[1]

    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    print(f"  Output features: {n_features}")

    return preprocessor, n_features


# =============================================================================
#  STEP 2 — Load trained model
# =============================================================================
def load_model(n_features):
    """Load the trained model from checkpoint. Detects architecture automatically."""
    print(f"\nLoading model from {CHECKPOINT}")
    print(f"  Device: {DEVICE}")

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)

    # Use n_features and config from checkpoint if available
    ckpt_n_features = ckpt.get("n_features", n_features)
    ckpt_config = ckpt.get("config", "base")
    architecture = ckpt.get("architecture", "FTTransformer")

    if architecture == "FTTransformerGAT":
        # Import from colab training script
        try:
            from colab_train_gat import FTTransformerGAT, build_model as build_gat_model
            from colab_train_gat import (
                GEGLU as GAT_GEGLU, PreNormResidual as GAT_PreNormResidual,
                FeatureTokenizer as GAT_FeatureTokenizer, GATFallback,
            )
            # Register for pickle compatibility
            __main__.FTTransformerGAT = FTTransformerGAT
            __main__.GATFallback = GATFallback
            __main__.GAT_GEGLU = GAT_GEGLU

            print(f"  Architecture: FTTransformerGAT")
            model = build_gat_model(ckpt_n_features, config=ckpt_config)
        except ImportError:
            print("  WARNING: colab_train_gat not found, falling back to FTTransformer")
            model = build_ft_transformer(ckpt_n_features, config=ckpt_config)
    else:
        print(f"  Architecture: FTTransformer")
        model = build_ft_transformer(ckpt_n_features, config=ckpt_config)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Print checkpoint metadata
    metrics = ckpt.get("metrics", {})
    if metrics:
        print(f"  Checkpoint metrics: MAE={metrics.get('MAE', '?')}, "
              f"RMSE={metrics.get('RMSE', '?')}, R²={metrics.get('R²', '?')}")
    else:
        epoch = ckpt.get("epoch", "?")
        val_mae = ckpt.get("val_mae", "?")
        print(f"  Checkpoint epoch: {epoch}, val_mae: {val_mae}")

    return model


# =============================================================================
#  STEP 3 — Prepare segment data for each time slot
# =============================================================================
def prepare_segment_batch(segments_df, hour, day_of_week):
    """
    Take the spatial features from OSM segments and add temporal
    features for a specific time slot.

    Returns a DataFrame matching the training data schema.
    """
    df = segments_df.copy()

    # Add temporal features
    df["time_of_night"] = f"{hour:02d}:00"
    df["day_of_week"] = day_of_week

    # Select only the columns the preprocessor expects
    model_cols = [
        "segment_id", "road_length_m", "luminosity_lux", "streetlight_density",
        "shadow_index", "footfall_density", "commercial_activity",
        "vehicular_volume", "transit_distance_m", "incident_density",
        "infrastructure_quality", "cctv_density", "safe_zone_distance_m",
        "time_of_night", "day_of_week", "zone_type", "speed_limit_kmh",
    ]

    # Apply time-based scaling to dynamic features
    activity_map = {
        18: 1.0, 19: 0.95, 20: 0.85, 21: 0.70, 22: 0.55, 23: 0.40,
        0: 0.25, 1: 0.15, 2: 0.10, 3: 0.08, 4: 0.07, 5: 0.12,
    }
    time_factor = activity_map[hour]
    weekend_boost = 1.3 if day_of_week in ("Friday", "Saturday", "Sunday") else 1.0

    # Modulate footfall and vehicular volume by time
    df["footfall_density"] = (df["footfall_density"] * time_factor * weekend_boost).astype(int)
    df["vehicular_volume"] = (df["vehicular_volume"] * time_factor * 0.8).astype(int)
    # Commercial activity decays at night (especially Old Delhi markets close by 22:00)
    df["commercial_activity"] = (df["commercial_activity"] * time_factor * weekend_boost).astype(int)

    return df[model_cols]


# =============================================================================
#  STEP 4 — Batch inference
# =============================================================================
@torch.no_grad()
def score_batch(model, preprocessor, df_batch):
    """
    Transform features and run model inference.
    Returns numpy array of safety scores [0-100].
    """
    # Transform features (drop segment_id before preprocessing)
    X = preprocessor.transform(df_batch)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    # Batch inference
    all_scores = []
    for start in range(0, len(X_tensor), BATCH_SIZE):
        batch = X_tensor[start:start + BATCH_SIZE]
        scores = model(batch).squeeze(-1).cpu().numpy()
        all_scores.append(scores)

    return np.concatenate(all_scores)


# =============================================================================
#  MAIN PIPELINE
# =============================================================================
def main():
    # --- 1. Load segments ---
    print("=" * 60)
    print("DELHI ROAD SEGMENT SCORING PIPELINE")
    print("=" * 60)

    if not os.path.exists(SEGMENTS_CSV):
        print(f"\nERROR: {SEGMENTS_CSV} not found.")
        print("Run delhi_osm_pipeline.py first to download and prepare segments.")
        sys.exit(1)

    segments = pd.read_csv(SEGMENTS_CSV)
    N = len(segments)
    print(f"\nLoaded {N} road segments")

    # --- 2. Fit preprocessor ---
    preprocessor, n_features = get_fitted_preprocessor()

    # --- 3. Load model ---
    model = load_model(n_features)

    # --- 4. Score at each time slot & day ---
    print(f"\nScoring {N} segments × {len(TIME_SLOTS)} hours × {len(DAYS)} day types...")
    print(f"  Total inferences: {N * len(TIME_SLOTS) * len(DAYS):,}")

    score_columns = {}  # "score_HH_Day" → numpy array

    for day in DAYS:
        for hour in tqdm(TIME_SLOTS, desc=f"  {day}"):
            col_name = f"score_{hour:02d}_{day[:3]}"

            # Prepare batch with time features
            df_batch = prepare_segment_batch(segments, hour, day)

            # Run inference
            scores = score_batch(model, preprocessor, df_batch)
            score_columns[col_name] = np.round(scores, 2)

    # --- 5. Compute aggregate scores ---
    score_df = pd.DataFrame(score_columns)

    # Worst-case (minimum safety = most dangerous time)
    weekday_cols = [c for c in score_df.columns if c.endswith("_Wed")]
    weekend_cols = [c for c in score_df.columns if c.endswith("_Sat")]

    segments["worst_score_weekday"] = score_df[weekday_cols].min(axis=1).round(2)
    segments["worst_score_weekend"] = score_df[weekend_cols].min(axis=1).round(2)
    segments["worst_score_overall"] = score_df.min(axis=1).round(2)
    segments["best_score_overall"] = score_df.max(axis=1).round(2)
    segments["avg_score"] = score_df.mean(axis=1).round(2)

    # Worst time identification
    segments["worst_time_col"] = score_df.idxmin(axis=1)

    # Add all individual time scores
    for col_name, scores in score_columns.items():
        segments[col_name] = scores

    # --- 6. Save results ---
    output_csv = os.path.join(DATA_DIR, "delhi_scored_roads.csv")
    segments.to_csv(output_csv, index=False)
    print(f"\nScored CSV saved: {output_csv}")

    # GeoJSON for map visualization (with geometry if available)
    try:
        import geopandas as gpd
        gpkg_path = os.path.join(DATA_DIR, "delhi_road_segments.gpkg")
        if os.path.exists(gpkg_path):
            geo_segments = gpd.read_file(gpkg_path)
            # Merge scores into geodataframe
            score_cols = ["worst_score_overall", "best_score_overall",
                          "avg_score", "worst_score_weekday", "worst_score_weekend",
                          "worst_time_col"]
            for col in score_cols:
                geo_segments[col] = segments[col].values

            geojson_path = os.path.join(DATA_DIR, "delhi_scored_roads.geojson")
            geo_segments.to_file(geojson_path, driver="GeoJSON")
            print(f"GeoJSON saved: {geojson_path}")
    except Exception as e:
        print(f"GeoJSON export skipped: {e}")

    # --- 7. Summary ---
    print(f"\n{'='*60}")
    print("SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"Total segments scored: {N}")
    print(f"Time slots: {len(TIME_SLOTS)} hours × {len(DAYS)} day types = {len(score_columns)} scores/segment")
    print(f"\nOverall safety score distribution:")
    print(segments["avg_score"].describe().round(2).to_string())
    print(f"\nBy Delhi zone:")
    zone_summary = segments.groupby("delhi_zone").agg({
        "worst_score_overall": ["mean", "min"],
        "avg_score": "mean",
        "best_score_overall": ["mean", "max"],
    }).round(2)
    print(zone_summary.to_string())
    print(f"\nMost dangerous segments (lowest worst-case score):")
    worst = segments.nsmallest(10, "worst_score_overall")[
        ["segment_id", "osm_name", "delhi_zone", "worst_score_overall", "avg_score", "worst_time_col"]
    ]
    print(worst.to_string())
    print(f"\nSafest segments (highest worst-case score):")
    safest = segments.nlargest(10, "worst_score_overall")[
        ["segment_id", "osm_name", "delhi_zone", "worst_score_overall", "avg_score", "worst_time_col"]
    ]
    print(safest.to_string())


if __name__ == "__main__":
    main()
