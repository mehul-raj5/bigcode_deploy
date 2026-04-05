"""
REST API for the Fear-Free Night Navigator
=============================================
Flask API serving the routing engine and map data.

Endpoints:
  GET  /api/route      — compute a safety-aware route
  GET  /api/heatmap    — get safety heatmap data
  POST /api/feedback   — submit user feedback (crime report / appreciation)
  GET  /api/feedback   — get aggregated feedback counts
  GET  /              — serve the frontend
"""

import os
import sys
import json
import time
import threading
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
from collections import defaultdict

DELHI_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DELHI_DIR)

from routing_engine import get_graph, _graph

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Background Thread for Loading the Graph Safely
def trigger_background_load():
    print("Background thread started. Loading the massive road graph...")
    get_graph()
    print("Graph fully loaded and ready to serve!")

# Start the thread immediately so Gunicorn doesn't block while binding the port
threading.Thread(target=trigger_background_load, daemon=True).start()


# =====================================================================
#  ROUTES
# =====================================================================

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/route")
def compute_route():
    """
    Compute a safety-aware route.

    Query params:
      start_lat, start_lon  — origin
      end_lat, end_lon      — destination
      mode                  — walking | bicycle | car
      profile               — standard | solo_women | family
      tier                  — fastest | balanced | safest | safe_short | pure_safety
      constraints           — comma-separated: no_dark_roads,no_alleys,no_unpaved
      leaving_time          — HH:00 format (18:00–05:00), optional
      leaving_day           — weekday | weekend, optional (defaults to current day)
    """
    try:
        start_lat = float(request.args.get("start_lat", 0))
        start_lon = float(request.args.get("start_lon", 0))
        end_lat = float(request.args.get("end_lat", 0))
        end_lon = float(request.args.get("end_lon", 0))
        mode = request.args.get("mode", "walking")
        profile = request.args.get("profile", "standard")
        tier = request.args.get("tier", "balanced")
        constraints_str = request.args.get("constraints", "")
        constraints = [c.strip() for c in constraints_str.split(",") if c.strip()]

        # Parse leaving time → score column name
        leaving_time = request.args.get("leaving_time", "")
        leaving_day = request.args.get("leaving_day", "")
        time_col = None
        if leaving_time:
            try:
                hour = int(leaving_time.split(":")[0])
                day_abbr = "Sat" if leaving_day == "weekend" else "Wed"
                time_col = f"score_{hour:02d}_{day_abbr}"
            except (ValueError, IndexError):
                pass

        # Validate
        if mode not in ("walking", "bicycle", "car"):
            return jsonify({"error": f"Invalid mode: {mode}"}), 400
        if profile not in ("standard", "solo_women", "family"):
            return jsonify({"error": f"Invalid profile: {profile}"}), 400
        if tier not in ("fastest", "balanced", "safest", "safe_short", "pure_safety"):
            return jsonify({"error": f"Invalid tier: {tier}"}), 400

        if not _graph._loaded:
            return jsonify({"error": "Server is still warming up and loading the massive road network into memory. Please try again in 30 seconds."}), 503

        graph = get_graph()
        result = graph.route(
            start_lat, start_lon,
            end_lat, end_lon,
            mode=mode,
            profile=profile,
            tier=tier,
            constraints=constraints,
            time_col=time_col,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/heatmap")
def safety_heatmap():
    """Return safety heatmap data for the map overlay."""
    try:
        scored_path = os.path.join(DELHI_DIR, "data", "delhi_scored_roads.csv")
        df = pd.read_csv(scored_path, usecols=[
            "mid_lat", "mid_lon", "avg_score", "worst_score_overall",
            "delhi_zone", "osm_name", "luminosity_lux"
        ])
        # Sample for performance (full 67K is too heavy for browser)
        if len(df) > 5000:
            df = df.sample(5000, random_state=42)

        points = []
        for _, row in df.iterrows():
            points.append({
                "lat": round(float(row["mid_lat"]), 6),
                "lon": round(float(row["mid_lon"]), 6),
                "score": round(float(row["avg_score"]), 1),
                "worst": round(float(row["worst_score_overall"]), 1),
                "zone": str(row["delhi_zone"]),
                "name": str(row["osm_name"]) if pd.notna(row["osm_name"]) else "",
            })

        return jsonify({"points": points})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bounds")
def map_bounds():
    """Return the bounding box of the Delhi network."""
    if not _graph._loaded:
        return jsonify({"error": "Graph still loading"}), 503

    graph = get_graph()
    coords = list(graph.node_coords.values())
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return jsonify({
        "south": min(lats),
        "north": max(lats),
        "west": min(lons),
        "east": max(lons),
        "center_lat": sum(lats) / len(lats),
        "center_lon": sum(lons) / len(lons),
    })


FEEDBACK_FILE = os.path.join(DELHI_DIR, "data", "user_feedback.json")

# Simple in-memory rate limiter: IP → list of timestamps
_rate_limit = defaultdict(list)
RATE_LIMIT_MAX = 10    # max requests
RATE_LIMIT_WINDOW = 60  # per 60 seconds


def _check_rate_limit(ip):
    """Return True if request is within rate limit."""
    now = time.time()
    # Clean old entries
    _rate_limit[ip] = [t for t in _rate_limit[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limit[ip].append(now)
    return True


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Submit user feedback (crime report or appreciation)."""
    try:
        ip = request.remote_addr or "unknown"
        if not _check_rate_limit(ip):
            return jsonify({"error": "Rate limit exceeded. Max 10 reports per minute."}), 429

        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        segment_id = str(data.get("segment_id", ""))
        fb_type = data.get("type", "")
        lat = data.get("lat")
        lon = data.get("lon")
        description = str(data.get("description", ""))[:500]  # cap description length

        if fb_type not in ("crime_report", "appreciation"):
            return jsonify({"error": "type must be 'crime_report' or 'appreciation'"}), 400
        if not segment_id:
            return jsonify({"error": "segment_id required"}), 400

        entry = {
            "segment_id": segment_id,
            "type": fb_type,
            "lat": float(lat) if lat is not None else None,
            "lon": float(lon) if lon is not None else None,
            "description": description,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Load existing feedback
        feedback = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedback = json.load(f)

        feedback.append(entry)

        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedback, f, indent=2)

        return jsonify({"status": "ok", "total_reports": len(feedback)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["GET"])
def get_feedback():
    """Get aggregated feedback counts per segment."""
    try:
        if not os.path.exists(FEEDBACK_FILE):
            return jsonify({"feedback": {}})

        with open(FEEDBACK_FILE, "r") as f:
            feedback = json.load(f)

        agg = {}
        for entry in feedback:
            sid = entry.get("segment_id", "")
            if sid not in agg:
                agg[sid] = {"crime_reports": 0, "appreciations": 0}
            if entry.get("type") == "crime_report":
                agg[sid]["crime_reports"] += 1
            elif entry.get("type") == "appreciation":
                agg[sid]["appreciations"] += 1

        return jsonify({"feedback": agg})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
