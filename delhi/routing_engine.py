"""
Safety-Aware Routing Engine for Delhi
=======================================
Implements a modified A* algorithm with:
  1. Multi-objective impedance: Cost_e = Time_e × (1 + (β_tier + β_profile) × β_mode × Risk_e^α)
  2. Dynamic weights for travel mode, user profile, and safety tier
  3. Graph pruning for hard constraints (no dark roads, no alleys)

Uses the Delhi OSM road network with pre-computed safety scores.

Author : Fear-Free Night Navigator Team
"""

import os
import json
import heapq
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict

DELHI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# =====================================================================
#  TRAVEL MODE WEIGHTS (β_mode)
# =====================================================================
MODE_WEIGHTS = {
    "walking":    0.6,   # Exposed but profile matters more
    "bicycle":    0.3,   # Fast but exposed
    "car":        0.05,  # Enclosed, minimal street-level risk
}

# Average speeds in km/h for time estimation
MODE_SPEEDS = {
    "walking":    5.0,
    "bicycle":   15.0,
    "car":       30.0,
}

# =====================================================================
#  USER PROFILE WEIGHTS (β_profile)
# =====================================================================
PROFILE_WEIGHTS = {
    "standard":  0.0,    # No extra safety bias
    "solo_women": 3.0,   # High sensitivity — major rerouting for safety
    "family":    2.0,    # Moderate — meaningful detours for safer roads
}

# =====================================================================
#  SAFETY TIER WEIGHTS (β_tier)
# =====================================================================
TIER_WEIGHTS = {
    "fastest":      0.0,    # Pure time routing
    "balanced":     0.8,    # Meaningful detours for safety
    "safest":       3.0,    # Heavy safety penalty — max safety preference
    "safe_short":   1.5,    # Safe but less penalty for extra distance
    "pure_safety":  0.0,    # Special: cost = Risk only, time ignored
}

# =====================================================================
#  HARD CONSTRAINT THRESHOLDS
# =====================================================================
PRUNE_THRESHOLDS = {
    "no_dark_roads":   15.0,    # luminosity_lux below this
    "no_alleys":       None,    # highway type filter
    "no_unpaved":      None,    # highway type filter
}

ALLEY_TYPES = {"service", "path", "track", "footway", "steps", "['footway', 'steps']"}
UNPAVED_TYPES = {"track", "path"}


# =====================================================================
#  GRAPH BUILDER
# =====================================================================
class DelhiRoutingGraph:
    """
    Builds and holds the routing graph from OSM + scored data.
    Supports dynamic A* with safety-aware impedance and graph pruning.
    """

    def __init__(self):
        self.adj = defaultdict(list)       # node -> [(neighbor, edge_idx)]
        self.edges = []                    # list of edge dicts
        self.node_coords = {}              # node_id -> (lat, lon)
        self._loaded = False
        self._node_ids = None
        self._node_lats = None
        self._node_lons = None

    def load(self):
        """Load OSM edges and merge with safety scores."""
        if self._loaded:
            return

        print("Loading Delhi routing graph...")

        # Load raw edges (has u, v, geometry)
        raw_path = os.path.join(DELHI_DIR, "delhi_osm_raw.gpkg")
        raw = gpd.read_file(raw_path, layer="edges")

        # Load scored data (has safety scores)
        scored_path = os.path.join(DELHI_DIR, "delhi_scored_roads.csv")
        scored = pd.read_csv(scored_path)

        print(f"  Edges: {len(raw):,}")
        print(f"  Scores: {len(scored):,}")

        # Build edge list and adjacency
        for idx in range(len(raw)):
            u = int(raw.iloc[idx]["u"])
            v = int(raw.iloc[idx]["v"])
            geom = raw.geometry.iloc[idx]

            # Edge attributes from raw
            length_m = float(raw.iloc[idx]["length"])
            highway = str(raw.iloc[idx].get("highway", ""))
            name = raw.iloc[idx].get("name", "")
            if pd.isna(name):
                name = ""

            # Safety attributes from scored data
            row = scored.iloc[idx] if idx < len(scored) else None
            if row is not None:
                avg_score = float(row.get("avg_score", 50))
                worst_score = float(row.get("worst_score_overall", 50))
                luminosity = float(row.get("luminosity_lux", 20))
                zone = str(row.get("delhi_zone", "General_Urban"))
                infra_quality = str(row.get("infrastructure_quality", "Both_Sides"))
                speed_limit = float(row.get("speed_limit_kmh", 40))
            else:
                avg_score = 50
                worst_score = 50
                luminosity = 20
                zone = "General_Urban"
                infra_quality = "Both_Sides"
                speed_limit = 40

            # Per-time-slot scores (score_18_Wed, score_19_Sat, etc.)
            time_scores = {}
            if row is not None:
                for col in scored.columns:
                    if col.startswith("score_") and col != "score":
                        parts = col.split("_")
                        if len(parts) == 3 and parts[1].isdigit():
                            time_scores[col] = float(row[col]) if pd.notna(row[col]) else avg_score

            # Extract coordinates from geometry
            coords = list(geom.coords)

            edge = {
                "idx": idx,
                "u": u,
                "v": v,
                "length_m": length_m,
                "highway": highway,
                "name": name,
                "avg_score": avg_score,
                "worst_score": worst_score,
                "luminosity": luminosity,
                "zone": zone,
                "infra_quality": infra_quality,
                "speed_limit": speed_limit,
                "coords": coords,
            }
            # Merge time-slot scores directly onto edge dict
            edge.update(time_scores)

            self.edges.append(edge)
            self.adj[u].append((v, idx))
            self.adj[v].append((u, idx))  # bidirectional

            # Store node coordinates (from first/last point of geometry)
            if u not in self.node_coords:
                self.node_coords[u] = (coords[0][1], coords[0][0])  # (lat, lon)
            if v not in self.node_coords:
                self.node_coords[v] = (coords[-1][1], coords[-1][0])

        self._loaded = True

        # Compute actual score range for risk normalization
        all_scores = [e["avg_score"] for e in self.edges]
        self._score_min = min(all_scores)
        self._score_max = max(all_scores)
        self._score_range = self._score_max - self._score_min
        print(f"  Score range: {self._score_min:.1f} – {self._score_max:.1f}")

        # Build numpy arrays for fast nearest-node lookup
        self._node_ids = np.array(list(self.node_coords.keys()))
        self._node_lats = np.array([self.node_coords[n][0] for n in self._node_ids])
        self._node_lons = np.array([self.node_coords[n][1] for n in self._node_ids])

        print(f"  Nodes: {len(self.node_coords):,}")
        self._load_feedback()
        print(f"  Graph ready.")

    def _load_feedback(self):
        """Adjust edge scores based on user feedback (crime reports / appreciations)."""
        feedback_path = os.path.join(DELHI_DIR, "user_feedback.json")
        if not os.path.exists(feedback_path):
            return

        try:
            with open(feedback_path, "r") as f:
                feedback = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        # Aggregate per edge index
        counts = {}
        for entry in feedback:
            sid = entry.get("segment_id", "")
            # segment_id is the edge index as string
            try:
                idx = int(sid)
            except (ValueError, TypeError):
                continue
            if idx not in counts:
                counts[idx] = {"crime": 0, "appreciate": 0}
            if entry.get("type") == "crime_report":
                counts[idx]["crime"] += 1
            elif entry.get("type") == "appreciation":
                counts[idx]["appreciate"] += 1

        # Apply adjustments
        adjusted = 0
        for idx, c in counts.items():
            if idx < len(self.edges):
                adjustment = -(c["crime"] * 2.0) + (c["appreciate"] * 1.0)
                adjustment = max(-20, min(10, adjustment))
                edge = self.edges[idx]
                edge["avg_score"] = max(0, min(100, edge["avg_score"] + adjustment))
                edge["worst_score"] = max(0, min(100, edge["worst_score"] + adjustment))
                adjusted += 1

        if adjusted > 0:
            print(f"  Feedback applied: {adjusted} edges adjusted from {len(feedback)} reports")

    def find_nearest_node(self, lat, lon):
        """Find the nearest graph node to a given lat/lon (vectorized)."""
        d = (self._node_lats - lat) ** 2 + (self._node_lons - lon) ** 2
        return int(self._node_ids[np.argmin(d)])

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Haversine distance in meters."""
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _edge_cost(self, edge, mode, profile, tier, use_time_of_day=None):
        """
        Multi-objective impedance function.

        Cost_e = Time_e × (1 + (β_tier + β_profile) × β_mode × Risk_e^α)

        Profile and tier are additive so each independently shifts routes.
        Mode acts as a scaling factor (walking amplifies, car dampens).

        Risk_e = (100 - safety_score) / 100
        α = 1.5 (super-linear penalty for dangerous roads)
        Time_e = length_m / speed (seconds)
        """
        beta_mode = MODE_WEIGHTS[mode]
        beta_profile = PROFILE_WEIGHTS[profile]
        beta_tier = TIER_WEIGHTS[tier]

        # Time in seconds
        speed_mps = MODE_SPEEDS[mode] * 1000 / 3600
        time_s = edge["length_m"] / speed_mps

        # Risk normalized to actual data range [score_min, score_max] → [1, 0]
        score = edge["avg_score"]
        if use_time_of_day and use_time_of_day in edge:
            score = edge[use_time_of_day]

        risk = 1.0 - (score - self._score_min) / self._score_range if self._score_range > 0 else 0.5
        risk = max(0.0, min(1.0, risk))
        alpha = 2.0  # quadratic penalty on normalized risk

        # Pure safety mode: cost = risk only, time is just a tiny tiebreaker
        if tier == "pure_safety":
            cost = (risk ** alpha) + time_s * 0.001
            return cost

        # Additive: profile + tier, scaled by mode exposure
        safety_penalty = (beta_tier + beta_profile) * beta_mode * (risk ** alpha)
        cost = time_s * (1.0 + safety_penalty)

        return cost

    def _heuristic(self, node, goal_lat, goal_lon, tier="balanced"):
        """A* heuristic: straight-line time estimate to goal."""
        if node not in self.node_coords:
            return 0
        lat, lon = self.node_coords[node]
        dist = self._haversine(lat, lon, goal_lat, goal_lon)
        # Pure safety: heuristic must be 0 (cost is risk-based, not distance-based)
        if tier == "pure_safety":
            return 0
        # Use walking speed as conservative lower bound
        speed_mps = 5.0 * 1000 / 3600
        return dist / speed_mps

    def route(self, start_lat, start_lon, end_lat, end_lon,
              mode="walking", profile="standard", tier="balanced",
              constraints=None, time_col=None):
        """
        Find optimal route using modified A* with safety-aware impedance.

        Parameters
        ----------
        start_lat, start_lon : float — origin coordinates
        end_lat, end_lon : float — destination coordinates
        mode : str — 'walking', 'bicycle', 'car'
        profile : str — 'standard', 'solo_women', 'family'
        tier : str — 'fastest', 'balanced', 'safest'
        constraints : list[str] — hard constraints like 'no_dark_roads', 'no_alleys'
        time_col : str or None — score column like 'score_22_Sat' for time-specific routing

        Returns
        -------
        dict with route geometry, stats, and step-by-step edges
        """
        self.load()

        if constraints is None:
            constraints = []

        # Find nearest nodes
        start_node = self.find_nearest_node(start_lat, start_lon)
        end_node = self.find_nearest_node(end_lat, end_lon)

        if start_node is None or end_node is None:
            return {"error": "Could not find nodes near the given coordinates"}

        goal_lat, goal_lon = self.node_coords[end_node]

        # ── Graph Pruning (Hard Constraints) ─────────────────────────
        pruned_edges = set()
        for constraint in constraints:
            if constraint == "no_dark_roads":
                threshold = PRUNE_THRESHOLDS["no_dark_roads"]
                for i, e in enumerate(self.edges):
                    if e["luminosity"] < threshold:
                        pruned_edges.add(i)
            elif constraint == "no_alleys":
                for i, e in enumerate(self.edges):
                    if e["highway"] in ALLEY_TYPES:
                        pruned_edges.add(i)
            elif constraint == "no_unpaved":
                for i, e in enumerate(self.edges):
                    if e["highway"] in UNPAVED_TYPES:
                        pruned_edges.add(i)

        # ── A* Search ────────────────────────────────────────────────
        # Priority queue: (f_score, node_id)
        open_set = [(0, start_node)]
        came_from = {}        # node -> (prev_node, edge_idx)
        g_score = {start_node: 0}
        visited = set()

        while open_set:
            f, current = heapq.heappop(open_set)

            if current == end_node:
                break

            if current in visited:
                continue
            visited.add(current)

            for neighbor, edge_idx in self.adj.get(current, []):
                # Skip pruned edges
                if edge_idx in pruned_edges:
                    continue

                edge = self.edges[edge_idx]
                cost = self._edge_cost(edge, mode, profile, tier, use_time_of_day=time_col)
                tentative_g = g_score[current] + cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = (current, edge_idx)
                    g_score[neighbor] = tentative_g
                    h = self._heuristic(neighbor, goal_lat, goal_lon, tier)
                    heapq.heappush(open_set, (tentative_g + h, neighbor))

        # ── Reconstruct path ─────────────────────────────────────────
        if end_node not in came_from and start_node != end_node:
            return {"error": "No route found between the given points"}

        path_edges = []
        node = end_node
        while node in came_from:
            prev_node, edge_idx = came_from[node]
            path_edges.append(edge_idx)
            node = prev_node
        path_edges.reverse()

        # ── Build route response ─────────────────────────────────────
        total_distance = 0
        total_time = 0
        route_coords = []
        edge_details = []
        safety_scores = []
        speed_mps = MODE_SPEEDS[mode] * 1000 / 3600

        # Determine which score column to use (time-specific or avg)
        _has_time = time_col and len(self.edges) > 0 and time_col in self.edges[0]
        score_key = time_col if _has_time else "avg_score"

        for edge_idx in path_edges:
            edge = self.edges[edge_idx]
            total_distance += edge["length_m"]
            total_time += edge["length_m"] / speed_mps
            safety_scores.append(edge[score_key])

            # Determine if we traverse forward or backward
            coords = edge["coords"]
            if route_coords and route_coords[-1] != coords[0]:
                coords = list(reversed(coords))

            if not route_coords:
                route_coords.extend(coords)
            else:
                route_coords.extend(coords[1:])  # skip duplicate join point

            edge_details.append({
                "name": edge["name"] or "Unnamed road",
                "length_m": round(edge["length_m"], 1),
                "safety_score": round(edge[score_key], 1),
                "worst_score": round(edge["worst_score"], 1),
                "luminosity": round(edge["luminosity"], 1),
                "highway": edge["highway"],
                "zone": edge["zone"],
            })

        # Route GeoJSON
        route_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lon, lat in route_coords],
            },
            "properties": {
                "total_distance_m": round(total_distance, 1),
                "total_distance_km": round(total_distance / 1000, 2),
                "total_time_s": round(total_time, 1),
                "total_time_min": round(total_time / 60, 1),
                "avg_safety_score": round(np.mean(safety_scores), 2) if safety_scores else 0,
                "min_safety_score": round(np.min(safety_scores), 2) if safety_scores else 0,
                "max_safety_score": round(np.max(safety_scores), 2) if safety_scores else 0,
                "segments_count": len(path_edges),
                "mode": mode,
                "profile": profile,
                "tier": tier,
                "constraints": constraints,
            },
        }

        # Safety-colored segments for frontend visualization
        colored_segments = []
        for edge_idx in path_edges:
            edge = self.edges[edge_idx]
            score = edge[score_key]

            # Color based on safety score
            if score >= 60:
                color = "#22c55e"  # green - safe
            elif score >= 45:
                color = "#eab308"  # yellow - moderate
            elif score >= 35:
                color = "#f97316"  # orange - caution
            else:
                color = "#ef4444"  # red - dangerous

            coords = edge["coords"]
            colored_segments.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lon, lat in coords],
                },
                "properties": {
                    "safety_score": round(score, 1),
                    "color": color,
                    "name": edge["name"] or "Unnamed",
                    "length_m": round(edge["length_m"], 1),
                },
            })

        return {
            "route": route_geojson,
            "colored_segments": {
                "type": "FeatureCollection",
                "features": colored_segments,
            },
            "summary": {
                "distance_km": round(total_distance / 1000, 2),
                "time_min": round(total_time / 60, 1),
                "avg_safety": round(np.mean(safety_scores), 1) if safety_scores else 0,
                "min_safety": round(np.min(safety_scores), 1) if safety_scores else 0,
                "segments": len(path_edges),
                "pruned_edges": len(pruned_edges),
            },
            "edges": edge_details,
        }


# Singleton graph instance
_graph = DelhiRoutingGraph()


def get_graph():
    """Get or create the singleton routing graph."""
    _graph.load()
    return _graph


if __name__ == "__main__":
    g = get_graph()

    # Demo: Route from Dwarka to Connaught Place
    result = g.route(
        start_lat=28.5921, start_lon=77.0460,  # Dwarka
        end_lat=28.6315, end_lon=77.2167,       # Connaught Place
        mode="walking",
        profile="solo_women",
        tier="safest",
        constraints=["no_dark_roads"],
    )

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        s = result["summary"]
        print(f"Route: {s['distance_km']} km, {s['time_min']} min")
        print(f"Safety: avg={s['avg_safety']}, min={s['min_safety']}")
        print(f"Segments: {s['segments']}, Pruned: {s['pruned_edges']}")
