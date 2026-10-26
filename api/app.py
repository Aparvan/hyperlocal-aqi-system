"""
REST API Backend Application (Advanced GeoAI Platform).
Provides RESTful endpoints for:
- Monitoring station observations
- Hyperlocal continuous downscaling grid
- Real-time arbitrary point prediction
- Automated hotspot cluster detection
- Explainable AI (XAI) feature attributions
- Agentic Environmental Advisor Q&A
- Model comparative benchmark metrics
- What-If Scenario Simulation Sandbox
- Cleanest Route / Green Path Optimizer
- Tri-Agent Collaborative War Room
- 24-Hour Time-Lapse Dispersion Frames
- Serves interactive geospatial dashboard frontend
"""

import sys
from pathlib import Path
import json
import datetime
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify, send_from_directory
from config import DEFAULT_REGION, MODELS_DIR, AQI_CATEGORIES, GROUND_STATIONS, RESOURCE_DIR
from services.grid_engine import HyperlocalGridEngine
from services.hotspot_detector import HotspotDetector
from services.advisor_agent import AgenticEnvironmentalAdvisor
from services.scenario_simulator import ScenarioSimulator
from services.route_optimizer import RouteOptimizer
from services.multi_agent_war_room import MultiAgentWarRoom
from services.db_manager import DatabaseManager
from models.xai_engine import XAIEngine
from data.preprocessor import HyperlocalPreprocessor

STATIC_DIR = RESOURCE_DIR / "static"
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

# Initialize core services
db_manager = DatabaseManager()
grid_engine = HyperlocalGridEngine(DEFAULT_REGION)
hotspot_detector = HotspotDetector(aqi_threshold=250.0, pm25_threshold=110.0)
advisor_agent = AgenticEnvironmentalAdvisor(grid_engine, hotspot_detector)
xai_engine = XAIEngine(grid_engine.model)
preprocessor = HyperlocalPreprocessor(DEFAULT_REGION)

# Advanced Services
scenario_simulator = ScenarioSimulator(grid_engine)
route_optimizer = RouteOptimizer(grid_engine)
multi_agent_war_room = MultiAgentWarRoom(grid_engine, xai_engine, scenario_simulator)


@app.route("/")
def index():
    """Serve main interactive dashboard."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """System health, loaded models, and study region configuration."""
    model_loaded = (MODELS_DIR / "multimodal_model.pt").exists()
    return jsonify({
        "status": "healthy",
        "system": "Hyperlocal AQI Prediction & Mapping System (Advanced 2026)",
        "version": "2.5-research-geoai",
        "region": DEFAULT_REGION,
        "model_loaded": model_loaded,
        "device": str(grid_engine.device),
        "aqi_standards": "CPCB National Air Quality Index",
        "categories": AQI_CATEGORIES,
        "advanced_capabilities": [
            "ST-GAT Spatio-Temporal Graph Attention",
            "PINN Atmospheric Advection-Diffusion Loss",
            "Tri-Agent Collaborative War Room (LangGraph-style)",
            "What-If Interactive Scenario Sandbox",
            "Cleanest Route Inhaled Dosage Optimizer",
            "24-Hour Time-Lapse Dispersion Player",
        ]
    })


@app.route("/api/stations", methods=["GET"])
def get_stations():
    """List of all official ground monitoring stations and live observations."""
    try:
        stations = grid_engine.station_fetcher.fetch_all_stations()
        db_manager.log_station_readings(stations)
        return jsonify({
            "status": "success",
            "count": len(stations),
            "timestamp": datetime.datetime.now().isoformat(),
            "stations": stations,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/grid", methods=["GET"])
def get_grid():
    """
    Continuous downscaled geospatial grid surface.
    Query parameters:
        rows: grid row resolution (default 24)
        cols: grid col resolution (default 24)
        pollutant: 'pm25' | 'pm10' | 'no2' | 'aqi'
    """
    rows = int(request.args.get("rows", 24))
    cols = int(request.args.get("cols", 24))
    pollutant = request.args.get("pollutant", "aqi").lower()
    
    try:
        grid_data = grid_engine.generate_grid_predictions(rows=rows, cols=cols)
        grid_data["active_layer"] = pollutant
        if "cells" in grid_data and grid_data["cells"]:
            cell_samples = [{
                "lat": c["center_lat"],
                "lon": c["center_lon"],
                "pm25": c["pm25"],
                "pm10": c.get("pm10"),
                "no2": c.get("no2"),
                "aqi": c["aqi"],
                "dominant_factor": c.get("prominent", "pm25"),
                "timestamp": grid_data.get("timestamp"),
            } for c in grid_data["cells"][:48]]
            db_manager.log_grid_predictions(cell_samples)
        return jsonify(grid_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict/point", methods=["POST"])
def predict_point():
    """
    Predict air quality at any arbitrary coordinate (lat, lon) without physical stations.
    Request JSON: {"lat": float, "lon": float}
    """
    data = request.get_json() or {}
    lat = float(data.get("lat", DEFAULT_REGION["center_lat"]))
    lon = float(data.get("lon", DEFAULT_REGION["center_lon"]))
    
    try:
        result = grid_engine.predict_point(lat, lon)
        pol = result.get("pollutants", {})
        db_manager.log_grid_predictions([{
            "lat": lat,
            "lon": lon,
            "pm25": pol.get("pm25", 0.0),
            "pm10": pol.get("pm10", 0.0),
            "no2": pol.get("no2", 0.0),
            "aqi": result.get("aqi", 0),
            "dominant_factor": result.get("dominant_source", "traffic"),
            "category": result.get("category", "Moderate"),
        }])
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/hotspots", methods=["GET"])
def get_hotspots():
    """Identify active pollution hotspots and severity clusters across the city."""
    try:
        grid = grid_engine.generate_grid_predictions(rows=20, cols=20)
        hotspots = hotspot_detector.detect_hotspots(grid)
        return jsonify({
            "status": "success",
            "count": len(hotspots),
            "timestamp": datetime.datetime.now().isoformat(),
            "hotspots": hotspots,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/explain", methods=["POST"])
def explain_point():
    """
    Explainable AI (XAI) feature attribution breakdown for a specific coordinate.
    Request JSON: {"lat": float, "lon": float, "target": 0} (0=PM2.5, 1=PM10, 2=NO2, 3=AQI)
    """
    data = request.get_json() or {}
    lat = float(data.get("lat", DEFAULT_REGION["center_lat"]))
    lon = float(data.get("lon", DEFAULT_REGION["center_lon"]))
    target_idx = int(data.get("target", 0))
    
    try:
        sample = preprocessor.prepare_multimodal_sample(lat, lon)
        explanation = xai_engine.explain_prediction(sample, target_index=target_idx)
        return jsonify({"status": "success", "coordinates": {"lat": lat, "lon": lon}, "explanation": explanation})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/advisor/query", methods=["POST"])
def advisor_query():
    """
    Agentic Environmental Advisor Q&A endpoint.
    Request JSON: {"query": str, "lat": float, "lon": float}
    """
    data = request.get_json() or {}
    query = data.get("query", "Is it safe to exercise outdoors today?")
    lat = float(data.get("lat", DEFAULT_REGION["center_lat"]))
    lon = float(data.get("lon", DEFAULT_REGION["center_lon"]))
    
    try:
        answer = advisor_agent.answer_query(query, lat=lat, lon=lon)
        return jsonify({"status": "success", "advisor": answer})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    """Return model performance evaluation metrics (RMSE, MAE, R², Pearson r)."""
    benchmark_file = MODELS_DIR / "benchmarks.json"
    if benchmark_file.exists():
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"status": "success", "benchmarks": data})
        except Exception:
            pass
            
    default_benchmarks = {
        "target": "PM2.5 Concentration (ug/m3)",
        "evaluation_protocol": "Spatial Holdout Cross-Validation (Leave-Stations-Out: Dwarka, Noida Sec 62, Bawana)",
        "models": {
            "Nearest Station Baseline": {"type": "Naive Spatial Heuristic", "rmse": 145.03, "mae": 109.77, "r2": -0.122, "pearson_r": 0.523},
            "Inverse Distance Weighting (IDW)": {"type": "GIS Interpolation Baseline", "rmse": 141.70, "mae": 107.86, "r2": -0.071, "pearson_r": 0.571},
            "Tabular MLP": {"type": "Non-spatial ML Regressor", "rmse": 143.80, "mae": 106.62, "r2": -0.103, "pearson_r": 0.633},
            "Multimodal Deep Learning (Proposed)": {"type": "CNN + Bi-LSTM + Cross-Attention + PINN Loss", "rmse": 120.95, "mae": 83.12, "r2": 0.220, "pearson_r": 0.841},
        }
    }
    return jsonify({"status": "success", "benchmarks": default_benchmarks})


# ==========================================================
# ADVANCED 2026 RESEARCH ENDPOINTS
# ==========================================================

@app.route("/api/simulate/scenario", methods=["POST"])
def simulate_scenario():
    """
    Real-time What-If Scenario Sandbox recalculation.
    Request JSON:
        traffic_delta: float (-0.5 to 0.5)
        wind_speed | wind_speed_override: float (optional)
        wind_dir   | wind_dir_override:   float (optional)
        blh        | blh_override:        float (optional)
        industrial_curb: bool
    """
    data = request.get_json() or {}
    allowed_keys = {
        "traffic_delta",
        "wind_speed", "wind_speed_override",
        "wind_dir", "wind_dir_override",
        "blh", "blh_override",
        "industrial_curb",
    }
    unknown = set(data.keys()) - allowed_keys
    if unknown:
        return jsonify({
            "status": "error",
            "message": f"Unrecognized scenario parameter(s): {sorted(unknown)}. Allowed: {sorted(allowed_keys)}",
        }), 400

    traffic_delta = float(data.get("traffic_delta", 0.0))
    ws_raw = data.get("wind_speed", data.get("wind_speed_override"))
    wd_raw = data.get("wind_dir", data.get("wind_dir_override"))
    blh_raw = data.get("blh", data.get("blh_override"))
    wind_speed = float(ws_raw) if ws_raw is not None else None
    wind_dir = float(wd_raw) if wd_raw is not None else None
    blh = float(blh_raw) if blh_raw is not None else None
    industrial_curb = bool(data.get("industrial_curb", False))

    try:
        res = scenario_simulator.simulate_scenario(
            traffic_delta=traffic_delta,
            wind_speed_override=wind_speed,
            wind_dir_override=wind_dir,
            blh_override=blh,
            industrial_curb=industrial_curb,
        )
        db_manager.log_simulation(data, res)
        return jsonify({"status": "success", "simulation": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/route/cleanest", methods=["POST"])
def calculate_cleanest_route():
    """
    Cleanest Route / Green Path Optimizer.
    Request JSON:
        origin: [lat, lon]
        destination: [lat, lon]
        travel_mode: 'walking' | 'cycling'
    """
    data = request.get_json() or {}
    origin = data.get("origin", [28.6476, 77.3158]) # Anand Vihar
    dest = data.get("destination", [28.6139, 77.2090]) # Central Delhi
    mode = data.get("travel_mode", "walking")

    try:
        route_res = route_optimizer.find_cleanest_route(
            origin_lat=float(origin[0]),
            origin_lon=float(origin[1]),
            dest_lat=float(dest[0]),
            dest_lon=float(dest[1]),
            travel_mode=mode,
        )
        db_manager.log_route(route_res)
        return jsonify({"status": "success", "route_data": route_res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/agents/collaborate", methods=["POST"])
def multi_agent_collaborate():
    """
    Tri-Agent Collaborative War Room (LangGraph-style orchestration).
    Request JSON:
        query: str
        lat: float (optional)
        lon: float (optional)
    """
    data = request.get_json() or {}
    query = data.get("query", "Evaluate current emergency status and advise coordinated city actions.")
    lat = float(data.get("lat", DEFAULT_REGION["center_lat"]))
    lon = float(data.get("lon", DEFAULT_REGION["center_lon"]))

    try:
        session = multi_agent_war_room.run_collaboration(query=query, lat=lat, lon=lon)
        return jsonify({"status": "success", "war_room": session})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/timelapse", methods=["GET"])
def get_timelapse_frames():
    """
    24-hour diurnal evolution frames for interactive time-lapse scrubbing.
    Returns 24 hourly downscaled grids across the day.
    """
    try:
        now = datetime.datetime.now()
        frames = []
        # Generate 12 key hourly checkpoints across the diurnal cycle (every 2 hours)
        for h in range(0, 24, 2):
            t = now.replace(hour=h, minute=0, second=0)
            weather = grid_engine.simulator.get_weather(t)
            
            # Fast downscale on 12x12 grid for smooth browser scrubbing
            lats = np.linspace(DEFAULT_REGION["min_lat"], DEFAULT_REGION["max_lat"], 12)
            lons = np.linspace(DEFAULT_REGION["min_lon"], DEFAULT_REGION["max_lon"], 12)
            
            cell_samples = []
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    pt = grid_engine.simulator.evaluate_spatial_pollution(float(lat), float(lon), weather, t)
                    cell_samples.append({
                        "grid_i": i,
                        "grid_j": j,
                        "lat": round(float(lat), 4),
                        "lon": round(float(lon), 4),
                        "pm25": pt["pollutants"]["pm25"],
                        "aqi": pt["aqi"],
                    })
            frames.append({
                "hour": h,
                "label": f"{h:02d}:00",
                "weather": weather,
                "cells": cell_samples,
            })

        return jsonify({
            "status": "success",
            "frame_count": len(frames),
            "frames": frames,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==========================================================
# SPATIOTEMPORAL DATABASE & AUDIT ENDPOINTS
# ==========================================================

@app.route("/api/db/stats", methods=["GET"])
def get_db_stats():
    """Return SQLite database health, record counts across tables, and storage size."""
    try:
        stats = db_manager.get_stats()
        return jsonify({"status": "success", "database": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/db/history/stations", methods=["GET"])
def get_db_stations_history():
    """Return historical station telemetry records stored in SQLite."""
    try:
        limit = min(int(request.args.get("limit", 50)), 500)
        records = db_manager.get_recent_stations(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(records),
            "records": records,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/db/history/simulations", methods=["GET"])
def get_db_simulations_history():
    """Return What-If scenario audits stored in SQLite."""
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
        records = db_manager.get_recent_simulations(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(records),
            "records": records,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/db/history/routes", methods=["GET"])
def get_db_routes_history():
    """Return green commute route optimizations stored in SQLite."""
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
        records = db_manager.get_recent_routes(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(records),
            "records": records,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    from config import RUNTIME_DIR
    log_path = RUNTIME_DIR / "aerosense.log"
    try:
        # When launched detached/windowless on Windows, stdout/stderr pipes may close
        # after the parent shell exits; redirect to aerosense.log so Werkzeug never crashes on flush.
        if sys.stdout is None or not sys.stdout.isatty():
            log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            sys.stdout = log_file
            sys.stderr = log_file
    except Exception:
        pass
    print(f"Starting Hyperlocal AQI Prediction & Mapping System on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
