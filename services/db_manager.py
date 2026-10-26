"""
Database Management Module for AeroSense AI.
Provides thread-safe embedded SQLite persistence for:
- Official ground station telemetry history
- Downscaled hyperlocal grid prediction snapshots
- Policy what-if simulation audits
- Green commute pathfinding route logs
"""

import sqlite3
import json
import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Any, Generator
from config import DB_PATH, DATA_DIR, GROUND_STATIONS


class DatabaseManager:
    """Thread-safe SQLite manager for AeroSense AI spatiotemporal logs and telemetry."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
        self.populate_initial_data()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that guarantees connection is closed cleanly on exit."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize tables and indices if not already present."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # 1. Ground Stations Telemetry History
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS stations_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                pm25 REAL NOT NULL,
                pm10 REAL,
                no2 REAL,
                aqi INTEGER NOT NULL,
                category TEXT,
                source TEXT DEFAULT 'CPCB_TELEMETRY',
                UNIQUE(station_id, timestamp)
            );
            """)

            # 2. Hyperlocal Downscaled Predictions Snapshots
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hyperlocal_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                pm25 REAL NOT NULL,
                pm10 REAL,
                no2 REAL,
                aqi INTEGER NOT NULL,
                category TEXT,
                dominant_factor TEXT,
                model_name TEXT DEFAULT 'Multimodal Deep Learning'
            );
            """)

            # 3. What-If Scenario Simulations Audit Log
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS whatif_simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                traffic_delta REAL,
                industrial_curb REAL,
                wind_speed REAL,
                boundary_layer REAL,
                avg_baseline_aqi REAL,
                avg_simulated_aqi REAL,
                avg_reduction_pct REAL,
                active_grap_stage TEXT
            );
            """)

            # 4. Cleanest Route Pathfinding Optimization History
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS green_routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                origin_lat REAL,
                origin_lon REAL,
                dest_lat REAL,
                dest_lon REAL,
                distance_km REAL,
                baseline_dose REAL,
                green_dose REAL,
                dose_reduction_pct REAL
            );
            """)

            # Create indices for sub-millisecond retrieval
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_st_time ON stations_history(station_id, timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_time ON hyperlocal_predictions(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_time ON whatif_simulations(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_route_time ON green_routes(timestamp);")
            conn.commit()

    def log_station_readings(self, stations: list[dict]) -> int:
        """Batch insert station observations; ignores duplicates."""
        inserted = 0
        now_iso = datetime.datetime.now().isoformat()
        with self._connection() as conn:
            cursor = conn.cursor()
            for st in stations:
                st_id = st.get("station_id") or st.get("id", "UNKNOWN")
                name = st.get("name", "Unknown Station")
                ts = st.get("timestamp") or now_iso
                lat = float(st.get("lat", 0.0))
                lon = float(st.get("lon", 0.0))
                
                pollutants = st.get("pollutants", {})
                pm25 = float(pollutants.get("pm25", st.get("pm25", 0.0)))
                pm10 = float(pollutants.get("pm10", st.get("pm10", 0.0)))
                no2 = float(pollutants.get("no2", st.get("no2", 0.0)))
                aqi = int(round(st.get("aqi", 0)))
                cat = st.get("aqi_category", st.get("category", "Moderate"))
                
                try:
                    cursor.execute("""
                    INSERT OR IGNORE INTO stations_history 
                    (station_id, name, timestamp, lat, lon, pm25, pm10, no2, aqi, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (st_id, name, ts, lat, lon, pm25, pm10, no2, aqi, cat))
                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception:
                    continue
            conn.commit()
        return inserted

    def log_grid_predictions(self, grid_points: list[dict], model_name: str = "Multimodal Deep Learning") -> int:
        """Log downscaled prediction snapshots across the spatial grid."""
        inserted = 0
        now_iso = datetime.datetime.now().isoformat()
        with self._connection() as conn:
            cursor = conn.cursor()
            for pt in grid_points:
                ts = pt.get("timestamp") or now_iso
                lat = float(pt.get("lat", 0.0))
                lon = float(pt.get("lon", 0.0))
                pm25 = float(pt.get("pm25", 0.0))
                pm10 = float(pt.get("pm10", 0.0)) if pt.get("pm10") is not None else None
                no2 = float(pt.get("no2", 0.0)) if pt.get("no2") is not None else None
                aqi = int(round(pt.get("aqi", 0)))
                cat = pt.get("aqi_category", pt.get("category", "Unknown"))
                dom = pt.get("dominant_factor", pt.get("dominant_source"))
                
                try:
                    cursor.execute("""
                    INSERT INTO hyperlocal_predictions
                    (timestamp, lat, lon, pm25, pm10, no2, aqi, category, dominant_factor, model_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (ts, lat, lon, pm25, pm10, no2, aqi, cat, dom, model_name))
                    inserted += 1
                except Exception:
                    continue
            conn.commit()
        return inserted

    def log_simulation(self, params: dict, result: dict) -> int:
        """Audit log for What-If scenario execution."""
        now_iso = datetime.datetime.now().isoformat()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO whatif_simulations
            (timestamp, traffic_delta, industrial_curb, wind_speed, boundary_layer,
             avg_baseline_aqi, avg_simulated_aqi, avg_reduction_pct, active_grap_stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso,
                float(params.get("traffic_delta", 0.0)),
                float(params.get("industrial_curb", 0.0)),
                float(params.get("wind_speed", 0.0)),
                float(params.get("boundary_layer", 0.0)),
                float(result.get("avg_baseline_aqi", 0.0)),
                float(result.get("avg_simulated_aqi", 0.0)),
                float(result.get("avg_reduction_pct", 0.0)),
                result.get("active_grap_stage", "Normal"),
            ))
            conn.commit()
            return cursor.lastrowid

    def log_route(self, route_result: dict) -> int:
        """Audit log for Cleanest Route optimization."""
        now_iso = datetime.datetime.now().isoformat()
        origin = route_result.get("origin", {})
        dest = route_result.get("destination", {})
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO green_routes
            (timestamp, origin_lat, origin_lon, dest_lat, dest_lon, distance_km,
             baseline_dose, green_dose, dose_reduction_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso,
                float(origin.get("lat", 0.0)),
                float(origin.get("lon", 0.0)),
                float(dest.get("lat", 0.0)),
                float(dest.get("lon", 0.0)),
                float(route_result.get("distance_km", 0.0)),
                float(route_result.get("baseline_exposure_dose", 0.0)),
                float(route_result.get("clean_exposure_dose", 0.0)),
                float(route_result.get("dose_reduction_pct", 0.0)),
            ))
            conn.commit()
            return cursor.lastrowid

    def get_stats(self) -> dict:
        """Return system database statistics and record counts."""
        with self._connection() as conn:
            cursor = conn.cursor()
            st_count = cursor.execute("SELECT COUNT(*) FROM stations_history").fetchone()[0]
            pred_count = cursor.execute("SELECT COUNT(*) FROM hyperlocal_predictions").fetchone()[0]
            sim_count = cursor.execute("SELECT COUNT(*) FROM whatif_simulations").fetchone()[0]
            route_count = cursor.execute("SELECT COUNT(*) FROM green_routes").fetchone()[0]
            
            latest_time_row = cursor.execute("SELECT MAX(timestamp) FROM stations_history").fetchone()
            latest_time = latest_time_row[0] if latest_time_row and latest_time_row[0] else None

        db_size_kb = 0.0
        if self.db_path.exists():
            db_size_kb = round(self.db_path.stat().st_size / 1024.0, 2)

        return {
            "status": "healthy",
            "database_engine": "SQLite 3 (ACID, WAL mode)",
            "db_path": str(self.db_path),
            "size_kb": db_size_kb,
            "size_mb": round(db_size_kb / 1024.0, 2),
            "tables": {
                "stations_history": st_count,
                "hyperlocal_predictions": pred_count,
                "whatif_simulations": sim_count,
                "green_routes": route_count,
            },
            "total_records": st_count + pred_count + sim_count + route_count,
            "latest_update": latest_time,
        }

    def get_recent_stations(self, limit: int = 50) -> list[dict]:
        """Fetch latest station telemetry rows."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT station_id, name, timestamp, lat, lon, pm25, pm10, no2, aqi, category
            FROM stations_history
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_simulations(self, limit: int = 10) -> list[dict]:
        """Fetch latest scenario simulations."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT id, timestamp, traffic_delta, industrial_curb, wind_speed, boundary_layer,
                   avg_baseline_aqi, avg_simulated_aqi, avg_reduction_pct, active_grap_stage
            FROM whatif_simulations
            ORDER BY id DESC
            LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_routes(self, limit: int = 10) -> list[dict]:
        """Fetch latest green route queries."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT id, timestamp, origin_lat, origin_lon, dest_lat, dest_lon, distance_km,
                   baseline_dose, green_dose, dose_reduction_pct
            FROM green_routes
            ORDER BY id DESC
            LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def populate_initial_data(self) -> None:
        """Pre-populate database with historical records and current station snapshots."""
        with self._connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM stations_history").fetchone()[0]
        
        # If already populated, return
        if count > 0:
            return

        print("Initializing SQLite Database with historical station telemetry...")
        json_file = DATA_DIR / "training_dataset.json"
        st_map = {s["id"]: s["name"] for s in GROUND_STATIONS}

        records_to_insert = []
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for rec in data:
                    if rec.get("is_station") == 1 and rec.get("station_id"):
                        targets = rec.get("targets", {})
                        st_id = rec["station_id"]
                        records_to_insert.append({
                            "station_id": st_id,
                            "name": st_map.get(st_id, f"Station {st_id}"),
                            "timestamp": rec.get("timestamp", datetime.datetime.now().isoformat()),
                            "lat": rec.get("lat", 28.61),
                            "lon": rec.get("lon", 77.21),
                            "pollutants": {
                                "pm25": targets.get("pm25", 90.0),
                                "pm10": targets.get("pm10", 180.0),
                                "no2": targets.get("no2", 45.0),
                            },
                            "aqi": rec.get("aqi", 200.0),
                            "category": "Preloaded Historical",
                        })
            except Exception as e:
                print(f"Warning: Could not pre-populate from training_dataset.json: {e}")

        # Also insert initial snapshot from GROUND_STATIONS
        now_iso = datetime.datetime.now().isoformat()
        for s in GROUND_STATIONS:
            records_to_insert.append({
                "station_id": s["id"],
                "name": s["name"],
                "timestamp": now_iso,
                "lat": s["lat"],
                "lon": s["lon"],
                "pollutants": {"pm25": 120.0, "pm10": 220.0, "no2": 55.0},
                "aqi": 300,
                "category": "Very Poor",
            })

        if records_to_insert:
            self.log_station_readings(records_to_insert)
            print(f"Pre-populated {len(records_to_insert)} records into SQLite stations_history table.")
