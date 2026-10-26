"""Unit tests for SQLite Database Persistence and Audit Logging."""

import sys
import tempfile
import shutil
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from services.db_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_aerosense.db"
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initialization(self):
        self.assertTrue(self.db_path.exists())
        stats = self.db.get_stats()
        self.assertEqual(stats["status"], "healthy")
        self.assertIn("stations_history", stats["tables"])
        self.assertIn("hyperlocal_predictions", stats["tables"])
        self.assertIn("whatif_simulations", stats["tables"])
        self.assertIn("green_routes", stats["tables"])

    def test_station_logging(self):
        future_ts = (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat()
        stations = [{
            "station_id": "TEST_001",
            "name": "Test Anand Vihar",
            "timestamp": future_ts,
            "lat": 28.6476,
            "lon": 77.3158,
            "pollutants": {"pm25": 140.5, "pm10": 260.0, "no2": 65.0},
            "aqi": 315,
            "category": "Very Poor",
        }]
        inserted = self.db.log_station_readings(stations)
        self.assertGreaterEqual(inserted, 1)

        recent = self.db.get_recent_stations(limit=10)
        self.assertGreaterEqual(len(recent), 1)
        station_ids = [r["station_id"] for r in recent]
        self.assertIn("TEST_001", station_ids)
        test_rec = next(r for r in recent if r["station_id"] == "TEST_001")
        self.assertEqual(test_rec["aqi"], 315)

    def test_simulation_logging(self):
        params = {"traffic_delta": -0.3, "industrial_curb": True, "wind_speed": 4.5, "boundary_layer": 1800}
        result = {"avg_baseline_aqi": 340.0, "avg_simulated_aqi": 272.0, "avg_reduction_pct": 20.0, "active_grap_stage": "Stage III"}
        sim_id = self.db.log_simulation(params, result)
        self.assertIsInstance(sim_id, int)

        sims = self.db.get_recent_simulations(limit=5)
        self.assertGreaterEqual(len(sims), 1)
        self.assertEqual(sims[0]["active_grap_stage"], "Stage III")
        self.assertEqual(sims[0]["avg_reduction_pct"], 20.0)

    def test_route_logging(self):
        route = {
            "origin": {"lat": 28.6476, "lon": 77.3158},
            "destination": {"lat": 28.6139, "lon": 77.2090},
            "distance_km": 12.4,
            "baseline_exposure_dose": 350.0,
            "clean_exposure_dose": 210.0,
            "dose_reduction_pct": 40.0,
        }
        route_id = self.db.log_route(route)
        self.assertIsInstance(route_id, int)

        routes = self.db.get_recent_routes(limit=5)
        self.assertGreaterEqual(len(routes), 1)
        self.assertEqual(routes[0]["distance_km"], 12.4)
        self.assertEqual(routes[0]["dose_reduction_pct"], 40.0)


if __name__ == "__main__":
    unittest.main()
