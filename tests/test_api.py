"""Integration tests for REST API endpoints."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from api.app import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")

    def test_stations_endpoint(self):
        res = self.client.get("/api/stations")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertGreater(data["count"], 0)

    def test_grid_endpoint(self):
        res = self.client.get("/api/grid?rows=4&cols=4&pollutant=pm25")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["grid_dimensions"]["total_cells"], 16)

    def test_predict_point_endpoint(self):
        payload = {"lat": 28.65, "lon": 77.25}
        res = self.client.post("/api/predict/point", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("prediction", data)

    def test_hotspots_endpoint(self):
        res = self.client.get("/api/hotspots")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("hotspots", data)

    def test_explain_endpoint(self):
        payload = {"lat": 28.65, "lon": 77.25, "target": 0}
        res = self.client.post("/api/explain", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("explanation", data)

    def test_advisor_endpoint(self):
        payload = {"query": "What are the main causes of pollution?", "lat": 28.65, "lon": 77.25}
        res = self.client.post("/api/advisor/query", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("advisor", data)

    def test_benchmarks_endpoint(self):
        res = self.client.get("/api/benchmarks")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("benchmarks", data)


if __name__ == "__main__":
    unittest.main()
