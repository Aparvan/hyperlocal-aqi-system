"""Unit tests for Grid Downscaling, Hotspot Detection, and Advisor Agent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from services.grid_engine import HyperlocalGridEngine
from services.hotspot_detector import HotspotDetector
from services.advisor_agent import AgenticEnvironmentalAdvisor


class TestServices(unittest.TestCase):
    def setUp(self):
        self.grid_engine = HyperlocalGridEngine()
        self.hotspot_detector = HotspotDetector(aqi_threshold=200.0, pm25_threshold=90.0)
        self.advisor = AgenticEnvironmentalAdvisor(self.grid_engine, self.hotspot_detector)

    def test_grid_generation(self):
        grid = self.grid_engine.generate_grid_predictions(rows=6, cols=6)
        self.assertEqual(grid["grid_dimensions"]["total_cells"], 36)
        self.assertIn("cells", grid)
        self.assertIn("statistics", grid)

    def test_point_prediction(self):
        point = self.grid_engine.predict_point(28.6139, 77.2090)
        self.assertIn("aqi", point)
        self.assertIn("category", point)
        self.assertIn("pollutants", point)
        self.assertGreater(point["aqi"], 0)

    def test_hotspot_detection(self):
        grid = self.grid_engine.generate_grid_predictions(rows=8, cols=8)
        hotspots = self.hotspot_detector.detect_hotspots(grid)
        self.assertIsInstance(hotspots, list)

    def test_advisor_query(self):
        res = self.advisor.answer_query("Is it safe to run in Delhi right now?", lat=28.62, lon=77.21)
        self.assertIn("response_markdown", res)
        self.assertIn("top_driver", res)
        self.assertIn("aqi", res)


if __name__ == "__main__":
    unittest.main()
