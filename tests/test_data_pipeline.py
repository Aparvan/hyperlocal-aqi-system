"""Unit tests for Data Ingestion, Simulation, and Preprocessing modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import datetime
import torch
import numpy as np

from data.generator import AtmosphericSimulator, calculate_sub_aqi, calculate_overall_aqi
from data.weather_fetcher import WeatherFetcher
from data.station_fetcher import StationFetcher
from data.satellite_fetcher import SatelliteDataProcessor
from data.preprocessor import HyperlocalPreprocessor
from config import DEFAULT_REGION


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.sim = AtmosphericSimulator(DEFAULT_REGION, seed=42)
        self.prep = HyperlocalPreprocessor(DEFAULT_REGION)

    def test_sub_aqi_calculation(self):
        # PM2.5 30 ug/m3 should map to Good (<=50)
        self.assertAlmostEqual(calculate_sub_aqi("pm25", 30.0), 50.0, places=1)
        # PM2.5 60 ug/m3 should map to Satisfactory (100)
        self.assertAlmostEqual(calculate_sub_aqi("pm25", 60.0), 100.0, places=1)
        # PM2.5 250 ug/m3 should map to Very Poor / Severe threshold (400)
        self.assertAlmostEqual(calculate_sub_aqi("pm25", 250.0), 400.0, places=1)

    def test_overall_aqi(self):
        pols = {"pm25": 110.0, "pm10": 90.0, "no2": 45.0}
        aqi, prominent = calculate_overall_aqi(pols)
        self.assertGreater(aqi, 200)
        self.assertEqual(prominent, "pm25")

    def test_weather_generation(self):
        now = datetime.datetime.now()
        weather = self.sim.get_weather(now)
        self.assertIn("temperature_c", weather)
        self.assertIn("boundary_layer_height_m", weather)
        self.assertGreater(weather["boundary_layer_height_m"], 100.0)

    def test_spatial_pollution_physics(self):
        now = datetime.datetime.now()
        weather = self.sim.get_weather(now)
        pt = self.sim.evaluate_spatial_pollution(28.64, 77.21, weather, now)
        self.assertIn("pollutants", pt)
        self.assertGreater(pt["pollutants"]["pm25"], 0)
        self.assertGreater(pt["satellite_aod"], 0)

    def test_preprocessor_tensors(self):
        sample = self.prep.prepare_multimodal_sample(28.6139, 77.2090)
        self.assertIn("spatial", sample)
        self.assertIn("temporal", sample)
        self.assertIn("tabular", sample)
        
        # Spatial: (1, 4, 5, 5)
        self.assertEqual(sample["spatial"].shape, (1, 4, 5, 5))
        # Temporal: (1, 24, 8)
        self.assertEqual(sample["temporal"].shape, (1, 24, 8))
        # Tabular: (1, 12)
        self.assertEqual(sample["tabular"].shape, (1, 12))


if __name__ == "__main__":
    unittest.main()
