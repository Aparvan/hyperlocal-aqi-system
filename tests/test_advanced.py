"""Comprehensive unit and integration tests for Advanced Capabilities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from unittest.mock import patch
import torch
import numpy as np

from models.st_gat import SpatioTemporalGAT, GraphAttentionLayer
from models.pinn_loss import PhysicsInformedLoss
from services.grid_engine import HyperlocalGridEngine
from services.scenario_simulator import ScenarioSimulator
from services.route_optimizer import RouteOptimizer
from services.multi_agent_war_room import MultiAgentWarRoom
from models.xai_engine import XAIEngine
from api.app import app

# Deterministic baseline weather fixture so unit tests never depend on live Open-Meteo wind/BLH
FIXED_WEATHER = {
    "timestamp": "2026-01-15T10:00:00",
    "source": "Deterministic Test Fixture",
    "latitude": 28.6139,
    "longitude": 77.2090,
    "temperature_c": 22.0,
    "relative_humidity_pct": 58.0,
    "wind_speed_ms": 3.0,
    "wind_direction_deg": 310.0,
    "wind_u": 2.30,
    "wind_v": -1.93,
    "pressure_hpa": 1013.0,
    "boundary_layer_height_m": 600.0,
    "rainfall_mm": 0.0,
}


class TestAdvancedModules(unittest.TestCase):
    def setUp(self):
        self.weather_patcher = patch(
            "data.weather_fetcher.WeatherFetcher.fetch_current_weather",
            return_value=dict(FIXED_WEATHER),
        )
        self.weather_patcher.start()
        self.addCleanup(self.weather_patcher.stop)

        self.grid_engine = HyperlocalGridEngine()
        self.xai_engine = XAIEngine(self.grid_engine.model)
        self.scenario_sim = ScenarioSimulator(self.grid_engine)
        self.route_opt = RouteOptimizer(self.grid_engine)
        self.war_room = MultiAgentWarRoom(self.grid_engine, self.xai_engine, self.scenario_sim)
        self.client = app.test_client()
        self.client.testing = True

    def test_st_gat_model(self):
        gat = SpatioTemporalGAT(node_features=8, hidden_dim=16, out_dim=16)
        nodes = np.array([[28.60, 77.20], [28.65, 77.25], [28.70, 77.30]], dtype=np.float32)
        adj = gat.build_dynamic_adjacency(nodes, wind_u=2.0, wind_v=-1.5)
        
        self.assertEqual(adj.shape, (1, 3, 3))
        # Self-loops should be 1.0
        self.assertAlmostEqual(adj[0, 0, 0].item(), 1.0)
        
        node_feats = torch.randn(1, 3, 8)
        out = gat(node_feats, adj)
        self.assertEqual(out.shape, (1, 3, 16))

    def test_pinn_loss(self):
        pinn = PhysicsInformedLoss(dx_meters=2000.0, dy_meters=2000.0)
        grid_c = torch.full((1, 1, 10, 10), 100.0, requires_grad=True)
        res = pinn(grid_c, wind_u=3.0, wind_v=-2.0)
        
        self.assertIn("pinn_loss", res)
        self.assertGreaterEqual(res["pinn_loss"].item(), 0.0)
        # Verify gradient flow
        res["pinn_loss"].backward()
        self.assertIsNotNone(grid_c.grad)

    def test_scenario_simulator(self):
        # 1. Pure emission cut (traffic -40% + industrial curb, keeping weather at baseline) must reduce PM2.5
        sim_res = self.scenario_sim.simulate_scenario(
            traffic_delta=-0.40,
            industrial_curb=True,
        )
        self.assertIn("impact_summary", sim_res)
        self.assertIn("percentage_change_pm25", sim_res["impact_summary"])
        self.assertLess(sim_res["impact_summary"]["percentage_change_pm25"], 0.0)

    def test_scenario_wind_direction(self):
        # Baseline fixture has wind_speed_ms = 3.0 m/s.
        # Higher wind (6.0 m/s) -> stronger ventilation -> lower PM2.5 (< 0%)
        high_wind = self.scenario_sim.simulate_scenario(wind_speed_override=6.0)
        self.assertLess(high_wind["impact_summary"]["percentage_change_pm25"], 0.0)

        # Lower wind (1.0 m/s) -> stagnation buildup -> higher PM2.5 (> 0%)
        low_wind = self.scenario_sim.simulate_scenario(wind_speed_override=1.0)
        self.assertGreater(low_wind["impact_summary"]["percentage_change_pm25"], 0.0)

    def test_scenario_blh_response(self):
        # Baseline fixture has boundary_layer_height_m = 600.0 m.
        # Deeper mixing layer (1500m) -> dilution -> lower PM2.5 (< 0%)
        deep_blh = self.scenario_sim.simulate_scenario(blh_override=1500.0)
        self.assertLess(deep_blh["impact_summary"]["percentage_change_pm25"], 0.0)

        # Shallow thermal inversion (200m) -> trapping -> higher PM2.5 (> 0%)
        shallow_blh = self.scenario_sim.simulate_scenario(blh_override=200.0)
        self.assertGreater(shallow_blh["impact_summary"]["percentage_change_pm25"], 0.0)

    def test_route_optimizer(self):
        # From Anand Vihar (East) to Central Delhi
        res = self.route_opt.find_cleanest_route(
            origin_lat=28.6476, origin_lon=77.3158,
            dest_lat=28.6139, dest_lon=77.2090,
            travel_mode="walking"
        )
        self.assertIn("fastest_route", res)
        self.assertIn("cleanest_route", res)
        self.assertIn("comparison", res)
        # Cleanest route should reduce inhaled dose
        self.assertGreaterEqual(res["comparison"]["exposure_reduction_pct"], 0.0)

    def test_multi_agent_war_room(self):
        session = self.war_room.run_collaboration(
            query="Critical air quality advisory needed for North Delhi industrial zone.",
            lat=28.73, lon=77.17
        )
        self.assertIn("debate_transcript", session)
        self.assertEqual(len(session["debate_transcript"]), 3)
        self.assertIn("consensus_summary", session)
        self.assertIn("grap_stage", session)

    def test_advanced_api_endpoints(self):
        # 1. Scenario Simulation Endpoint (including _override parameter aliases and unknown-key rejection)
        p1 = self.client.post("/api/simulate/scenario", json={
            "traffic_delta": -0.25,
            "wind_speed_override": 5.0,
            "industrial_curb": True,
        })
        self.assertEqual(p1.status_code, 200)
        self.assertIn("simulation", p1.get_json())
        self.assertEqual(p1.get_json()["simulation"]["scenario_parameters"]["wind_speed_ms"], 5.0)

        p1_bad = self.client.post("/api/simulate/scenario", json={"bogus_param": 123})
        self.assertEqual(p1_bad.status_code, 400)

        # 2. Route Optimizer Endpoint
        p2 = self.client.post("/api/route/cleanest", json={
            "origin": [28.65, 77.30],
            "destination": [28.61, 77.21],
            "travel_mode": "cycling"
        })
        self.assertEqual(p2.status_code, 200)
        self.assertIn("route_data", p2.get_json())

        # 3. Multi-Agent Collaboration Endpoint
        p3 = self.client.post("/api/agents/collaborate", json={"query": "Evaluate emergency actions for Anand Vihar."})
        self.assertEqual(p3.status_code, 200)
        self.assertIn("war_room", p3.get_json())

        # 4. Timelapse Endpoint
        p4 = self.client.get("/api/timelapse")
        self.assertEqual(p4.status_code, 200)
        data = p4.get_json()
        self.assertGreater(data["frame_count"], 0)


if __name__ == "__main__":
    unittest.main()
