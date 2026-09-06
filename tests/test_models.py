"""Unit tests for Baseline and Multimodal Deep Learning Models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import torch
import numpy as np

from models.baselines import IDWInterpolator, NearestStationPredictor, TabularMLPBaseline, compute_metrics
from models.spatial_cnn import SpatialCNNBranch
from models.temporal_lstm import TemporalLSTMBranch
from models.multimodal_fusion import MultimodalFusionNetwork
from models.xai_engine import XAIEngine
from data.preprocessor import HyperlocalPreprocessor


class TestModels(unittest.TestCase):
    def test_metrics_computation(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 31.0, 39.0])
        m = compute_metrics(y_true, y_pred)
        self.assertLess(m["rmse"], 2.0)
        self.assertGreater(m["r2"], 0.95)

    def test_idw_interpolator(self):
        idw = IDWInterpolator()
        stations = [
            {"lat": 28.60, "lon": 77.20, "pollutants": {"pm25": 100.0}},
            {"lat": 28.70, "lon": 77.30, "pollutants": {"pm25": 200.0}},
        ]
        idw.fit(stations)
        # Point exact at station 1 should yield close to 100
        pred = idw.predict_point(28.60, 77.20, "pm25")
        self.assertAlmostEqual(pred, 100.0, delta=1.0)

    def test_spatial_cnn_forward(self):
        cnn = SpatialCNNBranch(in_channels=4, out_dim=32)
        x = torch.randn(2, 4, 5, 5)
        out = cnn(x)
        self.assertEqual(out.shape, (2, 32))

    def test_temporal_lstm_forward(self):
        lstm = TemporalLSTMBranch(in_features=8, hidden_dim=64, num_layers=2, out_dim=32)
        x = torch.randn(2, 24, 8)
        out = lstm(x)
        self.assertEqual(out.shape, (2, 32))

    def test_multimodal_fusion_forward(self):
        model = MultimodalFusionNetwork(
            spatial_channels=4,
            temporal_features=8,
            tabular_features=12,
            embedding_dim=32,
            num_heads=4,
        )
        spat = torch.randn(2, 4, 5, 5)
        temp = torch.randn(2, 24, 8)
        tab = torch.randn(2, 12)
        
        res = model(spat, temp, tab, return_attention=True)
        self.assertIn("preds", res)
        self.assertEqual(res["preds"].shape, (2, 4))
        self.assertIn("cross_modal_attention", res)

    def test_xai_attribution(self):
        model = MultimodalFusionNetwork(
            spatial_channels=4,
            temporal_features=8,
            tabular_features=12,
            embedding_dim=32,
            num_heads=4,
        )
        prep = HyperlocalPreprocessor()
        sample = prep.prepare_multimodal_sample(28.6139, 77.2090)
        xai = XAIEngine(model)
        explanation = xai.explain_prediction(sample, target_index=0)
        
        self.assertIn("top_driver", explanation)
        self.assertIn("factor_contributions", explanation)
        self.assertIn("modality_influence", explanation)


if __name__ == "__main__":
    unittest.main()
