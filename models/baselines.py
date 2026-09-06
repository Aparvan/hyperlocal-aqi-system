"""
Baseline Spatial and Tabular Regression Models.
1. Inverse Distance Weighting (IDW): Traditional GIS spatial interpolation.
2. Nearest Station Baseline: Naive nearest-neighbor baseline.
3. Tabular Feature Regressor: PyTorch-based feedforward tabular baseline (comparable to XGBoost/RF).
Provides comparative benchmarks to demonstrate the quantitative superiority
of the Multimodal Deep Learning downscaling approach.
"""

import numpy as np
import torch
import torch.nn as nn
import math


class IDWInterpolator:
    """
    Inverse Distance Weighting (IDW) Spatial Interpolator.
    Estimates pollution at unmonitored location (lat, lon) using:
    z(x) = sum(w_i * z_i) / sum(w_i), where w_i = 1 / d(x, x_i)^p
    """
    def __init__(self, power: float = 2.0, epsilon: float = 1e-4):
        self.power = power
        self.epsilon = epsilon
        self.stations = []

    def fit(self, stations_data: list[dict]):
        """Fit with a snapshot of station locations and pollutant values."""
        self.stations = stations_data

    def predict_point(self, lat: float, lon: float, pollutant: str = "pm25") -> float:
        """Predict pollutant concentration at an unmonitored coordinate."""
        if not self.stations:
            return 80.0
            
        weights = []
        values = []
        
        for st in self.stations:
            val = st["pollutants"].get(pollutant, 80.0)
            dlat = st["lat"] - lat
            dlon = st["lon"] - lon
            dist = math.hypot(dlat, dlon)
            
            if dist < self.epsilon:
                return float(val)
                
            w = 1.0 / (dist ** self.power)
            weights.append(w)
            values.append(val)
            
        total_w = sum(weights)
        if total_w == 0:
            return float(np.mean(values))
            
        pred = sum(w * v for w, v in zip(weights, values)) / total_w
        return round(float(pred), 1)


class NearestStationPredictor:
    """Naive spatial baseline: Assumes the air quality equals that of the nearest monitoring station."""
    def __init__(self):
        self.stations = []

    def fit(self, stations_data: list[dict]):
        self.stations = stations_data

    def predict_point(self, lat: float, lon: float, pollutant: str = "pm25") -> float:
        if not self.stations:
            return 80.0
        min_dist = float("inf")
        nearest_val = 80.0
        for st in self.stations:
            dist = math.hypot(st["lat"] - lat, st["lon"] - lon)
            if dist < min_dist:
                min_dist = dist
                nearest_val = st["pollutants"].get(pollutant, 80.0)
        return round(float(nearest_val), 1)


class TabularMLPBaseline(nn.Module):
    """
    Feedforward Tabular Baseline (Deep MLP) operating solely on point-level features
    without spatial CNN or temporal LSTM context.
    """
    def __init__(self, in_features: int = 12, out_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression performance metrics: RMSE, MAE, R², and Pearson correlation."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    
    # Coefficient of Determination (R²)
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    r2 = max(-1.0, min(1.0, float(r2)))
    
    # Pearson Correlation Coefficient (r)
    if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = 0.0
        
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
        "pearson_r": round(pearson_r, 3),
    }
