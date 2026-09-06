"""
Configuration Module for Hyperlocal Air Quality Prediction System
Defines geographic bounding boxes, pollutant standards (CPCB / EPA AQI breakpoints),
feature schemas, neural network hyperparameters, and system paths.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_store"
MODELS_DIR = BASE_DIR / "saved_models"
CACHE_DIR = BASE_DIR / "cache"

for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default Study Region: Delhi-NCR Metropolitan Area
DEFAULT_REGION = {
    "name": "Delhi-NCR",
    "min_lat": 28.40,
    "max_lat": 28.88,
    "min_lon": 76.88,
    "max_lon": 77.48,
    "center_lat": 28.6139,
    "center_lon": 77.2090,
}

# Grid Downscaling Parameters
# 24 x 24 spatial grid cells over the bounding box (~2km downscaled cells for fast real-time inference)
GRID_CONFIG = {
    "lat_steps": 24,
    "lon_steps": 24,
    "spatial_resolution_km": 2.0,
    "patch_size": 5, # 5x5 spatial patch around target coordinates for CNN branch
}

# Monitored Pollutants and Indian CPCB AQI Breakpoints (standard calculation)
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# CPCB Sub-index Breakpoints [C_low, C_high, I_low, I_high]
AQI_BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),
        (30.1, 60, 51, 100),
        (60.1, 90, 101, 200),
        (90.1, 120, 201, 300),
        (120.1, 250, 301, 400),
        (250.1, 500, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (50.1, 100, 51, 100),
        (100.1, 250, 101, 200),
        (250.1, 350, 201, 300),
        (350.1, 430, 301, 400),
        (430.1, 600, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (40.1, 80, 51, 100),
        (80.1, 180, 101, 200),
        (180.1, 280, 201, 300),
        (280.1, 400, 301, 400),
        (400.1, 600, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (40.1, 80, 51, 100),
        (80.1, 380, 101, 200),
        (380.1, 800, 201, 300),
        (800.1, 1600, 301, 400),
        (1600.1, 2000, 401, 500),
    ],
    "co": [
        (0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50),
        (50.1, 100, 51, 100),
        (100.1, 168, 101, 200),
        (168.1, 208, 201, 300),
        (208.1, 748, 301, 400),
        (748.1, 1000, 401, 500),
    ],
}

# AQI Categories and Colors
AQI_CATEGORIES = [
    {"range": (0, 50), "category": "Good", "color": "#00E400", "advisory": "Air quality is satisfactory; air pollution poses little or no risk."},
    {"range": (51, 100), "category": "Satisfactory", "color": "#92D050", "advisory": "Minor breathing discomfort to sensitive people."},
    {"range": (101, 200), "category": "Moderate", "color": "#FFFF00", "advisory": "Breathing discomfort to people with lung disease, asthma, and heart diseases."},
    {"range": (201, 300), "category": "Poor", "color": "#FF7E00", "advisory": "Breathing discomfort to most people on prolonged exposure."},
    {"range": (301, 400), "category": "Very Poor", "color": "#FF0000", "advisory": "Respiratory illness on prolonged exposure. Severe effect on people with heart/lung disease."},
    {"range": (401, 500), "category": "Severe", "color": "#7E0023", "advisory": "Healthy people also affected; seriously impacts those with existing diseases. Emergency action recommended."},
]

# Neural Network Architecture Hyperparameters
MODEL_CONFIG = {
    # Temporal LSTM Branch
    "seq_length": 24, # 24 hours of sequence history
    "temporal_features": 8, # [temp, humidity, wind_speed, wind_dir, pressure, rain, blh, pm25_lag]
    "lstm_hidden_dim": 64,
    "lstm_num_layers": 2,
    
    # Spatial CNN Branch
    "spatial_channels": 4, # [satellite_aod, land_use, road_density, elevation]
    "patch_height": 5,
    "patch_width": 5,
    "cnn_channels": [16, 32],
    
    # Tabular Geo-Weather Branch
    "tabular_features": 12, # [lat, lon, dist_to_road, dist_to_station, elevation, temp, hum, wind_u, wind_v, blh, pop_density, industrial_proxy]
    "tabular_hidden_dim": 32,
    
    # Fusion & Training
    "fusion_hidden_dim": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 20,
}

# Known Ground-Monitoring Stations in Delhi-NCR (Real-world coordinates from CPCB/DPCC)
GROUND_STATIONS = [
    {"id": "DL_001", "name": "Anand Vihar, Delhi", "lat": 28.6476, "lon": 77.3158, "type": "Industrial/Traffic"},
    {"id": "DL_002", "name": "Punjabi Bagh, Delhi", "lat": 28.6683, "lon": 77.1167, "type": "Residential/Commercial"},
    {"id": "DL_003", "name": "ITO, Delhi", "lat": 28.6289, "lon": 77.2410, "type": "High Traffic Corridor"},
    {"id": "DL_004", "name": "R K Puram, Delhi", "lat": 28.5632, "lon": 77.1869, "type": "Urban Residential"},
    {"id": "DL_005", "name": "Dwarka Sector 8, Delhi", "lat": 28.5710, "lon": 77.0719, "type": "Suburban Residential"},
    {"id": "DL_006", "name": "Jahangirpuri, Delhi", "lat": 28.7328, "lon": 77.1706, "type": "Industrial/Urban"},
    {"id": "DL_007", "name": "Siri Fort, Delhi", "lat": 28.5504, "lon": 77.2159, "type": "Institutional/Green"},
    {"id": "DL_008", "name": "Bawana, Delhi", "lat": 28.7762, "lon": 77.0511, "type": "Heavy Industrial Area"},
    {"id": "UP_001", "name": "Sector 62, Noida", "lat": 28.6245, "lon": 77.3649, "type": "IT/Commercial Corridor"},
    {"id": "UP_002", "name": "Vasundhara, Ghaziabad", "lat": 28.6603, "lon": 77.3573, "type": "Residential/Highway"},
    {"id": "HR_001", "name": "Vikas Sadan, Gurugram", "lat": 28.4501, "lon": 77.0264, "type": "Commercial Hub"},
    {"id": "HR_002", "name": "Sector 16A, Faridabad", "lat": 28.4088, "lon": 77.3178, "type": "Industrial/Urban"},
]
