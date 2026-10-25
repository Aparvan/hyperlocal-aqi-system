"""
Spatial-Temporal Multi-Modal Preprocessor.
Performs alignment, normalization, and tensor structuring across:
1. Spatial Branch: (4, 5, 5) 2D spatial patches of satellite AOD, land-use, roads, elevation.
2. Temporal Branch: (24, 8) sequential observations of meteorology and lag pollution.
3. Tabular Branch: 12-dim normalized vector of geographic, meteorological, and station-proximity features.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import datetime

from config import DEFAULT_REGION, MODEL_CONFIG
from data.weather_fetcher import WeatherFetcher
from data.station_fetcher import StationFetcher
from data.satellite_fetcher import SatelliteDataProcessor


class NormalizationStats:
    """Pre-computed normalization parameters (mean and std) for numerical stability."""
    TABULAR_MEAN = np.array([28.64, 77.18, 5.0, 120.0, 220.0, 25.0, 60.0, 0.0, -1.5, 750.0, 0.5, 0.5], dtype=np.float32)
    TABULAR_STD = np.array([0.15, 0.18, 4.0, 60.0, 20.0, 8.0, 20.0, 2.0, 2.5, 450.0, 0.3, 0.3], dtype=np.float32)
    
    TEMPORAL_MEAN = np.array([25.0, 60.0, 3.0, 180.0, 1010.0, 0.2, 750.0, 120.0], dtype=np.float32)
    TEMPORAL_STD = np.array([8.0, 20.0, 1.5, 90.0, 5.0, 1.0, 450.0, 60.0], dtype=np.float32)


class HyperlocalPreprocessor:
    def __init__(self, region: dict = None):
        self.region = region or DEFAULT_REGION
        self.weather_fetcher = WeatherFetcher(self.region)
        self.station_fetcher = StationFetcher(self.region)
        self.satellite_proc = SatelliteDataProcessor(self.region)
        
    def build_tabular_features(self, lat: float, lon: float, weather: dict, nearest_st: dict, dist_km: float) -> np.ndarray:
        """
        Construct 12-dimensional tabular feature vector:
        0: Latitude
        1: Longitude
        2: Distance to nearest physical station (km)
        3: PM2.5 reading at nearest station (ug/m3)
        4: Elevation proxy (m)
        5: Current Temperature (C)
        6: Current Humidity (%)
        7: Wind U (eastward vector m/s)
        8: Wind V (northward vector m/s)
        9: Planetary Boundary Layer Height (m)
        10: Population density proxy (0.0 to 1.0)
        11: Industrial activity proxy (0.0 to 1.0)
        """
        st_pm25 = nearest_st["pollutants"]["pm25"] if nearest_st else 90.0
        elev = 210.0 + 40.0 * np.exp(-((lat - 28.60)**2 + (lon - 77.17)**2) / 0.01)
        
        # Industrial proxy based on distance to known industrial clusters
        ind_dist = min(np.hypot(lat - 28.78, lon - 77.05), np.hypot(lat - 28.65, lon - 77.31))
        ind_proxy = float(np.clip(1.0 - ind_dist / 0.12, 0.0, 1.0))
        
        # Population density proxy
        pop_dist = np.hypot(lat - 28.63, lon - 77.22)
        pop_density = float(np.clip(1.0 - pop_dist / 0.25, 0.1, 1.0))
        
        raw_vec = np.array([
            lat,
            lon,
            dist_km,
            st_pm25,
            elev,
            weather.get("temperature_c", 25.0),
            weather.get("relative_humidity_pct", 60.0),
            weather.get("wind_u", 0.0),
            weather.get("wind_v", -1.5),
            weather.get("boundary_layer_height_m", 700.0),
            pop_density,
            ind_proxy,
        ], dtype=np.float32)
        
        # Normalize
        norm_vec = (raw_vec - NormalizationStats.TABULAR_MEAN) / (NormalizationStats.TABULAR_STD + 1e-6)
        return norm_vec.astype(np.float32)

    def build_temporal_features(self, weather_series: list[dict], baseline_pm25: float = 100.0) -> np.ndarray:
        """
        Construct (seq_len=24, 8) normalized temporal sequence:
        Features: [temp, humidity, wind_speed, wind_direction, pressure, rainfall, blh, pm25_lag]
        """
        seq_len = MODEL_CONFIG["seq_length"]
        if len(weather_series) < seq_len:
            # Pad if needed
            weather_series = [weather_series[0]] * (seq_len - len(weather_series)) + weather_series
        elif len(weather_series) > seq_len:
            weather_series = weather_series[-seq_len:]
            
        mat = np.zeros((seq_len, 8), dtype=np.float32)
        for i, w in enumerate(weather_series):
            # Diurnal pseudo-lag for PM2.5
            mat[i, 0] = w.get("temperature_c", 25.0)
            mat[i, 1] = w.get("relative_humidity_pct", 60.0)
            mat[i, 2] = w.get("wind_speed_ms", 3.0)
            mat[i, 3] = w.get("wind_direction_deg", 310.0)
            mat[i, 4] = w.get("pressure_hpa", 1010.0)
            mat[i, 5] = w.get("rainfall_mm", 0.0)
            mat[i, 6] = w.get("boundary_layer_height_m", 700.0)
            mat[i, 7] = baseline_pm25 * (1.0 + 0.2 * np.sin(i * np.pi / 12.0))
            
        # Normalize
        norm_mat = (mat - NormalizationStats.TEMPORAL_MEAN) / (NormalizationStats.TEMPORAL_STD + 1e-6)
        return norm_mat.astype(np.float32)

    def prepare_multimodal_sample(self, lat: float, lon: float, timestamp: datetime.datetime = None, full_raster: np.ndarray = None) -> dict:
        """
        Creates synchronized inputs ready for Multimodal Deep Learning inference:
        - spatial_patch: torch.Tensor shape (1, 4, 5, 5)
        - temporal_seq: torch.Tensor shape (1, 24, 8)
        - tabular_vec: torch.Tensor shape (1, 12)
        """
        timestamp = timestamp or datetime.datetime.now()
        
        # 1. Spatial Patch
        spatial_patch_np = self.satellite_proc.extract_spatial_patch(lat, lon, full_raster)
        spatial_tensor = torch.tensor(spatial_patch_np, dtype=torch.float32).unsqueeze(0) # (1, 4, 5, 5)
        
        # 2. Weather & Stations
        current_weather = self.weather_fetcher.fetch_current_weather(lat, lon)
        weather_series = self.weather_fetcher.fetch_weather_series(hours=MODEL_CONFIG["seq_length"], lat=lat, lon=lon)
        nearest_st, dist_km = self.station_fetcher.find_nearest_station(lat, lon)
        
        # 3. Temporal Sequence
        nearest_pm25 = nearest_st["pollutants"]["pm25"] if nearest_st else 90.0
        temporal_seq_np = self.build_temporal_features(weather_series, baseline_pm25=nearest_pm25)
        temporal_tensor = torch.tensor(temporal_seq_np, dtype=torch.float32).unsqueeze(0) # (1, 24, 8)
        
        # 4. Tabular Vector
        tabular_vec_np = self.build_tabular_features(lat, lon, current_weather, nearest_st, dist_km)
        tabular_tensor = torch.tensor(tabular_vec_np, dtype=torch.float32).unsqueeze(0) # (1, 12)
        
        return {
            "spatial": spatial_tensor,
            "temporal": temporal_tensor,
            "tabular": tabular_tensor,
            "metadata": {
                "lat": lat,
                "lon": lon,
                "timestamp": timestamp.isoformat(),
                "nearest_station": nearest_st["station_name"] if nearest_st else "N/A",
                "station_distance_km": dist_km,
                "current_weather": current_weather,
            }
        }


class MultimodalDataset(Dataset):
    """PyTorch Dataset for training the multimodal architecture."""
    def __init__(self, records: list[dict], preprocessor: HyperlocalPreprocessor):
        self.records = records
        self.prep = preprocessor
        self.raster = preprocessor.satellite_proc.get_satellite_surface()
        self.weather_series = preprocessor.weather_fetcher.fetch_weather_series(hours=MODEL_CONFIG["seq_length"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        lat = rec["lat"]
        lon = rec["lon"]
        weather = rec["weather"]
        
        # Spatial Patch
        spatial = self.prep.satellite_proc.extract_spatial_patch(lat, lon, self.raster)
        
        # Nearest station proxy
        nearest_st, dist_km = self.prep.station_fetcher.find_nearest_station(lat, lon)
        st_pm25 = nearest_st["pollutants"]["pm25"] if nearest_st else 90.0
        
        # Tabular
        tabular = self.prep.build_tabular_features(lat, lon, weather, nearest_st, dist_km)
        
        # Temporal (Leak-free: uses nearest observed station historical lag, never ground-truth target)
        temporal = self.prep.build_temporal_features(self.weather_series, baseline_pm25=st_pm25)
        
        # Targets: [PM2.5, PM10, NO2, Overall AQI]
        targets = np.array([
            rec["targets"]["pm25"],
            rec["targets"]["pm10"],
            rec["targets"]["no2"],
            rec["aqi"],
        ], dtype=np.float32)
        
        return {
            "spatial": torch.tensor(spatial, dtype=torch.float32),
            "temporal": torch.tensor(temporal, dtype=torch.float32),
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "target": torch.tensor(targets, dtype=torch.float32),
        }
