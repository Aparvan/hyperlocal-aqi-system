"""
Hyperlocal Grid Downscaling Engine.
Generates fine-grained continuous spatial pollution fields across the region.
Computes bounding polygons for each cell (GeoJSON format) for interactive Leaflet map rendering.
Performs batch multi-modal neural network inference for fast real-time response.
"""

import numpy as np
import torch
import datetime
from pathlib import Path

from config import DEFAULT_REGION, GRID_CONFIG, MODELS_DIR, MODEL_CONFIG
from data.generator import AtmosphericSimulator, calculate_overall_aqi, calculate_sub_aqi
from data.preprocessor import HyperlocalPreprocessor
from data.station_fetcher import StationFetcher
from data.weather_fetcher import WeatherFetcher
from data.satellite_fetcher import SatelliteDataProcessor
from models.multimodal_fusion import MultimodalFusionNetwork


class HyperlocalGridEngine:
    def __init__(self, region: dict = None):
        self.region = region or DEFAULT_REGION
        self.prep = HyperlocalPreprocessor(self.region)
        self.weather_fetcher = WeatherFetcher(self.region)
        self.station_fetcher = StationFetcher(self.region)
        self.satellite_proc = SatelliteDataProcessor(self.region)
        self.simulator = AtmosphericSimulator(self.region)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load trained PyTorch Multimodal weights if available."""
        model_path = MODELS_DIR / "multimodal_model.pt"
        self.model = MultimodalFusionNetwork(
            spatial_channels=MODEL_CONFIG["spatial_channels"],
            temporal_features=MODEL_CONFIG["temporal_features"],
            tabular_features=MODEL_CONFIG["tabular_features"],
            embedding_dim=32,
            num_heads=4,
            dropout=0.0,
        ).to(self.device)
        
        if model_path.exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Successfully loaded trained Multimodal weights.")
            except Exception as e:
                print(f"Warning: Could not load model checkpoint ({e}). Initialized in inference mode.")
        self.model.eval()

    def generate_grid_predictions(self, rows: int = None, cols: int = None, timestamp: datetime.datetime = None) -> dict:
        """
        Produce a dense grid prediction over the entire study region.
        Returns:
            Dictionary with grid cells, bounding boxes for Leaflet polygons,
            summary statistics, and current meteorological state.
        """
        rows = rows or GRID_CONFIG["lat_steps"]
        cols = cols or GRID_CONFIG["lon_steps"]
        timestamp = timestamp or datetime.datetime.now()
        
        lats = np.linspace(self.region["min_lat"], self.region["max_lat"], rows)
        lons = np.linspace(self.region["min_lon"], self.region["max_lon"], cols)
        
        d_lat = (lats[1] - lats[0]) / 2.0
        d_lon = (lons[1] - lons[0]) / 2.0
        
        weather = self.weather_fetcher.fetch_current_weather()
        stations = self.station_fetcher.fetch_all_stations()
        raster = self.satellite_proc.get_satellite_surface(timestamp)
        weather_series = self.weather_fetcher.fetch_weather_series(hours=MODEL_CONFIG["seq_length"])
        
        cells = []
        spatial_batch = []
        temporal_batch = []
        tabular_batch = []
        coords = []
        
        # Batch construct inputs
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                coords.append((i, j, float(lat), float(lon)))
                
                spat_patch = self.satellite_proc.extract_spatial_patch(lat, lon, raster)
                nearest_st, dist_km = self.station_fetcher.find_nearest_station(lat, lon)
                tab_vec = self.prep.build_tabular_features(lat, lon, weather, nearest_st, dist_km)
                nearest_pm = nearest_st["pollutants"]["pm25"] if nearest_st else 90.0
                temp_seq = self.prep.build_temporal_features(weather_series, baseline_pm25=nearest_pm)
                
                spatial_batch.append(spat_patch)
                temporal_batch.append(temp_seq)
                tabular_batch.append(tab_vec)
                
        # Run batch inference through Multimodal Deep Learning network
        spat_tensor = torch.tensor(np.array(spatial_batch), dtype=torch.float32).to(self.device)
        temp_tensor = torch.tensor(np.array(temporal_batch), dtype=torch.float32).to(self.device)
        tab_tensor = torch.tensor(np.array(tabular_batch), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            out = self.model(spat_tensor, temp_tensor, tab_tensor)
            preds = out["preds"].cpu().numpy()
            
        all_pm25 = []
        all_aqi = []
        
        for idx, (i, j, lat, lon) in enumerate(coords):
            pm25 = round(max(5.0, float(preds[idx, 0])), 1)
            pm10 = round(max(10.0, float(preds[idx, 1])), 1)
            no2 = round(max(5.0, float(preds[idx, 2])), 1)
            
            # Recalculate true CPCB composite AQI
            pol_dict = {"pm25": pm25, "pm10": pm10, "no2": no2}
            aqi, prominent = calculate_overall_aqi(pol_dict)
            
            all_pm25.append(pm25)
            all_aqi.append(aqi)
            
            # Cell polygon bounding coordinates for Leaflet GeoJSON/Polygon:
            # [[lat_min, lon_min], [lat_max, lon_max]]
            bounds = [
                [round(lat - d_lat, 4), round(lon - d_lon, 4)],
                [round(lat + d_lat, 4), round(lon + d_lon, 4)],
            ]
            
            cells.append({
                "id": f"cell_{i}_{j}",
                "grid_i": i,
                "grid_j": j,
                "center_lat": round(lat, 4),
                "center_lon": round(lon, 4),
                "bounds": bounds,
                "pm25": pm25,
                "pm10": pm10,
                "no2": no2,
                "aqi": aqi,
                "prominent": prominent,
                "satellite_aod": round(float(raster[0, i, j]), 3),
            })
            
        return {
            "timestamp": timestamp.isoformat(),
            "region": self.region["name"],
            "bounds": [
                [self.region["min_lat"], self.region["min_lon"]],
                [self.region["max_lat"], self.region["max_lon"]],
            ],
            "grid_dimensions": {"rows": rows, "cols": cols, "total_cells": len(cells)},
            "statistics": {
                "pm25_min": round(float(np.min(all_pm25)), 1),
                "pm25_max": round(float(np.max(all_pm25)), 1),
                "pm25_mean": round(float(np.mean(all_pm25)), 1),
                "aqi_min": round(float(np.min(all_aqi)), 1),
                "aqi_max": round(float(np.max(all_aqi)), 1),
                "aqi_mean": round(float(np.mean(all_aqi)), 1),
            },
            "weather": weather,
            "cells": cells,
        }

    def predict_point(self, lat: float, lon: float) -> dict:
        """
        On-demand inference for an exact clicked coordinate anywhere in the region.
        Returns predicted pollutant values, AQI category, and nearest station info.
        """
        sample = self.prep.prepare_multimodal_sample(lat, lon)
        spat = sample["spatial"].to(self.device)
        temp = sample["temporal"].to(self.device)
        tab = sample["tabular"].to(self.device)
        
        with torch.no_grad():
            out = self.model(spat, temp, tab)
            preds = out["preds"].cpu().numpy()[0]
            
        pm25 = round(max(5.0, float(preds[0])), 1)
        pm10 = round(max(10.0, float(preds[1])), 1)
        no2 = round(max(5.0, float(preds[2])), 1)
        
        pol_dict = {"pm25": pm25, "pm10": pm10, "no2": no2}
        aqi, prominent = calculate_overall_aqi(pol_dict)
        
        from config import AQI_CATEGORIES
        cat_info = AQI_CATEGORIES[-1]
        for cat in AQI_CATEGORIES:
            c_min, c_max = cat["range"]
            if c_min <= aqi <= c_max:
                cat_info = cat
                break
                
        return {
            "lat": lat,
            "lon": lon,
            "pollutants": {
                "pm25": pm25,
                "pm10": pm10,
                "no2": no2,
            },
            "aqi": aqi,
            "prominent_pollutant": prominent,
            "category": cat_info["category"],
            "color": cat_info["color"],
            "health_advisory": cat_info["advisory"],
            "metadata": sample["metadata"],
        }
