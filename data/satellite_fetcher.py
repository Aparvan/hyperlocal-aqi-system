"""
Satellite Atmospheric Observation Processor.
Handles Sentinel-5P TROPOMI / MODIS Aerosol Optical Depth (AOD)
and tropospheric NO2 column density rasters.
Generates multi-channel spatial patches for CNN-based spatial feature learning.
"""

import numpy as np
import datetime
from config import DEFAULT_REGION, GRID_CONFIG
from data.generator import AtmosphericSimulator


class SatelliteDataProcessor:
    def __init__(self, region: dict = None):
        self.region = region or DEFAULT_REGION
        self.simulator = AtmosphericSimulator(self.region)
        self.grid_rows = GRID_CONFIG["lat_steps"]
        self.grid_cols = GRID_CONFIG["lon_steps"]
        self.patch_size = GRID_CONFIG["patch_size"] # default 5x5 patch

    def get_satellite_surface(self, timestamp: datetime.datetime = None) -> np.ndarray:
        """
        Generate or fetch 4-channel spatial raster over the bounding box:
        Channel 0: Satellite AOD (Aerosol Optical Depth)
        Channel 1: Land-use classification (0=Water/Forest, 0.5=Residential, 1.0=Industrial/Commercial)
        Channel 2: Road & Traffic Density Index (0.0 to 1.0)
        Channel 3: Topographic Elevation (normalized 0.0 to 1.0, Delhi ~200-300m)
        Returns: (4, grid_rows, grid_cols) numpy array
        """
        timestamp = timestamp or datetime.datetime.now()
        lats = np.linspace(self.region["min_lat"], self.region["max_lat"], self.grid_rows)
        lons = np.linspace(self.region["min_lon"], self.region["max_lon"], self.grid_cols)
        
        raster = np.zeros((4, self.grid_rows, self.grid_cols), dtype=np.float32)
        weather = self.simulator.get_weather(timestamp)
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                point = self.simulator.evaluate_spatial_pollution(float(lat), float(lon), weather, timestamp)
                
                # Channel 0: Satellite AOD
                raster[0, i, j] = point["satellite_aod"]
                
                # Channel 1: Land-use classification proxy
                # High near Anand Vihar, Bawana, Gurugram, Low near Ridge/Yamuna
                dist_ind = min([np.hypot(lat - s["lat"], lon - s["lon"]) for s in self.simulator.emission_sources])
                land_use = 1.0 - np.clip(dist_ind / 0.15, 0.0, 0.8)
                raster[1, i, j] = float(land_use)
                
                # Channel 2: Road network / traffic corridor density
                # Elevated along major latitudes and longitudes passing through Delhi center
                road_dist = min(abs(lat - 28.63), abs(lon - 77.22), abs(lat - 28.48), abs(lon - 77.08))
                road_density = float(np.exp(- (road_dist ** 2) / (2 * 0.03 ** 2)))
                raster[2, i, j] = road_density
                
                # Channel 3: Normalized Elevation (Delhi Ridge has slight elevation ~250m vs Yamuna floodplains ~205m)
                elev_m = 210.0 + 40.0 * np.exp(-((lat - 28.60)**2 + (lon - 77.17)**2) / 0.01)
                raster[3, i, j] = float((elev_m - 200.0) / 100.0)
                
        return raster

    def extract_spatial_patch(self, lat: float, lon: float, full_raster: np.ndarray = None) -> np.ndarray:
        """
        Extract a normalized (4, patch_size, patch_size) spatial window around (lat, lon)
        to feed into the 2D-CNN feature extraction branch.
        """
        if full_raster is None:
            full_raster = self.get_satellite_surface()
            
        # Map lat, lon to grid row and column index
        lat_ratio = (lat - self.region["min_lat"]) / (self.region["max_lat"] - self.region["min_lat"])
        lon_ratio = (lon - self.region["min_lon"]) / (self.region["max_lon"] - self.region["min_lon"])
        
        r_center = int(np.clip(round(lat_ratio * (self.grid_rows - 1)), 0, self.grid_rows - 1))
        c_center = int(np.clip(round(lon_ratio * (self.grid_cols - 1)), 0, self.grid_cols - 1))
        
        half = self.patch_size // 2
        
        # Pad raster borders with reflection to handle edges
        padded = np.pad(full_raster, ((0, 0), (half, half), (half, half)), mode='reflect')
        
        r_start = r_center
        c_start = c_center
        patch = padded[:, r_start : r_start + self.patch_size, c_start : c_start + self.patch_size]
        return patch.astype(np.float32)
