"""
High-Fidelity Physical Environment and Pollution Dataset Generator.
Simulates realistic urban atmospheric dynamics:
- Boundary Layer Height (BLH) diurnal inversion
- Diurnal traffic rush-hour emission cycles
- Wind advection-diffusion transport (Gaussian plume dispersion principles)
- Satellite column-integrated Aerosol Optical Depth (AOD)
- Local land-use and road density proxies
Used for benchmark evaluations, model pre-training, and offline demonstration.
"""

import math
import numpy as np
import datetime
from pathlib import Path
import json

from config import DEFAULT_REGION, GROUND_STATIONS, POLLUTANTS, DATA_DIR, AQI_BREAKPOINTS


def calculate_sub_aqi(pollutant: str, concentration: float) -> float:
    """Calculate standard CPCB sub-index for a given pollutant concentration."""
    if pollutant not in AQI_BREAKPOINTS or concentration is None:
        return 0.0
    
    concentration = max(0.0, float(concentration))
    breakpoints = AQI_BREAKPOINTS[pollutant]
    
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            sub_aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(sub_aqi, 1)
            
    # If exceeding max breakpoint, extrapolate linearly
    c_low, c_high, i_low, i_high = breakpoints[-1]
    sub_aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
    return min(500.0, round(sub_aqi, 1))


def calculate_overall_aqi(pollutant_values: dict) -> tuple[float, str]:
    """Overall AQI is determined by the maximum sub-index among monitored criteria pollutants."""
    max_aqi = 0.0
    prominent = "pm25"
    for pol, val in pollutant_values.items():
        sub = calculate_sub_aqi(pol, val)
        if sub > max_aqi:
            max_aqi = sub
            prominent = pol
    return max_aqi, prominent


class AtmosphericSimulator:
    def __init__(self, region: dict = None, seed: int = 42):
        self.region = region or DEFAULT_REGION
        self.rng = np.random.default_rng(seed)
        
        # Spatial emission source clusters (Latitude, Longitude, Intensity, Type)
        self.emission_sources = [
            {"name": "Anand Vihar Transport & Industrial", "lat": 28.650, "lon": 77.310, "intensity": 2.4, "radius": 0.06},
            {"name": "Bawana Industrial Area", "lat": 28.780, "lon": 77.050, "intensity": 2.8, "radius": 0.07},
            {"name": "ITO & Ring Road Arterial Corridor", "lat": 28.630, "lon": 77.240, "intensity": 2.1, "radius": 0.05},
            {"name": "Noida Industrial Phase II", "lat": 28.580, "lon": 77.370, "intensity": 1.9, "radius": 0.06},
            {"name": "Gurugram Cyber City Highway Hub", "lat": 28.490, "lon": 77.090, "intensity": 1.8, "radius": 0.05},
            {"name": "Faridabad Manufacturing Belt", "lat": 28.380, "lon": 77.310, "intensity": 2.0, "radius": 0.07},
        ]
        
        # Green Buffers (Ridge, Forests, Large Parks) which mitigate pollution
        self.green_zones = [
            {"name": "Central Ridge Reserve", "lat": 28.600, "lon": 77.170, "mitigation": 0.55, "radius": 0.05},
            {"name": "Southern Ridge & Asola", "lat": 28.490, "lon": 77.220, "mitigation": 0.60, "radius": 0.07},
            {"name": "Yamuna Riverfront Corridor", "lat": 28.640, "lon": 77.260, "mitigation": 0.70, "radius": 0.04},
        ]

    def get_weather(self, timestamp: datetime.datetime) -> dict:
        """
        Simulate realistic meteorological conditions for the given hour.
        Incorporates seasonal variation, solar cycle, boundary layer dynamics.
        """
        hour = timestamp.hour + timestamp.minute / 60.0
        day_of_year = timestamp.timetuple().tm_yday
        
        # Diurnal Solar Cycle (Max temp around 14:30, Min temp around 05:30)
        temp_cycle = math.sin((hour - 9.0) * math.pi / 12.0)
        # Seasonal base (Delhi: warm in summer ~35C, winter ~16C)
        is_winter = (day_of_year > 300 or day_of_year < 60)
        base_temp = 16.0 if is_winter else 30.0
        temperature = round(base_temp + 6.0 * temp_cycle + self.rng.normal(0, 0.8), 1)
        
        # Relative Humidity (inversely related to temperature)
        base_rh = 75.0 if is_winter else 50.0
        humidity = max(20.0, min(98.0, round(base_rh - 25.0 * temp_cycle + self.rng.normal(0, 2.0), 1)))
        
        # Planetary Boundary Layer Height (BLH in meters)
        # Drops to ~200-400m at night/dawn (trapping inversion), rises to 1500-2200m at midday
        blh_norm = max(0.05, 0.5 * (math.sin((hour - 8.0) * math.pi / 12.0) + 1.0))
        if hour < 6.0 or hour > 20.0:
            blh = 250.0 + 150.0 * self.rng.uniform(0.7, 1.2)
        else:
            blh = 350.0 + 1600.0 * (blh_norm ** 1.3) + self.rng.normal(0, 50.0)
        blh = max(180.0, round(blh, 1))
        
        # Wind: Prevailing North-Westerly in Delhi winters (approx 300°-320°), speeds 1.5 - 6.5 m/s
        wind_dir_deg = (305.0 + self.rng.normal(0, 20.0)) % 360.0
        wind_speed_ms = max(0.5, round(2.8 + 1.2 * temp_cycle + self.rng.normal(0, 0.7), 2))
        
        # Convert to vector components u (eastward) and v (northward)
        rad = math.radians(wind_dir_deg)
        wind_u = round(-wind_speed_ms * math.sin(rad), 2)
        wind_v = round(-wind_speed_ms * math.cos(rad), 2)
        
        # Surface atmospheric pressure
        pressure = round(1012.0 - (temperature - 20.0) * 0.3 + self.rng.normal(0, 0.4), 1)
        rainfall = 0.0 if not self.rng.random() < 0.03 else round(self.rng.uniform(0.5, 8.0), 1)
        
        return {
            "timestamp": timestamp.isoformat(),
            "temperature_c": temperature,
            "relative_humidity_pct": humidity,
            "wind_speed_ms": wind_speed_ms,
            "wind_direction_deg": round(wind_dir_deg, 1),
            "wind_u": wind_u,
            "wind_v": wind_v,
            "pressure_hpa": pressure,
            "boundary_layer_height_m": blh,
            "rainfall_mm": rainfall,
        }

    def evaluate_spatial_pollution(self, lat: float, lon: float, weather: dict, timestamp: datetime.datetime) -> dict:
        """
        Estimate ground-level concentration at an exact coordinate (lat, lon)
        using atmospheric physics, advection from sources, boundary layer ventilation, and road proximity.
        """
        hour = timestamp.hour + timestamp.minute / 60.0
        
        # 1. Traffic surge coefficient (morning 8-10am, evening 6-9pm)
        rush_morning = math.exp(-((hour - 9.0) ** 2) / 3.0)
        rush_evening = math.exp(-((hour - 19.5) ** 2) / 4.0)
        traffic_factor = 1.0 + 0.85 * rush_morning + 0.95 * rush_evening
        
        # 2. Boundary Layer Inversion Trap Factor (lower BLH = less volume = higher ground concentration)
        blh = weather["boundary_layer_height_m"]
        inversion_factor = (800.0 / blh) ** 0.65
        
        # 3. Wind Ventilation Dilution (higher wind = dispersion, calm wind = stagnation)
        wind_speed = max(0.8, weather["wind_speed_ms"])
        ventilation_factor = 2.5 / wind_speed
        
        # 4. Regional baseline background (Delhi regional winter/autumn load)
        base_pm25 = 65.0
        base_pm10 = 115.0
        base_no2 = 35.0
        base_so2 = 14.0
        base_co = 1.2
        base_o3 = 28.0
        
        # 5. Local source contributions with wind advection
        local_source_pm25 = 0.0
        local_source_no2 = 0.0
        
        wind_u = weather["wind_u"]
        wind_v = weather["wind_v"]
        
        for src in self.emission_sources:
            # Vector from source to point
            dlat = lat - src["lat"]
            dlon = lon - src["lon"]
            dist_sq = dlat * dlat + dlon * dlon
            dist = math.sqrt(dist_sq)
            
            # Advection alignment: is the point downwind of the source?
            # Displacement vector (dlon, dlat) dotted with wind vector (wind_u, wind_v)
            downwind_align = (dlon * wind_u + dlat * wind_v) / (max(0.1, dist) * wind_speed + 1e-6)
            advection_boost = 1.0 + max(0.0, downwind_align) * 1.6
            
            # Gaussian spatial decay
            sigma = src["radius"]
            decay = math.exp(-dist_sq / (2.0 * sigma * sigma))
            
            contrib = src["intensity"] * decay * advection_boost
            local_source_pm25 += contrib * 55.0
            local_source_no2 += contrib * 25.0
            
        # 6. Green buffer mitigation
        mitigation_multiplier = 1.0
        for gz in self.green_zones:
            dist = math.hypot(lat - gz["lat"], lon - gz["lon"])
            if dist < gz["radius"]:
                factor = 1.0 - (1.0 - gz["mitigation"]) * (1.0 - dist / gz["radius"])
                mitigation_multiplier = min(mitigation_multiplier, factor)
                
        # 7. Secondary aerosol hygroscopic formation (humidity > 65% boosts PM2.5)
        rh = weather["relative_humidity_pct"]
        rh_secondary_factor = 1.0 + (max(0.0, rh - 60.0) / 40.0) ** 1.5 * 0.4
        
        # 8. Photochemical Ozone dynamics (O3 formed in sunlight, destroyed at night)
        temp = weather["temperature_c"]
        solar_potential = max(0.0, math.sin((hour - 6.0) * math.pi / 12.0))
        o3_concentration = (base_o3 + 35.0 * solar_potential * (temp / 25.0)) / (1.0 + local_source_no2 * 0.02)
        
        # Final Concentrations
        pm25 = (base_pm25 * inversion_factor * ventilation_factor * rh_secondary_factor + local_source_pm25 * traffic_factor) * mitigation_multiplier
        pm10 = pm25 * 1.75 + self.rng.uniform(10.0, 35.0)
        no2 = (base_no2 * inversion_factor * ventilation_factor * traffic_factor + local_source_no2 * traffic_factor) * mitigation_multiplier
        so2 = (base_so2 * inversion_factor + local_source_pm25 * 0.15)
        co = (base_co * inversion_factor * traffic_factor + local_source_no2 * 0.03)
        
        # Add micro-scale random perturbation
        pm25 = max(5.0, round(pm25 + self.rng.normal(0, 3.5), 1))
        pm10 = max(10.0, round(pm10 + self.rng.normal(0, 6.0), 1))
        no2 = max(5.0, round(no2 + self.rng.normal(0, 2.5), 1))
        so2 = max(2.0, round(so2 + self.rng.normal(0, 1.0), 1))
        co = max(0.1, round(co + self.rng.normal(0, 0.05), 2))
        o3 = max(4.0, round(o3_concentration + self.rng.normal(0, 2.0), 1))
        
        pollutants = {
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "so2": so2,
            "co": co,
            "o3": o3,
        }
        
        aqi_val, prominent = calculate_overall_aqi(pollutants)
        
        # Satellite AOD proxy (column integral, ~0.2 to 1.8 unitless)
        satellite_aod = round(min(2.5, max(0.1, (pm25 / 140.0) * 0.8 + 0.15 + self.rng.normal(0, 0.04))), 3)
        
        return {
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "pollutants": pollutants,
            "aqi": aqi_val,
            "prominent_pollutant": prominent,
            "satellite_aod": satellite_aod,
            "mitigation_factor": round(mitigation_multiplier, 3),
        }

    def generate_station_snapshot(self, timestamp: datetime.datetime = None) -> list[dict]:
        """Generate current readings for all physical ground stations."""
        timestamp = timestamp or datetime.datetime.now()
        weather = self.get_weather(timestamp)
        
        results = []
        for st in GROUND_STATIONS:
            point_data = self.evaluate_spatial_pollution(st["lat"], st["lon"], weather, timestamp)
            results.append({
                "station_id": st["id"],
                "station_name": st["name"],
                "station_type": st["type"],
                "lat": st["lat"],
                "lon": st["lon"],
                "timestamp": timestamp.isoformat(),
                "pollutants": point_data["pollutants"],
                "aqi": point_data["aqi"],
                "prominent": point_data["prominent_pollutant"],
                "satellite_aod": point_data["satellite_aod"],
                "weather": weather,
            })
        return results

    def generate_spatial_grid(self, steps_lat: int = 24, steps_lon: int = 24, timestamp: datetime.datetime = None) -> dict:
        """
        Generate continuous downscaled spatial grid over the region bounding box.
        Used for full-city pollution surface visualization and hotspot detection.
        """
        timestamp = timestamp or datetime.datetime.now()
        weather = self.get_weather(timestamp)
        
        lats = np.linspace(self.region["min_lat"], self.region["max_lat"], steps_lat)
        lons = np.linspace(self.region["min_lon"], self.region["max_lon"], steps_lon)
        
        grid_cells = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                data = self.evaluate_spatial_pollution(float(lat), float(lon), weather, timestamp)
                grid_cells.append({
                    "grid_i": i,
                    "grid_j": j,
                    "lat": float(round(lat, 4)),
                    "lon": float(round(lon, 4)),
                    "pm25": data["pollutants"]["pm25"],
                    "pm10": data["pollutants"]["pm10"],
                    "no2": data["pollutants"]["no2"],
                    "so2": data["pollutants"]["so2"],
                    "co": data["pollutants"]["co"],
                    "o3": data["pollutants"]["o3"],
                    "aqi": data["aqi"],
                    "prominent": data["prominent_pollutant"],
                    "satellite_aod": data["satellite_aod"],
                })
                
        return {
            "timestamp": timestamp.isoformat(),
            "region": self.region["name"],
            "bounds": [
                [self.region["min_lat"], self.region["min_lon"]],
                [self.region["max_lat"], self.region["max_lon"]],
            ],
            "lat_steps": steps_lat,
            "lon_steps": steps_lon,
            "weather": weather,
            "cells": grid_cells,
        }

    def generate_training_dataset(self, num_days: int = 7, samples_per_hour: int = 20) -> dict:
        """
        Generate a multi-day training sequence across arbitrary space and time
        for training multimodal deep learning networks.
        """
        start_time = datetime.datetime.now() - datetime.timedelta(days=num_days)
        records = []
        
        total_hours = num_days * 24
        print(f"Generating synthetic training dataset for {num_days} days ({total_hours} hours)...")
        
        for h in range(total_hours):
            current_time = start_time + datetime.timedelta(hours=h)
            weather = self.get_weather(current_time)
            
            # Stations
            for st in GROUND_STATIONS:
                pt = self.evaluate_spatial_pollution(st["lat"], st["lon"], weather, current_time)
                records.append({
                    "timestamp": current_time.isoformat(),
                    "hour": current_time.hour,
                    "day_of_week": current_time.weekday(),
                    "is_station": 1,
                    "station_id": st["id"],
                    "lat": st["lat"],
                    "lon": st["lon"],
                    "weather": weather,
                    "satellite_aod": pt["satellite_aod"],
                    "targets": pt["pollutants"],
                    "aqi": pt["aqi"],
                })
                
            # Random unmonitored coordinates to emulate spatial downscaling training points
            for _ in range(samples_per_hour):
                rand_lat = float(self.rng.uniform(self.region["min_lat"], self.region["max_lat"]))
                rand_lon = float(self.rng.uniform(self.region["min_lon"], self.region["max_lon"]))
                pt = self.evaluate_spatial_pollution(rand_lat, rand_lon, weather, current_time)
                records.append({
                    "timestamp": current_time.isoformat(),
                    "hour": current_time.hour,
                    "day_of_week": current_time.weekday(),
                    "is_station": 0,
                    "station_id": None,
                    "lat": round(rand_lat, 5),
                    "lon": round(rand_lon, 5),
                    "weather": weather,
                    "satellite_aod": pt["satellite_aod"],
                    "targets": pt["pollutants"],
                    "aqi": pt["aqi"],
                })
                
        output_file = DATA_DIR / "training_dataset.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f)
            
        print(f"Dataset generated: {len(records)} records saved to {output_file}")
        return {"total_records": len(records), "output_path": str(output_file)}


if __name__ == "__main__":
    sim = AtmosphericSimulator()
    stations = sim.generate_station_snapshot()
    print(f"Generated snapshot for {len(stations)} stations. Sample AQI: {stations[0]['station_name']} -> {stations[0]['aqi']}")
    grid = sim.generate_spatial_grid(steps_lat=10, steps_lon=10)
    print(f"Generated spatial grid with {len(grid['cells'])} cells.")
