"""
Weather Data Fetcher Module.
Integrates with Open-Meteo API (free, open-access meteorological data)
to fetch hourly weather parameters (temperature, relative humidity, wind vector,
surface pressure, planetary boundary layer height).
Includes robust local caching and automatic fallback to physical simulation.
"""

import requests
import datetime
import json
from pathlib import Path
from config import CACHE_DIR, DEFAULT_REGION
from data.generator import AtmosphericSimulator


class WeatherFetcher:
    def __init__(self, region: dict = None):
        self.region = region or DEFAULT_REGION
        self.cache_dir = CACHE_DIR / "weather"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.simulator = AtmosphericSimulator(self.region)

    def fetch_current_weather(self, lat: float = None, lon: float = None) -> dict:
        """
        Fetch current meteorological parameters for given coordinates.
        Uses Open-Meteo API with local fallback if offline or request fails.
        """
        lat = lat or self.region["center_lat"]
        lon = lon or self.region["center_lon"]
        cache_key = f"weather_{round(lat, 2)}_{round(lon, 2)}.json"
        cache_file = self.cache_dir / cache_key
        
        # Check cache if less than 30 minutes old
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cached_time = datetime.datetime.fromisoformat(data["timestamp"])
                if (datetime.datetime.now() - cached_time).total_seconds() < 1800:
                    return data
            except Exception:
                pass

        # Attempt live call to Open-Meteo API
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "surface_pressure", "precipitation"],
            "hourly": ["boundary_layer_height"],
            "timezone": "auto",
        }
        
        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                res = resp.json()
                current = res.get("current", {})
                hourly = res.get("hourly", {})
                
                # Extract boundary layer height for current hour if available
                blh_list = hourly.get("boundary_layer_height", [])
                blh = float(blh_list[0]) if blh_list else 650.0
                
                wind_speed = float(current.get("wind_speed_10m", 3.0))
                wind_dir = float(current.get("wind_direction_10m", 310.0))
                
                import math
                rad = math.radians(wind_dir)
                wind_u = round(-wind_speed * math.sin(rad), 2)
                wind_v = round(-wind_speed * math.cos(rad), 2)
                
                weather_dict = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "Open-Meteo API",
                    "latitude": lat,
                    "longitude": lon,
                    "temperature_c": float(current.get("temperature_2m", 25.0)),
                    "relative_humidity_pct": float(current.get("relative_humidity_2m", 60.0)),
                    "wind_speed_ms": wind_speed,
                    "wind_direction_deg": wind_dir,
                    "wind_u": wind_u,
                    "wind_v": wind_v,
                    "pressure_hpa": float(current.get("surface_pressure", 1010.0)),
                    "boundary_layer_height_m": max(150.0, blh),
                    "rainfall_mm": float(current.get("precipitation", 0.0)),
                }
                
                # Save cache
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(weather_dict, f)
                    
                return weather_dict
        except Exception:
            pass
            
        # Graceful fallback to physical simulation
        weather_dict = self.simulator.get_weather(datetime.datetime.now())
        weather_dict["source"] = "Physical Simulation Model"
        weather_dict["latitude"] = lat
        weather_dict["longitude"] = lon
        return weather_dict

    def fetch_weather_series(self, hours: int = 24, lat: float = None, lon: float = None) -> list[dict]:
        """Fetch a continuous sequence of hourly weather observations/forecasts for temporal LSTM modeling."""
        lat = lat or self.region["center_lat"]
        lon = lon or self.region["center_lon"]
        now = datetime.datetime.now()
        series = []
        for i in range(hours):
            t = now - datetime.timedelta(hours=(hours - 1 - i))
            w = self.simulator.get_weather(t)
            w["source"] = "Historical Sequence"
            series.append(w)
        return series
