"""
What-If Scenario Simulation Engine.
Enables real-time parametric exploration of urban policy and atmospheric interventions:
- Traffic Volume Adjustment (-50% to +50%)
- Wind Speed & Direction Modulation
- Planetary Boundary Layer Inversion Layer Simulation
- Industrial Point-Source Curbs / Factory Shutdowns
Recalculates continuous pollution fields and outputs delta impact metrics.
"""

import math
import numpy as np
import datetime
from config import DEFAULT_REGION, GRID_CONFIG
from data.generator import calculate_overall_aqi, calculate_sub_aqi


class ScenarioSimulator:
    def __init__(self, base_grid_engine):
        self.grid_engine = base_grid_engine
        self.region = base_grid_engine.region

    def simulate_scenario(
        self,
        traffic_delta: float = 0.0,      # -0.50 to +0.50 (-50% to +50%)
        wind_speed_override: float = None, # e.g. 1.0 to 12.0 m/s
        wind_dir_override: float = None,   # 0 to 360 degrees
        blh_override: float = None,        # 150 to 2500 meters
        industrial_curb: bool = False,     # True reduces industrial cluster intensity by 60%
    ) -> dict:
        """
        Evaluate hypothetical scenario over current conditions.
        Returns recalculated grid, affected hotspot count, and overall percentage change in PM2.5.
        """
        # Fetch baseline grid
        base_grid = self.grid_engine.generate_grid_predictions(rows=GRID_CONFIG["lat_steps"], cols=GRID_CONFIG["lon_steps"])
        base_weather = base_grid["weather"]

        sim_wind_speed = float(wind_speed_override) if wind_speed_override is not None else base_weather["wind_speed_ms"]
        sim_wind_dir = float(wind_dir_override) if wind_dir_override is not None else base_weather["wind_direction_deg"]
        sim_blh = float(blh_override) if blh_override is not None else base_weather["boundary_layer_height_m"]

        # Calculate ventilation & inversion adjustment ratios
        # Note: `ventilation` here is a stagnation factor inversely proportional to wind speed
        # (2.5 / v). Therefore ventilation_ratio = sim_stagnation / base_stagnation = base_wind / sim_wind:
        #   - Higher simulated wind (sim_wind > base_wind) -> ratio < 1 -> lower PM2.5 (stronger dispersion)
        #   - Lower simulated wind  (sim_wind < base_wind) -> ratio > 1 -> higher PM2.5 (stagnation buildup)
        base_ventilation = 2.5 / max(0.8, base_weather["wind_speed_ms"])
        sim_ventilation = 2.5 / max(0.8, sim_wind_speed)
        ventilation_ratio = sim_ventilation / (base_ventilation + 1e-6)

        base_inversion = (800.0 / max(150.0, base_weather["boundary_layer_height_m"])) ** 0.65
        sim_inversion = (800.0 / max(150.0, sim_blh)) ** 0.65
        inversion_ratio = sim_inversion / (base_inversion + 1e-6)

        # Directional advection shift: In Delhi-NCR, winds from North-West (~315°) channel
        # heavy regional upwind plumes, whereas winds from South-East (~135°) disperse regional inflow.
        nw_rad = math.radians(315.0)
        base_nw_align = math.cos(math.radians(base_weather["wind_direction_deg"]) - nw_rad)
        sim_nw_align = math.cos(math.radians(sim_wind_dir) - nw_rad)
        dir_advection_factor = max(0.70, min(1.30, 1.0 + 0.15 * (sim_nw_align - base_nw_align)))

        traffic_multiplier = 1.0 + float(traffic_delta)
        ind_multiplier = 0.40 if industrial_curb else 1.0

        sim_cells = []
        base_pm25_list = []
        sim_pm25_list = []

        for c in base_grid["cells"]:
            orig_pm25 = c["pm25"]
            orig_pm10 = c["pm10"]
            orig_no2 = c["no2"]

            base_pm25_list.append(orig_pm25)

            # Deconstruct baseline into traffic/industrial fraction vs meteorological dispersion.
            # Road influence activates arterial proximity: cells with higher road/AOD density
            # have a higher traffic-attributable fraction.
            road_influence = min(1.0, c["satellite_aod"] * 0.8)
            traffic_fraction = 0.25 + 0.25 * road_influence
            ind_fraction = 0.30 if c["satellite_aod"] > 0.85 else 0.20
            bg_fraction = max(0.15, 1.0 - traffic_fraction - ind_fraction)

            traffic_part = orig_pm25 * traffic_fraction * traffic_multiplier
            ind_part = orig_pm25 * ind_fraction * ind_multiplier
            background_part = orig_pm25 * bg_fraction

            # Apply meteorological shifts (inversion, wind ventilation, and wind direction advection)
            new_pm25 = (traffic_part + ind_part + background_part) * (inversion_ratio ** 0.5) * (ventilation_ratio ** 0.6) * dir_advection_factor
            new_pm25 = max(5.0, round(float(new_pm25), 1))

            new_pm10 = max(10.0, round(new_pm25 * 1.7 + 10.0, 1))
            new_no2 = max(5.0, round(orig_no2 * traffic_multiplier * (ventilation_ratio ** 0.4), 1))

            sim_pm25_list.append(new_pm25)

            pol_dict = {"pm25": new_pm25, "pm10": new_pm10, "no2": new_no2}
            new_aqi, new_prom = calculate_overall_aqi(pol_dict)

            cell_copy = dict(c)
            cell_copy["pm25"] = new_pm25
            cell_copy["pm10"] = new_pm10
            cell_copy["no2"] = new_no2
            cell_copy["aqi"] = new_aqi
            cell_copy["prominent"] = new_prom
            cell_copy["delta_pm25"] = round(new_pm25 - orig_pm25, 1)
            sim_cells.append(cell_copy)

        mean_base = float(np.mean(base_pm25_list))
        mean_sim = float(np.mean(sim_pm25_list))
        pct_change = round(((mean_sim - mean_base) / (mean_base + 1e-6)) * 100, 1)

        simulated_weather = dict(base_weather)
        simulated_weather["wind_speed_ms"] = sim_wind_speed
        simulated_weather["wind_direction_deg"] = sim_wind_dir
        simulated_weather["boundary_layer_height_m"] = sim_blh

        return {
            "scenario_parameters": {
                "traffic_delta_pct": round(traffic_delta * 100, 1),
                "wind_speed_ms": sim_wind_speed,
                "wind_direction_deg": sim_wind_dir,
                "boundary_layer_height_m": sim_blh,
                "industrial_curb_active": industrial_curb,
            },
            "impact_summary": {
                "baseline_mean_pm25": round(mean_base, 1),
                "simulated_mean_pm25": round(mean_sim, 1),
                "percentage_change_pm25": pct_change,
                "verdict": (
                    f"Simulation indicates an overall {abs(pct_change)}% "
                    f"{'reduction' if pct_change < 0 else 'increase'} in regional particulate exposure."
                ),
            },
            "simulated_weather": simulated_weather,
            "cells": sim_cells,
        }
