"""
Agentic Environmental Advisor (2026 Trend Booster).
Autonomous multi-step reasoning agent that synthesizes:
1. Hyperlocal Multimodal Deep Learning predictions
2. Explainable AI (XAI) feature attribution drivers
3. Spatial Hotspot cluster metrics
4. Atmospheric physics and health standards
Provides domain-specific diagnostic explanations, citizen health advisories,
and smart city municipal mitigation recommendations.
"""

import math
from datetime import datetime

from services.grid_engine import HyperlocalGridEngine
from services.hotspot_detector import HotspotDetector
from models.xai_engine import XAIEngine
from data.preprocessor import HyperlocalPreprocessor


class AgenticEnvironmentalAdvisor:
    def __init__(self, grid_engine: HyperlocalGridEngine, hotspot_detector: HotspotDetector):
        self.grid_engine = grid_engine
        self.hotspot_detector = hotspot_detector
        self.xai_engine = XAIEngine(self.grid_engine.model)
        self.prep = HyperlocalPreprocessor(self.grid_engine.region)

    def answer_query(self, query: str, lat: float = None, lon: float = None) -> dict:
        """
        Process citizen or municipal policy inquiries with grounded environmental intelligence.
        """
        lat = lat or self.grid_engine.region["center_lat"]
        lon = lon or self.grid_engine.region["center_lon"]
        
        # 1. Hyperlocal Inference
        point_pred = self.grid_engine.predict_point(lat, lon)
        aqi = point_pred["aqi"]
        pm25 = point_pred["pollutants"]["pm25"]
        pm10 = point_pred["pollutants"]["pm10"]
        no2 = point_pred["pollutants"]["no2"]
        category = point_pred["category"]
        weather = point_pred["metadata"]["current_weather"]
        
        # 2. XAI Attribution Breakdown
        sample = self.prep.prepare_multimodal_sample(lat, lon)
        xai_res = self.xai_engine.explain_prediction(sample, target_index=0) # PM2.5 attribution
        top_driver = xai_res["top_driver"]
        top_driver_pct = xai_res["sorted_drivers"][0]["percentage"]
        second_driver = xai_res["sorted_drivers"][1]["factor"]
        second_driver_pct = xai_res["sorted_drivers"][1]["percentage"]
        
        # 3. Hotspot Context Check
        grid = self.grid_engine.generate_grid_predictions(rows=12, cols=12)
        hotspots = self.hotspot_detector.detect_hotspots(grid)
        in_hotspot = None
        for hs in hotspots:
            c_lat = hs["centroid"]["lat"]
            c_lon = hs["centroid"]["lon"]
            dist_km = math.hypot((c_lat - lat) * 111.0, (c_lon - lon) * 111.0 * math.cos(math.radians(lat)))
            if dist_km < math.sqrt(hs["approx_area_km2"] / math.pi) + 1.5:
                in_hotspot = hs
                break
                
        # 4. Synthesize Agentic Multi-Step Reasoning
        q_lower = query.lower()
        
        # Determine intent
        is_exercise_or_health = any(w in q_lower for w in ["jog", "run", "walk", "exercise", "safe", "health", "mask", "children", "asthma"])
        is_cause_or_why = any(w in q_lower for w in ["why", "cause", "reason", "driver", "spike", "factor", "high", "explain"])
        is_policy_or_action = any(w in q_lower for w in ["action", "policy", "mitigation", "authority", "government", "grap", "control", "sprinkler"])
        is_what_if = any(w in q_lower for w in ["what if", "scenario", "wind", "rain", "traffic reduced", "forecast"])
        
        # Formulate Advisory Responses
        health_advice = []
        if aqi > 300:
            health_advice.append("🚫 **Avoid Outdoor Cardio**: Heavy exertion will cause deep inhalation of fine particulate matter (PM2.5).")
            health_advice.append("😷 **Wear N95/FFP2 Respirators**: Cloth or surgical masks do not filter sub-micron particles effectively.")
            health_advice.append("🏠 **Indoor Air Quality**: Keep windows sealed during morning and late evening hours; run HEPA air filtration.")
        elif aqi > 200:
            health_advice.append("⚠️ **Reduce Prolonged Outdoor Exposure**: Vulnerable groups (children, elderly, asthmatics) should stay indoors.")
            health_advice.append("😷 **Protective Masks**: Use certified N95 masks when commuting along major corridors.")
        else:
            health_advice.append("✅ **Moderate Outdoor Activity**: Air quality is acceptable for healthy individuals; sensitive persons should monitor symptoms.")
            
        policy_actions = []
        if in_hotspot:
            policy_actions.append(f"🚨 **Targeted Hotspot Response ({in_hotspot['name']})**: Centroid peak AQI is {in_hotspot['peak_aqi']}.")
            policy_actions.append(f"💧 **Smog Guns & Water Cannons**: Deploy immediately along the high-traffic arterial grid within a {round(math.sqrt(in_hotspot['approx_area_km2']), 1)} km radius.")
        if "Traffic" in top_driver:
            policy_actions.append("🚗 **Traffic Management**: Synchronize corridor traffic lights and enforce strict anti-idling at key intersections.")
        if "Boundary Layer" in top_driver:
            policy_actions.append("🌡️ **Meteorological Warning**: Atmospheric inversion layer is shallow (~" + str(weather["boundary_layer_height_m"]) + "m), trapping emissions near surface.")
        if "Industrial" in top_driver:
            policy_actions.append("🏭 **Industrial Scrutiny**: Inspect nearby non-conforming industrial clusters for unauthorized fuel combustion.")
            
        # Agent Reasoning Formulation
        response_sections = []
        
        response_sections.append(
            f"### 📍 Hyperlocal Status at ({lat:.3f}°N, {lon:.3f}°E)\n"
            f"- **Estimated AQI**: **{aqi}** ({category})\n"
            f"- **PM2.5**: **{pm25} µg/m³** | **PM10**: **{pm10} µg/m³** | **NO2**: **{no2} µg/m³**\n"
            f"- **Current Weather**: Temp {weather['temperature_c']}°C, Humidity {weather['relative_humidity_pct']}%, Wind {weather['wind_speed_ms']} m/s (from {weather['wind_direction_deg']}°), BLH {weather['boundary_layer_height_m']}m"
        )
        
        if in_hotspot:
            response_sections.append(
                f"\n> ⚠️ **Hotspot Alert**: This location falls inside active hotspot **{in_hotspot['name']}** (Peak AQI: {in_hotspot['peak_aqi']}, Severity: {in_hotspot['severity']})."
            )
            
        response_sections.append(
            f"\n### 🔬 AI Diagnostic Root Cause (Explainable AI)\n"
            f"- **Primary Driver**: **{top_driver}** (contributing **{top_driver_pct}%** to the pollution elevation).\n"
            f"- **Secondary Driver**: **{second_driver}** (**{second_driver_pct}%**).\n"
            f"- *Physical Mechanism*: {xai_res['diagnostic_summary']}"
        )
        
        if is_exercise_or_health or not (is_policy_or_action or is_what_if):
            response_sections.append("\n### 🏥 Citizen Health & Protective Advisory\n" + "\n".join(f"- {a}" for a in health_advice))
            
        if is_policy_or_action or in_hotspot or not is_exercise_or_health:
            response_sections.append("\n### 🏛️ Municipal & Environmental Policy Interventions\n" + "\n".join(f"- {a}" for a in policy_actions))
            
        if is_what_if or "what if" in q_lower:
            response_sections.append(
                f"\n### 🔮 What-If Scenario Simulation\n"
                f"- **If Wind Speed doubles to {round(weather['wind_speed_ms'] * 2, 1)} m/s**: Enhanced atmospheric ventilation would disperse particulate loading, expected to decrease local PM2.5 by ~28–35% (Estimated AQI: ~{round(aqi * 0.7)}).\n"
                f"- **If Daytime Solar Heating elevates BLH to 1800m**: Ground-level concentrations will dilute vertically by ~40% by afternoon hours."
            )
            
        final_answer = "\n".join(response_sections)
        
        return {
            "query": query,
            "coordinates": {"lat": lat, "lon": lon},
            "aqi": aqi,
            "category": category,
            "pollutants": point_pred["pollutants"],
            "top_driver": top_driver,
            "top_driver_pct": top_driver_pct,
            "in_hotspot": bool(in_hotspot),
            "hotspot_info": in_hotspot,
            "response_markdown": final_answer,
        }
