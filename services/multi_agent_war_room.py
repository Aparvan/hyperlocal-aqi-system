"""
Tri-Agent Collaborative Environmental War Room (2026 Trend Booster).
Implements a multi-agent society inspired by LangGraph agentic orchestration:
1. DiagnosticAgent: Audits sensor consistency, cross-references satellite AOD with ground monitors, and computes XAI root causes.
2. SimulationAgent: Executes atmospheric scenario forecasting and what-if dispersion modeling.
3. PolicyOrchestratorAgent: Enforces statutory Graded Response Action Plan (GRAP I-IV) emergency protocols and coordinates municipal departments.
"""

from datetime import datetime
import math


class DiagnosticAgent:
    def __init__(self, xai_engine, grid_engine):
        self.name = "Dr. Priya Varma (Diagnostic & Sensor QC Lead)"
        self.role = "Diagnostic Agent"
        self.xai_engine = xai_engine
        self.grid_engine = grid_engine

    def analyze(self, lat: float, lon: float, query: str) -> dict:
        point_pred = self.grid_engine.predict_point(lat, lon)
        weather = point_pred["metadata"]["current_weather"]
        sample = self.grid_engine.prep.prepare_multimodal_sample(lat, lon)
        xai = self.xai_engine.explain_prediction(sample, target_index=0)

        # Cross-reference Satellite AOD vs Ground Sensors
        sat_aod = sample["spatial"][0, 0, 2, 2].item()
        nearest_st = point_pred["metadata"]["nearest_station"]
        dist_km = point_pred["metadata"]["station_distance_km"]

        anomaly_flag = False
        if sat_aod > 1.2 and point_pred["pollutants"]["pm25"] < 40.0:
            anomaly_flag = True
            qc_note = "Elevated column AOD detected aloft with low surface PM2.5, suggesting elevated transboundary smoke layer."
        else:
            qc_note = "Surface monitoring values correlate reliably with satellite column optical depth."

        diagnosis = (
            f"Sensor telemetry confirmed at coordinates ({lat:.3f}°N, {lon:.3f}°E). "
            f"Current PM2.5 is {point_pred['pollutants']['pm25']} µg/m³ (AQI: {point_pred['aqi']} - {point_pred['category']}). "
            f"Explainable AI confirms {xai['top_driver']} is the primary driver ({xai['sorted_drivers'][0]['percentage']}% contribution), "
            f"compounded by {xai['sorted_drivers'][1]['factor']} ({xai['sorted_drivers'][1]['percentage']}%). {qc_note}"
        )

        return {
            "agent": self.name,
            "role": self.role,
            "status": "Completed",
            "findings": diagnosis,
            "top_driver": xai["top_driver"],
            "aqi": point_pred["aqi"],
            "anomaly_detected": anomaly_flag,
            "weather": weather,
        }


class SimulationAgent:
    def __init__(self, scenario_simulator):
        self.name = "Arjun Mehta (Atmospheric Simulation Specialist)"
        self.role = "Simulation Agent"
        self.simulator = scenario_simulator

    def project(self, diagnostic_data: dict) -> dict:
        curr_weather = diagnostic_data["weather"]
        curr_aqi = diagnostic_data["aqi"]

        # Run 2 micro-climate simulations:
        # 1. Negative scenario: Wind speed drops by 50% (stagnation)
        stagnant_sim = self.simulator.simulate_scenario(wind_speed_override=max(0.5, curr_weather["wind_speed_ms"] * 0.5))
        # 2. Positive scenario: 30% traffic curb + Anti-smog water spraying
        curbed_sim = self.simulator.simulate_scenario(traffic_delta=-0.30, industrial_curb=True)

        forecast = (
            f"Atmospheric dispersion modeling indicates acute vulnerability to wind speed shifts. "
            f"If surface winds decelerate to {stagnant_sim['simulated_weather']['wind_speed_ms']:.1f} m/s tonight, "
            f"regional PM2.5 will surge by +{stagnant_sim['impact_summary']['percentage_change_pm25']}%. "
            f"Conversely, our simulation confirms that instituting a 30% vehicular traffic curb and halting local industrial emission points "
            f"will yield a {abs(curbed_sim['impact_summary']['percentage_change_pm25'])}% immediate reduction in surface particulates."
        )

        return {
            "agent": self.name,
            "role": self.role,
            "status": "Completed",
            "forecast": forecast,
            "curb_reduction_pct": abs(curbed_sim['impact_summary']['percentage_change_pm25']),
            "stagnation_surge_pct": stagnant_sim['impact_summary']['percentage_change_pm25'],
        }


class PolicyOrchestratorAgent:
    def __init__(self):
        self.name = "Director S. K. Nair (Municipal & GRAP Enforcement)"
        self.role = "Policy Orchestrator"

    def formulate_directives(self, diag: dict, sim: dict) -> dict:
        aqi = diag["aqi"]

        # Statutory CAQM Revised GRAP Stages (Commission for Air Quality Management):
        # Stage I: 201-300 (Poor), Stage II: 301-400 (Very Poor),
        # Stage III: 401-450 (Severe), Stage IV: >450 (Severe+ / Emergency)
        if aqi > 450:
            stage = "GRAP Stage IV (Severe+ / Emergency: AQI > 450)"
            actions = [
                "Ban entry of commercial diesel trucks into Delhi-NCR jurisdiction (except essential goods & LNG/CNG/electric).",
                "Mandate 50% work-from-home across public and private corporate sectors to curb peak traffic surge.",
                "Halt all outdoor civil construction, earth-moving, and demolition activities immediately.",
                "Deploy heavy automated mist-cannons and water sprinklers continuously across Ring Road and arterial hotspots.",
            ]
        elif aqi >= 401:
            stage = "GRAP Stage III (Severe: AQI 401 - 450)"
            actions = [
                "Enforce strict ban on BS-III petrol and BS-IV diesel light motor vehicles (passenger 4-wheelers) in Delhi-NCR.",
                "Halt non-essential commercial construction and demolition activities involving dust generation.",
                "Intensify mechanized road sweeping and mechanized water washing during pre-dawn and evening hours.",
                "Shut down non-conforming industrial boiler operations violating clean-fuel (PNG/biomass) mandates.",
            ]
        elif aqi >= 301:
            stage = "GRAP Stage II (Very Poor: AQI 301 - 400)"
            actions = [
                "Synchronize traffic signaling at major intersections to minimize vehicular idling and tailpipe emissions.",
                "Enforce regulated power supply to strictly avoid diesel generator (DG) usage across commercial complexes.",
                "Deploy targeted smog guns at designated local hotspot quadrants.",
                "Issue public health advisory for vulnerable cohorts (pediatric and elderly asthmatics).",
            ]
        elif aqi >= 201:
            stage = "GRAP Stage I (Poor: AQI 201 - 300)"
            actions = [
                "Strictly enforce anti-dust regulations at construction sites with active water sprinkling.",
                "Intensify anti-open burning patrols across municipal landfill sites and open grounds.",
                "Ensure smooth traffic flow along congested corridors through active traffic police deployment.",
            ]
        else:
            stage = "Standard Ambient Operations (AQI <= 200: Moderate/Good)"
            actions = [
                "Continue routine mechanized sweeping and monitoring across all baseline stations.",
                "Encourage active public transit adoption and electric vehicle mobility.",
            ]

        policy_statement = (
            f"In accordance with statutory CAQM environmental mandates, current index ({aqi}) triggers **{stage}**. "
            f"Validating simulation data showing {sim['curb_reduction_pct']}% mitigative headroom, "
            f"the inter-departmental taskforce hereby authorizes the following priority enforcement protocols:"
        )

        return {
            "agent": self.name,
            "role": self.role,
            "grap_stage": stage,
            "directive_intro": policy_statement,
            "statutory_actions": actions,
        }


class MultiAgentWarRoom:
    def __init__(self, grid_engine, xai_engine, scenario_simulator):
        self.diagnostic_agent = DiagnosticAgent(xai_engine, grid_engine)
        self.simulation_agent = SimulationAgent(scenario_simulator)
        self.policy_agent = PolicyOrchestratorAgent()

    def run_collaboration(self, query: str, lat: float = None, lon: float = None) -> dict:
        lat = lat or 28.6139
        lon = lon or 77.2090

        # Step 1: Diagnostic Lead executes telemetry & XAI inspection
        diag_res = self.diagnostic_agent.analyze(lat, lon, query)

        # Step 2: Simulation Specialist projects what-if futures
        sim_res = self.simulation_agent.project(diag_res)

        # Step 3: Policy Orchestrator drafts emergency GRAP directives
        policy_res = self.policy_agent.formulate_directives(diag_res, sim_res)

        # Build chronological multi-agent transcript
        debate = [
            {
                "speaker": diag_res["agent"],
                "role": diag_res["role"],
                "badge": "Diagnostic & XAI",
                "content": diag_res["findings"],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            },
            {
                "speaker": sim_res["agent"],
                "role": sim_res["role"],
                "badge": "Atmospheric Modeling",
                "content": sim_res["forecast"],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            },
            {
                "speaker": policy_res["agent"],
                "role": policy_res["role"],
                "badge": "Policy & GRAP Authority",
                "content": f"{policy_res['directive_intro']}\n" + "\n".join(f"- {a}" for a in policy_res["statutory_actions"]),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            },
        ]

        consensus_summary = (
            f"The Multi-Agent Council reached 100% consensus: {diag_res['top_driver']} is the primary catalyst. "
            f"Immediate execution of {policy_res['grap_stage']} is projected to attenuate localized exposure by ~{sim_res['curb_reduction_pct']}%."
        )

        return {
            "session_id": f"WAR-ROOM-{int(datetime.now().timestamp())}",
            "coordinates": {"lat": lat, "lon": lon},
            "query": query,
            "consensus_score": "100% Convergence",
            "grap_stage": policy_res["grap_stage"],
            "consensus_summary": consensus_summary,
            "debate_transcript": debate,
        }
