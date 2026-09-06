"""
Spatial Pollution Hotspot Detection Engine.
Identifies contiguous spatial zones with statutory air quality violations
(e.g., CPCB AQI > 300 / Severe or PM2.5 > 120 ug/m3).
Clusters adjacent violating cells, determines peak pollution epicenters,
estimates affected geographic area, and generates priority municipal alerts.
"""

import numpy as np
from config import AQI_BREAKPOINTS


class HotspotDetector:
    def __init__(self, aqi_threshold: float = 250.0, pm25_threshold: float = 110.0):
        self.aqi_threshold = aqi_threshold
        self.pm25_threshold = pm25_threshold

    def detect_hotspots(self, grid_data: dict) -> list[dict]:
        """
        Scan grid cells, perform spatial 8-connected component clustering,
        and return prioritized list of detected pollution hotspots.
        """
        cells = grid_data.get("cells", [])
        rows = grid_data.get("grid_dimensions", {}).get("rows", 24)
        cols = grid_data.get("grid_dimensions", {}).get("cols", 24)
        
        # Build 2D matrix of binary breach flags
        breach_map = np.zeros((rows, cols), dtype=bool)
        cell_lookup = {}
        
        for c in cells:
            r, col = c["grid_i"], c["grid_j"]
            cell_lookup[(r, col)] = c
            if c["aqi"] >= self.aqi_threshold or c["pm25"] >= self.pm25_threshold:
                breach_map[r, col] = True
                
        # Connected Components Labeling (BFS / Flood Fill)
        visited = np.zeros((rows, cols), dtype=bool)
        clusters = []
        
        for r in range(rows):
            for col in range(cols):
                if breach_map[r, col] and not visited[r, col]:
                    # Start new cluster
                    queue = [(r, col)]
                    visited[r, col] = True
                    cluster_cells = []
                    
                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        cluster_cells.append(cell_lookup[(curr_r, curr_c)])
                        
                        # 8-connected neighbors
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if breach_map[nr, nc] and not visited[nr, nc]:
                                        visited[nr, nc] = True
                                        queue.append((nr, nc))
                                        
                    clusters.append(cluster_cells)
                    
        # Analyze and format each detected hotspot cluster
        hotspots = []
        for idx, cluster in enumerate(clusters):
            # Sort by AQI descending
            cluster_sorted = sorted(cluster, key=lambda x: x["aqi"], reverse=True)
            peak_cell = cluster_sorted[0]
            
            lats = [c["center_lat"] for c in cluster]
            lons = [c["center_lon"] for c in cluster]
            pm25s = [c["pm25"] for c in cluster]
            aqis = [c["aqi"] for c in cluster]
            
            centroid_lat = round(float(np.mean(lats)), 4)
            centroid_lon = round(float(np.mean(lons)), 4)
            
            # Area estimate (each cell is approx 2km x 2km = 4 km²)
            cell_area_km2 = 4.0
            total_area_km2 = len(cluster) * cell_area_km2
            
            # Severity classification
            peak_aqi = peak_cell["aqi"]
            if peak_aqi >= 400:
                severity = "Emergency / Severe"
                color = "#7E0023"
                priority = 1
            elif peak_aqi >= 300:
                severity = "Very Poor"
                color = "#FF0000"
                priority = 2
            else:
                severity = "Poor / High Warning"
                color = "#FF7E00"
                priority = 3
                
            # Name cluster based on geographic quadrant
            quadrant = ""
            if centroid_lat > 28.66:
                quadrant += "North"
            elif centroid_lat < 28.56:
                quadrant += "South"
            if centroid_lon > 77.25:
                quadrant += "-East"
            elif centroid_lon < 77.15:
                quadrant += "-West"
            else:
                quadrant += " Central"
                
            cluster_name = f"{quadrant.strip()} Industrial & Traffic Hotspot"
            
            hotspots.append({
                "hotspot_id": f"HS-{idx+1:02d}",
                "name": cluster_name,
                "priority": priority,
                "severity": severity,
                "color": color,
                "peak_aqi": peak_aqi,
                "peak_pm25": peak_cell["pm25"],
                "avg_pm25": round(float(np.mean(pm25s)), 1),
                "cell_count": len(cluster),
                "approx_area_km2": total_area_km2,
                "centroid": {"lat": centroid_lat, "lon": centroid_lon},
                "peak_location": {"lat": peak_cell["center_lat"], "lon": peak_cell["center_lon"]},
                "satellite_aod_peak": peak_cell["satellite_aod"],
                "recommended_action": self._generate_recommendation(severity, centroid_lat, centroid_lon),
            })
            
        # Sort hotspots by peak AQI descending
        hotspots.sort(key=lambda x: x["peak_aqi"], reverse=True)
        return hotspots

    def _generate_recommendation(self, severity: str, lat: float, lon: float) -> str:
        if "Severe" in severity:
            return "Immediate GRAP-IV emergency protocols: Deploy automated water misters on arterial corridors, restrict heavy commercial diesel vehicles, halt open construction dust."
        elif "Very Poor" in severity:
            return "Intensify mechanical road sweeping, activate localized smog guns, issue high-risk respiratory warnings for outdoor schools and elder centers."
        else:
            return "Enforce strict anti-idling at traffic bottlenecks and monitor waste combustion."
