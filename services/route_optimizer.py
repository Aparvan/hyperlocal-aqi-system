"""
Cleanest Route / Green Path Optimizer.
Calculates and compares:
1. Fastest Route: Standard shortest-distance travel path.
2. Cleanest Route: Pollution-weighted path minimizing cumulative toxic PM2.5 lung exposure.
Computes minute-by-minute inhaled particulate dosage based on adult respiratory ventilation rates:
Dose (ug) = Sum( C_k * VentilationRate * delta_t_k )
"""

import heapq
import math
import numpy as np
from config import DEFAULT_REGION, GRID_CONFIG


class RouteOptimizer:
    def __init__(self, grid_engine):
        self.grid_engine = grid_engine
        self.region = grid_engine.region

    def find_cleanest_route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        travel_mode: str = "walking", # 'walking' (5 km/h) | 'cycling' (15 km/h)
    ) -> dict:
        """
        Compute Fastest vs Cleanest route between coordinates.
        Returns GeoJSON coordinates, travel times, and inhaled PM2.5 dosage comparison.
        """
        # Speed in km/h and minute respiratory ventilation rate in m3/min
        if travel_mode == "cycling":
            speed_kmh = 16.0
            vent_rate_m3_min = 0.030 # 30 L/min during cycling
        else:
            speed_kmh = 4.8
            vent_rate_m3_min = 0.016 # 16 L/min during walking

        # Fetch current pollution grid
        grid = self.grid_engine.generate_grid_predictions(rows=20, cols=20)
        cells = grid["cells"]
        rows = 20
        cols = 20

        # Build 2D lookup of concentrations
        pm25_matrix = np.zeros((rows, cols), dtype=np.float32)
        coord_matrix = np.zeros((rows, cols, 2), dtype=np.float32)

        for c in cells:
            r, col = c["grid_i"], c["grid_j"]
            pm25_matrix[r, col] = c["pm25"]
            coord_matrix[r, col] = [c["center_lat"], c["center_lon"]]

        # Find closest grid indices for origin and destination
        start_r, start_c = self._closest_node(origin_lat, origin_lon, coord_matrix)
        end_r, end_c = self._closest_node(dest_lat, dest_lon, coord_matrix)

        # 1. Dijkstra for Fastest Route (Weights purely distance)
        path_fastest, dist_fastest = self._dijkstra(
            start_r, start_c, end_r, end_c, coord_matrix, pm25_matrix, exposure_penalty=0.0
        )

        # 2. Dijkstra for Cleanest Route (Weights distance + non-linear pollution penalty)
        path_cleanest, dist_cleanest = self._dijkstra(
            start_r, start_c, end_r, end_c, coord_matrix, pm25_matrix, exposure_penalty=3.5
        )

        # Evaluate detailed metrics for both routes
        fastest_metrics = self._evaluate_path_metrics(
            path_fastest, coord_matrix, pm25_matrix, speed_kmh, vent_rate_m3_min, [origin_lat, origin_lon], [dest_lat, dest_lon]
        )
        cleanest_metrics = self._evaluate_path_metrics(
            path_cleanest, coord_matrix, pm25_matrix, speed_kmh, vent_rate_m3_min, [origin_lat, origin_lon], [dest_lat, dest_lon]
        )

        # Dose reduction percentage
        dose_fastest = fastest_metrics["total_inhaled_dose_ug"]
        dose_cleanest = cleanest_metrics["total_inhaled_dose_ug"]
        dose_reduction_pct = round(((dose_fastest - dose_cleanest) / max(0.01, dose_fastest)) * 100, 1)

        time_delta_mins = round(cleanest_metrics["duration_minutes"] - fastest_metrics["duration_minutes"], 1)

        return {
            "origin": {"lat": origin_lat, "lon": origin_lon},
            "destination": {"lat": dest_lat, "lon": dest_lon},
            "travel_mode": travel_mode,
            "fastest_route": fastest_metrics,
            "cleanest_route": cleanest_metrics,
            "comparison": {
                "exposure_reduction_pct": max(0.0, dose_reduction_pct),
                "extra_time_minutes": max(0.0, time_delta_mins),
                "summary": (
                    f"The Cleanest Route reduces inhaled toxic particulate dose by {dose_reduction_pct}% "
                    f"with only {time_delta_mins} mins extra travel time."
                ),
            },
        }

    def _closest_node(self, lat, lon, coord_matrix):
        rows, cols, _ = coord_matrix.shape
        min_d = float("inf")
        best = (0, 0)
        for r in range(rows):
            for c in range(cols):
                clat, clon = coord_matrix[r, c]
                d = math.hypot(clat - lat, clon - lon)
                if d < min_d:
                    min_d = d
                    best = (r, c)
        return best

    def _dijkstra(self, start_r, start_c, end_r, end_c, coords, pm25, exposure_penalty=0.0):
        rows, cols, _ = coords.shape
        dist = { (r, c): float("inf") for r in range(rows) for c in range(cols) }
        parent = {}
        dist[(start_r, start_c)] = 0.0

        pq = [(0.0, start_r, start_c)]

        while pq:
            cost, r, c = heapq.heappop(pq)
            if (r, c) == (end_r, end_c):
                break
            if cost > dist[(r, c)]:
                continue

            # 8-connected neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        # Physical distance in km
                        p1 = coords[r, c]
                        p2 = coords[nr, nc]
                        d_km = math.hypot((p2[0] - p1[0]) * 111.0, (p2[1] - p1[1]) * 111.0 * math.cos(math.radians(p1[0])))
                        
                        # Pollution penalty
                        avg_pm = (pm25[r, c] + pm25[nr, nc]) / 2.0
                        # Weight formula: Distance * (1 + penalty * (PM2.5 / 100)^2)
                        edge_weight = d_km * (1.0 + exposure_penalty * ((avg_pm / 90.0) ** 2))

                        if dist[(r, c)] + edge_weight < dist[(nr, nc)]:
                            dist[(nr, nc)] = dist[(r, c)] + edge_weight
                            parent[(nr, nc)] = (r, c)
                            heapq.heappush(pq, (dist[(nr, nc)], nr, nc))

        # Reconstruct path
        path = []
        curr = (end_r, end_c)
        while curr in parent:
            path.append(curr)
            curr = parent[curr]
        path.append((start_r, start_c))
        path.reverse()
        return path, dist[(end_r, end_c)]

    def _evaluate_path_metrics(self, path, coords, pm25, speed_kmh, vent_rate, orig_coord, dest_coord):
        # Build coordinate list with origin and dest snapped
        route_coords = [orig_coord]
        for r, c in path:
            route_coords.append([float(coords[r, c][0]), float(coords[r, c][1])])
        route_coords.append(dest_coord)

        total_distance_km = 0.0
        total_inhaled_dose_ug = 0.0
        pm25_samples = []

        for i in range(len(route_coords) - 1):
            p1 = route_coords[i]
            p2 = route_coords[i+1]
            seg_dist_km = math.hypot((p2[0] - p1[0]) * 111.0, (p2[1] - p1[1]) * 111.0 * math.cos(math.radians(p1[0])))
            total_distance_km += seg_dist_km

            # Travel time for segment in minutes
            seg_time_min = (seg_dist_km / speed_kmh) * 60.0

            # Concentration approximation
            idx = min(len(path) - 1, i)
            r, c = path[idx]
            seg_pm25 = float(pm25[r, c])
            pm25_samples.append(seg_pm25)

            # Inhaled dose: C (ug/m3) * Ventilation (m3/min) * Time (min)
            seg_dose = seg_pm25 * vent_rate * seg_time_min
            total_inhaled_dose_ug += seg_dose

        total_time_min = (total_distance_km / speed_kmh) * 60.0
        avg_pm25 = float(np.mean(pm25_samples)) if pm25_samples else 100.0

        return {
            "coordinates": route_coords,
            "distance_km": round(total_distance_km, 2),
            "duration_minutes": round(total_time_min, 1),
            "average_pm25": round(avg_pm25, 1),
            "total_inhaled_dose_ug": round(total_inhaled_dose_ug, 1),
        }
