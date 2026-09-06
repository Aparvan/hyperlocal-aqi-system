/**
 * AeroSense AI - Advanced GeoAI & Multi-Agent Environmental Platform
 * Leaflet GIS, Wind Particles, What-If Sandbox, Cleanest Route, and Multi-Agent War Room.
 */

document.addEventListener("DOMContentLoaded", () => {
    // App State
    const state = {
        map: null,
        gridLayerGroup: null,
        stationLayerGroup: null,
        hotspotLayerGroup: null,
        routeLayerGroup: null,
        inspectorMarker: null,
        activeLayer: "aqi",
        gridOpacity: 0.7,
        showStations: true,
        showWind: true,
        inspectedPoint: { lat: 28.6139, lon: 77.2090 },
        stations: [],
        hotspots: [],
        gridData: null,
        weather: { wind_speed_ms: 3.0, wind_direction_deg: 305 },
        
        // Timelapse
        timelapseFrames: [],
        timelapseIndex: 0,
        timelapsePlaying: false,
        timelapseTimer: null,
        
        // Travel Mode
        travelMode: "walking",
    };

    // DOM Elements
    const elements = {
        map: document.getElementById("map"),
        windCanvas: document.getElementById("windCanvas"),
        mapLoader: document.getElementById("mapLoader"),
        weatherRegion: document.getElementById("weatherRegion"),
        weatherTemp: document.getElementById("weatherTemp"),
        weatherWind: document.getElementById("weatherWind"),
        weatherBLH: document.getElementById("weatherBLH"),
        opacitySlider: document.getElementById("opacitySlider"),
        opacityVal: document.getElementById("opacityVal"),
        toggleStations: document.getElementById("toggleStations"),
        toggleWindParticles: document.getElementById("toggleWindParticles"),
        stationMiniList: document.getElementById("stationMiniList"),
        hotspotCount: document.getElementById("hotspotCount"),
        hotspotsList: document.getElementById("hotspotsList"),
        
        // Timelapse
        timelapseBar: document.getElementById("timelapseBar"),
        timelapsePlayBtn: document.getElementById("timelapsePlayBtn"),
        timelapseHourLabel: document.getElementById("timelapseHourLabel"),
        timelapseSlider: document.getElementById("timelapseSlider"),

        // Inspector
        inspLat: document.getElementById("inspLat"),
        inspLon: document.getElementById("inspLon"),
        inspStation: document.getElementById("inspStation"),
        inspAqi: document.getElementById("inspAqi"),
        inspCategory: document.getElementById("inspCategory"),
        inspAdvisory: document.getElementById("inspAdvisory"),
        inspPm25: document.getElementById("inspPm25"),
        inspPm10: document.getElementById("inspPm10"),
        inspNo2: document.getElementById("inspNo2"),
        xaiBarsContainer: document.getElementById("xaiBarsContainer"),
        xaiSummaryText: document.getElementById("xaiSummaryText"),
        modalityWeightsRow: document.getElementById("modalityWeightsRow"),
        btnAskAdvisorAboutPoint: document.getElementById("btnAskAdvisorAboutPoint"),

        // Tabs
        navTabs: document.querySelectorAll(".nav-tab"),
        tabPanes: document.querySelectorAll(".tab-pane"),

        // What-If Sandbox
        sliderTraffic: document.getElementById("sliderTraffic"),
        valTrafficDelta: document.getElementById("valTrafficDelta"),
        sliderWindSpeed: document.getElementById("sliderWindSpeed"),
        valWindSpeed: document.getElementById("valWindSpeed"),
        sliderBLH: document.getElementById("sliderBLH"),
        valBLH: document.getElementById("valBLH"),
        toggleIndustrialCurb: document.getElementById("toggleIndustrialCurb"),
        btnRunSimulation: document.getElementById("btnRunSimulation"),
        btnResetSimulation: document.getElementById("btnResetSimulation"),
        simDeltaPct: document.getElementById("simDeltaPct"),
        simDeltaVerdict: document.getElementById("simDeltaVerdict"),
        simBasePm25: document.getElementById("simBasePm25"),
        simProjectedPm25: document.getElementById("simProjectedPm25"),
        simHotspotDelta: document.getElementById("simHotspotDelta"),

        // Cleanest Route
        routeOriginInput: document.getElementById("routeOriginInput"),
        routeDestInput: document.getElementById("routeDestInput"),
        btnCalculateRoute: document.getElementById("btnCalculateRoute"),
        presetRouteBtns: document.querySelectorAll(".preset-route-btn"),
        routeExposureReduction: document.getElementById("routeExposureReduction"),
        routeSummaryText: document.getElementById("routeSummaryText"),
        fastestDist: document.getElementById("fastestDist"),
        fastestTime: document.getElementById("fastestTime"),
        fastestAvgPm: document.getElementById("fastestAvgPm"),
        fastestDose: document.getElementById("fastestDose"),
        cleanestDist: document.getElementById("cleanestDist"),
        cleanestTime: document.getElementById("cleanestTime"),
        cleanestAvgPm: document.getElementById("cleanestAvgPm"),
        cleanestDose: document.getElementById("cleanestDose"),

        // War Room
        warRoomForm: document.getElementById("warRoomForm"),
        warRoomInput: document.getElementById("warRoomInput"),
        debateStream: document.getElementById("debateStream"),
        warRoomSessionId: document.getElementById("warRoomSessionId"),

        // Benchmarks Table
        benchmarkTableBody: document.getElementById("benchmarkTableBody"),
    };

    // 1. Initialize Map
    function initMap() {
        state.map = L.map("map", {
            center: [28.6139, 77.2090],
            zoom: 11,
            minZoom: 9,
            maxZoom: 16,
        });

        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: '&copy; CartoDB OpenStreetMap contributors',
            subdomains: "abcd",
            maxZoom: 19,
        }).addTo(state.map);

        state.gridLayerGroup = L.layerGroup().addTo(state.map);
        state.hotspotLayerGroup = L.layerGroup().addTo(state.map);
        state.stationLayerGroup = L.layerGroup().addTo(state.map);
        state.routeLayerGroup = L.layerGroup().addTo(state.map);

        state.map.on("click", (e) => {
            const { lat, lng } = e.latlng;
            inspectLocation(lat, lng);
        });

        initWindParticles();
    }

    // 2. Animated Wind Particle Streamlines (Canvas)
    let particles = [];
    function initWindParticles() {
        const canvas = elements.windCanvas;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");

        function resize() {
            canvas.width = elements.map.clientWidth;
            canvas.height = elements.map.clientHeight;
        }
        resize();
        window.addEventListener("resize", resize);
        state.map.on("move", resize);
        state.map.on("zoom", resize);

        // Spawn 80 particles
        for (let i = 0; i < 90; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                age: Math.floor(Math.random() * 80),
                maxAge: 70 + Math.random() * 40,
            });
        }

        function animate() {
            if (!state.showWind) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                requestAnimationFrame(animate);
                return;
            }

            ctx.fillStyle = "rgba(11, 15, 25, 0.12)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const windDeg = (state.weather.wind_direction_deg + 180) % 360; // downwind direction
            const rad = (windDeg * Math.PI) / 180.0;
            const speed = Math.min(6.0, Math.max(1.2, state.weather.wind_speed_ms * 0.7));

            const dx = Math.sin(rad) * speed;
            const dy = -Math.cos(rad) * speed;

            ctx.strokeStyle = "rgba(6, 182, 212, 0.45)";
            ctx.lineWidth = 1.2;

            particles.forEach(p => {
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                p.x += dx;
                p.y += dy;
                p.age++;
                ctx.lineTo(p.x, p.y);
                ctx.stroke();

                if (p.age > p.maxAge || p.x < 0 || p.x > canvas.width || p.y < 0 || p.y > canvas.height) {
                    p.x = Math.random() * canvas.width;
                    p.y = Math.random() * canvas.height;
                    p.age = 0;
                }
            });

            requestAnimationFrame(animate);
        }
        animate();
    }

    // Color Scales
    function getAqiColor(aqi) {
        if (aqi <= 50) return "#00E400";
        if (aqi <= 100) return "#92D050";
        if (aqi <= 200) return "#FFFF00";
        if (aqi <= 300) return "#FF7E00";
        if (aqi <= 400) return "#FF0000";
        return "#7E0023";
    }

    function getPollutantColor(type, val) {
        if (type === "pm25") {
            if (val <= 30) return "#00E400";
            if (val <= 60) return "#92D050";
            if (val <= 90) return "#FFFF00";
            if (val <= 120) return "#FF7E00";
            if (val <= 250) return "#FF0000";
            return "#7E0023";
        }
        if (type === "pm10") {
            if (val <= 50) return "#00E400";
            if (val <= 100) return "#92D050";
            if (val <= 250) return "#FFFF00";
            if (val <= 350) return "#FF7E00";
            if (val <= 430) return "#FF0000";
            return "#7E0023";
        }
        if (type === "no2") {
            if (val <= 40) return "#00E400";
            if (val <= 80) return "#92D050";
            if (val <= 180) return "#FFFF00";
            if (val <= 280) return "#FF7E00";
            return "#FF0000";
        }
        return getAqiColor(val);
    }

    // 3. Load Base System Data
    async function loadData() {
        showLoader(true);
        try {
            const statusRes = await fetch("/api/status");
            const statusData = await statusRes.json();
            if (statusData.region) elements.weatherRegion.textContent = statusData.region.name;

            const stRes = await fetch("/api/stations");
            const stData = await stRes.json();
            state.stations = stData.stations || [];
            renderStations();

            const gridRes = await fetch(`/api/grid?rows=22&cols=22&pollutant=${state.activeLayer}`);
            state.gridData = await gridRes.json();
            state.weather = state.gridData.weather;
            renderWeatherBar(state.weather);
            renderGrid();

            const hsRes = await fetch("/api/hotspots");
            const hsData = await hsRes.json();
            state.hotspots = hsData.hotspots || [];
            renderHotspots();

            await inspectLocation(state.inspectedPoint.lat, state.inspectedPoint.lon);
            loadBenchmarks();
            loadTimelapseFrames();
        } catch (err) {
            console.error("Failed loading data:", err);
        } finally {
            showLoader(false);
        }
    }

    function renderWeatherBar(w) {
        if (!w) return;
        elements.weatherTemp.textContent = `${w.temperature_c}°C / ${w.relative_humidity_pct}%`;
        elements.weatherWind.textContent = `${w.wind_speed_ms} m/s (${w.wind_direction_deg}°)`;
        elements.weatherBLH.textContent = `${w.boundary_layer_height_m} m`;
    }

    // 4. Render Grid Polygons
    function renderGrid(overrideCells = null) {
        state.gridLayerGroup.clearLayers();
        const cells = overrideCells || (state.gridData ? state.gridData.cells : []);
        if (!cells) return;

        cells.forEach(cell => {
            const val = state.activeLayer === "aqi" ? cell.aqi : cell[state.activeLayer];
            const color = getPollutantColor(state.activeLayer, val);

            const polygon = L.rectangle(cell.bounds, {
                color: color,
                weight: 0.5,
                fillColor: color,
                fillOpacity: state.gridOpacity,
            });

            polygon.bindTooltip(`
                <div style="font-family: Inter, sans-serif; font-size: 11px;">
                    <strong>Hyperlocal Cell (${cell.center_lat}, ${cell.center_lon})</strong><br>
                    AQI: <b>${cell.aqi}</b> (${cell.prominent})<br>
                    PM2.5: ${cell.pm25} µg/m³ | NO2: ${cell.no2} µg/m³
                </div>
            `, { sticky: true, opacity: 0.95 });

            polygon.on("click", (e) => {
                L.DomEvent.stopPropagation(e);
                inspectLocation(cell.center_lat, cell.center_lon);
            });

            state.gridLayerGroup.addLayer(polygon);
        });
    }

    // 5. Render Stations
    function renderStations() {
        state.stationLayerGroup.clearLayers();
        elements.stationMiniList.innerHTML = "";
        if (!state.showStations) return;

        state.stations.forEach(st => {
            const aqiColor = getAqiColor(st.aqi);
            const customIcon = L.divIcon({
                className: "custom-station-icon",
                html: `
                    <div style="
                        background-color: ${aqiColor}; width: 22px; height: 22px; border-radius: 50%;
                        border: 2px solid white; box-shadow: 0 0 8px rgba(0,0,0,0.8);
                        display: flex; align-items: center; justify-content: center;
                        color: black; font-weight: 800; font-size: 9px; font-family: 'JetBrains Mono', monospace;
                    ">●</div>
                `,
                iconSize: [22, 22],
                iconAnchor: [11, 11],
            });

            const marker = L.marker([st.lat, st.lon], { icon: customIcon });
            marker.bindPopup(`
                <div style="font-family: Inter, sans-serif; min-width: 170px;">
                    <strong style="font-size: 13px;">${st.station_name}</strong><br>
                    <span style="font-size: 10px; color: #666;">Type: ${st.station_type}</span>
                    <hr style="margin: 6px 0; border: none; border-top: 1px solid #ddd;">
                    <div style="font-size: 11px;">
                        <b>AQI:</b> <span style="color: ${aqiColor}; font-weight: 700;">${st.aqi}</span> (${st.prominent})<br>
                        <b>PM2.5:</b> ${st.pollutants.pm25} µg/m³<br>
                        <b>PM10:</b> ${st.pollutants.pm10} µg/m³<br>
                        <b>NO2:</b> ${st.pollutants.no2} µg/m³
                    </div>
                </div>
            `);
            state.stationLayerGroup.addLayer(marker);

            const row = document.createElement("div");
            row.className = "st-row";
            row.innerHTML = `
                <span class="st-name" title="${st.station_name}">${st.station_name}</span>
                <span class="st-aqi" style="background:${aqiColor}; color:${st.aqi > 200 ? '#fff':'#000'}">${st.aqi}</span>
            `;
            row.addEventListener("click", () => {
                state.map.flyTo([st.lat, st.lon], 13);
                inspectLocation(st.lat, st.lon);
            });
            elements.stationMiniList.appendChild(row);
        });
    }

    // 6. Render Hotspots
    function renderHotspots() {
        state.hotspotLayerGroup.clearLayers();
        elements.hotspotsList.innerHTML = "";
        elements.hotspotCount.textContent = state.hotspots.length;

        state.hotspots.forEach(hs => {
            const radiusMeters = Math.sqrt(hs.approx_area_km2 / Math.PI) * 1000;
            const circle = L.circle([hs.centroid.lat, hs.centroid.lon], {
                radius: radiusMeters,
                color: hs.color,
                fillColor: hs.color,
                fillOpacity: 0.15,
                weight: 2,
                dashArray: "5, 5",
            });

            circle.bindTooltip(`
                <div style="font-family: Inter, sans-serif;">
                    <b>🚨 ${hs.name}</b><br>
                    Peak AQI: <b style="color:${hs.color}">${hs.peak_aqi}</b> (${hs.severity})<br>
                    Area: ~${hs.approx_area_km2} km²
                </div>
            `);
            state.hotspotLayerGroup.addLayer(circle);

            const item = document.createElement("div");
            item.className = "hotspot-item";
            item.innerHTML = `
                <div class="hs-header">
                    <span>${hs.name}</span>
                    <span class="hs-badge" style="background:${hs.color}">${hs.peak_aqi} AQI</span>
                </div>
                <div class="hs-detail">Severity: <strong>${hs.severity}</strong> • Area: ~${hs.approx_area_km2} km²</div>
                <div class="hs-action">${hs.recommended_action}</div>
            `;
            item.addEventListener("click", () => {
                state.map.flyTo([hs.centroid.lat, hs.centroid.lon], 12);
                inspectLocation(hs.centroid.lat, hs.centroid.lon);
            });
            elements.hotspotsList.appendChild(item);
        });
    }

    // 7. Location Inspection & XAI
    async function inspectLocation(lat, lon) {
        state.inspectedPoint = { lat, lon };
        if (state.inspectorMarker) {
            state.inspectorMarker.setLatLng([lat, lon]);
        } else {
            const inspIcon = L.divIcon({
                className: "custom-insp-icon",
                html: `<div style="width: 14px; height: 14px; background: #3B82F6; border: 3px solid white; border-radius: 50%; box-shadow: 0 0 10px rgba(59,130,246,0.9);"></div>`,
                iconSize: [14, 14],
                iconAnchor: [7, 7],
            });
            state.inspectorMarker = L.marker([lat, lon], { icon: inspIcon }).addTo(state.map);
        }

        elements.inspLat.textContent = `${lat.toFixed(4)}° N`;
        elements.inspLon.textContent = `${lon.toFixed(4)}° E`;

        try {
            const predRes = await fetch("/api/predict/point", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lat, lon }),
            });
            const predData = await predRes.json();
            const pred = predData.prediction;

            elements.inspAqi.textContent = pred.aqi;
            elements.inspAqi.style.color = pred.color;
            elements.inspCategory.textContent = pred.category;
            elements.inspCategory.style.backgroundColor = `${pred.color}22`;
            elements.inspCategory.style.color = pred.color;
            elements.inspCategory.style.borderColor = pred.color;
            elements.inspAdvisory.textContent = pred.health_advisory;

            elements.inspPm25.textContent = pred.pollutants.pm25;
            elements.inspPm10.textContent = pred.pollutants.pm10;
            elements.inspNo2.textContent = pred.pollutants.no2;

            const nearest = pred.metadata ? pred.metadata.nearest_station : "N/A";
            const dist = pred.metadata ? pred.metadata.station_distance_km : 0;
            elements.inspStation.textContent = `Nearest: ${nearest} (${dist} km)`;

            const xaiRes = await fetch("/api/explain", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lat, lon, target: 0 }),
            });
            const xaiData = await xaiRes.json();
            renderXAI(xaiData.explanation);
        } catch (err) {
            console.error("Failed inspecting point:", err);
        }
    }

    function renderXAI(explanation) {
        if (!explanation) return;
        elements.xaiSummaryText.textContent = explanation.diagnostic_summary;
        elements.xaiBarsContainer.innerHTML = "";

        const drivers = explanation.sorted_drivers || [];
        drivers.forEach(d => {
            const barItem = document.createElement("div");
            barItem.className = "xai-bar-item";
            barItem.innerHTML = `
                <div class="xai-bar-label"><span>${d.factor}</span><strong>${d.percentage}%</strong></div>
                <div class="xai-progress-track"><div class="xai-progress-fill" style="width: ${d.percentage}%;"></div></div>
            `;
            elements.xaiBarsContainer.appendChild(barItem);
        });

        if (explanation.modality_influence) {
            const m = explanation.modality_influence;
            elements.modalityWeightsRow.innerHTML = `
                <div class="modality-chip">Spatial CNN: <strong>${m['Spatial Raster (CNN)'] || 33}%</strong></div>
                <div class="modality-chip">Temporal LSTM: <strong>${m['Temporal Sequence (LSTM)'] || 36}%</strong></div>
                <div class="modality-chip">Tabular Geo: <strong>${m['Tabular Geo-Weather (Dense)'] || 31}%</strong></div>
            `;
        }
    }

    // 8. 24-Hour Timelapse Player
    async function loadTimelapseFrames() {
        try {
            const res = await fetch("/api/timelapse");
            const data = await res.json();
            state.timelapseFrames = data.frames || [];
        } catch (err) {
            console.error("Failed loading timelapse frames:", err);
        }
    }

    function setTimelapseIndex(idx) {
        if (!state.timelapseFrames || state.timelapseFrames.length === 0) return;
        state.timelapseIndex = idx;
        const frame = state.timelapseFrames[idx];
        elements.timelapseHourLabel.textContent = frame.label;
        elements.timelapseSlider.value = idx;
        renderWeatherBar(frame.weather);
        state.weather = frame.weather;
        // Map 12x12 cells to grid bounds
        const d_lat = (28.88 - 28.40) / 24.0;
        const d_lon = (77.48 - 76.88) / 24.0;
        const mappedCells = frame.cells.map(c => ({
            ...c,
            bounds: [
                [c.lat - d_lat, c.lon - d_lon],
                [c.lat + d_lat, c.lon + d_lon],
            ],
            prominent: "pm25",
            no2: round(c.pm25 * 0.4, 1),
        }));
        renderGrid(mappedCells);
    }

    function toggleTimelapsePlay() {
        state.timelapsePlaying = !state.timelapsePlaying;
        elements.timelapsePlayBtn.textContent = state.timelapsePlaying ? "⏸ Pause" : "▶ Play";
        if (state.timelapsePlaying) {
            state.timelapseTimer = setInterval(() => {
                let next = (state.timelapseIndex + 1) % state.timelapseFrames.length;
                setTimelapseIndex(next);
            }, 1000);
        } else {
            clearInterval(state.timelapseTimer);
        }
    }

    // 9. What-If Scenario Sandbox
    elements.sliderTraffic.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        elements.valTrafficDelta.textContent = `${val > 0 ? '+' : ''}${val}%`;
    });
    elements.sliderWindSpeed.addEventListener("input", (e) => {
        elements.valWindSpeed.textContent = `${parseFloat(e.target.value).toFixed(1)} m/s`;
    });
    elements.sliderBLH.addEventListener("input", (e) => {
        elements.valBLH.textContent = `${e.target.value} m`;
    });

    async function runScenarioSimulation() {
        showLoader(true);
        try {
            const traffic = parseInt(elements.sliderTraffic.value) / 100.0;
            const wind = parseFloat(elements.sliderWindSpeed.value);
            const blh = parseFloat(elements.sliderBLH.value);
            const curb = elements.toggleIndustrialCurb.checked;

            const res = await fetch("/api/simulate/scenario", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    traffic_delta: traffic,
                    wind_speed: wind,
                    blh: blh,
                    industrial_curb: curb,
                }),
            });
            const data = await res.json();
            const sim = data.simulation;

            const pct = sim.impact_summary.percentage_change_pm25;
            elements.simDeltaPct.textContent = `${pct > 0 ? '+' : ''}${pct}%`;
            elements.simDeltaPct.style.color = pct < 0 ? "#10B981" : "#EF4444";
            elements.simDeltaVerdict.textContent = sim.impact_summary.verdict;
            elements.simBasePm25.textContent = `${sim.impact_summary.baseline_mean_pm25} µg/m³`;
            elements.simProjectedPm25.textContent = `${sim.impact_summary.simulated_mean_pm25} µg/m³`;
            elements.simHotspotDelta.textContent = pct < 0 ? "Reduced -2" : "Surged +3";

            // Update Map with simulated cells
            renderGrid(sim.cells);
        } catch (err) {
            console.error("Simulation error:", err);
        } finally {
            showLoader(false);
        }
    }

    function resetSimulation() {
        elements.sliderTraffic.value = 0;
        elements.valTrafficDelta.textContent = "0% (Baseline)";
        elements.sliderWindSpeed.value = 3.0;
        elements.valWindSpeed.textContent = "Current (3.0 m/s)";
        elements.sliderBLH.value = 700;
        elements.valBLH.textContent = "Current (700 m)";
        elements.toggleIndustrialCurb.checked = false;
        elements.simDeltaPct.textContent = "0.0%";
        elements.simDeltaPct.style.color = "#06B6D4";
        renderGrid();
    }

    // 10. Cleanest Route Optimizer
    async function calculateCleanestRoute(originCoords, destCoords) {
        showLoader(true);
        state.routeLayerGroup.clearLayers();
        try {
            const res = await fetch("/api/route/cleanest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    origin: originCoords,
                    destination: destCoords,
                    travel_mode: state.travelMode,
                }),
            });
            const data = await res.json();
            const route = data.route_data;

            elements.routeExposureReduction.textContent = `-${route.comparison.exposure_reduction_pct}%`;
            elements.routeSummaryText.textContent = route.comparison.summary;

            elements.fastestDist.textContent = `${route.fastest_route.distance_km} km`;
            elements.fastestTime.textContent = `${route.fastest_route.duration_minutes} mins`;
            elements.fastestAvgPm.textContent = `${route.fastest_route.average_pm25} µg/m³`;
            elements.fastestDose.textContent = `${route.fastest_route.total_inhaled_dose_ug} µg`;

            elements.cleanestDist.textContent = `${route.cleanest_route.distance_km} km`;
            elements.cleanestTime.textContent = `${route.cleanest_route.duration_minutes} mins`;
            elements.cleanestAvgPm.textContent = `${route.cleanest_route.average_pm25} µg/m³`;
            elements.cleanestDose.textContent = `${route.cleanest_route.total_inhaled_dose_ug} µg`;

            // Draw both routes on map
            // Fastest Route (dashed orange-red)
            const fastLine = L.polyline(route.fastest_route.coordinates, {
                color: "#EF4444",
                weight: 4,
                opacity: 0.85,
                dashArray: "6, 6",
            }).bindTooltip("⚡ Fastest Route (Shortest Distance)");

            // Cleanest Route (solid emerald green)
            const cleanLine = L.polyline(route.cleanest_route.coordinates, {
                color: "#10B981",
                weight: 5,
                opacity: 0.95,
            }).bindTooltip("🌿 Cleanest Route (-" + route.comparison.exposure_reduction_pct + "% Less PM2.5 Inhaled)");

            state.routeLayerGroup.addLayer(fastLine);
            state.routeLayerGroup.addLayer(cleanLine);

            state.map.fitBounds(cleanLine.getBounds(), { padding: [40, 40] });
        } catch (err) {
            console.error("Route calculation error:", err);
        } finally {
            showLoader(false);
        }
    }

    // 11. Multi-Agent War Room
    async function conveneWarRoom(query) {
        if (!query.trim()) return;
        elements.debateStream.innerHTML = `
            <div class="empty-state" style="color: var(--accent-cyan)">
                <div class="spinner" style="margin: 0 auto 0.75rem"></div>
                Convening Dr. Priya Varma, Arjun Mehta, and Director S.K. Nair...
            </div>
        `;
        try {
            const res = await fetch("/api/agents/collaborate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    lat: state.inspectedPoint.lat,
                    lon: state.inspectedPoint.lon,
                }),
            });
            const data = await res.json();
            const session = data.war_room;

            elements.warRoomSessionId.textContent = session.session_id;
            elements.debateStream.innerHTML = "";

            // Render Consensus Banner
            const consensusBanner = document.createElement("div");
            consensusBanner.style.background = "rgba(16, 185, 129, 0.15)";
            consensusBanner.style.border = "1px solid rgba(16, 185, 129, 0.4)";
            consensusBanner.style.padding = "0.75rem 1rem";
            consensusBanner.style.borderRadius = "8px";
            consensusBanner.style.fontSize = "0.82rem";
            consensusBanner.innerHTML = `<strong>🤝 Council Resolution:</strong> ${session.consensus_summary}`;
            elements.debateStream.appendChild(consensusBanner);

            // Render Each Agent's Speech Bubble
            session.debate_transcript.forEach(speech => {
                const bubble = document.createElement("div");
                bubble.className = "agent-speech-bubble";
                bubble.innerHTML = `
                    <div class="bubble-header">
                        <span class="speaker-title">${speech.speaker}</span>
                        <span class="authority-pill">${speech.badge}</span>
                    </div>
                    <div class="speech-content">${speech.content}</div>
                    <span class="speech-time">${speech.timestamp}</span>
                `;
                elements.debateStream.appendChild(bubble);
            });

            elements.debateStream.scrollTop = elements.debateStream.scrollHeight;
        } catch (err) {
            console.error("War room error:", err);
        }
    }

    // 12. Load Benchmarks
    async function loadBenchmarks() {
        try {
            const res = await fetch("/api/benchmarks");
            const data = await res.json();
            const models = data.benchmarks?.models || {};

            elements.benchmarkTableBody.innerHTML = "";
            Object.entries(models).forEach(([name, m]) => {
                const isProposed = name.includes("Multimodal");
                const tr = document.createElement("tr");
                if (isProposed) tr.className = "highlight-row";
                tr.innerHTML = `
                    <td><strong>${name}</strong></td>
                    <td>${m.type}</td>
                    <td><b>${m.rmse}</b></td>
                    <td>${m.mae}</td>
                    <td>${m.r2}</td>
                    <td>${m.pearson_r}</td>
                `;
                elements.benchmarkTableBody.appendChild(tr);
            });
        } catch (err) {
            console.error("Failed loading benchmarks:", err);
        }
    }

    function showLoader(show) {
        elements.mapLoader.style.display = show ? "flex" : "none";
    }

    // Event Listeners
    // Layer switcher
    document.querySelectorAll(".pill-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            if (btn.dataset.layer) {
                document.querySelectorAll(".pill-btn[data-layer]").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                state.activeLayer = btn.dataset.layer;
                renderGrid();
            } else if (btn.dataset.mode) {
                document.querySelectorAll(".pill-btn[data-mode]").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                state.travelMode = btn.dataset.mode;
            }
        });
    });

    elements.opacitySlider.addEventListener("input", (e) => {
        state.gridOpacity = parseInt(e.target.value) / 100.0;
        elements.opacityVal.textContent = `${e.target.value}%`;
        renderGrid();
    });

    elements.toggleStations.addEventListener("change", (e) => {
        state.showStations = e.target.checked;
        renderStations();
    });

    elements.toggleWindParticles.addEventListener("change", (e) => {
        state.showWind = e.target.checked;
    });

    elements.timelapsePlayBtn.addEventListener("click", toggleTimelapsePlay);
    elements.timelapseSlider.addEventListener("input", (e) => {
        setTimelapseIndex(parseInt(e.target.value));
    });

    elements.btnRunSimulation.addEventListener("click", runScenarioSimulation);
    elements.btnResetSimulation.addEventListener("click", resetSimulation);

    elements.btnCalculateRoute.addEventListener("click", () => {
        const orig = elements.routeOriginInput.value.split(",").map(x => parseFloat(x.trim()));
        const dest = elements.routeDestInput.value.split(",").map(x => parseFloat(x.trim()));
        calculateCleanestRoute(orig, dest);
        const mapTab = document.querySelector('.nav-tab[data-tab="map-view"]');
        if (mapTab) mapTab.click();
    });

    elements.presetRouteBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const p = btn.dataset.preset;
            if (p === "1") {
                elements.routeOriginInput.value = "28.6476, 77.3158";
                elements.routeDestInput.value = "28.6139, 77.2090";
            } else if (p === "2") {
                elements.routeOriginInput.value = "28.5710, 77.0719";
                elements.routeDestInput.value = "28.6300, 77.2200";
            } else if (p === "3") {
                elements.routeOriginInput.value = "28.6245, 77.3649";
                elements.routeDestInput.value = "28.5504, 77.2159";
            }
            elements.btnCalculateRoute.click();
        });
    });

    elements.warRoomForm.addEventListener("submit", (e) => {
        e.preventDefault();
        conveneWarRoom(elements.warRoomInput.value);
    });

    elements.btnAskAdvisorAboutPoint.addEventListener("click", () => {
        const tab = document.querySelector('.nav-tab[data-tab="warroom-view"]');
        if (tab) tab.click();
        conveneWarRoom(`Urgent assessment required for coordinates (${state.inspectedPoint.lat.toFixed(3)}, ${state.inspectedPoint.lon.toFixed(3)}) under current weather inversion.`);
    });

    elements.navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            elements.navTabs.forEach(t => t.classList.remove("active"));
            elements.tabPanes.forEach(p => p.classList.remove("active"));
            tab.classList.add("active");
            const target = document.getElementById(tab.dataset.tab);
            if (target) {
                target.classList.add("active");
                if (tab.dataset.tab === "map-view" && state.map) {
                    setTimeout(() => state.map.invalidateSize(), 150);
                }
            }
        });
    });

    function round(val, dec) {
        return Number(Math.round(val + 'e' + dec) + 'e-' + dec);
    }

    initMap();
    loadData();
});
