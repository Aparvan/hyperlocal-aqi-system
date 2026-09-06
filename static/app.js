/**
 * AeroSense AI - Frontend Controller & Geospatial Visualization
 * Leaflet Map, Multimodal Grid Renderer, XAI Attributions, and Agentic Advisor.
 */

document.addEventListener("DOMContentLoaded", () => {
    // App State
    const state = {
        map: null,
        gridLayerGroup: null,
        stationLayerGroup: null,
        hotspotLayerGroup: null,
        inspectorMarker: null,
        activeLayer: "aqi", // 'aqi' | 'pm25' | 'pm10' | 'no2'
        gridOpacity: 0.7,
        showStations: true,
        inspectedPoint: { lat: 28.6139, lon: 77.2090 },
        stations: [],
        hotspots: [],
        gridData: null,
    };

    // DOM Elements
    const elements = {
        map: document.getElementById("map"),
        mapLoader: document.getElementById("mapLoader"),
        weatherRegion: document.getElementById("weatherRegion"),
        weatherTemp: document.getElementById("weatherTemp"),
        weatherWind: document.getElementById("weatherWind"),
        weatherBLH: document.getElementById("weatherBLH"),
        opacitySlider: document.getElementById("opacitySlider"),
        opacityVal: document.getElementById("opacityVal"),
        toggleStations: document.getElementById("toggleStations"),
        stationMiniList: document.getElementById("stationMiniList"),
        hotspotCount: document.getElementById("hotspotCount"),
        hotspotsList: document.getElementById("hotspotsList"),
        
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

        // Tabs & Navigation
        navTabs: document.querySelectorAll(".nav-tab"),
        tabPanes: document.querySelectorAll(".tab-pane"),

        // Benchmarks Table
        benchmarkTableBody: document.getElementById("benchmarkTableBody"),

        // Advisor
        chatMessages: document.getElementById("chatMessages"),
        advisorForm: document.getElementById("advisorForm"),
        advisorInput: document.getElementById("advisorInput"),
        promptChips: document.querySelectorAll(".chip"),
    };

    // 1. Initialize Leaflet Map
    function initMap() {
        state.map = L.map("map", {
            center: [28.6139, 77.2090],
            zoom: 11,
            minZoom: 9,
            maxZoom: 16,
        });

        // Dark Matter / CartoDB Base Tiles for sleek modern GIS contrast
        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: '&copy; <a href="https://carto.com/">CartoDB</a> OpenStreetMap contributors',
            subdomains: "abcd",
            maxZoom: 19,
        }).addTo(state.map);

        state.gridLayerGroup = L.layerGroup().addTo(state.map);
        state.hotspotLayerGroup = L.layerGroup().addTo(state.map);
        state.stationLayerGroup = L.layerGroup().addTo(state.map);

        // Map Click Listener -> Trigger Hyperlocal Point Prediction & XAI
        state.map.on("click", (e) => {
            const { lat, lng } = e.latlng;
            inspectLocation(lat, lng);
        });
    }

    // Color mapper for AQI values (CPCB Standards)
    function getAqiColor(aqi) {
        if (aqi <= 50) return "#00E400";
        if (aqi <= 100) return "#92D050";
        if (aqi <= 200) return "#FFFF00";
        if (aqi <= 300) return "#FF7E00";
        if (aqi <= 400) return "#FF0000";
        return "#7E0023";
    }

    // Color mapper for specific pollutants
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

    // 2. Fetch Core Data
    async function loadData() {
        showLoader(true);
        try {
            // Weather and Status
            const statusRes = await fetch("/api/status");
            const statusData = await statusRes.json();
            if (statusData.region) {
                elements.weatherRegion.textContent = statusData.region.name;
            }

            // Ground Monitoring Stations
            const stRes = await fetch("/api/stations");
            const stData = await stRes.json();
            state.stations = stData.stations || [];
            renderStations();

            // Downscaled Grid
            const gridRes = await fetch(`/api/grid?rows=22&cols=22&pollutant=${state.activeLayer}`);
            state.gridData = await gridRes.json();
            renderWeatherBar(state.gridData.weather);
            renderGrid();

            // Detected Hotspots
            const hsRes = await fetch("/api/hotspots");
            const hsData = await hsRes.json();
            state.hotspots = hsData.hotspots || [];
            renderHotspots();

            // Initial inspection of center coordinate
            await inspectLocation(state.inspectedPoint.lat, state.inspectedPoint.lon);

            // Load Benchmarks table
            loadBenchmarks();

        } catch (err) {
            console.error("Failed loading system data:", err);
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

    // 3. Render Downscaled Grid Polygons
    function renderGrid() {
        state.gridLayerGroup.clearLayers();
        if (!state.gridData || !state.gridData.cells) return;

        state.gridData.cells.forEach(cell => {
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
                    PM2.5: ${cell.pm25} µg/m³ | NO2: ${cell.no2} µg/m³<br>
                    Satellite AOD: ${cell.satellite_aod}
                </div>
            `, { sticky: true, opacity: 0.95 });

            polygon.on("click", (e) => {
                L.DomEvent.stopPropagation(e);
                inspectLocation(cell.center_lat, cell.center_lon);
            });

            state.gridLayerGroup.addLayer(polygon);
        });
    }

    // 4. Render Physical Ground Stations
    function renderStations() {
        state.stationLayerGroup.clearLayers();
        elements.stationMiniList.innerHTML = "";

        if (!state.showStations) return;

        state.stations.forEach(st => {
            const aqiColor = getAqiColor(st.aqi);

            // Custom Marker Icon with pulsating border
            const customIcon = L.divIcon({
                className: "custom-station-icon",
                html: `
                    <div style="
                        background-color: ${aqiColor};
                        width: 22px;
                        height: 22px;
                        border-radius: 50%;
                        border: 2px solid white;
                        box-shadow: 0 0 8px rgba(0,0,0,0.8);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: black;
                        font-weight: 800;
                        font-size: 9px;
                        font-family: 'JetBrains Mono', monospace;
                    ">
                        ●
                    </div>
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

            // Mini List Row in sidebar
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

    // 5. Render Hotspots
    function renderHotspots() {
        state.hotspotLayerGroup.clearLayers();
        elements.hotspotsList.innerHTML = "";
        elements.hotspotCount.textContent = state.hotspots.length;

        if (state.hotspots.length === 0) {
            elements.hotspotsList.innerHTML = `<div class="empty-state">No critical statutory breaches detected.</div>`;
            return;
        }

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

            // Hotspot Card in sidebar
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

    // 6. Inspect Location (Multimodal Inference & XAI)
    async function inspectLocation(lat, lon) {
        state.inspectedPoint = { lat, lon };

        // Move or create inspector marker
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
            // Call point prediction endpoint
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

            // Call XAI explanation endpoint
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

    // 7. Render XAI Attribution Bars
    function renderXAI(explanation) {
        if (!explanation) return;
        elements.xaiSummaryText.textContent = explanation.diagnostic_summary;
        elements.xaiBarsContainer.innerHTML = "";

        const drivers = explanation.sorted_drivers || [];
        drivers.forEach(d => {
            const barItem = document.createElement("div");
            barItem.className = "xai-bar-item";
            barItem.innerHTML = `
                <div class="xai-bar-label">
                    <span>${d.factor}</span>
                    <strong>${d.percentage}%</strong>
                </div>
                <div class="xai-progress-track">
                    <div class="xai-progress-fill" style="width: ${d.percentage}%;"></div>
                </div>
            `;
            elements.xaiBarsContainer.appendChild(barItem);
        });

        // Modality Weights
        if (explanation.modality_influence) {
            const m = explanation.modality_influence;
            elements.modalityWeightsRow.innerHTML = `
                <div class="modality-chip">Spatial CNN: <strong>${m['Spatial Raster (CNN)'] || 33}%</strong></div>
                <div class="modality-chip">Temporal LSTM: <strong>${m['Temporal Sequence (LSTM)'] || 36}%</strong></div>
                <div class="modality-chip">Tabular Geo: <strong>${m['Tabular Geo-Weather (Dense)'] || 31}%</strong></div>
            `;
        }
    }

    // 8. Load Benchmarks
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

    // 9. Agentic Environmental Advisor Q&A
    async function askAdvisor(queryText) {
        if (!queryText.trim()) return;

        // Append User Message
        appendChatMessage("You", queryText, "user-msg");
        elements.advisorInput.value = "";

        // Append Loading Indicator
        const loadingId = "loading-" + Date.now();
        const loadingMsg = document.createElement("div");
        loadingMsg.id = loadingId;
        loadingMsg.className = "chat-msg bot-msg";
        loadingMsg.innerHTML = `<div class="msg-author">AeroSense Advisor</div><div class="msg-body">Synthesizing multimodal models, XAI attributions, and health guidelines...</div>`;
        elements.chatMessages.appendChild(loadingMsg);
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

        try {
            const res = await fetch("/api/advisor/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: queryText,
                    lat: state.inspectedPoint.lat,
                    lon: state.inspectedPoint.lon,
                }),
            });
            const data = await res.json();
            const advisor = data.advisor;

            // Remove loading
            const elem = document.getElementById(loadingId);
            if (elem) elem.remove();

            // Format markdown-like text to HTML
            const formatted = formatMarkdown(advisor.response_markdown);
            appendChatMessage("AeroSense Advisor", formatted, "bot-msg", true);

        } catch (err) {
            const elem = document.getElementById(loadingId);
            if (elem) elem.remove();
            appendChatMessage("AeroSense Advisor", "Apologies, encountered an error processing your query. Please retry.", "bot-msg");
        }
    }

    function appendChatMessage(author, content, typeClass, isHtml = false) {
        const msg = document.createElement("div");
        msg.className = `chat-msg ${typeClass}`;
        if (isHtml) {
            msg.innerHTML = `<div class="msg-author">${author}</div><div class="msg-body">${content}</div>`;
        } else {
            msg.innerHTML = `<div class="msg-author">${author}</div><div class="msg-body"></div>`;
            msg.querySelector(".msg-body").textContent = content;
        }
        elements.chatMessages.appendChild(msg);
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }

    function formatMarkdown(md) {
        if (!md) return "";
        let html = md
            .replace(/### (.*?)\n/g, '<h4 style="margin: 0.6rem 0 0.3rem; font-size: 0.95rem; color: #06B6D4;">$1</h4>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/> ⚠️ (.*?)\n/g, '<div style="background: rgba(239,68,68,0.15); border-left: 3px solid #EF4444; padding: 0.4rem 0.6rem; margin: 0.4rem 0; border-radius: 4px; font-size: 0.8rem;">⚠️ $1</div>')
            .replace(/- (.*?)\n/g, '<li style="margin-left: 1.2rem; margin-bottom: 0.2rem;">$1</li>')
            .replace(/\n\n/g, '<br>');
        return html;
    }

    function showLoader(show) {
        elements.mapLoader.style.display = show ? "flex" : "none";
    }

    // Event Listeners
    // Layer switcher
    document.querySelectorAll(".pill-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".pill-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state.activeLayer = btn.dataset.layer;
            renderGrid();
        });
    });

    // Opacity slider
    elements.opacitySlider.addEventListener("input", (e) => {
        state.gridOpacity = parseInt(e.target.value) / 100.0;
        elements.opacityVal.textContent = `${e.target.value}%`;
        renderGrid();
    });

    // Toggle stations
    elements.toggleStations.addEventListener("change", (e) => {
        state.showStations = e.target.checked;
        renderStations();
    });

    // Navigation Tabs
    elements.navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            elements.navTabs.forEach(t => t.classList.remove("active"));
            elements.tabPanes.forEach(p => p.classList.remove("active"));
            tab.classList.add("active");
            const targetPane = document.getElementById(tab.dataset.tab);
            if (targetPane) {
                targetPane.classList.add("active");
                if (tab.dataset.tab === "map-view" && state.map) {
                    setTimeout(() => state.map.invalidateSize(), 150);
                }
            }
        });
    });

    // Advisor form
    elements.advisorForm.addEventListener("submit", (e) => {
        e.preventDefault();
        askAdvisor(elements.advisorInput.value);
    });

    // Prompt chips
    elements.promptChips.forEach(chip => {
        chip.addEventListener("click", () => {
            askAdvisor(chip.dataset.query);
        });
    });

    // Inspect -> Ask Advisor button
    elements.btnAskAdvisorAboutPoint.addEventListener("click", () => {
        const tab = document.querySelector('.nav-tab[data-tab="advisor-view"]');
        if (tab) tab.click();
        askAdvisor(`Please diagnose air quality at coordinates (${state.inspectedPoint.lat.toFixed(3)}, ${state.inspectedPoint.lon.toFixed(3)}) and explain the health risks.`);
    });

    // Bootstrap
    initMap();
    loadData();
});
