/* ==========================================================================
   GeoVision AI — dashboard application logic.

   Data strategy: on load, try the live FastAPI backend (same origin,
   `/api/...`) with a short timeout. If it's not running -- e.g. this file
   is being previewed as a static export -- fall back to the bundled
   `demo_data.json` snapshot produced by scripts/run_demo.py. Either path
   feeds the exact same render functions below.
   ========================================================================== */

const BAND_KEYS = ["air_quality", "vegetation", "water", "climate"];
const BAND_COLOR = {
  air_quality: "#E2765B",
  vegetation: "#6FCB78",
  water: "#4FB7C9",
  climate: "#8C8FE0",
  ehi: "#E7B84C",
};
const BAND_LABEL = {
  air_quality: "Air quality",
  vegetation: "Vegetation",
  water: "Water",
  climate: "Climate",
  ehi: "Composite EHI",
};
const ARCH_COLOR = { lstm: "#4FB7C9", cnn: "#E2765B", transformer: "#8C8FE0", ensemble: "#E7B84C" };
const ARCH_LABEL = { lstm: "LSTM", cnn: "CNN", transformer: "Transformer", ensemble: "Ensemble" };
const VARIABLE_LABEL = {
  pm25: "PM2.5 (µg/m³)",
  ndvi: "NDVI (index)",
  precipitation_total: "Precipitation (mm)",
  temperature_2m: "Temperature (K)",
  soil_moisture: "Soil moisture (m³/m³)",
};

const state = {
  data: null,
  regionMeta: {},          // region_id -> {name, level, ...}
  levelFilter: "all",
  currentRegion: null,
  forecastVariable: null,
};

/* ---------------------------------------------------------------------- */
/* Data loading                                                            */
/* ---------------------------------------------------------------------- */

async function fetchJSON(url, { timeoutMs = 4000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function tryLoadLiveCore() {
  const regions = await fetchJSON("/api/regions", { timeoutMs: 2500 });
  const ids = regions.map((r) => r.region_id);

  // Show live-loading progress in the hero so the user knows we're fetching
  // real satellite data in the background.
  const modeEl = document.getElementById("dataMode");
  let loadedCount = 0;
  function updateProgress() {
    loadedCount++;
    if (modeEl && modeEl.classList.contains("live-loading")) {
      modeEl.textContent = `live API pipeline — syncing ${loadedCount}/${ids.length} regions…`;
    }
  }

  // Use allSettled so one slow/failed region doesn't abort the entire live load.
  const EHI_TIMEOUT_MS = 180000; // 3 minutes per region
  const settled = await Promise.allSettled(
    ids.map(async (id) => {
      const j = await fetchJSON(`/api/ehi?region_id=${encodeURIComponent(id)}&days=180`, { timeoutMs: EHI_TIMEOUT_MS });
      updateProgress();
      return [id, j.series];
    })
  );

  const successfulEntries = settled
    .filter((r) => r.status === "fulfilled" && Array.isArray(r.value[1]) && r.value[1].length > 0)
    .map((r) => r.value);
  const successfulIds = successfulEntries.map(([id]) => id);
  const ehi_series = Object.fromEntries(successfulEntries);

  const compareIds = successfulIds.length > 0 ? successfulIds : ids;
  const compareJson = await fetchJSON(`/api/compare?region_ids=${compareIds.join(",")}&days=180`, { timeoutMs: 30000 });
  const alertsJson = await fetchJSON("/api/alerts?days=180", { timeoutMs: 30000 });

  return {
    generated_at: new Date().toLocaleDateString("en-CA"),
    regions: regions.filter((r) => successfulIds.length === 0 || successfulIds.includes(r.region_id)),
    ehi_series,
    comparison: compareJson.table || [],
    alerts: alertsJson.alerts || [],
    mode: "live",
  };
}

async function loadDemoData() {
  const data = await fetchJSON("demo_data.json", { timeoutMs: 8000 });
  data.mode = "demo";
  return data;
}

/* ---------------------------------------------------------------------- */
/* Small helpers                                                          */
/* ---------------------------------------------------------------------- */

function clamp01to100(v) {
  return Math.max(0, Math.min(100, v ?? 0));
}

function fmt1(v) {
  return typeof v === "number" ? v.toFixed(1) : "—";
}

function fmt2(v) {
  return typeof v === "number" ? v.toFixed(2) : "—";
}

function latestRecord(series) {
  if (!series || series.length === 0) return null;
  return series[series.length - 1];
}

function bandBarsHTML(record, { size = "sm" } = {}) {
  const cls = size === "lg" ? "rb-bar" : "bandstack-bar";
  const fillCls = size === "lg" ? "rb-fill" : "bandstack-fill";
  return BAND_KEYS.map((key) => {
    const pct = clamp01to100(record ? record[key] : 0);
    return `<div class="${cls}"><div class="${fillCls}" style="width:${pct}%;background:${BAND_COLOR[key]}"></div></div>`;
  }).join("");
}

/* ---------------------------------------------------------------------- */
/* Canvas line chart (shared by the EHI trend chart and the forecast chart)*/
/* ---------------------------------------------------------------------- */

function createLineChart(canvas, tooltipEl) {
  const ctx = canvas.getContext("2d");
  let current = null; // { labels, series: [{key,label,color,values,width,dashed}] }
  let hoverIdx = null;

  function niceHeight() {
    // canvas.getAttribute("height") reflects the *pixel buffer* size (cssHeight * dpr) —
    // NOT the CSS display height. After the first draw() it returns e.g. 520 instead of 260,
    // causing every mousemove to double the canvas height → white screen + infinite scroll.
    // Read style.height (set by draw() to the correct CSS px) when available.
    const fromStyle = parseFloat(canvas.style.height);
    if (fromStyle > 0) return fromStyle;
    return parseInt(canvas.getAttribute("height"), 10) || canvas.clientHeight || 220;
  }


  function layout(cssWidth, cssHeight) {
    return { left: 44, right: 14, top: 14, bottom: 26, width: cssWidth, height: cssHeight };
  }

  function valueRange() {
    let lo = Infinity, hi = -Infinity;
    current.series.forEach((s) => {
      s.values.forEach((v) => {
        if (v == null || Number.isNaN(v)) return;
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      });
    });
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.12;
    return [lo - pad, hi + pad];
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 600;
    const cssHeight = niceHeight();
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
    canvas.style.height = cssHeight + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    if (!current || !current.labels.length) {
      ctx.fillStyle = "#56635E";
      ctx.font = "12px 'IBM Plex Mono', monospace";
      ctx.fillText("no data available", 14, cssHeight / 2);
      return;
    }

    const L = layout(cssWidth, cssHeight);
    const plotW = L.width - L.left - L.right;
    const plotH = L.height - L.top - L.bottom;
    const [lo, hi] = valueRange();
    const n = current.labels.length;
    const xAt = (i) => L.left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
    const yAt = (v) => L.top + plotH - ((v - lo) / (hi - lo)) * plotH;

    // gridlines + y labels
    ctx.strokeStyle = "#1B2620";
    ctx.lineWidth = 1;
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "#56635E";
    // Pick decimal precision based on the magnitude of the value range
    const range = hi - lo;
    const yFmt = range < 1 ? (n) => n.toFixed(2) : range < 10 ? (n) => n.toFixed(1) : (n) => n.toFixed(0);
    const gridLines = 4;
    for (let g = 0; g <= gridLines; g++) {
      const v = lo + ((hi - lo) * g) / gridLines;
      const y = yAt(v);
      ctx.beginPath();
      ctx.moveTo(L.left, y);
      ctx.lineTo(L.left + plotW, y);
      ctx.stroke();
      ctx.fillText(yFmt(v), 4, y + 3);
    }

    // x labels: first / middle / last
    const xIdxs = n > 1 ? [0, Math.floor((n - 1) / 2), n - 1] : [0];
    xIdxs.forEach((i) => {
      ctx.textAlign = n === 1 ? "center" : i === 0 ? "left" : i === n - 1 ? "right" : "center";
      ctx.fillText(current.labels[i], xAt(i), L.top + plotH + 18);
    });
    ctx.textAlign = "left";

    // series lines
    current.series.forEach((s) => {
      ctx.beginPath();
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width || 1.6;
      if (s.dashed) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
      let started = false;
      s.values.forEach((v, i) => {
        if (v == null || Number.isNaN(v)) return;
        const x = xAt(i), y = yAt(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
      });
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // hover guide + dots
    if (hoverIdx != null && hoverIdx >= 0 && hoverIdx < n) {
      const x = xAt(hoverIdx);
      ctx.beginPath();
      ctx.strokeStyle = "#24322D";
      ctx.moveTo(x, L.top);
      ctx.lineTo(x, L.top + plotH);
      ctx.stroke();
      current.series.forEach((s) => {
        const v = s.values[hoverIdx];
        if (v == null || Number.isNaN(v)) return;
        const y = yAt(v);
        ctx.beginPath();
        ctx.fillStyle = s.color;
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    canvas._layout = { L, plotW, plotH, n, xAt };
  }

  function updateTooltip(evt) {
    if (!current || !current.labels.length || !tooltipEl) return;
    const rect = canvas.getBoundingClientRect();
    const mx = evt.clientX - rect.left;
    const { n, xAt } = canvas._layout || {};
    if (!n) return;
    let nearest = 0, best = Infinity;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(xAt(i) - mx);
      if (d < best) { best = d; nearest = i; }
    }
    hoverIdx = nearest;
    draw();

    const rows = current.series
      .map((s) => {
        const v = s.values[nearest];
        if (v == null || Number.isNaN(v)) return "";
        return `<span class="tt-row"><span class="tt-dot" style="background:${s.color}"></span>${s.label}: ${v.toFixed(2)}</span>`;
      })
      .join("");
    tooltipEl.innerHTML = `<span class="tt-date">${current.labels[nearest]}</span>${rows}`;
    tooltipEl.style.left = xAt(nearest) + "px";
    tooltipEl.style.top = "10px";
    tooltipEl.hidden = false;
  }

  canvas.addEventListener("mousemove", updateTooltip);
  canvas.addEventListener("mouseleave", () => {
    hoverIdx = null;
    if (tooltipEl) tooltipEl.hidden = true;
    draw();
  });
  window.addEventListener("resize", () => draw());

  return {
    render(data) {
      current = data;
      hoverIdx = null;
      if (tooltipEl) tooltipEl.hidden = true;
      draw();
    },
  };
}

/* ---------------------------------------------------------------------- */
/* Section renderers                                                       */
/* ---------------------------------------------------------------------- */

function renderScan() {
  const track = document.getElementById("scanTrack");
  const { regions, ehi_series } = state.data;
  track.innerHTML = regions
    .map((r) => {
      const latest = latestRecord(ehi_series[r.region_id]);
      const ehiVal = latest ? clamp01to100(latest.ehi) : 0;
      return `
        <button class="scan-chip" data-region="${r.region_id}" role="tab" aria-selected="false">
          <span class="scan-chip-name">${r.name}</span>
          <div class="bandstack">${bandBarsHTML(latest)}</div>
          <span class="scan-chip-ehi" style="color:${ehiVal >= 60 ? BAND_COLOR.vegetation : ehiVal >= 40 ? BAND_COLOR.ehi : "#E4573D"}">${ehiVal.toFixed(0)}</span>
        </button>`;
    })
    .join("");

  track.querySelectorAll(".scan-chip").forEach((chip) => {
    chip.addEventListener("click", () => selectRegion(chip.dataset.region));
  });
}

function renderLevelFilter() {
  const levels = ["all", ...new Set(Object.values(state.regionMeta).map((r) => r.level))];
  const el = document.getElementById("levelFilter");
  el.innerHTML = levels
    .map(
      (lvl) =>
        `<button class="level-btn ${lvl === state.levelFilter ? "active" : ""}" data-level="${lvl}" role="tab">${lvl}</button>`
    )
    .join("");
  el.querySelectorAll(".level-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.levelFilter = btn.dataset.level;
      renderLevelFilter();
      renderComparison();
    });
  });
}

function renderComparison() {
  const list = document.getElementById("comparisonList");
  let rows = [...(state.data.comparison || [])];
  rows.sort((a, b) => (b.ehi ?? 0) - (a.ehi ?? 0));
  if (state.levelFilter !== "all") {
    rows = rows.filter((r) => (state.regionMeta[r.region_id] || {}).level === state.levelFilter);
  }
  if (!rows.length) {
    list.innerHTML = `<div class="alert-empty">No regions match this filter.</div>`;
    return;
  }
  list.innerHTML = rows
    .map((r) => {
      const meta = state.regionMeta[r.region_id] || {};
      return `
        <button class="comparison-row" data-region="${r.region_id}">
          <span>
            <span class="region-name">${meta.name || r.region_id}</span>
            <span class="region-level">${meta.level || ""}</span>
          </span>
          <span class="row-bands">${BAND_KEYS.map(
            (k) => `<span class="row-band"><span class="row-band-fill" style="width:${clamp01to100(r[k])}%;background:${BAND_COLOR[k]}"></span></span>`
          ).join("")}</span>
          <span class="row-ehi" style="color:${BAND_COLOR.ehi}">${fmt1(r.ehi)}</span>
        </button>`;
    })
    .join("");

  list.querySelectorAll(".comparison-row").forEach((row) => {
    row.addEventListener("click", () => selectRegion(row.dataset.region));
  });
}

function populateRegionSelect() {
  const select = document.getElementById("regionSelect");
  const byLevel = {};
  Object.entries(state.regionMeta).forEach(([id, meta]) => {
    byLevel[meta.level] = byLevel[meta.level] || [];
    byLevel[meta.level].push({ id, name: meta.name });
  });
  select.innerHTML = Object.entries(byLevel)
    .map(
      ([level, items]) =>
        `<optgroup label="${level}">${items
          .map((it) => `<option value="${it.id}">${it.name}</option>`)
          .join("")}</optgroup>`
    )
    .join("");
  select.addEventListener("change", () => selectRegion(select.value));
}

let ehiChart;

function renderRegionDetail(regionId) {
  const meta = state.regionMeta[regionId] || { name: regionId };
  // Show region name in heading; level badge in the gold accent span
  const levelBadge = meta.level ? ` <span id="regionTitleName">${meta.level}</span>` : `<span id="regionTitleName"></span>`;
  document.getElementById("regionTitle").innerHTML = `${meta.name}${levelBadge}`;
  document.getElementById("regionSelect").value = regionId;

  document.querySelectorAll(".scan-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.region === regionId);
  });

  const series = state.data.ehi_series[regionId] || [];
  const latest = latestRecord(series);

  const bandEl = document.getElementById("regionBandstack");
  const items = [{ key: "ehi", label: "Composite EHI" }, ...BAND_KEYS.map((k) => ({ key: k, label: BAND_LABEL[k] }))];
  bandEl.innerHTML = items
    .map((it) => {
      const v = latest ? latest[it.key] : null;
      return `
        <div class="rb-item">
          <span class="rb-label">${it.label}</span>
          <div class="rb-value" style="color:${BAND_COLOR[it.key]}">${fmt1(v)}</div>
          <div class="rb-bar"><div class="rb-fill" style="width:${clamp01to100(v)}%;background:${BAND_COLOR[it.key]}"></div></div>
        </div>`;
    })
    .join("");

  const asof = meta.ndvi_asof;
  document.getElementById("regionNote").textContent =
    `Data through ${latest ? latest.date : "—"}: temperature, rainfall, soil moisture (ERA5/live) and PM2.5 (CAMS) are daily means over land sample points inside the region. ` +
    `Vegetation = MODIS NDVI (5 km windows at up to 5 land points, clear pixels only), latest usable composite ${asof || "n/a"} (published with a 1–2 month lag and dropped when cloudy, so newer days carry that value forward).`;

  const chartData = {
    labels: series.map((r) => r.date),
    series: [
      { key: "ehi", label: "Composite EHI", color: BAND_COLOR.ehi, width: 2.4, values: series.map((r) => r.ehi) },
      ...BAND_KEYS.map((k) => ({
        key: k, label: BAND_LABEL[k], color: BAND_COLOR[k], width: 1.3,
        values: series.map((r) => r[k]),
      })),
    ],
  };
  ehiChart.render(chartData);

  document.getElementById("chartLegend").innerHTML = chartData.series
    .map((s) => `<span class="legend-item"><span class="legend-dot" style="background:${s.color}"></span>${s.label}</span>`)
    .join("");
}

function selectRegion(regionId) {
  if (!regionId || !state.data.ehi_series[regionId]) return;
  state.currentRegion = regionId;
  renderRegionDetail(regionId);
  renderModelComparison(regionId);
}

/* -- model comparison ---------------------------------------------------- */

let forecastChart;

function renderModelComparison(regionId) {
  // Training results are precomputed for every region by scripts/run_demo.py (GPU, real data)
  const entry = (state.data.training_by_region || {})[regionId];
  const report = (entry && entry.report) || {};
  document.getElementById("trainRegionName").textContent =
    (state.regionMeta[regionId] || {}).name || regionId || "—";
  const meta = state.data.training_meta;
  document.getElementById("trainMeta").textContent = meta
    ? `${meta.device.toUpperCase()} · ${meta.mode || "per-region"} model across regions${meta.seeds ? ` (${meta.seeds} seeds)` : ""} · ${meta.train_days}d · ${meta.epochs} epochs · trained ${meta.trained_on}`
    : "—";

  renderClassification(report.classification || {});
  renderForecastTabs(report.forecasting || {});
}

function renderClassification(classification) {
  const el = document.getElementById("classificationCards");
  const entries = Object.entries(classification);
  if (!entries.length) {
    el.innerHTML = `<div class="alert-empty">No classification results in this snapshot.</div>`;
    return;
  }
  const bestModel = entries.reduce((a, b) => (b[1].metrics.kappa > a[1].metrics.kappa ? b : a))[0];

  el.innerHTML = entries
    .map(([modelType, result]) => {
      const m = result.metrics;
      const maxFeat = Math.max(...Object.values(result.top_features || { x: 1 }));
      const featureRows = Object.entries(result.top_features || {})
        .map(
          ([name, val]) => `
        <div class="feature-bar-row">
          <span class="feature-name">${name}</span>
          <div class="feature-track"><div class="feature-fill" style="width:${(val / maxFeat) * 100}%"></div></div>
          <span class="feature-pct">${(val * 100).toFixed(0)}%</span>
        </div>`
        )
        .join("");
      return `
        <div class="model-card ${modelType === bestModel ? "best" : ""}">
          <div class="model-card-top">
            <span class="model-card-name">${modelType === "random_forest" ? "Random Forest" : "XGBoost"}</span>
          </div>
          <div class="model-metrics">
            <span class="metric"><span class="metric-val">${(m.accuracy * 100).toFixed(1)}%</span><span class="metric-label">Accuracy</span></span>
            <span class="metric"><span class="metric-val">${fmt2(m.kappa)}</span><span class="metric-label">Kappa</span></span>
            <span class="metric"><span class="metric-val">${m.roc_auc != null ? fmt2(m.roc_auc) : "—"}</span><span class="metric-label">ROC-AUC</span></span>
            <span class="metric"><span class="metric-val">${m.baseline_accuracy != null ? (m.baseline_accuracy * 100).toFixed(1) + "%" : "—"}</span><span class="metric-label">Naive acc.</span></span>
          </div>
          ${(() => {
            const d = m.baseline_accuracy != null ? (m.accuracy - m.baseline_accuracy) * 100 : null;
            if (d == null) return "";
            if (Math.abs(d) < 0.05) return `<div class="verdict">on par with the naive predictor</div>`;
            return `<div class="verdict ${d > 0 ? "win" : "lose"}">${d > 0 ? "beats" : "below"} naive predictor by ${Math.abs(d).toFixed(1)} pts</div>`;
          })()}
          <div class="feature-bars">${featureRows}</div>
        </div>`;
    })
    .join("");
}

function renderForecastTabs(forecasting) {
  const variables = Object.keys(forecasting);
  const tabsEl = document.getElementById("forecastTabs");
  if (!variables.length) {
    tabsEl.innerHTML = "";
    document.getElementById("forecastMetrics").innerHTML =
      `<div class="alert-empty">No forecasting results in this snapshot.</div>`;
    forecastChart.render({ labels: [], series: [] });
    return;
  }
  if (!state.forecastVariable || !variables.includes(state.forecastVariable)) {
    state.forecastVariable = variables[0];
  }
  tabsEl.innerHTML = variables
    .map(
      (v) =>
        `<button class="forecast-tab ${v === state.forecastVariable ? "active" : ""}" data-var="${v}">${VARIABLE_LABEL[v] || v}</button>`
    )
    .join("");
  tabsEl.querySelectorAll(".forecast-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.forecastVariable = btn.dataset.var;
      renderForecastTabs(forecasting);
    });
  });
  renderForecastDetail(forecasting[state.forecastVariable]);
}

/* Live provider forecast (Open-Meteo NWP/CAMS, fetched in the browser) drawn next to our models */
const _providerCache = {};
async function providerForecast(regionId, variable) {
  const r = state.regionMeta[regionId];
  if (!r || r.lat == null) return null;
  const key = `${regionId}:${variable}`;
  if (_providerCache[key] !== undefined) return _providerCache[key];
  const q = `latitude=${r.lat}&longitude=${r.lon}&timezone=UTC&past_days=7&forecast_days=16`;
  let byDate = {};
  try {
    if (variable === "pm25") {
      const j = await fetchJSON(`https://air-quality-api.open-meteo.com/v1/air-quality?${q.replace("forecast_days=16", "forecast_days=5")}&hourly=pm2_5`, { timeoutMs: 8000 });
      const acc = {};
      j.hourly.time.forEach((t, i) => { const v = j.hourly.pm2_5[i]; if (v != null) (acc[t.slice(0, 10)] ||= []).push(v); });
      Object.entries(acc).forEach(([d, a]) => { byDate[d] = a.reduce((x, y) => x + y, 0) / a.length; });
    } else if (variable === "precipitation_total" || variable === "temperature_2m") {
      const f = variable === "precipitation_total" ? "precipitation_sum" : "temperature_2m_mean";
      const j = await fetchJSON(`https://api.open-meteo.com/v1/forecast?${q}&daily=${f}`, { timeoutMs: 8000 });
      j.daily.time.forEach((d, i) => { const v = j.daily[f][i]; if (v != null) byDate[d] = variable === "temperature_2m" ? v + 273.15 : v; });
    } else {
      return (_providerCache[key] = null);
    }
  } catch (e) {
    return null; // offline / blocked: just no overlay
  }
  return (_providerCache[key] = byDate);
}

async function addProviderOverlay(regionId, variable, labels, chartSeries) {
  const byDate = await providerForecast(regionId, variable);
  if (!byDate || regionId !== state.currentRegion || variable !== state.forecastVariable) return;
  const base = new Date((state.data.training_meta || {}).trained_on + "T00:00:00Z");
  if (isNaN(base)) return;
  const values = labels.map((_, i) => {
    const d = new Date(base.getTime() + (i + 1) * 86400000).toISOString().slice(0, 10);
    return byDate[d] ?? null;
  });
  if (values.every((v) => v == null)) return;
  const all = [...chartSeries, { key: "provider", label: "Open-Meteo live forecast", color: "#F4F1E8", width: 2.2, dashed: false, values }];
  forecastChart.render({ labels, series: all });
  const legendEl = document.getElementById("forecastLegend");
  if (legendEl) legendEl.innerHTML += `<span class="legend-item"><span class="legend-dot" style="background:#F4F1E8"></span>Open-Meteo live forecast (${values.filter((v) => v != null).length}d)</span>`;
}

function renderForecastDetail(archResults) {
  const archs = Object.keys(archResults || {});
  if (!archs.length) {
    forecastChart.render({ labels: [], series: [] });
    document.getElementById("forecastMetrics").innerHTML = `<div class="alert-empty">No forecast results available.</div>`;
    return;
  }
  const bestArch = archs.reduce((a, b) => (archResults[b].metrics.rmse < archResults[a].metrics.rmse ? b : a));
  const horizon = archResults[archs[0]].forecast.length;
  const labels = Array.from({ length: horizon }, (_, i) => `Day ${i + 1}`);

  const chartSeries = archs.map((a) => ({
    key: a,
    label: ARCH_LABEL[a] || a,
    color: ARCH_COLOR[a] || "#999",
    width: a === bestArch ? 2.2 : 1.3,
    dashed: a !== bestArch,
    values: archResults[a].forecast,
  }));

  forecastChart.render({ labels, series: chartSeries });
  addProviderOverlay(state.currentRegion, state.forecastVariable, labels, chartSeries);

  // Populate the forecast chart legend
  const legendEl = document.getElementById("forecastLegend");
  if (legendEl) {
    legendEl.innerHTML = chartSeries
      .map((s) => `<span class="legend-item"><span class="legend-dot" style="background:${s.color}"></span>${s.label}${s.key === bestArch ? " ✓" : ""}</span>`)
      .join("");
  }

  document.getElementById("forecastMetrics").innerHTML = archs
    .map((a) => {
      const m = archResults[a].metrics;
      return `
        <div class="fm-arch ${a === bestArch ? "best" : ""}">
          <div class="fm-arch-name"><span>${ARCH_LABEL[a] || a}</span>${a === bestArch ? "<span>best</span>" : ""}</div>
          <div class="fm-arch-metrics">
            <span class="metric"><span class="metric-val">${fmt2(m.rmse)}</span><span class="metric-label">RMSE</span></span>
            <span class="metric"><span class="metric-val">${fmt2(m.mae)}</span><span class="metric-label">MAE</span></span>
            <span class="metric"><span class="metric-val">${fmt2(m.r2)}</span><span class="metric-label">R²</span></span>
            <span class="metric"><span class="metric-val">${fmt2(m.baseline_rmse)}</span><span class="metric-label">Naive RMSE</span></span>
            <span class="metric"><span class="metric-val">${m.shrink != null ? fmt2(m.shrink) : "—"}</span><span class="metric-label">Model weight</span></span>
          </div>
          ${(() => {
            if (m.baseline_rmse == null || !m.baseline_rmse) return "";
            const imp = (1 - m.rmse / m.baseline_rmse) * 100;
            return `<div class="verdict ${imp > 0 ? "win" : "lose"}">${imp > 0 ? "" : "−"}${Math.abs(imp).toFixed(0)}% ${imp > 0 ? "better" : "worse"} than naive</div>`;
          })()}
        </div>`;
    })
    .join("");
}

/* -- alerts ---------------------------------------------------------------- */

function renderAlerts() {
  const alerts = [...(state.data.alerts || [])];
  const rank = { critical: 0, warning: 1, watch: 2 };
  alerts.sort((a, b) => (rank[a.severity] ?? 9) - (rank[b.severity] ?? 9));

  document.getElementById("alertCount").textContent = `${alerts.length} active`;
  const log = document.getElementById("alertLog");
  if (!alerts.length) {
    log.innerHTML = `<div class="alert-empty">No active alerts across monitored regions.</div>`;
    return;
  }
  log.innerHTML = alerts
    .map(
      (a) => `
        <div class="alert-row">
          <span class="alert-sev alert-sev-${a.severity}">${a.severity}</span>
          <span class="alert-region">${a.region_name || a.region_id}</span>
          <span class="alert-message">${a.message}</span>
          <span class="alert-date">${a.date}</span>
        </div>`
    )
    .join("");
}

/* -- pipeline strip (authored, static content describing the methodology) - */

const PIPELINE_STEPS = [
  { title: "Near real-time ingestion", desc: "Real data, no keys: Open-Meteo (ERA5 temperature & precipitation, ERA5-Land soil moisture, CAMS PM2.5 & NO₂) and NASA MODIS NDVI, pulled per region." },
  { title: "Preprocessing", desc: "Physical-range validation and gap-fill interpolation clean every time series; MODIS NDVI keeps only clear-pixel composites." },
  { title: "Feature engineering", desc: "7/30-day rolling statistics, anomaly scores, season and PCA compression build the model-ready matrix." },
  { title: "Parallel ML/DL training", desc: "Random Forest and XGBoost classifiers; LSTM, CNN and Transformer forecasters plus their ensemble, trained per region on GPU." },
  { title: "Validation", desc: "Chronological hold-out; RMSE, MAE, R² vs a naive repeat-last-value forecast; Accuracy, Kappa, ROC-AUC vs a naive predictor." },
  { title: "Index fusion", desc: "Air quality, vegetation, water and climate sub-indices combine into one weighted Environmental Health Index." },
  { title: "Interactive dashboard", desc: "This interface: country/state/city/continent comparison, trends, forecasts and model diagnostics." },
  { title: "Deployment & alerting", desc: "Threshold and rapid-decline detectors watch every region and raise watch / warning / critical alerts." },
];

function renderPipelineSteps() {
  document.getElementById("pipelineSteps").innerHTML = PIPELINE_STEPS.map(
    (s, i) => `
      <li class="pipeline-step">
        <div class="pipeline-step-num">${String(i + 1).padStart(2, "0")}</div>
        <div class="pipeline-step-title">${s.title}</div>
        <div class="pipeline-step-desc">${s.desc}</div>
      </li>`
  ).join("");
}

/* ---------------------------------------------------------------------- */
/* Init                                                                     */
/* ---------------------------------------------------------------------- */

async function init() {
  ehiChart = createLineChart(document.getElementById("ehiChart"), document.getElementById("chartTooltip"));
  forecastChart = createLineChart(document.getElementById("forecastChart"), document.getElementById("forecastTooltip"));

  const modeEl = document.getElementById("dataMode");
  const genEl = document.getElementById("generatedAt");

  // Step 1: Render instantly using cached/bundled snapshot (< 50ms)
  let initialData = null;
  try {
    initialData = await loadDemoData();
    state.data = initialData;
    state.regionMeta = Object.fromEntries(initialData.regions.map((r) => [r.region_id, r]));

    if (genEl) genEl.textContent = `snapshot generated ${initialData.generated_at}`;
    if (modeEl) {
      modeEl.classList.add("live-loading");
      modeEl.textContent = "connecting to live satellite pipeline…";
    }

    renderScan();
    renderLevelFilter();
    renderComparison();
    populateRegionSelect();
    renderPipelineSteps();
    renderAlerts();

    const preferredRegion = state.regionMeta["IN"] ? "IN" : initialData.regions[0].region_id;
    selectRegion(preferredRegion);
  } catch (err) {
    console.warn("Failed to load initial snapshot:", err);
  }

  // Step 2: Asynchronously revalidate with live API in the background
  try {
    const liveCore = await tryLoadLiveCore();
    if (liveCore && liveCore.regions && liveCore.regions.length > 0) {
      state.data = {
        ...state.data,
        ...liveCore,
        mode: "live",
      };
      state.regionMeta = Object.fromEntries(liveCore.regions.map((r) => [r.region_id, { ...(state.regionMeta[r.region_id] || {}), ...r }]));

      if (genEl) genEl.textContent = `live snapshot ${liveCore.generated_at}`;
      if (modeEl) {
        modeEl.classList.remove("live-loading");
        modeEl.textContent = "live API pipeline (synced)";
      }

      renderScan();
      renderComparison();
      renderAlerts();
      populateRegionSelect();
      if (state.currentRegion && state.regionMeta[state.currentRegion]) {
        selectRegion(state.currentRegion);
      }

      return;
    }
  } catch (err) {
    console.info("Live API unavailable, continuing with demo snapshot:", err.message);
  }

  // If live API unavailable, finalize demo mode badge
  if (modeEl) {
    modeEl.classList.remove("live-loading");
    modeEl.textContent = "static demo snapshot";
  }
}

init();
