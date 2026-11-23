# GeoVision AI — Complete Technical Report

Everything about the project as it exists now: architecture, every module, the data and where it comes from, preprocessing, features, the Environmental Health Index, all models and how they are trained and validated, the API, the caching layer, the dashboard, tests, results, the audits that shaped the current design, performance, security, known limitations, and how to run and reproduce it.

*Snapshot date: 2026-10-01. All numbers in §15 are generated directly from `dashboard/demo_data.json` (the artefact the dashboard displays), so they match what you see on screen. Companion document: `GeoVision_ML_DL_Data_Report.md` (the audit trail of the investigation, written while the project was being corrected).*

---

## Table of contents
1. [Executive summary](#1-executive-summary)
2. [Repository layout and file inventory](#2-repository-layout-and-file-inventory)
3. [System architecture and data flow](#3-system-architecture-and-data-flow)
4. [Configuration and the region registry](#4-configuration-and-the-region-registry)
5. [Data layer (ingestion)](#5-data-layer-ingestion)
6. [Preprocessing](#6-preprocessing)
7. [Feature engineering](#7-feature-engineering)
8. [Environmental Health Index (EHI)](#8-environmental-health-index-ehi)
9. [Machine-learning and deep-learning models](#9-machine-learning-and-deep-learning-models)
10. [Alerting](#10-alerting)
11. [Service layer, caching and REST API](#11-service-layer-caching-and-rest-api)
12. [Dashboard (frontend)](#12-dashboard-frontend)
13. [Scripts](#13-scripts)
14. [Tests](#14-tests)
15. [Results (generated from the snapshot)](#15-results-generated-from-the-snapshot)
16. [Data validation: are the numbers true?](#16-data-validation-are-the-numbers-true)
17. [Performance and operations](#17-performance-and-operations)
18. [Security and robustness](#18-security-and-robustness)
19. [History of problems found and fixed; open issues](#19-history-of-problems-found-and-fixed-open-issues)
20. [How to run, reproduce and troubleshoot](#20-how-to-run-reproduce-and-troubleshoot)
21. [Appendix: formulae and glossary](#21-appendix-formulae-and-glossary)

---

## 1. Executive summary

**What it is.** GeoVision AI is a near-real-time environmental intelligence platform. For 13 regions (6 countries, 2 states, 2 cities, 3 continents) it ingests **real, free, keyless** earth-observation and climate data, cleans it, engineers features, fuses four sub-indices into one 0–100 **Environmental Health Index (EHI)**, raises threshold and rapid-decline **alerts**, trains **machine-learning classifiers** (Random Forest, XGBoost) and **deep-learning forecasters** (LSTM, 1-D CNN, Transformer, plus an ensemble) and presents everything in an interactive web dashboard served by a FastAPI backend.

**Data (all real).**

| Variable | Source | What it is |
|---|---|---|
| Temperature, precipitation | Open-Meteo archive → ERA5 reanalysis (ECMWF) | daily mean 2 m temperature, daily rainfall |
| Soil moisture | Open-Meteo archive → ERA5-Land | volumetric soil water 0–100 cm (root zone) |
| PM2.5, NO₂ | Open-Meteo air-quality API → CAMS (Copernicus) reanalysis | hourly pollutant concentration, averaged to daily |
| NDVI | NASA ORNL DAAC → MODIS MOD13Q1 | 250 m, 16-day vegetation composites, clear-pixel filtered |
| Last ~2 days of weather | Open-Meteo forecast API (`past_days`) | live analysis, fills the archive's ~2-day lag |

No API keys. No synthetic fallback: if a source is unreachable the request fails rather than showing fabricated numbers (`synthetic.py` exists only as a unit-test fixture).

**Models.** One model per variable is trained on the history of **all regions pooled together** (about 13× the data of a per-region model), each region normalised separately and evaluated on its own untouched chronological hold-out. Forecasters predict 14 days ahead, are ensembles over 3 architectures × 3 random seeds, and include a shrinkage safeguard that pulls forecasts toward the naive "repeat last value" baseline when the network adds nothing.

**Honest headline results** (details in §15): the forecasters beat the naive baseline for PM2.5, rainfall and temperature in most regions; they have **no skill** for soil moisture and NDVI (slow signals dominated by persistence); the classifiers are modestly above the naive "condition stays the same" predictor on average but below it in a few regions. The dashboard shows the naive baseline next to every model so this is visible.

**Environment used.** Python 3.13, PyTorch 2.6 (CUDA 12.4), scikit-learn 1.7, XGBoost 3.1, pandas 2.3, FastAPI 0.139, NVIDIA RTX 4050 Laptop GPU (6 GB), Windows 11.

---

## 2. Repository layout and file inventory

```
.
├── README.md                          quick start, API table, data sources
├── GeoVision_Project_Technical_Report.md   (this file)
├── GeoVision_ML_DL_Data_Report.md          audit trail of the investigation and fixes
├── requirements.txt / pytest.ini / .env.example / .gitignore
├── usermanual.md, caching.md, challenges.md, changelog.md, project_summary.md
│                                      written for an earlier version; marked OUTDATED
├── geovision/                         the Python package (≈2 700 lines)
│   ├── config.py                      settings, thresholds, region registry
│   ├── ingestion/
│   │   ├── base.py                    Observation, BaseIngestionClient, get_json (HTTP+disk cache+retry), sample_points
│   │   ├── open_meteo_client.py       ERA5 / ERA5-Land / CAMS / live feed, multi-point land average
│   │   ├── modis_ndvi_client.py       MODIS NDVI, clear-pixel filter, multi-point land average
│   │   ├── pipeline.py                IngestionPipeline: runs both clients, tidy DataFrame
│   │   └── synthetic.py               seeded generator — TEST FIXTURE ONLY
│   ├── preprocessing/
│   │   ├── cloud_mask.py              range validation + daily gap-fill (name is historical)
│   │   └── normalize.py               min_max_normalize (used by the EHI)
│   ├── features/spectral_indices.py   pivot, rolling stats, anomalies, PCA
│   ├── index/health_index.py          sub-indices, composite EHI, region comparison
│   ├── alerting/alerts.py             threshold + rapid-decline alerts
│   ├── models/
│   │   ├── classifiers.py             condition labels, RF/XGBoost, per-region and pooled
│   │   ├── forecasting.py             LSTM/CNN/Transformer, shrinkage, per-region and pooled
│   │   ├── evaluate.py                RMSE/MAE/R², Accuracy/Kappa/ROC-AUC (+ naive baselines)
│   │   └── train.py                   FORECAST_CONFIG, run_training, run_pooled_training, best_model_summary
│   └── api/
│       ├── main.py                    FastAPI app, system/cache endpoints, /cache-status HTML page, static mount
│       ├── service.py                 orchestration + two-tier cache + JSON sanitising
│       └── routers/{environment,alerts,models}.py   /api/regions, /ehi, /compare, /alerts, /train
├── dashboard/                         static front end (vanilla HTML/CSS/JS, canvas charts)
│   ├── index.html · styles.css (≈490 lines) · app.js (≈780 lines)
│   └── demo_data.json                 precomputed snapshot (≈0.9 MB) incl. all models' results
├── scripts/
│   ├── run_demo.py                    fetch all regions → pooled training → write demo_data.json
│   └── test_live.py                   smoke test against a running server
└── tests/                             32 pytest tests (alerts, api, models, pipeline)
```

Line counts (code only): ingestion ≈ 430, preprocessing ≈ 125, features ≈ 90, index ≈ 125, alerting ≈ 135, models ≈ 650, api ≈ 850 (of which ≈ 460 is the embedded HTML of the cache-status page), dashboard ≈ 1 260, scripts ≈ 220, tests ≈ 350.

Generated at run time (git-ignored): `data/cache/*.pkl` (computed results), `data/cache/http/*.json` (raw API responses).

---

## 3. System architecture and data flow

```
 ┌─────────────── free public APIs ────────────────┐
 │ Open-Meteo archive (ERA5, ERA5-Land)            │
 │ Open-Meteo air-quality (CAMS)                   │
 │ Open-Meteo forecast (live analysis, past_days)  │
 │ NASA ORNL DAAC  MODIS MOD13Q1 NDVI + reliability│
 └───────────────────────┬─────────────────────────┘
                         │  get_json(): disk cache (TTL), 429 back-off, fail-fast on 4xx
                         ▼
   OpenMeteoClient / ModisNdviClient      ← 5 sample points per region (centroid + 2×2 grid in bbox)
        land mask · multi-point mean · unit conversion
                         ▼
   IngestionPipeline  →  tidy DataFrame (region, date, source, variable, value, unit, flag, meta)
                         ▼
   Preprocessing: range validation → daily resample → time interpolation (+ffill/bfill)
                         ▼
   Features: pivot → 7/30-day rolling mean/std → anomalies → PCA(3)      (41 columns)
                         ▼
   EHI: 4 sub-indices (0–100) → weighted composite                       (46 columns)
        ├──▶ Alerts (threshold, rapid decline)
        └──▶ Models  (per request: 1 region │ offline: pooled across all regions on GPU)
                         ▼
   Service layer (memory + pickle cache) → FastAPI REST  ──▶  Dashboard (canvas charts)
                                              ▲                  ▲
                                              └── demo_data.json (precomputed snapshot incl. models)
```

Two execution modes share the same code:

* **Live API mode** — `uvicorn geovision.api.main:app`: each endpoint runs the pipeline (cached 60 min) for the requested region(s). `/api/train` trains **one region** on demand.
* **Snapshot mode** — `python scripts/run_demo.py` runs the pipeline for all regions, trains the **pooled** models and writes `dashboard/demo_data.json`. The dashboard always paints this snapshot first (instantly, even offline), then tries to refresh EHI/alerts from the live API. The model panel always shows the snapshot's pooled results.

---

## 4. Configuration and the region registry

`geovision/config.py` (no secrets, no API keys):

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `ingestion_interval_minutes` | `INGESTION_INTERVAL_MIN` | 60 | TTL of the in-memory/disk result cache |
| `forecast_horizon_days` | `FORECAST_HORIZON_DAYS` | 14 | default forecast horizon (API `/train`) |
| `ehi_weights` | – | air 0.30, vegetation 0.25, water 0.25, climate 0.20 | composite weights (sum 1) |
| `alert_thresholds` | – | air 40, vegetation 35, water 35, climate 40, composite 45 | alert floors on the 0–100 scale |
| `CACHE_DIR` | – | `<repo>/data/cache` | results (`*.pkl`) and raw API responses (`http/*.json`) |

**Region registry** (13 regions; `bbox = [min_lon, min_lat, max_lon, max_lat]`, `lat/lon` = representative centroid):

| ID | Name | Level | Centroid (lat, lon) | Bounding box |
|---|---|---|---|---|
| IN | India | country | 22.35, 78.66 | 68.1, 6.5, 97.4, 35.5 |
| US | United States | country | 39.83, −98.58 | −124.8, 24.4, −66.9, 49.4 |
| BR | Brazil | country | −14.24, −51.93 | −73.9, −33.7, −34.8, 5.3 |
| AU | Australia | country | −25.27, 133.78 | 113.3, −43.6, 153.6, −10.7 |
| KE | Kenya | country | −0.02, 37.91 | 33.9, −4.7, 41.9, 4.6 |
| ID | Indonesia | country | −0.79, 113.92 | 95.0, −11.0, 141.0, 6.0 |
| MH-IN | Maharashtra, India | state | 19.75, 75.71 | 72.6, 15.6, 80.9, 22.1 |
| CA-US | California, USA | state | 36.78, −119.42 | −124.4, 32.5, −114.1, 42.0 |
| MUM-IN | Mumbai, India | city | 19.076, 72.877 | 72.77, 18.89, 72.98, 19.27 |
| SP-BR | São Paulo, Brazil | city | −23.55, −46.63 | −46.83, −23.73, −46.36, −23.36 |
| AS | Asia | continent | 34.0, 100.0 | 26.0, −11.0, 180.0, 77.0 |
| AF | Africa | continent | 1.0, 20.0 | −17.0, −35.0, 51.0, 37.0 |
| SA | South America | continent | −15.0, −60.0 | −81.0, −56.0, −34.0, 13.0 |

Adding a region = one new entry; everything else (API, dashboard, training) picks it up automatically.

---

## 5. Data layer (ingestion)

### 5.1 The `Observation` contract
Every client returns `Observation(region_id, date, source, variable, value, unit, quality_flag="ok", meta=None)` (frozen dataclass, `ingestion/base.py`). The pipeline concatenates observations, builds a DataFrame, converts dates to timestamps, drops duplicates on `(region_id, date, source, variable)` and sorts by `(variable, date)`.

### 5.2 Spatial sampling: why and how
A single centroid is a poor proxy for a country (the Australian centroid is desert; Mumbai's is a city block). `sample_points(region, n=2)` returns **5 points**: the centroid plus a 2×2 grid at ¼ and ¾ of the bounding box in each direction. Each client then keeps only **land** points and averages them:

* **Open-Meteo:** all 5 points are requested in one call (comma-separated coordinates). ERA5-Land soil moisture is *exactly 0* over the ocean, so points whose soil-moisture series averages ≤ 0.001 are treated as ocean and dropped (if none remain, the centroid is used).
* **MODIS:** each candidate point is probed with a single-pixel request for the latest composite; the no-data fill value (−3000) means water/no tile (the service answers HTTP 400 for open ocean, treated as "not land"). Up to 5 land points are kept (centroid first).

Averages are unweighted means over the surviving points. This is still a *sample* (≤5 points), not every pixel — coarse for continents (§19).

### 5.3 Open-Meteo client (`open_meteo_client.py`)
Three endpoints, keyless, JSON:

| Endpoint | Variables requested | Used for |
|---|---|---|
| `archive-api.open-meteo.com/v1/archive` | daily `temperature_2m_mean`, `precipitation_sum`; hourly `soil_moisture_0_to_100cm` | temperature, precipitation, soil moisture (history up to today−2 days) |
| `air-quality-api.open-meteo.com/v1/air-quality` | hourly `pm2_5`, `nitrogen_dioxide` | PM2.5, NO₂ (history through today) |
| `api.open-meteo.com/v1/forecast` (`past_days=5, forecast_days=1`) | same weather variables | **live fill** of the days after the archive end; never overwrites archive values |

Processing: hourly series → daily means (`_daily_mean`, skipping nulls); temperature °C → K (+273.15); per-point daily series → mean over land points. Units: K, mm, m³/m³, µg/m³. The archive end date is clamped to `today − 2 days` (the ERA5 archive rejects later dates with HTTP 400).

*Why 0–100 cm soil moisture:* the 0–7 cm surface layer is extremely volatile (Mumbai 0.26 → 0.07 in one week at monsoon end while the root zone stayed ≈ 0.26 and NASA POWER's normalised wetness read 0.84–0.86), which produced a misleading "water index 6.6".

### 5.4 MODIS NDVI client (`modis_ndvi_client.py`)
Source: NASA ORNL DAAC MODIS web service, product **MOD13Q1** (Terra, 250 m, 16-day composites), band `250m_16_days_NDVI` (scale 0.0001) and `250m_16_days_pixel_reliability`.

1. `GET /dates` for the centroid → composite calendar; keep composites from `start − 200 days` to `end` (the look-back lets a cloudy window inherit the latest clear composite).
2. For each land point and each chunk of ≤ 10 composites (service limit): request a **5×5 km window** (`kmAboveBelow = kmLeftRight = 2`) for NDVI and for pixel reliability.
3. Keep pixels with reliability 0 (good) or 1 (marginal) and value > −3000; discard the composite if fewer than 25 % of pixels are clear.
4. Composite value = mean of clear pixels × 0.0001; region value for a date = mean over the points that have a clear composite that date.
5. Composites dated before the window: only the latest is kept and re-dated to the window start; `meta = {"points": n, "composite": "YYYY-MM-DD"}` records provenance (used for the "NDVI as of" note).

*Why the reliability filter matters:* without it, monsoon cloud contaminated the composites (India July–August: 289 of 289 pixels flagged cloudy; NDVI fell to 0.09 instead of ≈ 0.6). Composites are published with a 1–2 month lag, so the newest weeks are carried forward by the gap-fill; cloudy seasons extend this (India's latest usable composite is 26 June).

### 5.5 `get_json` — HTTP, caching, retry (`ingestion/base.py`)
* **Disk cache:** key = SHA-1 of URL + sorted params; files `data/cache/http/<key>.json`; per-call TTL (Open-Meteo archive 6 h, air quality and live feed 1 h, MODIS 12 h). Atomic writes (`.tmp` → rename). Re-running after a crash or reloading the dashboard costs nothing.
* **429 handling:** HTTP 429 ("Minutely API request limit exceeded") → wait 25 s × attempt and retry (up to 5 tries).
* **4xx other than 429:** raise `IngestionUnavailableError` immediately (retrying cannot help).
* **Other failures:** retry after 2 s; after the last try raise `IngestionUnavailableError`.
* **Never returns fabricated data.**

### 5.6 Free-tier limits (important operationally)
Open-Meteo's free tier is ≈ 600 calls/minute, 5 000/hour, 10 000/day, with long or multi-location requests counted as several calls. A full 13-region run issues ≈ 160 weighted calls per region; the HTTP cache, 5-point sampling, random start-up staggering in `run_demo.py` and 429 back-off keep a run within limits. Running many full rebuilds in an hour can still hit the daily/hourly limit — the cache is the mitigation.

### 5.7 Synthetic generator (`synthetic.py`) — test fixture only
Seeded (SHA-256 of region id) annual sinusoids per variable with a repeating 24-day noise pattern; produces all six core variables including `soil_moisture`. Used by unit tests that must run offline. **The pipeline never uses it.**

### 5.8 Data-quality properties
| Aspect | Property |
|---|---|
| Time span in snapshot | 731 daily rows per region (2024-10-01 → 2026-10-01) |
| Latency | weather/air: today; NDVI: up to ~3 months in cloudy regions (dashboard shows the date) |
| Spatial support | mean of ≤ 5 land points; NDVI windows 5×5 km |
| PM2.5 / NO₂ | modelled (CAMS), not station measurements |
| Precipitation | reanalysis point/area rainfall: noisy day to day |
| Missing data | none after gap-fill, but gap-filled spans are interpolated/carried forward (flag `interpolated`) |

---

## 6. Preprocessing (`preprocessing/cloud_mask.py`, `normalize.py`)

The module name is historical; there is **no pixel cloud masking here** (cloud screening happens in the MODIS client).

1. **`flag_out_of_range`:** values outside physical ranges become NaN and are flagged `invalid`: `ndvi ∈ [−1, 1]`, `soil_moisture ∈ [0, 1]`, `temperature_2m ∈ [200, 330] K`, `precipitation_total ∈ [0, 500] mm`, `pm25 ∈ [0, 1000]`, `no2 ∈ [0, 500]`.
2. **`fill_gaps`:** for each `(region, source, variable)` series: reindex to a complete **daily** grid from the earliest to the latest date in the frame, `interpolate(method="time", limit_direction="both")`, then `bfill()`/`ffill()`. Sparse 16-day NDVI composites become a daily series (piecewise-linear between composites, flat after the last one). Quality flags become `interpolated` or `ok`.
3. `run_preprocessing` = validate → fill → sort. `normalize.min_max_normalize(series, lo, hi, invert)` clips to `[lo, hi]` and scales to 0–100 (inverted for "higher is worse"); it is used by the EHI. (`kelvin_to_celsius` exists but is unused.)

---

## 7. Feature engineering (`features/spectral_indices.py`)

The module name is historical (it no longer computes spectral indices; NDVI arrives ready-made).

* **`to_wide`:** pivot tidy rows to one row per `(region_id, date)`, columns = variables (mean if several sources).
* **`add_rolling_features`:** per core variable `ndvi, soil_moisture, temperature_2m, precipitation_total, pm25, no2`: 7- and 30-day rolling **mean** (`min_periods=1`) and **std** (`min_periods=2`, NaN→0), plus `<var>_anomaly = value − 30-day rolling mean`. (6 × (4 + 1) = 30 columns.)
* **`apply_pca`:** standardise the 6 core variables (NaN → column mean), PCA with 3 components (`random_state=42`) → `pca_1..3`. Fitted on the region's own history each time.
* **Result:** 41 columns = `date`, `region_id`, 6 raw + 30 rolling/anomaly + 3 PCA. After EHI: 46 columns (`air_quality, vegetation, water, climate, ehi`).

---

## 8. Environmental Health Index (EHI) (`index/health_index.py`)

Four sub-indices on a common 0 (poor) – 100 (excellent) scale, then a weighted composite.

| Sub-index | Components (weight within sub-index) | Range mapped to 0–100 | Direction |
|---|---|---|---|
| **air_quality** | PM2.5 (0.6), NO₂ (0.4) | PM2.5 0–150 µg/m³, NO₂ 0–200 µg/m³ | inverted (cleaner = higher) |
| **vegetation** | NDVI (1.0) | −0.1 → 0.9 | higher = better |
| **water** | root-zone soil moisture 0–100 cm (1.0) | 0.05 → 0.45 m³/m³ | higher = better |
| **climate** | 30-day mean rainfall (0.5), temperature comfort (0.5) | rain 0–15 mm/day; comfort = \|T − 295 K\| over 0–15 K | rain higher = better; distance from 22 °C inverted |

```
EHI = 0.30·air_quality + 0.25·vegetation + 0.25·water + 0.20·climate
```

* If a sub-index is NaN on a row, the remaining weights are renormalised to 1.
* Rainfall uses the **30-day rolling mean** (not single-day rainfall), so a dry day is not a drought.
* The bounds and weights are hand-chosen engineering assumptions, **not** calibrated against ground truth. The absolute soil-moisture range (0.05–0.45) ignores soil type (sandy soils are "dry" at values that are normal for them); NASA's porosity-normalised wetness would be an alternative.
* `latest_snapshot` returns the newest row per region; `compare_regions` builds the ranked comparison table.
* **Verified:** a hand recomputation from raw values (India, 2026-10-01) matched the code exactly (air 88.49, vegetation 71.64, water 76.69, climate 45.17, EHI 72.66).

Worked example — Australia, today: air 98.3, vegetation 58.4, water 40.6, climate 36.9 → EHI = 0.30·98.3 + 0.25·58.4 + 0.25·40.6 + 0.20·36.9 ≈ 61.6.

---

## 9. Machine-learning and deep-learning models

### 9.1 Overview
| Task | Models | Input | Output |
|---|---|---|---|
| **Condition classification, 7 days ahead** | Random Forest, XGBoost | 18 features on day t (12 environmental + 6 "today's condition" indicators) | class of day t+7: `degraded` / `stressed` / `healthy` |
| **Forecasting, 14 days ahead** (5 variables) | LSTM, 1-D CNN, Transformer, **ensemble** | last 30 days of the variable (+ drivers for NDVI) | the next 14 daily values |

Everything is *validated chronologically* and compared with a **naive baseline** (classification: "condition stays the same"; forecasting: "repeat the last value").

### 9.2 Classification (`models/classifiers.py`)

**Labels (proxy, rule-derived).** Real labelled data does not exist. `derive_condition_labels` computes, per region, a score from within-region z-scores:
`score = z(NDVI) − z(PM2.5) + 0.5·clip(z(precipitation), −1.5, 1.5)`, then `degraded` if score < −0.5, `stressed` if −0.5…0.5, `healthy` if > 0.5. Labels are therefore **region-relative** (about a third of days are "healthy" in every region regardless of absolute quality).

**Task (no leakage).** Features on day t predict the label of day **t+7** (`LEAD_DAYS = 7`), not day t's own label (which would just re-learn the rule from the same columns).

**Features (18).** Twelve environmental features: `ndvi, soil_moisture, temperature_2m, precipitation_total, pm25, no2, ndvi_anomaly, pm25_anomaly, precipitation_total_anomaly, pca_1, pca_2, pca_3` — each **standardised per region with training-portion statistics** so a pooled model sees comparable scales — plus six indicators of **today's region-relative condition** computed with the same rule as the target (`now_code` 0/1/2, one-hot `now_deg/now_str/now_hea`, 7-day mean `now_roll7`, 7-day lag `now_lag7`). The condition indicators matter for pooling: labels are relative to each region's own history, so without them a pooled model cannot tell that an NDVI of 0.45 is "healthy" for Mumbai but "degraded" for Indonesia, nor use the strong day-to-day persistence of the condition. (They use only information available on day t; like the labels themselves they are computed with the region's whole-window z-score statistics.)

**Models.**
* *Random Forest* (`sklearn`): 300 trees, unlimited depth, Gini, `max_features="sqrt"`, `random_state=42`.
* *XGBoost*: 300 rounds, depth 6, learning rate 0.1, `multi:softprob`, `num_class=3`, `mlogloss`, `seed=42`.
* `n_jobs` = env `GEO_N_JOBS` (default 4).

**Split and pooling.** Chronological: first 75 % of each region's rows train, last 25 % test. **Pooled training** (`train_and_compare_pooled`): *one* RF and *one* XGBoost are fitted on the concatenated training portions of all 13 regions and evaluated separately on each region's own test portion (so no region's test period is ever seen). The single-region API path (`train_and_compare`) trains per region.

**Metrics:** accuracy, Cohen's kappa, ROC-AUC (one-vs-rest, macro), and `baseline_accuracy` = accuracy of "condition at t+7 equals condition today". Feature importances (top 5) are reported.

**Experiments (13 regions, mean accuracy; naive = 0.653):**

| Variant | Random Forest | XGBoost | Worst region vs naive (RF) |
|---|---|---|---|
| per-region models | 0.634 | 0.623 | − |
| pooled, raw features | 0.659 | 0.642 | −20.4 pts |
| pooled + today's condition | 0.688 | 0.658 | −6.1 pts |
| **pooled + today's condition + per-region standardisation (adopted)** | **0.688** | **0.683** | **−3.9 pts** |

(The experiment ran on a cached frame set; in the final snapshot run, which uses freshly fetched data, the adopted setup gives RF 0.686 / XGBoost 0.681 vs naive 0.652 and the worst region is Indonesia at −12.2 pts — see §15.2 for exact per-region numbers.)

Adding season sin/cos or lagged environmental features to *per-region* models did not help (within ±0.5 points). Pooling plus the condition indicators is what lifts the classifiers above the naive predictor.

### 9.3 Forecasting (`models/forecasting.py`)

**Input/output.** Sliding windows: lookback `L = 30` days → horizon `H = 14` days. The target variable is always column 0 of the input matrix.

**Per-variable setup (`train.FORECAST_CONFIG`, chosen by a 13-region backtest):**

| Variable | Extra inputs | Residual learning |
|---|---|---|
| PM2.5 | none (univariate) | no |
| precipitation | none | no |
| temperature | none | no |
| soil moisture | none | yes |
| NDVI | the other 5 variables + day-of-year sin/cos (8 channels) | yes |

**Architectures** (all accept `F` input channels, shape `(batch, 30, F)` → `(batch, 14)`; the output layer is **zero-initialised** so training starts from "predict no change"):

| Model | Structure | ≈ Parameters (F=1) |
|---|---|---|
| **LSTM** | 2 layers, hidden 64, dropout 0.1, final hidden state → `Linear(64→14)` | 51 k |
| **1-D CNN** | `Conv1d(F→32,k=3)`, ReLU, `Conv1d(32→32,k=3)`, ReLU, global average pool → `Linear(32→14)` (receptive field 5 days) | 3.7 k |
| **Transformer** | `Linear(F→32)` + learned positional embedding `(1,30,32)`, 2 encoder layers (d_model 32, 4 heads, FF 128, dropout 0.1, post-LayerNorm), mean pool → `Linear(32→14)` | 27 k |

**Normalisation.** Each region's series is z-scored with the mean/std of its **first 85 %** only (no look-ahead into the validation period). Predictions are de-normalised to physical units.

**Residual learning.** When enabled the network predicts `y − last_observed_value` (normalised), and the output is added back — a form of "learn only what improves on naive".

**Training.** Adam, lr 1e-3, MSE loss. Per-region path: batch 32, up to 40 epochs. **Pooled path: batch 128, 30 epochs.** Whole training tensor lives on the GPU with manual batching (no DataLoader overhead). `torch.manual_seed(seed)` for reproducibility. Device: CUDA if available (`GEO_DEVICE=cpu` overrides).

**Shrinkage safeguard.** The training windows of each region are split chronologically 85/15 into *fit* and *inner* parts. The network is trained on the fit part only. A weight `α ∈ [0,1]` is then fitted by least squares on the inner part:
`α = clip( Σ (m − n)(y − n) / Σ (m − n)² , 0, 1 )` where `m` = model prediction, `n` = naive prediction, `y` = truth. Forecasts are `n + α·(m − n)`. Where the model adds nothing α→0 and the forecast becomes naive, so it cannot be much worse than naive. α is fitted on *training-period* data; the final validation window is never used for it (unbiased metrics). It is shown on the dashboard as "Model weight".

**Window bookkeeping (731-day series, pooled).** 688 windows per region → 584 train+inner / 104 validation; the 584 split into 496 fit / 88 inner. Pooled fit set = 13 × 496 = 6 448 windows.

**Pooled training** (`PooledForecaster`, `train_and_compare_pooled`): one network per (variable, architecture, seed) trained on the fit windows of all regions; each region keeps its own normalisation, validation windows and α. Each architecture's reported forecast is the **mean over 3 seeds (42, 43, 44)**; the **ensemble** is the mean of the three architectures. Validation windows (each region's last 15 %) are never trained on by any region's model.

**Metrics:** RMSE, MAE, R² (on the flattened validation windows, original units), `baseline_rmse` (naive, same windows), `shrink` (α). "% improvement" in this document = `100·(1 − RMSE_model / RMSE_naive)`.

### 9.4 Experiments that selected this design (13 regions, median % improvement over naive)
| Change tested | Result |
|---|---|
| Residual learning vs absolute targets | helps soil moisture; slightly hurts PM2.5/rain/temperature |
| Extra drivers + season as inputs | only NDVI benefits (−40 % → +3 %) |
| 3 years instead of 2 of history | no gain |
| Lookback 60 vs 30 | mixed; 30 kept |
| Classification: pool regions + today's condition indicators | RF +0.6 → +3.5 pts over naive in the experiment (final run: +3.4 pts), worst region −20 → −4 pts (final run: −12 pts) |
| Shrinkage toward naive | removes large negative outliers (soil moisture/NDVI ≈ 0 instead of −40 … −100 %) |
| **Pooling all regions** | **PM2.5 regions beating naive 10 → 12 of 13; rainfall mean +5.7 → +16 %; NDVI mean −39 → −2 %** |
| 3 seeds per architecture | small, consistent variance reduction |

### 9.5 Single-region path (API) vs pooled path (dashboard)
`GET /api/train` calls `run_training(features, horizon, epochs)`: trains RF/XGBoost and the forecasters (with ensemble and shrinkage, but *without* pooling and with one seed) for one region. The dashboard's model panel shows the **pooled** results from `demo_data.json` because they are more accurate and cheaper per region. Both return the same JSON shape.

### 9.6 Why the numbers are modest (and what that means)
Predicting a *persistent* environmental variable two weeks ahead is hard to beat a repeat-last-value forecast on; soil moisture and NDVI change slowly, so persistence is near-optimal. The classification target is a rule-derived label with strong day-to-day persistence. This is a property of the problem, not only of the models, and the dashboard reports the naive baseline so users can judge model value directly.

---

## 10. Alerting (`alerting/alerts.py`)

Rule-based; not machine learning.

| Trigger | Rule | Severity |
|---|---|---|
| **Threshold** (latest day per region, for each sub-index and the composite) | value below its floor: air 40, vegetation 35, water 35, climate 40, composite 45 | `critical` if value < 50 % of the floor, `warning` if < 80 %, else `watch` |
| **Rapid decline** (air, vegetation, water, climate, EHI) | drop of ≥ 15 points between the latest value and the last value ≥ 7 days earlier | `critical` if drop ≥ 22.5, else `warning` |

Alerts carry region, date, severity, trigger, indicator, value and a human-readable message (indicator names are title-cased; `ehi` → "EHI"). The list is sorted most-severe first. Alerts depend only on the EHI history — the forecasts are not used to predict future alerts.

---

## 11. Service layer, caching and REST API

### 11.1 Service layer (`api/service.py`)
* `region_ehi_dataframe(region, days)`: pipeline → preprocessing → features → EHI, cached under key `ehi:{region}:{days}:{today}`.
* `ehi_series`, `comparison_table`, `alerts_for`, `training_report`: build JSON-safe results; `_sanitize` turns NaN/Inf into `null`.
* **Two-tier cache** (`_cached`): (1) in-memory dict with timestamp, TTL = `ingestion_interval_minutes × 60` (60 min); (2) pickle files in `data/cache/` (survive restarts, TTL from file mtime), written atomically. A miss runs the builder and stores both tiers. `clear_all_caches()` empties memory, `*.pkl` and `http/*.json`.
* Raw API responses additionally have their own disk cache (§5.5), so even a result-cache miss often needs no network.

### 11.2 Endpoints
| Method & path | Parameters (default; range) | Returns |
|---|---|---|
| `GET /api/health` | – | `{status, service}` |
| `GET /api/regions` | – | region registry (id, name, level, lat, lon, bbox) |
| `GET /api/ehi` | `region_id` (required), `days` (180; 14–730) | `{region_id, series:[{date, ehi, air_quality, vegetation, water, climate}]}`; 404 for unknown region |
| `GET /api/compare` | `region_ids` (comma list), `days` (180; 14–730) | `{regions, table:[…latest snapshot per region, ranked by EHI…]}`; 404 for unknown ids |
| `GET /api/alerts` | `region_ids` (optional), `days` (180; 14–365) | `{alerts:[{region_id, region_name, date, severity, trigger, indicator, value, message}]}` |
| `GET /api/train` | `region_id`, `days` (730; 90–1000), `horizon` (14; 3–60), `epochs` (40; 3–200) | `{region_id, report:{classification, forecasting}}` — one region, synchronous, cached per day |
| `GET /api/cache/status` | – | memory entries (age/expiry/records), disk file names, TTL |
| `GET /api/regions/status` | – | fast per-region cache + latest EHI summary (reads memory/disk only, never ingests) |
| `POST /api/cache/clear` | – | flushes memory, `*.pkl` and raw HTTP cache; next request re-ingests |
| `GET /cache-status` | – | HTML status page (auto-refresh every 10 s) |
| `GET /docs` | – | Swagger UI |
| `GET /` | – | static dashboard |

`/api/train` is synchronous (blocks the request for tens of seconds) — a job queue would be the production design. CORS is open (`*`). Startup cache warm-up is intentionally disabled (it saturated the interpreter in earlier versions); the disk cache serves the first request.

### 11.3 Response shapes
`report.classification.<model>`: `{metrics:{accuracy, kappa, roc_auc, baseline_accuracy}, top_features:{name: importance ×5}}`.
`report.forecasting.<variable>.<arch>`: `{metrics:{rmse, mae, r2, baseline_rmse, shrink}, forecast:[14 numbers]}` for `arch ∈ {lstm, cnn, transformer, ensemble}`.

---

## 12. Dashboard (frontend)

Vanilla HTML/CSS/JS, no framework; charts drawn on `<canvas>`; fonts from Google Fonts (Big Shoulders Display, IBM Plex Sans/Mono); dark theme.

**Panels (top to bottom):** hero (title, "snapshot generated", data-mode badge) → **Live band scan** (a chip per region: four band bars + EHI) → **01 Global comparison** (all regions ranked, level filter all/country/state/city/continent, four sub-index bars + EHI) → **02 180-day trend** (region selector, five metric cards, freshness note, interactive multi-line chart with hover tooltip) → **03 Model comparison** (per-region classification cards with feature-importance bars and a "beats/below naive predictor by x pts" verdict; forecasting tabs for the 5 variables, 14-day chart of all four models, per-model cards: RMSE, MAE, R², naive RMSE, model weight, and "x % better/worse than naive") → **04 Alert log** (severity-sorted) → **05 Eight-stage methodology**.

**Data strategy.** `init()` first loads `demo_data.json` and renders everything within milliseconds (works offline). It then tries the live API: `/api/regions` (2.5 s timeout), `/api/ehi?days=180` for each region in parallel (`Promise.allSettled`, 180 s per request, with a "syncing x/13" badge), `/api/compare` and `/api/alerts?days=180`. On success the EHI/comparison/alerts are replaced by live values (region metadata merged so lat/lon and NDVI date survive); the model results always come from the snapshot. Status badge reads "static demo snapshot" or "live API pipeline (synced)".

**Charts.** `createLineChart` (canvas, DPR-aware, y-axis precision adapts to the range, hover guide + dots + tooltip, handles nulls). The forecast chart draws all four models (best solid, others dashed) and an extra **live provider line**: `providerForecast()` fetches Open-Meteo's *live* forecast in the browser (CORS open) — PM2.5 from the CAMS air-quality forecast (≈ 3–5 days), rainfall and temperature from the NWP forecast (up to 16 days) — aligned by date to the snapshot's training date. It silently disappears when offline.

**Freshness and honesty features:** a note under the trend chart states the data-through date and the latest usable NDVI composite; every model card shows its naive baseline and verdict; the model panel header states device, "pooled model across regions (3 seeds)", history length, epochs and training date.

---

## 13. Scripts

### 13.1 `scripts/run_demo.py` — build the snapshot
```
python scripts/run_demo.py [--regions IN,US,…] [--days 180] [--train-days 730] [--epochs 30] [--horizon 14] [--workers 4] [--out dashboard/demo_data.json]
```
1. **Fetch (parallel processes, one per region, 0–20 s random stagger):** ingest 730 days → preprocess → features → EHI; also records the latest MODIS composite date (`ndvi_asof`). A failing region is reported and skipped.
2. **Pooled training (main process, GPU):** `run_pooled_training` — pooled classifiers + 5 variables × 3 architectures × 3 seeds + ensemble; per-region reports.
3. **Assemble:** EHI series (last 181 days), comparison table, alerts (`generate_alerts`), per-region training results, `training_meta` (device, days, epochs, horizon, mode "pooled", seeds, fetch/train/wall seconds, date). NaN/Inf → `null`; written with `allow_nan=False`.

### 13.2 `scripts/test_live.py`
Smoke test against a running server: health, regions, 15-day Mumbai EHI.

---

## 14. Tests (`pytest`, 32 tests, ≈ 5 minutes because they use the real APIs)

| File | Tests |
|---|---|
| `test_alerts.py` (5) | low sub-index flagged; healthy → no alert; rapid drop detected; severity sorting; empty input |
| `test_api.py` (8) | health; regions; EHI for known region; 404 unknown; compare; compare 404; alerts; `/api/train` end-to-end (marked slow) |
| `test_models.py` (10) | label classes; RF/XGBoost train & compare; regression & classification metric sanity; `build_sequences` shapes; LSTM/CNN/Transformer train and predict (parametrised ×3); insufficient-data error; **pooled forecaster returns shrunk per-region results** |
| `test_pipeline.py` (9) | synthetic variables; **pipeline fetches real sources**; unknown region raises; out-of-range flagging; gap filling; feature matrix & PCA; EHI bounded 0–100; region ranking; **sample points inside bbox** |

`pytest -m "not slow"` skips the end-to-end training test. Tests need internet access; transient 429s are retried by `get_json`.

---

## 15. Results (generated from the snapshot)

* Snapshot generated: **2026-10-01**; regions: **13**; training mode: **pooled** (3 seeds per architecture); device: **cuda**; history: **730 days**; epochs: **30**; horizon: **14 days**.
* Timing of the run that produced it: fetch 39 s, training 248 s, total 286 s.

### 15.1 EHI by region (latest day)
| Rank | Region | Level | EHI | Air | Vegetation | Water | Climate | Latest NDVI composite |
|---|---|---|---|---|---|---|---|---|
| 1 | Indonesia | country | **81.8** | 84.9 | 95.2 | 78.2 | 65.2 | 2026-08-13 |
| 2 | Asia | continent | **71.7** | 98.5 | 83.6 | 76.3 | 11.0 | 2026-08-13 |
| 3 | Brazil | country | **71.6** | 94.2 | 80.6 | 64.1 | 35.8 | 2026-08-13 |
| 4 | United States | country | **70.1** | 97.6 | 79.4 | 43.8 | 50.0 | 2026-08-13 |
| 5 | South America | continent | **68.9** | 97.2 | 71.4 | 44.2 | 54.1 | 2026-08-13 |
| 6 | Maharashtra, India | state | **66.9** | 83.9 | 64.8 | 69.9 | 40.3 | 2026-08-13 |
| 7 | India | country | **66.3** | 90.0 | 54.2 | 64.1 | 48.7 | 2026-08-13 |
| 8 | São Paulo, Brazil | city | **66.0** | 73.0 | 43.3 | 74.2 | 73.8 | 2026-08-13 |
| 9 | Australia | country | **62.8** | 97.5 | 58.4 | 40.7 | 43.6 | 2026-08-13 |
| 10 | California, USA | state | **56.1** | 95.4 | 47.8 | 26.1 | 44.8 | 2026-08-13 |
| 11 | Mumbai, India | city | **55.6** | 72.4 | 44.8 | 53.9 | 46.3 | 2026-06-26 |
| 12 | Kenya | country | **53.4** | 97.2 | 42.5 | 28.0 | 32.7 | 2026-08-13 |
| 13 | Africa | continent | **50.1** | 91.2 | 44.9 | 16.9 | 36.4 | 2026-08-13 |

### 15.2 Classification — condition 7 days ahead (pooled, chronological test set)
Mean over 13 regions — **Random Forest** accuracy 0.686 (kappa 0.41, ROC-AUC 0.79), beats the naive predictor in 8/13 regions; **XGBoost** accuracy 0.681 (kappa 0.40, ROC-AUC 0.78), beats naive in 9/13; **naive** ("condition stays the same") 0.652. Mean margin RF +3.4 pts, XGBoost +2.8 pts.

| Region | RF acc | RF κ | RF AUC | XGB acc | XGB κ | XGB AUC | Naive acc | RF vs naive |
|---|---|---|---|---|---|---|---|---|
| India | 0.646 | 0.44 | 0.77 | 0.646 | 0.45 | 0.79 | 0.646 | +0.0 pts |
| United States | 0.762 | 0.48 | 0.80 | 0.691 | 0.32 | 0.75 | 0.613 | +14.9 pts |
| Brazil | 0.895 | 0.83 | 0.96 | 0.901 | 0.84 | 0.94 | 0.840 | +5.5 pts |
| Australia | 0.829 | -0.02 | 0.58 | 0.845 | 0.00 | 0.57 | 0.740 | +8.8 pts |
| Kenya | 0.613 | 0.41 | 0.83 | 0.646 | 0.46 | 0.81 | 0.619 | -0.6 pts |
| Indonesia | 0.597 | 0.41 | 0.84 | 0.536 | 0.31 | 0.79 | 0.718 | -12.2 pts |
| Maharashtra, India | 0.873 | 0.77 | 0.92 | 0.895 | 0.81 | 0.92 | 0.845 | +2.8 pts |
| California, USA | 0.586 | 0.38 | 0.76 | 0.624 | 0.43 | 0.79 | 0.558 | +2.8 pts |
| Mumbai, India | 0.580 | 0.32 | 0.82 | 0.608 | 0.34 | 0.79 | 0.619 | -3.9 pts |
| São Paulo, Brazil | 0.475 | 0.18 | 0.64 | 0.470 | 0.16 | 0.65 | 0.420 | +5.5 pts |
| Asia | 0.779 | 0.48 | 0.78 | 0.740 | 0.43 | 0.82 | 0.669 | +11.0 pts |
| Africa | 0.569 | 0.23 | 0.66 | 0.558 | 0.21 | 0.66 | 0.453 | +11.6 pts |
| South America | 0.713 | 0.47 | 0.85 | 0.691 | 0.42 | 0.85 | 0.740 | -2.8 pts |

Top-5 feature importances (one pooled model, so identical for every region) — Random Forest: `ndvi` 13 % · `now_roll7` 11 % · `ndvi_anomaly` 7 % · `soil_moisture` 7 % · `pca_1` 6 %. XGBoost: `now_roll7` 40 % · `now_str` 8 % · `now_code` 7 % · `ndvi` 6 % · `ndvi_anomaly` 4 %.

### 15.3 Forecasting — 14 days ahead (pooled, 3 seeds, % RMSE improvement over naive)
Median % RMSE improvement over the naive forecast across regions (positive = better than "repeat the last value").

| Variable | LSTM | CNN | Transformer | **Ensemble** | Ensemble: regions beating naive | Ensemble mean % | mean model weight α |
|---|---|---|---|---|---|---|---|
| PM2.5 | +16.6 % | +15.4 % | +15.0 % | **+17.3 %** | 12/13 | +12.5 % | 0.81 |
| Precipitation | +17.7 % | +18.4 % | +16.2 % | **+18.6 %** | 11/13 | +14.9 % | 0.82 |
| Temperature | +13.1 % | +9.0 % | +9.3 % | **+11.6 %** | 13/13 | +12.1 % | 0.64 |
| Soil moisture | -3.5 % | -0.0 % | -3.2 % | **-5.4 %** | 5/13 | -17.5 % | 0.78 |
| NDVI | -5.7 % | -0.0 % | +1.4 % | **+3.1 %** | 7/13 | +0.9 % | 0.40 |

**PM2.5 (µg/m³)** — per region

| Region | Naive RMSE | Ensemble RMSE | Improvement | Best single model (improvement) | α (ensemble) |
|---|---|---|---|---|---|
| India | 6.25 | 5.62 | +10.0 % | cnn (+11.2 %) | 0.86 |
| United States | 2.67 | 2.21 | +17.3 % | lstm (+19.0 %) | 0.85 |
| Brazil | 5.66 | 4.87 | +13.9 % | transformer (+14.1 %) | 0.78 |
| Australia | 2.05 | 1.6 | +21.9 % | cnn (+23.0 %) | 0.82 |
| Kenya | 3.04 | 2.44 | +19.7 % | cnn (+21.0 %) | 0.61 |
| Indonesia | 31.4 | 43.6 | -38.9 % | lstm (-35.5 %) | 0.94 |
| Maharashtra, India | 6.2 | 5.87 | +5.3 % | cnn (+7.0 %) | 0.70 |
| California, USA | 3.73 | 3.22 | +13.6 % | transformer (+13.9 %) | 0.81 |
| Mumbai, India | 6.09 | 5.7 | +6.3 % | lstm (+9.9 %) | 0.63 |
| São Paulo, Brazil | 17.5 | 12.7 | +27.5 % | lstm (+28.6 %) | 0.93 |
| Asia | 2.81 | 2.25 | +19.9 % | transformer (+21.0 %) | 0.92 |
| Africa | 11.6 | 8.66 | +25.4 % | transformer (+26.0 %) | 0.84 |
| South America | 2.1 | 1.66 | +20.8 % | cnn (+22.9 %) | 0.81 |

**Precipitation (mm/day)** — per region

| Region | Naive RMSE | Ensemble RMSE | Improvement | Best single model (improvement) | α (ensemble) |
|---|---|---|---|---|---|
| India | 4.6 | 3.75 | +18.6 % | cnn (+20.4 %) | 0.77 |
| United States | 2.86 | 2.03 | +29.0 % | cnn (+29.6 %) | 0.99 |
| Brazil | 1.89 | 1.7 | +9.9 % | lstm (+13.6 %) | 0.96 |
| Australia | 1.61 | 1.43 | +11.4 % | cnn (+15.1 %) | 0.51 |
| Kenya | 0.798 | 0.804 | -0.7 % | transformer (+1.7 %) | 0.89 |
| Indonesia | 4.09 | 3.85 | +5.9 % | transformer (+7.9 %) | 0.94 |
| Maharashtra, India | 9.06 | 7.26 | +19.9 % | lstm (+22.1 %) | 0.56 |
| California, USA | 0.823 | 0.959 | -16.5 % | cnn (-8.6 %) | 0.91 |
| Mumbai, India | 32.1 | 28.2 | +12.1 % | transformer (+16.2 %) | 0.31 |
| São Paulo, Brazil | 9.53 | 7.25 | +23.9 % | cnn (+23.6 %) | 0.96 |
| Asia | 3.96 | 2.85 | +27.9 % | cnn (+28.1 %) | 0.96 |
| Africa | 4.23 | 3.03 | +28.3 % | cnn (+28.5 %) | 0.99 |
| South America | 3.09 | 2.36 | +23.7 % | transformer (+25.1 %) | 0.94 |

**Temperature (K)** — per region

| Region | Naive RMSE | Ensemble RMSE | Improvement | Best single model (improvement) | α (ensemble) |
|---|---|---|---|---|---|
| India | 0.868 | 0.748 | +13.7 % | lstm (+16.7 %) | 0.63 |
| United States | 1.57 | 1.48 | +6.1 % | transformer (+9.3 %) | 0.43 |
| Brazil | 1.62 | 1.39 | +14.6 % | lstm (+15.1 %) | 0.94 |
| Australia | 2.26 | 1.87 | +17.2 % | transformer (+20.3 %) | 0.88 |
| Kenya | 0.602 | 0.561 | +6.8 % | cnn (+8.9 %) | 0.76 |
| Indonesia | 0.667 | 0.52 | +22.0 % | transformer (+21.5 %) | 0.98 |
| Maharashtra, India | 1.5 | 1.43 | +4.9 % | cnn (+4.8 %) | 0.32 |
| California, USA | 3.54 | 3.25 | +8.1 % | cnn (+9.0 %) | 0.34 |
| Mumbai, India | 1.37 | 1.21 | +11.6 % | cnn (+11.6 %) | 0.54 |
| São Paulo, Brazil | 3.56 | 2.98 | +16.5 % | lstm (+18.9 %) | 0.58 |
| Asia | 3.42 | 2.93 | +14.5 % | transformer (+17.3 %) | 0.56 |
| Africa | 0.986 | 0.877 | +11.0 % | lstm (+14.2 %) | 0.43 |
| South America | 1.59 | 1.42 | +10.7 % | lstm (+11.4 %) | 0.89 |

**Soil moisture (m³/m³)** — per region

| Region | Naive RMSE | Ensemble RMSE | Improvement | Best single model (improvement) | α (ensemble) |
|---|---|---|---|---|---|
| India | 0.0121 | 0.0119 | +2.0 % | cnn (+3.5 %) | 0.97 |
| United States | 0.00679 | 0.00745 | -9.7 % | lstm (+1.3 %) | 0.83 |
| Brazil | 0.00934 | 0.00985 | -5.4 % | transformer (-3.8 %) | 0.94 |
| Australia | 0.00807 | 0.0087 | -7.9 % | cnn (+2.3 %) | 0.94 |
| Kenya | 0.00258 | 0.0028 | -8.5 % | lstm (-3.5 %) | 0.71 |
| Indonesia | 0.0111 | 0.0302 | -171.3 % | cnn (-154.9 %) | 0.97 |
| Maharashtra, India | 0.026 | 0.0251 | +3.3 % | transformer (+3.9 %) | 0.65 |
| California, USA | 0.00289 | 0.00331 | -14.7 % | cnn (+7.7 %) | 0.93 |
| Mumbai, India | 0.0385 | 0.0385 | +0.0 % | lstm (+0.0 %) | 0.00 |
| São Paulo, Brazil | 0.00984 | 0.00932 | +5.3 % | lstm (+7.4 %) | 0.63 |
| Asia | 0.0156 | 0.0189 | -20.7 % | transformer (-12.9 %) | 0.82 |
| Africa | 0.00308 | 0.00299 | +3.0 % | lstm (+5.8 %) | 0.89 |
| South America | 0.0089 | 0.00921 | -3.5 % | transformer (-0.5 %) | 0.85 |

**NDVI (index)** — per region

| Region | Naive RMSE | Ensemble RMSE | Improvement | Best single model (improvement) | α (ensemble) |
|---|---|---|---|---|---|
| India | 0.0124 | 0.0182 | -46.2 % | transformer (-42.8 %) | 0.47 |
| United States | 0.022 | 0.0232 | -5.5 % | transformer (-4.9 %) | 0.46 |
| Brazil | 0.00723 | 0.00887 | -22.6 % | transformer (+9.0 %) | 0.45 |
| Australia | 0.00526 | 0.00508 | +3.4 % | lstm (+9.2 %) | 0.02 |
| Kenya | 0.0148 | 0.011 | +25.9 % | lstm (+24.9 %) | 0.45 |
| Indonesia | 0.00359 | 0.00359 | -0.0 % | lstm (-0.0 %) | 0.00 |
| Maharashtra, India | 0.0332 | 0.0382 | -15.3 % | cnn (-8.5 %) | 0.60 |
| California, USA | 0.00446 | 0.00271 | +39.2 % | lstm (+51.0 %) | 0.34 |
| Mumbai, India | 0.00172 | 0.00165 | +4.1 % | transformer (+11.0 %) | 0.11 |
| São Paulo, Brazil | 0.00899 | 0.00858 | +4.6 % | cnn (+10.5 %) | 0.30 |
| Asia | 0.0274 | 0.0286 | -4.3 % | transformer (+3.7 %) | 0.87 |
| Africa | 0.00932 | 0.00694 | +25.6 % | transformer (+33.0 %) | 0.89 |
| South America | 0.0607 | 0.0589 | +3.1 % | lstm (+6.2 %) | 0.25 |

### 15.4 Active alerts
| Severity | Region | Indicator | Trigger | Value | Message |
|---|---|---|---|---|---|
| critical | Asia | climate | threshold | 11.03 | Climate at 11.0 is below the 40 threshold for Asia. |
| critical | Africa | water | threshold | 16.91 | Water at 16.9 is below the 35 threshold for Africa. |
| warning | California, USA | water | threshold | 26.05 | Water at 26.1 is below the 35 threshold for California, USA. |
| warning | São Paulo, Brazil | air_quality | rapid_decline | 72.97 | Air Quality for São Paulo, Brazil dropped 17.3 points in 7 days. |
| watch | Brazil | climate | threshold | 35.77 | Climate at 35.8 is below the 40 threshold for Brazil. |
| watch | Africa | climate | threshold | 36.35 | Climate at 36.4 is below the 40 threshold for Africa. |
| watch | Kenya | water | threshold | 28.05 | Water at 28.0 is below the 35 threshold for Kenya. |
| watch | Kenya | climate | threshold | 32.74 | Climate at 32.7 is below the 40 threshold for Kenya. |

### 15.5 How to read these results
* **Positive % improvement** = lower RMSE than "repeat the last value". Negative = worse. 0 % often means the shrinkage safeguard reverted to the naive forecast.
* Medians are used because a region with an almost constant series has a tiny naive RMSE and produces huge percentage outliers.
* The "best" marker in the dashboard is chosen on the validation data and is therefore slightly optimistic; the ensemble is the unbiased headline.
* R² is often negative even when the model beats naive: R² compares with the validation-period *mean*, a different (and for trending seasons harsher) yardstick.

---

## 16. Data validation: are the numbers true?

| Check | Method | Outcome |
|---|---|---|
| Temperature | Open-Meteo (ERA5) vs NASA POWER (MERRA-2), last 150 days, all 13 centroids | Correlation 0.93–0.97 in seasonal regions; mean absolute difference 0.7–3 K. **Genuine** |
| Precipitation | same | Correlation 0.34–0.91 (point rainfall is noisy between products); period means agree in magnitude. **Genuine, noisy** |
| NDVI | raw MODIS values + pixel-reliability flag | India Jul–Aug composites 100 % cloud-flagged → NDVI 0.09 (false). Fixed by the clear-pixel filter; India vegetation 18.5 → 71.6 → now a 5-point average |
| Soil moisture | 0–7 cm vs 0–100 cm vs NASA POWER wetness | surface layer volatile (Mumbai 0.26 → 0.07 in a week); root zone steady → switched to 0–100 cm |
| Representativeness | centroid vs 3×3/5×5 grids (Australia: centroid NDVI 0.25, land-grid mean 0.45; desert-centroid water index 0) | replaced single-point sampling with a land-point average |
| EHI arithmetic | hand recomputation | identical |
| Live vs archive | overlap of the live feed and the archive for the same days | identical values (same ERA5-Land source), so the live fill introduces no discontinuity |
| Dashboard vs API | live-mode run vs snapshot | identical EHI values for all regions |
| PM2.5/NO₂ | no free keyless ground truth | CAMS model output; plausibility only (e.g. Mumbai 730-day mean 36 µg/m³) |

---

## 17. Performance and operations

| Stage | Time (this machine) | Notes |
|---|---|---|
| Fetch, 13 regions, cold cache, 6 parallel workers | ≈ 8–10 min | dominated by ORNL (≈ 10–15 requests per point); warm cache: seconds |
| Pooled classifiers (RF + XGBoost, 13 regions) | ≈ 25 s | CPU |
| Pooled forecasters (5 variables × 3 archs × 3 seeds = 45 networks) | ≈ 4 min when the GPU is in full-speed state; **25–40 s per network (~20–30 min total) when the laptop is power-throttled** | observed: same fit 5 s vs 25–40 s depending on power state |
| Single-region `/api/train` | tens of seconds | synchronous |
| Cached API response | < 1 ms (memory), ≈ 10–50 ms (disk) | TTL 60 min |

**GPU vs CPU.** LSTM training on 600 windows × 40 epochs: 86 s on CPU, 4.6 s on GPU (19×); the CNN and Transformer gain less. Models are tiny, so launch overhead dominates; the whole training set is kept on the device and batches are large (pooled: 128). Earlier attempts to train 7 regions concurrently on one GPU were *slower* than sequential because every process grabbed all CPU threads — workers are now capped (`torch.set_num_threads(1)`, `GEO_N_JOBS=1`) and training is done once in the main process after a parallel fetch phase.

**Caches to know about:** `data/cache/*.pkl` (results, 60 min), `data/cache/http/*.json` (raw responses, 1–12 h). `POST /api/cache/clear` removes both. They are unbounded in count (stale files are not auto-deleted).

---

## 18. Security and robustness

* **No secrets** are required or stored. `.env` is git-ignored; `.env.example` only lists tuning variables.
* **CORS is open (`*`)** and **`POST /api/cache/clear` is unauthenticated** — acceptable for a demo on localhost, not for public hosting (add auth, restrict origins).
* **Pickle caches:** result caches are loaded with `pickle`; they are written only by the application itself into the local `data/cache/` directory. Do not place untrusted files there (pickle can execute code on load).
* **Input validation:** FastAPI validates ranges/types; unknown region ids return 404; query lengths are bounded.
* **Robustness:** rate-limit back-off, fail-fast on 4xx, atomic cache writes, per-region failure isolation in `run_demo.py`, NaN/Inf sanitised before JSON, the dashboard degrades to the snapshot when the API is unreachable, and the provider overlay degrades silently.
* **Reproducibility:** fixed seeds for PCA, RF, XGBoost and all networks; deterministic outputs for identical inputs (median re-run difference 0).

---

## 19. History of problems found and fixed; open issues

### 19.1 Fixed (chronological)
| Problem found | Fix |
|---|---|
| Satellite NDVI/NDWI were a single flat, mis-scaled value; temperature, PM2.5, NO₂ were synthetic | Real keyless sources (Open-Meteo, MODIS); synthetic fallback removed |
| Classifier "95 % accuracy" was label leakage (label computed from its own input columns) | Predict the condition 7 days ahead; chronological split; naive baseline |
| Forecasters trained on ~240 days, no baseline, normalised with look-ahead | 730 days, naive baseline reported, train-only normalisation, seeds fixed |
| Parallel GPU training slower than sequential (thread oversubscription) | One thread per worker; fetch in parallel, train once on the GPU |
| Dashboard live mode never completed (uncached `/api/alerts?days=90` re-ingested all regions) | Dashboard and API use 180 days (cache hit) |
| MODIS NDVI cloud-contaminated (India monsoon NDVI 0.09) | Pixel-reliability filter, 5 km windows |
| Soil-moisture surface layer too volatile (Mumbai water 6.6) | Root-zone 0–100 cm |
| Single-centroid sampling (Australia desert centroid) | Multi-point land average |
| EHI rainfall term spiky | 30-day mean rainfall |
| 429 rate limits during multi-region runs | disk HTTP cache, back-off, staggering, 5 sample points |
| MODIS 400 for open-ocean points aborted regions | fail-fast 4xx; treat as "not land" |
| Models below naive on smooth series | shrinkage safeguard, pooled training, 3-seed averaging |
| Stale docs/docstrings (GEE, Planetary Computer, CHIRPS, ERA5/OpenAQ stub) | code docstrings/README corrected; outdated upstream docs bannered; this report |
| Clipped dashboard cards, truncated axis label, "Ehi", missing favicon | fixed |

### 19.2 Open issues and limitations
1. **Spatial coarseness:** ≤ 5 sample points per region; continents (Asia, Africa, South America) are rough averages. More points cost proportionally more API quota.
2. **NDVI latency:** newest composites lag 1–2 months and cloudy seasons extend it (Mumbai/India show a June composite). A truly fresh vegetation signal needs a different (non-free or processed) source.
3. **Air quality is modelled (CAMS), not measured.** Ground stations need an API key (OpenAQ v3).
4. **Classification labels are a proxy rule**, not field truth; accuracy is bounded by strong persistence. Replace with real labels (e.g. land-cover/condition surveys) before trusting classification as more than a demonstration.
5. **No forecasting skill for soil moisture and NDVI** (slow signals); the dashboard shows this openly.
6. **Single-point future inputs are not used.** The most promising accuracy gain — feeding the provider's *future* weather (and CAMS PM2.5 forecast) into the forecasters — needs archived forecasts for training (Open-Meteo "previous runs"/historical-forecast APIs) and is not implemented.
7. **Models are not persisted;** `/api/train` retrains each uncached request; the snapshot is regenerated by `run_demo.py` (≈ 5–30 min depending on GPU state).
8. **EHI bounds/weights uncalibrated;** soil-moisture bounds ignore soil type; rainfall cap 15 mm/day treats floods as healthy.
9. **Operational:** unbounded cache files; free-tier rate limits; unauthenticated cache-clear; open CORS (§18).
10. **Timezones:** ingestion uses local `date.today()` for window ends while the APIs use UTC days; at most a one-day edge effect.
11. **Name debt:** modules `cloud_mask.py` and `spectral_indices.py` keep historical names although they no longer do cloud masking or spectral-index computation.

### 19.3 Recommended next steps (priority order)
1. Use the provider's future weather/air-quality forecasts as forecaster inputs (archived forecasts for training).
2. Real condition labels; calibrate EHI bounds/weights against reference indices.
3. More sample points (or area products) and a population/land-cover weighting for continents.
4. Persist trained models, add a job queue for training, schedule `run_demo.py` (cron) to keep the snapshot fresh.
5. Authentication and restricted CORS for any public deployment; cache pruning.

---

## 20. How to run, reproduce and troubleshoot

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows (use source .venv/bin/activate elsewhere)
pip install -r requirements.txt                       # torch: install the CUDA build from pytorch.org for GPU use

python scripts/run_demo.py                            # fetch 13 regions + pooled training -> dashboard/demo_data.json
uvicorn geovision.api.main:app                        # dashboard + API at http://127.0.0.1:8000  (docs at /docs)
pytest                                                # ≈ 5 min, needs internet (pytest -m "not slow" to skip the training test)
```

Useful switches: `GEO_DEVICE=cpu` (force CPU), `GEO_N_JOBS=n` (tree-model threads), `INGESTION_INTERVAL_MIN`, `FORECAST_HORIZON_DAYS`; `run_demo.py --regions IN,AU --workers 2 --epochs 20` for a quick partial run.

**Troubleshooting**

| Symptom | Cause / fix |
|---|---|
| HTTP 429 "Minutely/Hourly/Daily API request limit exceeded" | Open-Meteo free-tier limit. Wait (the client already backs off) or rerun later; the HTTP cache makes reruns cheap |
| `IngestionUnavailableError … HTTP 400` for MODIS | open-ocean sample point (handled) or genuinely no data for the location/date |
| Training suddenly 5–8× slower | laptop on a power-saving state (GPU in low clocks / CPU throttled): plug in, use a high-performance power plan |
| Dashboard shows "static demo snapshot" | API not running (expected for the static page) or live sync failed; the snapshot is still valid |
| Old numbers after code changes | `POST /api/cache/clear` (result and HTTP caches) |
| Windows `Remove-Item` on `data\cache\*` blocked | use `POST /api/cache/clear` |

---

## 21. Appendix: formulae and glossary

**Normalisation to the sub-index scale:** `s = clip((x − lo)/(hi − lo), 0, 1) · 100`, inverted as `100 − s` for "higher is worse".
**Temperature comfort:** `|T − 295 K|` mapped over 0–15 K, inverted.
**Rolling features:** `roll_w_mean`, `roll_w_std` for w ∈ {7, 30}; `anomaly = x − roll30_mean`.
**Condition label:** `score = z(NDVI) − z(PM2.5) + 0.5·clip(z(precip), −1.5, 1.5)`; bins at −0.5 and 0.5.
**Residual target:** `y_res = y_norm − x_norm[last]`; prediction `ŷ = x_norm[last] + f(x)`.
**Shrinkage:** `ŷ_final = n + α (ŷ − n)`, `α = clip(⟨ŷ−n, y−n⟩ / ⟨ŷ−n, ŷ−n⟩, 0, 1)` fitted on held-out training windows.
**% improvement:** `100 (1 − RMSE_model / RMSE_naive)`.
**RMSE / MAE / R²:** standard definitions on the flattened validation windows.
**Cohen's kappa:** agreement corrected for chance (1 = perfect, 0 = chance).
**ROC-AUC (OvR macro):** area under the one-vs-rest ROC curves averaged over the three classes.

| Term | Meaning |
|---|---|
| ERA5 / ERA5-Land | ECMWF atmospheric / land reanalysis (hourly, ~25 km / ~9 km) |
| CAMS | Copernicus Atmosphere Monitoring Service (air-quality reanalysis/forecast) |
| MOD13Q1 | MODIS Terra 16-day 250 m vegetation-index product |
| NDVI | Normalised Difference Vegetation Index, (NIR − Red)/(NIR + Red) |
| Naive baseline | "repeat the last value" (forecast) / "condition stays the same" (classification) |
| Pooled model | one model trained on all regions' training data, evaluated per region on its own hold-out |
| Shrinkage (α) | weight on the model's deviation from naive, fitted on held-out training data |
| EHI | Environmental Health Index, 0–100 |
