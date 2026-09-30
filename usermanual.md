# User Manual — GeoVision AI Environmental Intelligence Platform

Welcome to the **GeoVision AI User Manual**. This document guides you through running the platform, understanding the architecture, operating the interactive dashboard, and interpreting the deep learning & machine learning graphs and metrics.

---

## Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Quickstart & Operations](#2-quickstart--operations)
3. [Understanding the Dashboard Interface](#3-understanding-the-dashboard-interface)
   - [Live Band Scan Ticker](#live-band-scan-ticker)
   - [01 — Application Services Global Comparison](#01--application-services-global-comparison)
   - [02 — 180-Day Trend & Region Detail](#02--180-day-trend--region-detail)
   - [03 — Stage 4/5 Parallel ML/DL Training & Model Comparison](#03--stage-45-parallel-mldl-training--model-comparison)
   - [04 — Automated Alert Log](#04--automated-alert-log)
   - [05 — Eight-Stage Methodology](#05--eight-stage-methodology)
4. [Interpreting Machine Learning & Deep Learning Graphs](#4-interpreting-machine-learning--deep-learning-graphs)
   - [Condition Classification (Random Forest vs. XGBoost)](#condition-classification-random-forest-vs-xgboost)
   - [Time-Series Forecasting (CNN vs. LSTM vs. Transformer)](#time-series-forecasting-cnn-vs-lstm-vs-transformer)
   - [Validation Metrics Reference Table](#validation-metrics-reference-table)
5. [API Operations & Cache Diagnostics](#5-api-operations--cache-diagnostics)

---

## 1. Overview & Architecture

GeoVision AI is an end-to-end environmental intelligence platform that ingests multi-source Earth observation data (optical satellites, reanalysis models, ground sensors) and fuses them into a single, comprehensive **Environmental Health Index (EHI)** (0–100 scale).

```
   [ Planetary Computer (Sentinel-2, Landsat-8, MODIS) ]
   [ NASA POWER (Daily Precipitation)                  ]  ──▶  [ Ingestion Pipeline ]
   [ OpenAQ / Copernicus ERA5                          ]
                                                                     │
                                                                     ▼
                                                          [ Preprocessing & Cloud-Masking ]
                                                                     │
                                                                     ▼
                                                          [ Feature Engineering & PCA ]
                                                                     │
                                                                     ▼
                                      ┌──────────────────────────────┴──────────────────────────────┐
                                      ▼                                                             ▼
                        [ ML/DL Models & Forecasting ]                             [ Composite EHI Fusion ]
                        - RF & XGBoost Classifiers                                 - 30% Air Quality
                        - CNN, LSTM, Transformer Forecasters                       - 25% Vegetation
                                      │                                            - 25% Water
                                      │                                            - 20% Climate
                                      └──────────────────────────────┬──────────────────────────────┘
                                                                     ▼
                                                         [ FastAPI Presentation Layer ]
                                                                     │
                                                                     ▼
                                                     [ Interactive Browser Dashboard ]
```

---

## 2. Quickstart & Operations

### Starting the System
1. Open PowerShell or terminal in the project directory:
   ```powershell
   cd "c:\Users\rijul\Desktop\College\SEM-7\DAV\Code 2\geovision-ai"
   ```
2. Activate the virtual environment:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
3. Launch the API server & web dashboard:
   ```powershell
   python -m uvicorn geovision.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
4. Access the dashboard in your web browser:
   👉 **`http://127.0.0.1:8000/`**

---

## 3. Understanding the Dashboard Interface

### Live Band Scan Ticker
- **Location:** Top banner below the header.
- **Function:** A carousel of monitored regions displaying their real-time snapshot score.
- **Components:**
  - **Mini Band Stacks:** 4 stacked colored bars representing sub-indices:
    - 🟧 **Air Quality** (PM2.5 / NO2)
    - 🟩 **Vegetation** (NDVI)
    - 🟦 **Water** (NDWI)
    - 🟪 **Climate** (Precipitation & Temperature)
  - **Score:** Composite EHI (0–100). Green (≥60: Good), Gold (40–59: Moderate), Red (<40: Degraded/Alert).
- **Interaction:** Click any chip to immediately switch the detail view to that region.

---

### 01 — Application Services Global Comparison
- **Location:** First major content panel.
- **Function:** Multi-scale spatial benchmarking across countries, states, cities, and continents.
- **Components:**
  - **Level Filters:** Filter regions by `all`, `country`, `state`, `city`, or `continent`.
  - **Color-Coded Tracks:** Proportional fill bars representing the 4 sub-indices side-by-side for each region.
  - **Composite EHI Value:** High-precision index score used to rank regional environmental health.

---

### 02 — 180-Day Trend & Region Detail
- **Location:** Second panel.
- **Function:** In-depth historical time-series analysis for a selected region.
- **Components:**
  - **Region Selector Dropdown:** Switch between any monitored world region.
  - **Sub-Index Metric Cards:** Current values for EHI, Air Quality, Vegetation, Water, and Climate.
  - **Interactive 180-Day Line Chart:**
    - **Gold Thick Line:** Composite EHI trend.
    - **Colored Lines:** Individual sub-indices over time.
    - **Hover Tooltip:** Move your mouse over any point in time to inspect exact historical values and dates.

---

### 03 — Stage 4/5 Parallel ML/DL Training & Model Comparison
- **Location:** Third panel.
- **Function:** Real-time machine learning benchmark comparing classical ML classifiers and deep learning sequence forecasters trained on regional time-series data.
- *(See Section 4 below for full details on reading these graphs and metrics).*

---

### 04 — Automated Alert Log
- **Location:** Fourth panel.
- **Function:** Rule-based and anomaly-driven early warning detection.
- **Severity Levels:**
  - `CRITICAL` (Red): Sub-index or EHI drops below emergency threshold (e.g. severe air pollution spike or extreme drought).
  - `WARNING` (Gold): Moderate degradation or rapid decline observed over a 7-day window.
  - `WATCH` (Gray): Metric approaching caution boundary.

---

### 05 — Eight-Stage Methodology
- **Location:** Fifth panel.
- **Function:** Interactive reference detailing each step in the scientific pipeline:
  1. *Near real-time ingestion*
  2. *Preprocessing & cloud masking*
  3. *Feature engineering (Rolling stats & PCA)*
  4. *Parallel ML/DL training*
  5. *Validation & benchmarking*
  6. *Index fusion (EHI calculation)*
  7. *Interactive dashboard*
  8. *Deployment & automated alerting*

---

## 4. Interpreting Machine Learning & Deep Learning Graphs

### Condition Classification (Random Forest vs. XGBoost)

The left side of Panel 03 benchmarks two ensemble algorithms trained to classify the region's environmental condition into discrete health classes (e.g., *Healthy*, *Moderate Stress*, *Degraded*):

```
┌────────────────────────────────────────────────────────┐
│  Random Forest                             ✓ best      │
│  Accuracy: 94.2%  |  Kappa: 0.88  |  ROC-AUC: 0.96     │
│  ────────────────────────────────────────────────────  │
│  Feature Importance:                                   │
│  ndvi_roll30_mean   ██████████████████████ 38%         │
│  pm25_anomaly       ████████████ 22%                   │
│  pca_1              ████████ 16%                       │
│  temp_distance      █████ 11%                          │
└────────────────────────────────────────────────────────┘
```

#### What this tells us:
- **Accuracy:** The percentage of days correctly classified into the proper environmental status.
- **Cohen's Kappa ($\kappa$):** Measures classification agreement adjusted for chance ($>0.80$ represents near-perfect agreement).
- **ROC-AUC:** Area under the ROC curve — evaluates model discriminative capability between normal and degraded states ($1.0$ is perfect).
- **Feature Importance Bars:** Shows which physical indicators are driving environmental classification decisions. For example, if `ndvi_roll30_mean` is top, long-term vegetation canopy health is the primary indicator of regional stability.

---

### Time-Series Forecasting (CNN vs. LSTM vs. Transformer)

The right side of Panel 03 provides 14-day forward predictions for key environmental variables:
- **`pm25` (PM2.5 Air Quality in $\mu g/m^3$)**
- **`ndvi` (Vegetation Index, 0–1 scale)**
- **`precipitation_total` (Precipitation in mm)**

```
Forecast Chart (14-Day Horizon):
Value
  ▲
  │              CNN (Dashed Orange) 
  │             /                    Transformer (Dashed Purple)
  │            /  LSTM (Solid Cyan - ✓ Best)  /
  │    ───────●──────────────────────────────●─────►
  └────────────────────────────────────────────────► Day (1 to 14)
```

#### Architecture Roles:
1. **CNN (1D Temporal Convolutional Network):** Uses causal dilated convolutions to detect localized temporal patterns and abrupt step changes.
2. **LSTM (Long Short-Term Memory Network):** Recurrent neural network with memory gates that capture medium-term seasonal trends and cyclical decay.
3. **Transformer (Self-Attention Encoder):** Multi-head self-attention network that models complex non-local dependencies and cross-variable correlations across long horizons.

---

### Validation Metrics Reference Table

| Metric | Full Name | Ideal Value | What it tells us |
|---|---|---|---|
| **RMSE** | Root Mean Squared Error | Lower is better ($0$) | Heavily penalizes large prediction errors; shows worst-case variance. |
| **MAE** | Mean Absolute Error | Lower is better ($0$) | Average magnitude of errors in physical units (e.g. $\pm 2.4\,\mu g/m^3$ for PM2.5). |
| **$R^2$** | Coefficient of Determination | Closer to $1.0$ ($100\%$) | Proportion of variance in physical measurements explained by the model. |
| **Accuracy** | Overall Accuracy | Closer to $100\%$ | Proportion of correct state classifications. |
| **Kappa** | Cohen's Kappa | $>0.80$ | Agreement adjusted for class imbalance. |
| **ROC-AUC** | Area Under Curve | Closer to $1.0$ | Sensitivity vs. specificity diagnostic performance. |

---

## 5. API Operations & Cache Diagnostics

### Live Ingestion Cache
The backend automatically caches live planetary calculations for **60 minutes** (`INGESTION_INTERVAL_MIN=60`).

- **Inspect Cache Status & TTL Countdown:**
  Navigate to: `http://127.0.0.1:8000/api/cache/status`
  ```json
  {
    "cached_entries": 13,
    "ttl_seconds": 3600,
    "entries": [
      {
        "key": "ehi:MUM-IN:180:2026-08-18",
        "age_seconds": 310,
        "expires_in_seconds": 3290,
        "records": 181
      }
    ]
  }
  ```

- **Interactive Cache Status Dashboard:**
  View real-time memory and disk cache states for all 13 regions at:
  👉 **`http://127.0.0.1:8000/cache-status`**

- **Ultra-Fast Regional Status API:**
  Query all 13 regions' live cache and EHI scores in `<60ms`:
  👉 **`http://127.0.0.1:8000/api/regions/status`**

- **Detailed Caching Guide:**
  For full details on the two-tier cache architecture, key schemas, atomic writes, and troubleshooting, read:
  👉 **[caching.md](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/caching.md)**

- **Force Instant Data Refresh:**
  Execute via terminal:
  ```powershell
  curl.exe -X POST http://127.0.0.1:8000/api/cache/clear
  ```

- **Interactive Swagger / OpenAPI Docs:**
  Explore all endpoints and live schemas at:
  👉 **`http://127.0.0.1:8000/docs`**
