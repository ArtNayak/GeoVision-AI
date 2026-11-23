# GeoVision AI — ML / DL & Data Technical Report

Everything about the data, the models, how they are trained, what they output, and how they plug into the project. Every statement below was taken from the source code at the repo root (`geovision/`, `dashboard/`, `scripts/`, `tests/`), as of upstream commit `9309a15`. The upstream restructure only changed `geovision/api/main.py` (cache endpoints, startup warm-up disabled); no ML/DL, data or EHI code changed. Numbers marked **(demo snapshot)** come from `dashboard/demo_data.json` (generated 2026-08-16, training region = India, synthetic data).

> **STATUS (2026-10-01): the project was fixed after this audit.** §§0–9 document the state **before** the fix (synthetic/flat data, label leakage) and are kept as the audit trail. **§10 describes the fixed code; §11 (audit) supersedes the numbers in §10.3 and corrects the NDVI data; §12 is the latest state.** Where §10 and earlier sections disagree, §10 is current.

---

## 0. Big picture

| # | Component | Type | Learns from data? | File |
|---|---|---|---|---|
| 1 | **Environmental Health Index (EHI)** | Hand-built weighted fusion formula | No (fixed bounds + weights) | `index/health_index.py` |
| 2 | **Random Forest classifier** | Classical ML | Yes | `models/classifiers.py` |
| 3 | **XGBoost classifier** | Classical ML (gradient boosting) | Yes | `models/classifiers.py` |
| 4 | **LSTM forecaster** | Deep learning (PyTorch) | Yes | `models/forecasting.py` |
| 5 | **1-D CNN forecaster** | Deep learning (PyTorch) | Yes | `models/forecasting.py` |
| 6 | **Transformer forecaster** | Deep learning (PyTorch) | Yes | `models/forecasting.py` |
| 7 | **PCA + StandardScaler** | Unsupervised feature compression | Yes (fit per region) | `features/spectral_indices.py` |
| 8 | **Alert detectors** | Rule-based (threshold + 7-day drop) | No | `alerting/alerts.py` |

Important: the EHI, the alerts and the classification labels are **rule-based**. Only items 2–7 are trained. The EHI and alerts do **not** use any ML model output; the forecasters' outputs are shown on the dashboard but are not fed back into the EHI or alerts.

Data flow:

```
Ingestion (4 live clients + synthetic fallback)
  → Preprocessing (range check, daily resample, interpolation)
  → Feature engineering (pivot, rolling stats, anomalies, PCA)
  → EHI fusion (sub-indices → composite)          → Alerts
  → /api/train: RF, XGBoost (classification) + LSTM, CNN, Transformer (forecasting)
  → FastAPI → dashboard
```

---

## 1. The data

### 1.1 Six core variables

| Variable | Meaning | Unit | Live source | Used by |
|---|---|---|---|---|
| `ndvi` | Vegetation greenness (NIR − Red)/(NIR + Red) | index, −1…1 | Planetary Computer: Sentinel-2, Landsat-8, MODIS | Vegetation sub-index, classifier, forecaster |
| `ndwi` | Water/moisture (Green − NIR)/(Green + NIR) | index, −1…1 | Planetary Computer: Sentinel-2 | Water sub-index, classifier |
| `temperature_2m` | Air temperature at 2 m | Kelvin | ERA5 (Copernicus CDS) — **not implemented** | Climate sub-index, classifier |
| `precipitation_total` | Daily precipitation | mm | NASA POWER `PRECTOTCORR` (free, no key) | Climate sub-index, classifier, forecaster |
| `pm25` | Fine particulate matter | µg/m³ | OpenAQ v3 (needs `OPENAQ_API_KEY`) | Air-quality sub-index, classifier, forecaster |
| `no2` | Nitrogen dioxide | µg/m³ | OpenAQ v3 (needs key) | Air-quality sub-index, classifier |

Other variables can be produced (Landsat `lst`, ERA5 wind, OpenAQ `pm10`/`o3`) but they are **not** in `CORE_VARIABLES`, so they never reach the features or models.

### 1.2 Which data is actually real today

| Source | Real data possible? | Reality in the code |
|---|---|---|
| Planetary Computer (NDVI/NDWI) | Yes — public, no key, needs `pystac-client planetary-computer rioxarray` installed | **Verified live** once those libs are installed (they are NOT in a default environment → synthetic). Returns one flat value per index (§1.11) |
| NASA POWER (precipitation) | Yes — always "configured", no key | **Verified live**, 177 daily values |
| ERA5 (temperature) | **No** — `fetch()` ends in `raise NotImplementedError` | **Always synthetic** (verified) |
| OpenAQ (PM2.5, NO2) | Only with `OPENAQ_API_KEY` | **Synthetic without the key** (verified 181/181 rows) |

The pipeline (`ingestion/pipeline.py`) tries each client, and on `IngestionUnavailableError` or `NotImplementedError` substitutes synthetic rows **for that source only**. So a "live" run is a mixture: real NDVI/NDWI/precipitation + synthetic temperature/PM2.5/NO2 (unless keys are set). The demo snapshot is **100 % synthetic**.

### 1.3 How the live satellite values are computed (Planetary Computer client)

- For each of 3 collections it searches the STAC catalog for the region bbox and date window (max 2 least-cloudy scenes, cloud cover < 20 %; MODIS has no cloud filter).
- Per scene it reads each needed band with `rasterio` **windowed** to the bbox (overview level 4 ≈ 16× downsampled) and takes the **mean pixel value** of the whole region. NDVI/NDWI are then computed from the *regional mean bands*, not per pixel.
- Landsat: surface temperature `lwir × 0.00341802 + 149.0` (K). MODIS NDVI × 0.0001.
- The result is **one observation per index per collection**, stamped at the **midpoint date** of the window. The preprocessing step then interpolates/forward-fills that single point into a flat line across the whole window. Consequence: live NDVI/NDWI are close to constant over the window (no real time variation).
- Landsat/MODIS/Sentinel-2 NDVI all end up as separate rows (`source` differs) but share the `ndvi` variable name; they get averaged together in the pivot (`aggfunc="mean"`).

### 1.4 Synthetic data generator (`ingestion/synthetic.py`)

Used for the demo, tests, and as the per-source fallback. Deterministic and region-seeded.

- Seed = SHA-256 of the region id mod 2³².
- Each variable = `base + amplitude · sin(2π · dayOfYear/365 + phase) + noise`, clipped to a valid range.
- "Tropical" if |latitude| < 15°. Arid bias for `AU`, `KE`.

| Variable | Formula (base ± amplitude, phase) | Clip |
|---|---|---|
| ndvi | 0.55 (tropical) or 0.40, ±0.12, phase 0.3, noise ×0.03 | 0 … 0.95 |
| ndwi | 0.10 (tropical) or −0.05, ±0.10, phase 1.8, noise ×0.04 | −0.6 … 0.6 |
| temperature_2m | 300 K (tropical) or 288 K, ±8 K, phase π in southern hemisphere, noise ±1.5 | none |
| precipitation_total | 4 − 1.2 (arid) mm, ±6, phase 2.5, noise ±3 | ≥ 0 |
| pm25 | 55 for IN, MH-IN, MUM-IN, ID, else 18; ±15, phase 4.0, noise ±6 | ≥ 2 |
| no2 | 22 µg/m³ for **every** region, ±8, phase 1.2, noise ±4 | ≥ 1 |

Caveats you should know:
- The "noise" is `((seed >> (k % 24)) % 1000)/1000 − 0.5) × 2 × scale` — it depends only on the region seed and `k % 24`, so it repeats with a **24-day period**; it is not random day to day. Different days look like smooth sinusoid + a fixed 24-day pattern.
- Every variable is a pure annual sine wave. Variables are therefore strongly (and artificially) correlated with each other through the season — this matters for feature importances (§5.4).
- NO2 has the same base for all regions; only PM2.5 differs by region.
- The code comment says it was built to look like the region's climate, but there is no real weather: no rain events, droughts, fires or pollution episodes.

### 1.5 Tidy observation schema

Every client returns `Observation(region_id, date, source, variable, value, unit, quality_flag, meta)`. The pipeline builds a DataFrame with those columns and drops duplicates on (region, date, source, variable).

### 1.6 Preprocessing (`preprocessing/cloud_mask.py`)

1. **Range validation** — readings outside the physical ranges are set to NaN and flagged `invalid`:
   `ndvi, ndwi ∈ [−1, 1]`, `temperature_2m ∈ [200, 330] K`, `precipitation_total ∈ [0, 500]`, `pm25 ∈ [0, 1000]`, `no2 ∈ [0, 500]`.
2. **Gap filling** — for each (region, source, variable) series: reindex to a full **daily** grid from the earliest to the latest date, `interpolate(method="time", limit_direction="both")`, then backfill/forward-fill. Flags become `interpolated` or `ok`.

Note: the function is named "cloud mask" but performs no per-pixel cloud masking; cloud screening happens at STAC search time (`eo:cloud_cover < 20`).

### 1.7 Feature engineering (`features/spectral_indices.py`)

1. **Pivot** tidy rows to one row per (region, date), columns = variables (mean if several sources).
2. **Rolling features** per core variable: 7-day and 30-day rolling mean (`min_periods=1`) and std (`min_periods=2`, NaN→0), plus `<var>_anomaly = value − 30-day rolling mean`.
   For 6 variables → 6 raw + 24 rolling (6×2 windows×2 stats) + 6 anomaly = 36 columns.
3. **PCA** on the six standardised core variables (missing filled with the column mean), keeping **3 components** → `pca_1, pca_2, pca_3`. The scaler and PCA are fitted on the region's own full history each time (random_state 42).

The classifiers only use a 12-column subset of these (§3.3). The forecasters use just one raw column at a time.

### 1.8 Data volume

A standard request uses 180–240 days of history → about **181–241 daily rows per region** (the demo has exactly 241 per region for 9 regions). This is **tiny** for ML — the classifier test set is 61 rows, the forecaster validation set is 30 windows.

### 1.9 Regions

13 regions in `config.py` (`REGION_REGISTRY`): countries IN, US, BR, AU, KE, ID; states MH-IN, CA-US; cities MUM-IN, SP-BR; continents AS, AF, SA. Each has a bbox (used for satellite/OpenAQ queries) and a centroid lat/lon (used for NASA POWER point queries). Models are trained **per region, one region at a time** — there is no global model across regions.

### 1.10 Caching

`api/service.py` keeps results in memory and as pickles in `data/cache/` (TTL 60 min; key includes today's date). Training reports are cached per (region, days, horizon, epochs, date). Startup warm-up (a 4-worker thread pool that pre-computed the EHI for all regions) is now **disabled** upstream; the disk cache in `data/cache/` (git-ignored) is used instead, and a region is computed on its first request. Cache endpoints: `POST /api/cache/clear`, `GET /api/cache/status`, `GET /api/regions/status` (instant, reads memory/disk only, never ingests), and `GET /cache-status` (HTML status page). These only report EHI cache state; none involve the ML/DL models.

### 1.11 Measured: what is really live vs synthetic (real run, 2026-10-01)

A real run was performed on this machine (Windows, Python 3.13, network available, **no** `OPENAQ_API_KEY` / `CDS_API_KEY`). `pystac-client`, `planetary-computer` and `rioxarray` were **not installed by default**, so a fresh checkout silently falls back to synthetic for satellite data too; they were installed in an isolated venv for the test. Region Mumbai (`MUM-IN`, small bbox), 180 days. Each source was called directly, and rows in the merged pipeline output were compared with `SyntheticClient` output (a row is synthetic if its value is identical to the generator's).

| Variable | Source | Result | Live? |
|---|---|---|---|
| ndvi | Sentinel-2, Landsat-8, MODIS | 1 value each, all dated 2026-09-01 (the window midpoint) | **Real** (but see quality issues) |
| ndwi | Sentinel-2 | 1 value, dated 2026-09-01 | **Real** (see issues) |
| lst | Landsat-8 | 1 value = 233.5 K | Real, but not used by any model/EHI |
| precipitation_total | NASA POWER | 177 daily values (no key needed), mean 13.4 mm, max 178 mm | **Real, daily** |
| temperature_2m | ERA5 | `CDS_API_KEY` missing and the fetch is unimplemented | **100 % synthetic** (181/181 rows) |
| pm25 | OpenAQ | `OPENAQ_API_KEY` missing | **100 % synthetic** (181/181 rows) |
| no2 | OpenAQ | `OPENAQ_API_KEY` missing | **100 % synthetic** (181/181 rows) |

So a live EHI is a **hybrid**: the vegetation, water and (half of the) climate sub-indices use real data; the air-quality sub-index (30 % of the EHI) and the temperature half of the climate sub-index are synthetic. This reproduces the Mumbai example in `caching.md` exactly (EHI 40.7, air 71.3, vegetation 21.4, water 53.0, climate 3.5), confirming that example was such a hybrid run.

**Quality problems found in the live data**
1. **NDVI/NDWI are flat.** One observation per index is forward/back-filled over 181 days. In the final daily frame `ndvi` has 1 unique value (0.114) and `ndwi` 1 unique value (0.071), standard deviation 0. The vegetation and water sub-indices therefore never change over the whole window (vegetation 21.4, water 53.0 on every day).
2. **Three NDVI sources are averaged.** 0.114 is the mean of Landsat 0.019, MODIS 0.355 and Sentinel-2 −0.033 — they disagree by 0.39, and a monsoon-season Mumbai NDVI of −0.03 (Sentinel-2) is implausible. The likely cause is that reflectance scale/offset factors are not applied before the ratio and that the bbox includes sea, but the exact cause was not verified.
3. **Landsat LST = 233 K (−40 °C)** for Mumbai is physically implausible (not used downstream).
4. **`overview_level=4` fails** with a GDAL error on Sentinel-2 and MODIS tiles; the code falls back to a full-resolution windowed read, so the "16× bandwidth saving" does not always apply.
5. **Speed.** Mumbai (small bbox): 26–51 s per region. **India (country-sized bbox) did not finish within a 30-minute limit** and was stopped — no result. Continent-scale regions should be expected to be slower still; the demo's "instant" experience depends on the disk cache, not on live ingestion.
6. Precipitation is real and daily, but its values (up to 178 mm) saturate the EHI climate scale, which caps at 15 mm, so the precipitation half of the climate sub-index is ~100 on rainy days and the climate score is dominated by the synthetic temperature term.

**Effect on the models (trained on this hybrid Mumbai data, 240 days, 10 epochs)**

| Model | Result | Meaning |
|---|---|---|
| RF / XGBoost classification | Accuracy 1.00, Kappa 1.00, ROC-AUC 1.00 (RF top features: pm25 38 %, pca_1 18 %, no2 12 %, temperature 10 %; XGBoost: pm25 **95 %**) | The label is computed from `ndvi`, `pm25`, `precipitation_total`. With NDVI constant, the label depends on PM2.5 and precipitation, and PM2.5 is **synthetic** — the classifier is learning a synthetic signal it can reproduce perfectly. Not evidence of skill. |
| NDVI forecasters | RMSE 0.000 (LSTM, CNN), 0.007 (Transformer), R² = 0 | The input series is a constant line; a model that predicts the constant is "perfect". Meaningless. |
| PM2.5 forecasters | RMSE 4.1 / 5.3 / 4.7 (LSTM/CNN/Transformer), R² −0.15 / −0.88 / −0.49 | Forecasting a synthetic sinusoid; all worse than predicting the mean. |
| Precipitation forecasters | RMSE 9.3 / 7.8 / 25.2, R² −2.0 / −1.1 / −21.4 | Real data, real monsoon variability: **all three fail badly**; the Transformer is far worse than predicting the mean. Only ~240 daily points and no seasonal context. |

**Bottom line:** in the current project no trained-model output is based on fully real data. The only genuinely real inputs to any model are NASA POWER precipitation and (as one flat number) satellite NDVI/NDWI; every PM2.5, NO₂ and temperature value is generated.

---

## 2. Environmental Health Index (EHI) — the fusion "model"

Not learned. A fixed formula that turns raw variables into 0–100 scores (higher = healthier).

**Step 1 — rescale each variable** with `min_max_normalize(value, lo, hi, invert)`: clip to [lo, hi], scale to 0–100, flip if higher-is-worse.

| Sub-index | Components (weight inside sub-index) | lo → hi | Direction |
|---|---|---|---|
| air_quality | PM2.5 (0.6), NO2 (0.4) | PM2.5 0–150, NO2 0–200 | inverted (more pollution = worse) |
| vegetation | NDVI (1.0) | −0.1 → 0.9 | higher = better |
| water | NDWI (1.0) | −0.3 → 0.4 | higher = better |
| climate | precipitation (0.5), temperature comfort (0.5) | precip 0–15 mm; temp = \|T − 295 K\| over 0–15 | precip higher = better; temperature distance inverted |

**Step 2 — composite**

```
EHI = 0.30·air_quality + 0.25·vegetation + 0.25·water + 0.20·climate
```

(weights from `Settings.ehi_weights`). If a sub-index is NaN for a row, the remaining weights are renormalised to sum to 1.

**Output:** per region per day: `ehi, air_quality, vegetation, water, climate`, all 0–100.

**Design notes:** the bounds (e.g. PM2.5 cap 150) and weights are hand-chosen, not calibrated against ground truth. Precipitation is treated as "more is better" up to 15 mm/day, which is an assumption (floods are not penalised).

---

## 3. Classification models — Random Forest & XGBoost

### 3.1 What they do in the project

Predict a 3-class **environmental condition** for each day: `degraded`, `stressed`, `healthy`. The dashboard shows their Accuracy, Cohen's Kappa, ROC-AUC, and the top-5 most important features side by side, and highlights the model with the higher Kappa.

### 3.2 Training data and labels

- **Input table:** the feature matrix of **one region** (~240 daily rows for the API/demo).
- **Labels are not real ground truth.** They come from `derive_condition_labels`, a rule over within-region z-scores:

```
score = z(ndvi) − z(pm25) + 0.5 · clip(z(precipitation_total), −1.5, 1.5)
label = degraded  if score < −0.5
        stressed  if −0.5 ≤ score ≤ 0.5
        healthy   if score > 0.5
```

  z-scores are computed over the region's own 240-day window, so the classes are **relative to that window**, not absolute health levels (a polluted region still gets ~equal numbers of "healthy" days).
- Rows with NaN in `ndvi`, `pm25`, `precipitation_total` are dropped; missing feature columns are filled with 0.
- **Split:** `train_test_split(test_size=0.25, random_state=42, stratify=labels)` — **random** (not time-ordered). ≈180 train / ≈61 test rows.
- The same split is used for both models so the comparison is fair.

### 3.3 Input features (12)

`ndvi, ndwi, temperature_2m, precipitation_total, pm25, no2, ndvi_anomaly, pm25_anomaly, precipitation_total_anomaly, pca_1, pca_2, pca_3`

(NaN → 0 before fitting.)

### 3.4 Random Forest (`sklearn.ensemble.RandomForestClassifier`)

- **What it is:** an ensemble of decision trees, each trained on a bootstrap sample with a random feature subset at each split; the prediction is a majority vote (probabilities are averaged).
- **Hyperparameters:** `n_estimators=300`, `max_depth=None` (trees grow fully), `random_state=42`, `n_jobs=-1`; other sklearn defaults (Gini criterion, `max_features="sqrt"`).
- **Output:** predicted class label and class probabilities (3 columns, order degraded/stressed/healthy); feature importances (mean decrease in impurity, sum to 1).

### 3.5 XGBoost (`xgboost.XGBClassifier`)

- **What it is:** gradient-boosted decision trees — trees are added one after another, each correcting the errors of the previous ones.
- **Hyperparameters:** `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `objective="multi:softprob"`, `num_class=3`, `eval_metric="mlogloss"`, `seed=42`, `n_jobs=-1`; other xgboost defaults.
- **Output:** class probabilities (softmax over 3 classes) → argmax class; feature importances (gain-based default in the sklearn wrapper).

### 3.6 Training and evaluation flow

1. Labels → encode to integer codes (order `degraded=0, stressed=1, healthy=2`).
2. Fit on the 75 % train split (no validation set, no early stopping, no hyperparameter search).
3. Predict on the 25 % test split; compute metrics with `models/evaluate.py`:
   - **Accuracy** — fraction correct.
   - **Cohen's Kappa** — agreement corrected for chance (1 = perfect, 0 = chance).
   - **ROC-AUC** — one-vs-rest, macro-averaged over the 3 classes, from class probabilities (set to `None` if a class is missing from the test set).
4. `train.py` keeps the **top-5** feature importances for the report.

### 3.7 Results **(demo snapshot, India, synthetic)**

| Model | Accuracy | Kappa | ROC-AUC |
|---|---|---|---|
| Random Forest | 95.08 % (58/61) | 0.916 | 0.990 |
| XGBoost | 95.08 % (58/61) | 0.916 | 0.987 |

Best by Kappa: Random Forest (tied on accuracy/Kappa; RF has higher AUC).

### 3.8 Feature importance **(demo snapshot)**

| Rank | Random Forest | Importance | XGBoost | Importance |
|---|---|---|---|---|
| 1 | pm25 | 24.0 % | pm25 | 39.2 % |
| 2 | pca_1 | 23.2 % | pca_2 | 20.2 % |
| 3 | ndwi | 13.8 % | pca_1 | 15.9 % |
| 4 | no2 | 13.0 % | precipitation_total_anomaly | 12.6 % |
| 5 | ndvi | 11.1 % | ndvi_anomaly | 6.0 % |

### 3.9 How to read these results honestly

- The label is computed from `ndvi`, `pm25` and `precipitation_total`, and those same columns (plus their anomalies and PCA components built from them) are **inputs**. The models are largely **re-learning the labelling formula**, which is why accuracy is ~95 %. This is target leakage, not evidence of predictive skill on real conditions.
- `pm25` dominates because it enters the score with the largest effective weight after NDVI and has the widest, most separable variance.
- `ndwi` and `no2` are **not** in the label formula, yet rank high in RF. In the synthetic data every variable is a phase-shifted annual sine wave, so they are spuriously correlated with the label through the season; on real data they would likely rank lower.
- Because the split is random and rolling features overlap in time, adjacent days leak between train and test.
- With ~61 test rows, one extra error changes accuracy by 1.6 points; the RF/XGBoost tie is not meaningful.

---

## 4. Forecasting models — LSTM, CNN, Transformer

### 4.1 What they do in the project

For each of three variables — **PM2.5, NDVI, precipitation** — each architecture is trained on that variable's own past and predicts the **next 14 days** (the horizon is configurable, 3–60). The dashboard shows a chart of the three 14-day forecasts (best model solid, others dashed) and RMSE/MAE/R² per model. The best per variable is chosen by lowest RMSE.

### 4.2 Training data

- **Input:** a single univariate daily series (the region's preprocessed `pm25`, `ndvi` or `precipitation_total`, sorted by date, NaNs dropped). ~241 points for the default 240-day request.
- A variable is skipped if it has fewer than 60 points; the whole model is skipped on `ValueError` (not enough windows) or `ImportError` (PyTorch missing) — then that variable simply has no forecast.
- Only the variable's own history is used — **no exogenous features** (no weather → PM2.5 link, no other variables).

### 4.3 Data preparation (shared by all three)

1. **Z-score normalise** with the mean/std of the **whole series** (reversed at prediction time).
2. **Sliding windows:** lookback **L = 30** days → next **H = 14** days. With 241 points: 241 − 30 − 14 + 1 = **198 windows**.
3. **Split (time-ordered):** first 85 % of windows for training (≈168), last 15 % for validation (≈30). Needs ≥ 8 windows, else error.
4. Windows are shuffled within the training set each epoch (batch size 16).
5. Forecast: the **last 30 days** of the series are fed in to predict the 14 days after the data ends.

### 4.4 Shared training recipe

| Item | Value |
|---|---|
| Loss | MSE on normalised values |
| Optimiser | Adam, learning rate 1e-3 |
| Batch size | 16 |
| Epochs | 60 (class default), 40 (API default), 25 (dashboard request & `run_demo.py`), 3 (tests) |
| Regularisation | dropout 0.1 (LSTM between layers, Transformer layers); none for CNN |
| Early stopping / LR schedule / weight decay | none |
| Random seed | **not set** — results vary slightly between runs |
| Input/Output shape | input `(batch, 30)` → output `(batch, 14)` (direct multi-step, not recursive) |

### 4.5 LSTM

- **Idea:** a recurrent network that reads the 30 days in order and keeps a memory (hidden state) of the trend/seasonality.
- **Architecture:** input `(batch, 30)` → reshaped to `(batch, 30, 1)` → `nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)` → take the **final hidden state of the top layer** `h_n[-1]` (64 values) → `Linear(64 → 14)`.
- **Parameters:** ≈ 51 k (hand-computed).
- **Why it's here:** the literature cited in the code uses LSTMs for long-range drought/precipitation forecasting.

### 4.6 1-D CNN

- **Idea:** slides small filters over the 30-day window to pick up local shape patterns, then averages them.
- **Architecture:** `(batch, 30)` → `(batch, 1, 30)` → `Conv1d(1→32, k=3, pad=1)` → ReLU → `Conv1d(32→32, k=3, pad=1)` → ReLU → `AdaptiveAvgPool1d(1)` (global average over time) → `Linear(32 → 14)`.
- **Parameters:** ≈ 3.7 k — by far the smallest.
- **Limitation:** receptive field is only 5 days and global average pooling throws away *where* in the window features occurred, so it is effectively "average local pattern → forecast". It can capture level and local slope but not long seasonal phase well.

### 4.7 Transformer (encoder only)

- **Idea:** self-attention lets every day in the window look at every other day.
- **Architecture:** `(batch, 30)` → `(batch, 30, 1)` → `Linear(1→32)` (embedding) → **+ learned positional embedding** `(1, 30, 32)` → `TransformerEncoder` with **2 layers** (`d_model=32`, `nhead=4`, feed-forward width 128, dropout 0.1, ReLU, post-LayerNorm, batch-first) → **mean over the 30 positions** → `Linear(32 → 14)`.
- **Parameters:** ≈ 27 k (hand-computed).
- No decoder and no causal mask (the whole lookback window is visible, which is fine because it's all past data).

### 4.8 Metrics (original units, flattened over all val windows × 14 steps)

- **RMSE** — square-root of mean squared error (penalises large misses).
- **MAE** — mean absolute error.
- **R²** — 1 − SSE/SST on the validation targets; **negative R² means worse than predicting the validation mean**.

### 4.9 Results **(demo snapshot, India, synthetic, 25 epochs)**

| Variable | Model | RMSE | MAE | R² | Best |
|---|---|---|---|---|---|
| PM2.5 (µg/m³) | LSTM | 4.107 | 3.348 | −0.374 | |
| | **CNN** | **3.864** | 3.316 | −0.216 | ✔ |
| | Transformer | 4.183 | 3.411 | −0.425 | |
| NDVI | LSTM | 0.033 | 0.028 | −1.140 | (tie at 3 decimals) |
| | CNN | 0.050 | 0.043 | −3.748 | |
| | **Transformer** | **0.033** | 0.027 | −1.067 | ✔ (marginally lower unrounded RMSE) |
| Precipitation (mm) | **LSTM** | **2.642** | 2.050 | −0.999 | ✔ |
| | CNN | 2.721 | 2.119 | −1.121 | |
| | Transformer | 2.738 | 2.134 | −1.148 | |

The snapshot's `best_models` field lists CNN (pm25), Transformer (ndvi), LSTM (precipitation). Note the dashboard's own "best" badge uses the same lowest-RMSE rule.

### 4.10 How to read these results honestly

- **All R² values are negative.** None of the forecasters beat a naive "predict the validation-period mean" baseline. RMSE differences between models are small relative to the noise and come from one short validation slice (30 overlapping windows) on one training run, so the "best model" label is not statistically meaningful.
- Likely reasons: only ~170 training windows; a 240-day history contains less than one annual cycle so the models have never seen the seasonal phase they are asked to extrapolate; the validation block is the **last** stretch of the year (different level from the training block); normalisation uses the whole series (mild leakage).
- There is no baseline (persistence / seasonal-naive / mean) reported, so the dashboard cannot show whether a model adds value.
- Predictions are for a synthetic sinusoid + 24-day pattern, so even an ideal model would only recover the generator.
- Overlapping windows mean validation inputs partly overlap training targets at the boundary.

---

## 5. Alerting (rule-based, not ML)

Runs on the EHI history for one or more regions; output is sorted most-severe first.

| Trigger | Rule | Severity |
|---|---|---|
| **Threshold** (latest day per region) | sub-index/EHI below its floor: air 40, vegetation 35, water 35, climate 40, composite 45 | `critical` if value < 50 % of floor, `warning` if < 80 %, else `watch` |
| **Rapid decline** | latest value vs the last reading ≥ 7 days earlier drops by ≥ 15 points (applies to the 4 sub-indices and EHI) | `critical` if drop ≥ 22.5, else `warning` |

Each alert carries region, date, indicator, value, human-readable message. Alerts are computed from the EHI; the forecasters are **not** used to predict future alerts.

---

## 6. How everything is used in the running project

| Endpoint | Pipeline run | ML involved |
|---|---|---|
| `GET /api/ehi` | ingest → preprocess → features (incl. PCA) → EHI | PCA only; no trained model |
| `GET /api/compare` | same per region, latest snapshot ranked by EHI | none |
| `GET /api/alerts` | same + alert rules | none |
| `GET /api/train` | ingest → preprocess → features → `run_training` | RF, XGBoost, LSTM, CNN, Transformer trained **from scratch on every (uncached) request** |

- Models are **never saved or reloaded** — each call trains fresh; results (metrics + forecasts + top features) are cached as a dict for the day. There is no model file, no inference-only path and no retraining schedule.
- The dashboard first shows `demo_data.json`, then silently replaces it with live API data; the model panel calls `/api/train?region_id=<active>&days=240&horizon=14&epochs=25` with a 120-second timeout.
- Training is single-region; changing the region in the dropdown does not retrigger training in the current dashboard code (training is requested once after the live load).
- In `run_demo.py`, `run_training` receives the EHI frame (features + sub-indices + EHI); the API passes the features frame. The classifiers only use their 12 fixed columns either way, so results are equivalent.

---

## 7. Hyperparameter cheat-sheet

| Model | Key settings |
|---|---|
| Random Forest | 300 trees, unlimited depth, seed 42 |
| XGBoost | 300 rounds, depth 6, lr 0.1, softprob, seed 42 |
| LSTM | 2 layers × 64 hidden, dropout 0.1, head Linear(64→14) |
| CNN | 2 × Conv1d(32, k=3) + global avg pool, head Linear(32→14) |
| Transformer | d_model 32, 4 heads, 2 layers, FF 128, dropout 0.1, learned positions, mean pool, head Linear(32→14) |
| Forecast data | lookback 30, horizon 14, 85/15 time split, z-score, batch 16, Adam 1e-3, MSE |
| Classifier data | 75/25 stratified random split, seed 42, 12 features |
| PCA | 3 components on 6 standardised core variables |

---

## 8. Limitations and suggested next steps

1. **Replace proxy labels** with real ground truth (e.g. ESA WorldCover, field/expert labels) and **remove label-defining variables from the inputs**; otherwise classifier metrics mean nothing.
2. **Make the data real:** implement ERA5, set `OPENAQ_API_KEY`, and change the Planetary Computer client to return a time series (one value per scene date) instead of a single midpoint value.
3. **Give forecasters more history** (≥ 2–3 years) so they see full seasonal cycles, and add **baselines** (persistence, seasonal-naive, mean) to every metrics table.
4. **Time-aware splitting** for the classifiers and forecasters; compute normalisation statistics on the training portion only.
5. **Set torch seeds**, add early stopping, and report confidence over several runs.
6. **Persist trained models** and score new data instead of retraining on each request; move `/api/train` behind a job queue.
7. Calibrate EHI bounds/weights against reference data and consider feeding forecasts into the alert logic for genuine early warning.

---

## 9. Where the project documents disagree with the code

The upstream docs (`usermanual.md`, `project_summary.md`, `caching.md`, `changelog.md`, `challenges.md`, `README.md`) were read in full and checked against the source. This report follows the **code**. Differences to be aware of (e.g. if a viewer compares this report with the user manual):

| Document says | What the code actually does |
|---|---|
| User manual §4: RF example with accuracy 94.2 %, Kappa 0.88, top features `ndvi_roll30_mean` 38 %, `temp_distance` 11 % | That box is an **illustration**, not output. The classifiers never see `ndvi_roll30_mean` or any rolling-window/temperature-distance column; their 12 inputs are listed in §3.3. Real demo values: RF 95.1 % / Kappa 0.916, top feature `pm25` 24 %. |
| User manual: classes "Healthy / Moderate Stress / Degraded" | Classes are `degraded`, `stressed`, `healthy`. |
| User manual: CNN uses "causal dilated convolutions" | Plain `Conv1d(kernel=3, padding=1)` ×2 with no dilation and no causality, then global average pooling (§4.6). |
| User manual: Transformer models "cross-variable correlations" | All three forecasters are **univariate** — each sees only one variable's own history (§4.2). |
| User manual: LSTM = "best" in the example chart | "Best" is whichever architecture has the lowest validation RMSE per variable; it differs per variable (§4.9). |
| `project_summary.md` / `changelog.md` / `challenges.md`: parallel warm-up of all 13 regions on startup | Disabled in the current `main.py` (`startup_warmup` only logs). `caching.md` §5 documents this correctly. |
| `project_summary.md`: "5 live data clients", ERA5 as a working source | ERA5 raises `NotImplementedError`; temperature is always synthetic (§1.2). OpenAQ is live only with `OPENAQ_API_KEY`. |
| `caching.md`: pipeline decodes SCL cloud masks; Sentinel-5P NO₂ columns; ground-level PM2.5 | No SCL decoding exists (cloud screening is only the STAC `eo:cloud_cover < 20` filter). No Sentinel-5P client exists; NO₂/PM2.5 come from OpenAQ or the synthetic generator. |
| `project_summary.md`: feature stage computes NDVI/NDWI | NDVI/NDWI are computed in `planetary_computer_client.py` (ingestion) from regional-mean bands; the feature stage only pivots, adds rolling/anomaly columns and PCA (§1.3, §1.7). |
| `README.md` "Limitations": every number is synthetic | True for `demo_data.json` and offline runs. **Verified by a real run (§1.11):** with the satellite libs installed and network available, NDVI/NDWI/precipitation are real and `caching.md`'s Mumbai example (EHI 40.7, vegetation 21.4, climate 3.5) is reproduced exactly. Temperature, PM2.5 and NO₂ stay synthetic without keys/ERA5 code. |
| `challenges.md` / `changelog.md`: live ingestion completes "in seconds" after the COG-overview fix | Measured: Mumbai 26–51 s; `overview_level=4` errors on Sentinel-2/MODIS tiles and falls back to full-resolution reads; India did not finish in 30 min (§1.11). |
| `changelog.md` Change 4: gap-filling makes vegetation/water "complete for every day" | Complete, but only because one observation is copied across the whole window — the series is flat (std 0). |
| `project_summary.md`: `app.js` ≈ 1000 lines, `demo_data.json` ≈ 600 KB | Not verified against file sizes; the `app.js` read for this report is ~750 lines. Not relevant to ML/data. |

Everything else checked consistent with the code: EHI weights/formula, RF (300 trees), XGBoost `multi:softprob`, LSTM (2 layers, 64 hidden, dropout 0.1), Transformer (d_model 32, 4 heads, 2 layers), 30-day lookback, 14-day default horizon, the 13 regions, the endpoint list, alert thresholds, and the 60-minute two-tier cache. The repo's own `changelog.md` also confirms `classifiers.py`, `forecasting.py`, `evaluate.py`, `health_index.py`, `synthetic.py` and `spectral_indices.py` were **never modified** from the base code, so the ML/DL design here is the original, not the contributor's work; the contributor's changes were ingestion (Planetary Computer/NASA POWER), gap-filling, caching, API and frontend.


---

## 10. After the fix — current state (real data, 2026-10-01)

### 10.1 What changed
| Problem (from §§1–4) | Fix |
|---|---|
| Temperature, PM2.5, NO₂ synthetic; ERA5/OpenAQ unusable | New `open_meteo_client.py`: free, keyless **Open-Meteo** — ERA5 daily temperature & precipitation, ERA5-Land soil moisture, CAMS PM2.5 & NO₂ (hourly → daily mean) |
| Satellite NDVI = one flat, mis-scaled value; NDWI unusable | New `modis_ndvi_client.py`: **NASA ORNL MODIS MOD13Q1** point time series (250 m, 16-day composites, already cloud-screened, scale ×0.0001) → a real varying NDVI series. NDWI replaced by **soil moisture** (ERA5-Land) for the water sub-index (bounds 0.05–0.45 m³/m³) |
| Silent synthetic fallback | **Removed.** If a source fails the request fails. `synthetic.py` is a test fixture only |
| Heavy/slow raster code, API keys | `planetary_computer_client`, `era5_client`, `openaq_client`, `precipitation_client`, credentials and raster dependencies deleted. No keys needed |
| Classifier label leakage, random split | Target = condition **7 days ahead** (`LEAD_DAYS`), features from today, **chronological 75/25 split**, plus a "condition stays the same" naive baseline |
| Forecasters: no baseline, tiny history, look-ahead normalisation | **Naive (repeat-last-value) RMSE** reported next to every model; training history **730 days** (was 240); normalisation uses training portion only; 40 epochs |
| Stale dashboard text, old demo snapshot | Dashboard text updated, naive baselines shown, `demo_data.json` regenerated from real data for all 13 regions |

All 30 tests pass against the real APIs. Not changed: model architectures, EHI weights, alert rules.

### 10.2 Current data
Per region, at the region centroid: temperature_2m (K), precipitation_total (mm), soil_moisture (m³/m³), pm25, no2 (µg/m³) daily from Open-Meteo; ndvi from MODIS 16-day composites, linearly interpolated to daily by the gap-fill step. Mumbai, 730 days: NDVI 0.04–0.27 (monsoon cycle), PM2.5 mean 36 µg/m³ (range 7–118), precipitation up to 151 mm/day, temperature 296–305 K. Fetch time ≈ 30 s per region for 730 days.

**Known limitations of the real data**
- Single-point sampling at the centroid; poor representation for countries/continents (e.g. the Asia centroid lies in western China).
- MODIS composites are published with ~1–2 months delay: the latest weeks of NDVI are a carried-forward value; between composites NDVI is piecewise-linear.
- PM2.5/NO₂ are CAMS model reanalysis, not ground-station readings.
- ERA5 archive lags ~2 days; the last 1–2 days of temperature/precipitation/soil moisture are carried forward.
- Open-Meteo/ORNL availability is required at runtime (no fallback).

### 10.3 Current results — all 13 regions (730 days, 40 epochs, GPU, `dashboard/demo_data.json`)

Every region has its own trained models and its own validation results, shown in the dashboard's model panel when a region is selected. (The registry has 13 regions; there is no 14th.) Training is seeded, so a re-run reproduces the numbers (median RMSE difference 0.0, max 1 %).

**Classification — condition 7 days ahead, chronological test set (last 25 %)**

| Region | RF acc / Kappa | XGB acc / Kappa | Naive acc (condition unchanged) |
|---|---|---|---|
| India (IN) | 0.69 / 0.50 | 0.69 / 0.50 | 0.61 |
| United States (US) | 0.61 / 0.36 | 0.55 / 0.28 | 0.60 |
| Brazil (BR) | 0.78 / 0.66 | 0.77 / 0.63 | 0.82 |
| Australia (AU) | 0.86 / 0.00 | 0.86 / 0.00 | 0.78 |
| Kenya (KE) | 0.51 / 0.28 | 0.55 / 0.33 | 0.56 |
| Indonesia (ID) | 0.55 / 0.34 | 0.44 / 0.20 | 0.61 |
| Maharashtra (MH-IN) | 0.76 / 0.60 | 0.64 / 0.43 | 0.78 |
| California (CA-US) | 0.74 / 0.44 | 0.75 / 0.45 | 0.59 |
| Mumbai (MUM-IN) | 0.61 / 0.38 | 0.52 / 0.22 | 0.79 |
| São Paulo (SP-BR) | 0.41 / 0.11 | 0.40 / 0.10 | 0.36 |
| Asia (AS) | 0.77 / 0.53 | 0.75 / 0.52 | 0.70 |
| Africa (AF) | 0.63 / 0.38 | 0.55 / 0.28 | 0.57 |
| South America (SA) | 0.64 / 0.46 | 0.67 / 0.50 | 0.64 |

The best classifier beats the naive predictor in **8 of 13** regions; in the other 5 (BR, KE, ID, MH-IN, MUM-IN) "condition stays the same" is as good or better. Australia's Kappa of 0.00 means the model just predicts the majority class (the naive accuracy there is 0.78). Random Forest is the better classifier in 9 regions, XGBoost in 4. The label is still a rule-derived proxy.

**Forecasting — next 14 days: best of LSTM/CNN/Transformer vs naive (repeat last value)**

| Variable | Best model beats naive in | Typical RMSE improvement where it wins | Where it loses |
|---|---|---|---|
| PM2.5 | **11 of 13** regions | 3–29 % (e.g. US 27 %, SP-BR 29 %, IN 8 %) | ID (−43 %), MUM-IN (−14 %) |
| Precipitation | **11 of 13** regions | 4–31 % (e.g. BR 31 %, AU 29 %, IN 24 %) | ID (−21 %), CA-US (−209 %, dry region where naive ≈ 0) |
| NDVI | **2 of 13** regions (AS +9 %, AF +12 %) | — | 11 regions; naive is far better (e.g. MUM-IN −192 %, BR −154 %) |

Which architecture wins varies by region and variable: PM2.5 — CNN 9, LSTM 2, Transformer 2; precipitation — CNN 6, Transformer 4, LSTM 3; NDVI — LSTM 7, Transformer 5, CNN 1. No single architecture is best everywhere. R² is often negative (worse than predicting the validation-period mean) even where the model beats the naive forecast; the naive column is the fairer comparison.

**Why NDVI forecasting fails:** NDVI is 16-day composites interpolated to daily, with the newest ~6 weeks carried forward, so the series is smooth and "repeat the last value" is very hard to beat.

### 10.4 What to say when presenting
- Data is real and free (ERA5/CAMS reanalysis via Open-Meteo + NASA MODIS), keyless; all 13 regions have real trained models and real validation metrics.
- Classifiers are evaluated on a future-looking task against a naive baseline: they win in 8 of 13 regions — a modest but real signal, not 95 %.
- Forecasters beat the naive baseline for PM2.5 and precipitation in 11 of 13 regions each, but **not** for NDVI (2 of 13) — state this openly.
- Limitations: centroid sampling, CAMS modelled (not station) air quality, rule-derived labels, one training run per region on ~700 windows.

### 10.5 Still not fixed (deliberately, time-boxed)
Trained models are not saved (`/api/train` retrains per uncached call; the dashboard shows results precomputed by `scripts/run_demo.py`); univariate forecasters (no weather → PM2.5 links); region-average sampling; real ground-truth labels; EHI bounds/weights not calibrated (e.g. precipitation cap at 15 mm/day).

### 10.6 Training speed (RTX 4050 laptop GPU, 16 CPU threads)

| Step | Before | After |
|---|---|---|
| LSTM, 40 epochs, 600 windows, batch 16 | 86 s on CPU | **4.6 s on GPU** (19×); **1.0 s** at batch 64 |
| CNN / Transformer, same setup | 4.1 s / 16.7 s CPU | 3.2 s / 10.4 s GPU (bs 16); 0.7 s / 7.4 s (bs 64) |
| Whole training set | DataLoader on CPU | kept on the GPU, manual batching (no DataLoader overhead), batch 32 |
| 13 regions, end-to-end (fetch 2 years of data + train 5 models + validate) | 7 parallel workers, all CPU threads: **413 s** (workers thrashed: 20 s of training became 150–180 s) | 8 parallel workers, 1 CPU thread each, GPU shared: **150 s** (2.75× faster) |

What made it fast: (1) GPU for the recurrent/attention forecasters, (2) data resident on the GPU, (3) batch 32 instead of 16 (benchmarked), (4) one process per region so network fetches (~30 s each, the largest cost) overlap, (5) capping each worker to 1 CPU thread (`torch.set_num_threads(1)`, `GEO_N_JOBS=1` for RF/XGBoost) after finding that unrestricted threads made parallel training slower than sequential. Random Forest/XGBoost stay on the CPU (tiny data). Reproduce: `python scripts/run_demo.py --workers 8 --epochs 40` (override device with `GEO_DEVICE=cpu`).


---

## 11. Audit of the dashboard values, model improvements and live data (2026-10-01, latest)

### 11.1 Dashboard issues found and fixed
| Issue | Fix |
|---|---|
| Forecast metric cards clipped/overflowing after adding the naive-RMSE column | Responsive grid layout |
| Hero/method text still said "satellite spectral indices", "cloud masking"; forecasting heading listed only three models | Text updated to the real pipeline; heading now lists the ensemble |
| Alert text "Ehi for India dropped…" | Labelled "EHI" |
| Forecast chart x-axis label "Day 14" cut off | Right-aligned last label |
| Live mode never finished: the dashboard asked `/api/alerts?days=90`, which is not cached, so the server re-ingested all 13 regions one after another and the 30 s browser timeout aborted (the page silently stayed on the snapshot) | Dashboard and API default now use `days=180`, which reuses the cache. Live mode now syncs and matches the snapshot exactly |
| Live sync replaced region metadata and lost lat/lon/NDVI date; "snapshot date" used UTC | Metadata merged; local date used |
| No way to see data freshness or judge the models | Note under the trend chart (data-through date and latest usable NDVI composite), per-model "x % better/worse than naive" verdicts, **live Open-Meteo forecast overlay** on the forecast chart |

### 11.2 Are the dashboard values true? (independent checks)
| Value | Check | Verdict |
|---|---|---|
| Temperature | Open-Meteo (ERA5) vs NASA POWER (MERRA-2), last 150 days, all 13 centroids | **Genuine.** Correlation 0.93–0.97 in seasonal regions (IN 0.97, US 0.97, AU 0.95, MUM-IN 0.94); 0.34–0.65 only in near-constant tropical series (ID, KE, AF) where there is little variance; mean absolute difference 0.7–3 K |
| Precipitation | same comparison | **Genuine but noisy** (point rainfall differs between products): correlation 0.05–0.91 (MUM-IN 0.91, MH-IN 0.79, IN 0.71, AF 0.05); period means agree in magnitude (IN 5.9 vs 6.8 mm, MUM-IN 12.8 vs 15.7 mm) |
| **NDVI / Vegetation** | Raw MODIS values and the pixel-reliability flag | **Was FALSE, now fixed.** For India, June→August NDVI fell from 0.76 to 0.09 during the monsoon (it should be green). The July–August composites are 100 % cloud-flagged pixels (289 of 289 flagged cloudy), so the value was cloud noise, and the dashboard showed India vegetation 18.5. Mumbai and others were affected too. Fix: 5×5 km window, only good/marginal pixels, skip composites with <25 % clear pixels. India vegetation is now 71.6 (latest usable composite 26 Jun) |
| EHI arithmetic | Hand recomputation from raw values (India, today) | **Correct**: air 88.49, vegetation 71.64, water 76.69, climate 45.17, EHI 72.66, identical to the code |
| PM2.5 / NO₂ | No free keyless ground truth (OpenAQ needs a key); plausibility only | **Real CAMS model output, not measurements.** Mumbai 730-day mean 36 µg/m³ is plausible; treat as modelled |
| Region-level numbers | Method | **Point values at the region centroid**, not area averages — e.g. "Asia" is one point in western China, "Australia" a desert point (water index 0). Valid as a sample, not as a country/continent average |
| Live vs snapshot | Live API run vs `demo_data.json` | Identical EHI values for all 13 regions |

Other changes from this audit: the EHI climate sub-index now uses the **30-day mean rainfall** instead of single-day rainfall (single dry days made it spiky), and the last ~2 days of weather now come from the live feed (§11.5).

### 11.3 Model improvement experiments (13 regions, 5 variables, 4 setups, ~1000 models)
Median % RMSE improvement over the naive "repeat last value" forecast (higher is better; ensemble = average of LSTM/CNN/Transformer, no selection on the validation set):

| Variable | Univariate, absolute (old) | Univariate + residual learning | + other variables & season | + 3 years of history |
|---|---|---|---|---|
| PM2.5 | **14.1 %** (9/13) | 10.0 % | 12.9 % | 10.5 % |
| Precipitation | **23.9 %** (10/13) | 19.7 % | 16.0 % | 17.7 % |
| Temperature | **14.8 %** (9/13) | 9.3 % | 11.5 % | 12.6 % |
| Soil moisture | 13.6 % (8/13) | **12.2 %** (10/13) | 5.6 % | 6.5 % |
| NDVI | −39.8 % | −15.6 % | **+2.7 %** (7/13) | +2.8 % |

What was learned: (1) the ensemble was the most robust choice; (2) extra inputs and more history did **not** help PM2.5/rain/temperature (they made the CNN worse); (3) only NDVI needed weather/season inputs to get off −40 %; (4) residual learning helps soil moisture. Adopted per-variable setups (`FORECAST_CONFIG` in `train.py`): PM2.5/precipitation/temperature univariate; soil moisture with residual learning; NDVI with exogenous inputs + residual learning; all with the ensemble, 730 days, GPU, seeded. (These experiments used the pre-final NDVI frames, so the NDVI row is indicative; final figures are in §11.4.)
Classifier experiment (13 regions): adding season and lag features did **not** improve Random Forest or XGBoost (within ±0.4 pts), so the classifiers were left unchanged.

### 11.4 Final validation results (current `demo_data.json`, 13 regions)
**Classification (condition 7 days ahead, chronological 25 % test set):** Random Forest mean accuracy 0.691 vs naive 0.684 (beats naive in 11/13 regions, **mean margin only +0.7 pts**; Kappa 0.44); XGBoost 0.663 (beats naive in 7/13; Kappa 0.39). Exceptions: Indonesia 0.38 vs 0.70, California 0.51 vs 0.56. Honest summary: a 7-day-ahead prediction of a rule-derived class is barely better than "it stays the same".

**Forecasting (median % RMSE improvement over naive / regions beating naive):**

| Variable | LSTM | CNN | Transformer | Ensemble |
|---|---|---|---|---|
| PM2.5 | 9.6 % (8) | 10.0 % (10) | 9.9 % (8) | **12.0 % (9)** |
| Precipitation | 19.0 % (10) | 23.6 % (10) | 18.3 % (10) | **23.6 % (10)** |
| Temperature | 12.5 % (10) | 8.2 % (9) | 9.4 % (7) | **14.5 % (10)** |
| Soil moisture | −1.2 % (6) | 6.1 % (9) | −3.7 % (6) | **6.9 % (10)** |
| NDVI | −1.6 % (6) | −16.2 % (2) | −3.7 % (5) | −0.6 % (6) |

PM2.5, rainfall, temperature and soil moisture show real skill over naive (modest for soil moisture, good for rain); **NDVI forecasting has no skill over naive** (a slow 16-day composite signal whose newest 1–3 months are carried forward). The dashboard's "BEST" marker is chosen on the validation data and is therefore optimistic; the ensemble column is the unbiased figure.

### 11.5 Live feeds: what is used and what could be
**Implemented**
- **Live fill:** the Open-Meteo forecast API (`past_days`, no archive lag) now supplies the last ~2 days of temperature, precipitation and soil moisture, so series end today instead of carrying values forward.
- **Live forecast overlay:** the dashboard fetches Open-Meteo's live forecasts in the browser (CORS is open) and draws them next to our models for PM2.5 (CAMS, about 3–5 days ahead), precipitation and temperature (up to 16 days). This is a live benchmark against a professional numerical forecast.

**Possible next steps (not done)**
- Use the provider's *future* weather (and the CAMS PM2.5 forecast) as **input features** to the forecasters — the most promising accuracy gain for PM2.5, since weather drives pollution; Open-Meteo's historical-forecast API allows training on archived forecasts.
- Backtest the provider forecast against ours on the same validation window for a fair head-to-head.
- Ground-station air quality needs a free API key (OpenAQ v3); NASA FIRMS (fires) also needs a free key. NDVI cannot be made fresher with free keyless sources (MODIS/VIIRS lag 1–2 months, monsoon cloud).
- Scheduled refresh: re-running `scripts/run_demo.py` (about 5 minutes) on a schedule keeps the snapshot current.

### 11.6 Remaining limitations (for the presentation)
Centroid sampling; modelled (CAMS) air quality; rule-derived classification labels with a thin margin over naive; NDVI up to ~3 months old in cloudy regions (shown on the dashboard); trained models not saved; one seeded run per region (~700 windows), so individual region results carry sampling noise.


---

## 12. Latest state (after the Mumbai/Australia audit and the accuracy pass)

The complete, current description of the project is in **`GeoVision_Project_Technical_Report.md`**; this section only records what changed after §11.

**Data fixes (values that were too low for Mumbai and Australia)**
- *Water* used a 0–7 cm soil layer that is extremely volatile (Mumbai 0.26 → 0.07 in a week) → now the 0–100 cm root-zone layer. Mumbai water 6.6 → 54.
- *Single-centroid sampling* made Australia a desert point (water 0, vegetation 34.6, NDVI 0.25 vs a 0.45 land average) → every region is now the mean over up to 5 **land** points (centroid + 2×2 grid in the bbox; ocean points removed). Australia water 0 → 41, vegetation 35 → 58, EHI 44 → 62; Mumbai EHI 44 → 56. Continents and the US dropped (Africa 87 → 50, US 85 → 70) because they are now genuine multi-point averages (Sahara, deserts, forests together).
- Robustness: disk cache for raw API responses, HTTP-429 back-off, fail-fast on 4xx, ocean points in MODIS handled (they had aborted two regions).

**Accuracy improvements (all keep the data real; hold-outs untouched)**
| Change | Effect |
|---|---|
| One model per variable trained on **all regions pooled** (≈13× data) with per-region normalisation and per-region hold-out | PM2.5 regions beating naive 10 → 12/13; precipitation mean +5.7 → +14.9 %; NDVI mean −39 → +0.9 % |
| **Shrinkage** toward the naive forecast (weight α fitted on held-out *training* windows) | removes large negative outliers |
| 3 seeds per architecture, ensemble of 3 architectures | variance reduction |
| Classifiers: pooled + today's region-relative condition features + per-region standardisation | RF mean accuracy 0.634 → 0.686 (naive 0.652); worst region −20 → −12 pts |

**Final validation results (13 regions, pooled):** classification RF 0.686 / XGBoost 0.681 vs naive 0.652 (RF beats naive in 8/13, XGBoost 9/13). Forecasting, ensemble median improvement over naive: PM2.5 **+17.3 %** (12/13 regions), precipitation **+18.6 %** (11/13), temperature **+11.6 %** (13/13), NDVI +3.1 % (7/13, marginal), soil moisture **−5.4 %** (5/13, no skill — slow signal, honest negative).

**Still true:** PM2.5/NO₂ are CAMS model output; classification labels are a proxy rule; NDVI lags 1–2 months (June for India/Mumbai in the cloudy season); spatial support is ≤ 5 land points per region; models are not persisted.
