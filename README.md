# GeoVision AI

**A global near real-time environmental intelligence and forecasting platform.**

This is the implementation of the system described in
`GeoVision_AI_Project_Report.docx` (Introduction, Objectives, Literature
Survey, Methodology & System Architecture): a cloud-based platform that
ingests satellite and climate data in near real time, applies ML/DL models
for classification and forecasting, fuses the results into a composite
**Environmental Health Index (EHI)**, and presents everything through an
interactive dashboard with automated alerting.

```
Data sources → Ingestion/preprocessing → Feature engineering →
ML/DL analytics → Index fusion → API → Dashboard, with alerting throughout
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate a static demo snapshot from real data (needs internet, no API keys)
python scripts/run_demo.py   # fetches data + trains all models for all 13 regions in parallel (GPU if available, ~2.5 min)

# 2. Run the full stack (API + dashboard, served together)
uvicorn geovision.api.main:app --reload
# -> open http://127.0.0.1:8000
```

The dashboard tries the live API first and, if it can't reach one (e.g. the
file is opened as a static export), falls back to the bundled
`dashboard/demo_data.json` snapshot. Either way you get the same UI.

Run the tests:

```bash
# pytest and httpx come with requirements.txt; the suite calls the real (free) data APIs, ~5 min
pytest                    # full suite (~10s)
pytest -m "not slow"      # skip the end-to-end /api/train integration test
```

## Project layout

```
geovision/
  config.py                 Settings + the region registry (country/state/city/continent)
  ingestion/                 Data-source layer (Stage 1)
    base.py                  Observation contract + BaseIngestionClient
    open_meteo_client.py       Open-Meteo: temperature, precipitation, soil moisture, PM2.5, NO2
    modis_ndvi_client.py       NASA MODIS NDVI (ORNL web service)
    synthetic.py               Seeded synthetic generator (test fixture only)
    pipeline.py                 Orchestrates all sources with per-source fallback
  preprocessing/              Stage 2: range validation, gap-fill
  features/                    Stage 3: pivot, rolling stats, anomalies, PCA
  models/                      Stage 4/5: ML/DL training + validation
    classifiers.py              Random Forest & XGBoost (environmental condition class)
    forecasting.py               CNN, LSTM, Transformer (multi-horizon forecasting)
    evaluate.py                   RMSE, MAE, R², Accuracy, Kappa, ROC-AUC
    train.py                       Orchestrates + compares all models
  index/                        Stage 6: Environmental Health Index fusion
  alerting/                     Stage 8: threshold + rapid-decline alerts
  api/                           Stage 7: FastAPI app serving the dashboard's data
dashboard/                    Static frontend (vanilla HTML/CSS/JS, canvas charts)
scripts/run_demo.py            End-to-end CLI demo -> dashboard/demo_data.json
tests/                          pytest suite covering every layer above
```

## Data sources (all real, all free, no API keys)

| Variable | Source | Client |
|---|---|---|
| Temperature, precipitation | Open-Meteo archive (ERA5 reanalysis) | `ingestion/open_meteo_client.py` |
| Soil moisture 0-100 cm (water sub-index) | Open-Meteo archive (ERA5-Land) | `ingestion/open_meteo_client.py` |
| PM2.5, NO2 | Open-Meteo air-quality API (CAMS reanalysis) | `ingestion/open_meteo_client.py` |
| NDVI (vegetation) | NASA ORNL DAAC MODIS MOD13Q1, 250 m / 16-day composites | `ingestion/modis_ndvi_client.py` |

Values are the mean over the **land points of a grid inside each region's bounding box** (plus the centroid; ocean points are excluded, NDVI uses up to 5 land points with a 5 km window and clear-pixel filtering). There is **no synthetic fallback**:
if a source is down, the request fails rather than showing made-up numbers.
`ingestion/synthetic.py` remains only as a unit-test fixture.

## API

Once running (`uvicorn geovision.api.main:app`):

| Endpoint | Description |
|---|---|
| `GET /api/regions` | Region registry (id, name, level, coordinates) |
| `GET /api/ehi?region_id=IN&days=180` | EHI + sub-index time series for one region |
| `GET /api/compare?region_ids=IN,US,BR&days=180` | Latest snapshot across regions, for comparison |
| `GET /api/alerts?region_ids=IN,US&days=180` | Active threshold/rapid-decline alerts |
| `GET /api/train?region_id=IN&days=730&horizon=14&epochs=40` | Trains RF/XGBoost + LSTM/CNN/Transformer (+ensemble) for ONE region and returns validation metrics. The dashboard instead shows the pooled all-region results precomputed by `scripts/run_demo.py` |
| `GET /cache-status` | Interactive visual HTML dashboard showing live cache state for all 13 regions |
| `GET /api/regions/status` | Fast JSON status endpoint returning cache & EHI metrics for all 13 regions (<60ms) |
| `GET /api/cache/status` | In-memory and disk cache diagnostic overview |
| `POST /api/cache/clear` | Flushes in-memory cache and purges persistent disk `.pkl` files |
| `GET /api/health` | Liveness check |

`/api/train` runs the real training loop synchronously (~10-15s for the
default settings) and is cached per day; for production, put it behind a
task queue and a job-status endpoint instead of blocking the request.
Interactive docs are auto-served at `/docs`.

## Documentation

- **[GeoVision_Project_Technical_Report.md](GeoVision_Project_Technical_Report.md)** -- complete, current technical description of the whole project (architecture, data, models, API, dashboard, results, audits, limitations).
- **[GeoVision_ML_DL_Data_Report.md](GeoVision_ML_DL_Data_Report.md)** -- audit trail of the data/model investigation and the fixes made.
- `usermanual.md`, `caching.md`, `challenges.md`, `changelog.md`, `project_summary.md` -- written for an earlier version of the project (marked OUTDATED at the top); kept for history.

## Environmental Health Index

```
EHI = 0.30 · air_quality + 0.25 · vegetation + 0.25 · water + 0.20 · climate
```

Each sub-index is rescaled to 0 (poor) - 100 (excellent) from its raw
variables (PM2.5/NO2, NDVI, root-zone soil moisture, 30-day rainfall/temperature) before
weighting -- see `geovision/index/health_index.py` for the exact bounds,
which you should tune against ground truth for your deployment regions.
Weights live in `Settings.ehi_weights` (`geovision/config.py`).

## Extending

- **New region**: add an entry to `REGION_REGISTRY` in `geovision/config.py`
  (bbox, centroid, level).
- **New data source**: implement `BaseIngestionClient`, register it in
  `IngestionPipeline._clients` (it must return `Observation`s for the core variables).
- **New model architecture**: for forecasting, add a class to
  `forecasting.py` implementing `forward(x) -> (batch, horizon)` and wire
  it into `_make_model`; for classification, extend
  `EnvironmentalClassifier.__init__`.
- **Real ground-truth labels**: swap `derive_condition_labels()` in
  `classifiers.py` (currently a literature-informed proxy rule) for actual
  labeled data once available.

## Limitations

- Regional values are a mean over a handful of land sample points (not every pixel); coarse for continents.
- MODIS NDVI is published with ~1-2 months lag; the newest weeks are carried forward, and
  the series is piecewise-linear between 16-day composites.
- PM2.5/NO2 are CAMS model reanalysis, not ground-station measurements.
- The classifiers predict a rule-derived condition label one week ahead (not field-verified ground truth).
- Forecasters are univariate and trained per request on ~2 years of daily data; the dashboard shows a
  naive (repeat-last-value) baseline next to every model -- for NDVI the naive forecast wins.
- `/api/train` is synchronous and retrains on every uncached request.
