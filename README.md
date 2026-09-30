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

# 1. Generate a static demo snapshot (works with zero credentials --
#    ingestion falls back to a seeded synthetic generator, see "Data modes")
python scripts/run_demo.py

# 2. Run the full stack (API + dashboard, served together)
uvicorn geovision.api.main:app --reload
# -> open http://127.0.0.1:8000
```

The dashboard tries the live API first and, if it can't reach one (e.g. the
file is opened as a static export), falls back to the bundled
`dashboard/demo_data.json` snapshot. Either way you get the same UI.

Run the tests:

```bash
pip install pytest
pytest                    # full suite (~10s)
pytest -m "not slow"      # skip the end-to-end /api/train integration test
```

## Project layout

```
geovision/
  config.py                 Settings + the region registry (country/state/city/continent)
  ingestion/                 Data-source layer (Stage 1)
    base.py                  Observation contract + BaseIngestionClient
    planetary_computer_client.py  Microsoft Planetary Computer (STAC): Sentinel-2, Landsat-8/9, MODIS
    era5_client.py             ERA5 climate reanalysis (Copernicus CDS)
    precipitation_client.py    Precipitation via NASA POWER (free, no key)
    openaq_client.py           OpenAQ ground air-quality stations
    synthetic.py               Seeded synthetic generator (offline dev/demo fallback)
    pipeline.py                 Orchestrates all sources with per-source fallback
  preprocessing/              Stage 2: cloud masking, range validation, gap-fill
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

## Data modes: live vs. synthetic

Every ingestion client implements the same `fetch(region_id, start, end) ->
list[Observation]` interface (`geovision/ingestion/base.py`). Each checks
`is_configured()` before hitting its live API; if credentials are missing,
`IngestionPipeline` transparently substitutes the seeded synthetic
generator **for that source only**, so the rest of the pipeline never
needs to know which mode it's in.

Planetary Computer and NASA POWER need **no credentials at all** -- their
data is public. Set whichever of the rest you have:

| Variable | Source | Used by |
|---|---|---|
| `PC_SDK_SUBSCRIPTION_KEY` *(optional)* | Microsoft Planetary Computer | `planetary_computer_client.py` |
| `CDS_API_KEY` | Copernicus Climate Data Store (ERA5) | `era5_client.py` |
| `OPENAQ_API_KEY` | OpenAQ v3 | `openaq_client.py` |
| *(none needed)* | NASA POWER (precipitation) | `precipitation_client.py` |

Notes:
- **Planetary Computer**: works immediately with `pip install
  pystac-client planetary-computer rioxarray odc-stac` -- no signup
  required. A free subscription key from
  [planetarycomputer.microsoft.com/account/request](https://planetarycomputer.microsoft.com/account/request)
  just raises your request-rate limit.
- **ERA5**: `era5_client.py` documents the CDS async request/download
  cycle needed for live retrieval (`cdsapi.Client().retrieve(...)`) --
  wire it up once you have a CDS account; it's intentionally left as a
  documented extension point rather than silently faked.
- **OpenAQ**: get a free API key at [explore.openaq.org](https://explore.openaq.org).
- **Precipitation**: CHIRPS itself isn't hosted on Planetary Computer, so
  this variable now comes from NASA POWER's public REST API instead --
  free forever, no key.

Without a network connection at all, `python scripts/run_demo.py` and
`uvicorn geovision.api.main:app` both still work out of the box on
synthetic data -- useful for development, demos, and this sandbox.

## API

Once running (`uvicorn geovision.api.main:app`):

| Endpoint | Description |
|---|---|
| `GET /api/regions` | Region registry (id, name, level, coordinates) |
| `GET /api/ehi?region_id=IN&days=180` | EHI + sub-index time series for one region |
| `GET /api/compare?region_ids=IN,US,BR&days=180` | Latest snapshot across regions, for comparison |
| `GET /api/alerts?region_ids=IN,US&days=90` | Active threshold/rapid-decline alerts |
| `GET /api/train?region_id=IN&days=240&horizon=14&epochs=40` | Runs RF/XGBoost + CNN/LSTM/Transformer and returns validation metrics |
| `GET /cache-status` | Interactive visual HTML dashboard showing live cache state for all 13 regions |
| `GET /api/regions/status` | Fast JSON status endpoint returning cache & EHI metrics for all 13 regions (<60ms) |
| `GET /api/cache/status` | In-memory and disk cache diagnostic overview |
| `POST /api/cache/clear` | Flushes in-memory cache and purges persistent disk `.pkl` files |
| `GET /api/health` | Liveness check |

`/api/train` runs the real training loop synchronously (~10-15s for the
default settings) and is cached per day; for production, put it behind a
task queue and a job-status endpoint instead of blocking the request.
Interactive docs are auto-served at `/docs`.

## Detailed Documentation Guides

- **[User Manual (`usermanual.md`)](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/usermanual.md):** Complete guide on operating the platform, understanding the dashboard, and interpreting ML/DL models.
- **[Caching Architecture Guide (`caching.md`)](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/caching.md):** Deep technical dive into the two-tier caching system (Memory + Disk `.pkl`), key schemas, atomic writes, and performance optimizations.
- **[Engineering Challenges & Solutions (`challenges.md`)](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/challenges.md):** Log of real-world bugs, architectural hurdles (GIL locks, NaN serialization, PyTorch multi-threading), and their resolutions.


## Environmental Health Index

```
EHI = 0.30 · air_quality + 0.25 · vegetation + 0.25 · water + 0.20 · climate
```

Each sub-index is rescaled to 0 (poor) - 100 (excellent) from its raw
variables (PM2.5/NO2, NDVI, NDWI, precipitation/temperature) before
weighting -- see `geovision/index/health_index.py` for the exact bounds,
which you should tune against ground truth for your deployment regions.
Weights live in `Settings.ehi_weights` (`geovision/config.py`).

## Extending

- **New region**: add an entry to `REGION_REGISTRY` in `geovision/config.py`
  (bbox, centroid, level).
- **New data source**: implement `BaseIngestionClient`, register it in
  `IngestionPipeline`, add its variables to `SOURCE_GROUPS`.
- **New model architecture**: for forecasting, add a class to
  `forecasting.py` implementing `forward(x) -> (batch, horizon)` and wire
  it into `_make_model`; for classification, extend
  `EnvironmentalClassifier.__init__`.
- **Real ground-truth labels**: swap `derive_condition_labels()` in
  `classifiers.py` (currently a literature-informed proxy rule) for actual
  labeled data once available.

## Limitations of this build

- No live credentials are wired into this environment, so every number
  you see (EHI values, alerts, model metrics) is generated from the
  seeded synthetic client, not real satellite/climate data. The
  ingestion/preprocessing/feature/model/index/alert *code paths* are real
  and unit-tested; only the upstream data is simulated.
- The classifiers train against a rule-based proxy label
  (`derive_condition_labels`), not field-verified ground truth -- replace
  it before trusting the classification metrics for anything real.
- `/api/train` is synchronous; fine for a demo, not for a production
  near-real-time SLA -- see the API section above.
