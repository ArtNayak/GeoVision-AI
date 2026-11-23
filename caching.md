> **OUTDATED (2026-10-01):** this document was written for an earlier version of the project (Planetary Computer / ERA5 / OpenAQ / CHIRPS data, single-point sampling, per-region models). The data sources, preprocessing, models and results have since changed. For the current, accurate description read **`GeoVision_Project_Technical_Report.md`**.

# Caching Architecture & Implementation Guide — GeoVision AI

This document provides a comprehensive technical guide to the **Two-Tier Caching System** in GeoVision AI. It explains why caching is necessary for planetary-scale satellite data, how in-memory and persistent disk caches work, how cache keys are structured, how edge cases (like DataFrame truthiness and JSON NaN compliance) are handled, and how to operate and monitor the cache via APIs and interactive web dashboards.

---

## Table of Contents
1. [Why Caching is Critical](#1-why-caching-is-critical)
2. [Two-Tier Cache Architecture](#2-two-tier-cache-architecture)
   - [Tier 1: In-Memory TTL Cache](#tier-1-in-memory-ttl-cache)
   - [Tier 2: Persistent Disk Cache](#tier-2-persistent-disk-cache)
3. [Cache Key Schema & Disk Filename Mapping](#3-cache-key-schema--disk-filename-mapping)
4. [Request Resolution Lifecycle](#4-request-resolution-lifecycle)
5. [Overcoming the Startup GIL Bottleneck](#5-overcoming-the-startup-gil-bottleneck)
6. [Cache Management & Monitoring Endpoints](#6-cache-management--monitoring-endpoints)
   - [Interactive Web Dashboard (`/cache-status`)](#interactive-web-dashboard-cache-status)
   - [Regional Status JSON API (`/api/regions/status`)](#regional-status-json-api-apiregionsstatus)
   - [System Cache Diagnostics (`/api/cache/status`)](#system-cache-diagnostics-apicachestatus)
   - [Cache Flush Endpoint (`/api/cache/clear`)](#cache-flush-endpoint-apicacheclear)
7. [Edge Cases & Technical Solutions](#7-edge-cases--technical-solutions)
   - [DataFrame Truthiness Ambiguity](#dataframe-truthiness-ambiguity)
   - [Safe Float & NaN/Inf JSON Serialization](#safe-float--naninf-json-serialization)
   - [Atomic Disk Writes](#atomic-disk-writes)
   - [Historical Snapshot Recovery](#historical-snapshot-recovery)
8. [Configuration & Tuning](#8-configuration--tuning)

---

## 1. Why Caching is Critical

GeoVision AI ingests multimodal planetary data from disparate Earth Observation sources:
- **Optical & Thermal Imagery:** Microsoft Planetary Computer STAC (Sentinel-2 L2A BOA reflectance, Landsat-8/9 thermal infrared, MODIS surface reflectance).
- **Atmospheric Reanalysis:** Copernicus CDS / ERA5 (2-metre air temperature, boundary layer dynamics).
- **Precipitation:** NASA POWER daily satellite-gauge precipitation records.
- **Air Quality:** Ground-level PM2.5 and Sentinel-5P tropospheric NO2 column densities.

### The Cost of Live Ingestion
Running the ingestion pipeline for a single region over a 180-day window requires:
1. Querying STAC item endpoints across multiple dates.
2. Streaming Cloud-Optimized GeoTIFF (COG) byte ranges over HTTP.
3. Decoding Scene Classification Layer (SCL) masks and masking clouds/shadows.
4. Computing band math (NDVI, NDWI, LST), rolling statistics, and PCA transformations.
5. Fusing composite sub-indices into the Environmental Health Index (EHI).

This raw pipeline takes **20 to 60+ seconds per region**. If the interactive dashboard or API queried live satellite providers on every user click or every 5-second polling tick, users would experience severe latency and rate-limit exhaustion.

**With the caching system:**
- Subsequent requests for the same region respond in **< 1 ms** (from memory).
- Cold requests after server restart respond in **~10–50 ms** (from persistent disk cache).
- External API calls and heavy array computations are eliminated during regular dashboard usage.

---

## 2. Two-Tier Cache Architecture

GeoVision AI utilizes a **hybrid two-tier caching architecture** implemented in [`geovision/api/service.py`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/geovision/api/service.py) and surfaced through [`geovision/api/main.py`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/geovision/api/main.py):

```
                       ┌─────────────────────────────────────────┐
                       │         Incoming Client Request         │
                       │   (Dashboard / API / Alert Evaluator)   │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │       Tier 1: In-Memory TTL Cache       │
                       │      service._CACHE (Python dict)       │
                       └────────────────────┬────────────────────┘
                                            │
                           Cache Hit? ──────┴────── Cache Miss?
                               │                         │
                               ▼                         ▼
                      ┌─────────────────┐       ┌────────────────────────────────┐
                      │  Return in <1ms │       │   Tier 2: Persistent Disk      │
                      └─────────────────┘       │   data/cache/*.pkl (Pickle)    │
                                                └────────────────┬───────────────┘
                                                                 │
                                                Disk Hit? ───────┴────── Disk Miss?
                                                    │                         │
                                                    ▼                         ▼
                                           ┌─────────────────┐       ┌─────────────────┐
                                           │ Read Disk .pkl  │       │ Live Ingestion  │
                                           │ Promote to Tier1│       │ Pipeline Run    │
                                           │ Return in ~20ms │       │ (20–60 seconds) │
                                           └─────────────────┘       └────────┬────────┘
                                                                              │
                                                                              ▼
                                                                     ┌─────────────────┐
                                                                     │ Save to Memory  │
                                                                     │ Atomic Disk Dump│
                                                                     └─────────────────┘
```

### Tier 1: In-Memory TTL Cache
- **Location:** In-process Python dictionary `service._CACHE: dict[str, tuple[float, Any]]`.
- **Payload:** Maps a cache key string to a tuple of `(creation_timestamp_epoch, pandas_dataframe)`.
- **Latency:** `< 0.5 ms`.
- **TTL (Time-To-Live):** Configurable via `settings.ingestion_interval_minutes * 60` (default: **3,600 seconds / 60 minutes**).
- **Eviction/Invalidation:** On lookup, if `time.time() - creation_timestamp >= TTL`, the entry is treated as expired. Calling `POST /api/cache/clear` immediately flushes the dictionary.

### Tier 2: Persistent Disk Cache
- **Location:** Stored on the local filesystem in the `data/cache/` directory.
- **Payload:** High-performance Python Pickle binary files (`.pkl`) containing serialized pandas DataFrames.
- **Latency:** `~1–3 ms` per file read.
- **Persistence:** Survives application shutdowns, process restarts, code updates, and Uvicorn `--reload` worker cycles.
- **Concurrency & Atomic Writes:** Uses atomic file replacement (writing to a `.tmp` file and renaming via `tmp_file.replace(disk_file)`) to prevent corrupted reads during concurrent worker access.

---

## 3. Cache Key Schema & Disk Filename Mapping

Each cache entry is partitioned by:
1. **Domain:** `ehi` (Environmental Health Index).
2. **Region ID:** Unique ISO / regional identifier (e.g., `MUM-IN`, `US`, `AF`, `CA-US`).
3. **Time Window (Days):** The temporal lookback window (e.g., `180`, `90`, `60`).
4. **Partition Date:** The ISO calendar date (`YYYY-MM-DD`) when the data was generated.

### In-Memory Key Format
```
ehi:{region_id}:{days}:{date}
```
**Examples:**
- `ehi:MUM-IN:180:2026-09-30`
- `ehi:CA-US:180:2026-09-30`
- `ehi:AF:180:2026-09-30`

### Disk Filename Mapping
Because filesystem paths cannot contain colons (`:`) on Windows and other operating systems, keys are sanitized using regular expressions (`re.sub(r"[^a-zA-Z0-9_.-]", "_", key)`):

| In-Memory Cache Key | Disk Cache Filename (`data/cache/`) |
| :--- | :--- |
| `ehi:MUM-IN:180:2026-09-30` | `ehi_MUM-IN_180_2026-09-30.pkl` |
| `ehi:CA-US:180:2026-09-30` | `ehi_CA-US_180_2026-09-30.pkl` |
| `ehi:IN:90:2026-08-19` | `ehi_IN_90_2026-08-19.pkl` |

### Why Date Partitioning?
Including the execution date (`YYYY-MM-DD`) in the key ensures:
1. Daily freshness: When midnight passes, the system naturally requests a new key, fetching fresh satellite scenes for the new day.
2. Historical snapshots: Past runs remain on disk as archives that can be referenced for offline analysis or trend verification.

---

## 4. Request Resolution Lifecycle

When a service method (such as `service.region_ehi_dataframe(region_id, days=180)`) is called:

1. **Calculate Cache Key:**
   ```python
   today = date.today().isoformat()
   cache_key = f"ehi:{region_id}:{days}:{today}"
   ```
2. **Tier 1 Check (Memory):**
   - Check if `cache_key` exists in `_CACHE`.
   - If present and `now - timestamp < TTL`, return the in-memory `DataFrame` immediately.
3. **Tier 2 Check (Disk):**
   - Compute `disk_file = _disk_cache_path(cache_key)`.
   - If `disk_file.exists()` and file modification time `now - mtime < TTL`:
     - Load with `pickle.load(f)`.
     - Promote entry into `_CACHE[cache_key] = (mtime, disk_val)`.
     - Return the loaded `DataFrame`.
4. **Compute & Ingest (Miss):**
   - Execute the builder function: run `IngestionPipeline`, cloud-masking, feature matrix extraction, and composite EHI computation.
   - Store in memory: `_CACHE[cache_key] = (now, value)`.
   - Write to disk atomically:
     ```python
     tmp_file = disk_file.with_suffix(".tmp")
     with open(tmp_file, "wb") as f:
         pickle.dump(value, f)
     tmp_file.replace(disk_file)
     ```
   - Return fresh `DataFrame`.

---

## 5. Overcoming the Startup GIL Bottleneck

### The Previous Architecture Flaw
In earlier iterations, server startup included a background thread pool that attempted to pre-warm all 13 regions by calling the live ingestion pipeline on boot. This caused a critical issue:
- Ingestion involves heavy GDAL/Rasterio C-extension calls, NumPy matrix math, and HTTP streaming.
- These operations saturate the Python Global Interpreter Lock (GIL) and CPU threads.
- As a result, the single Uvicorn event loop was blocked, and simple API requests (including health checks and status queries) hung or timed out for 10–15 minutes.

### The Modern Solution: On-Demand Persistence
1. **Disabled Blocking Warm-Up:** The startup hook in [`geovision/api/main.py`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/geovision/api/main.py) no longer triggers external ingestion on boot. The server starts **instantly in < 1 second**.
2. **Persistent Disk Reliance:** Because disk files (`.pkl`) persist across restarts, the server loads existing snapshots on first access.
3. **Historical Snapshot Recovery:** If the date has rolled over and a region has not yet completed today's ingestion, the system locates the latest available disk snapshot for that region (`ehi_{safe_rid}_*.pkl`), ensuring that metrics and health scores are available immediately without lag.

---

## 6. Cache Management & Monitoring Endpoints

### Interactive Web Dashboard (`/cache-status`)
- **URL:** [http://127.0.0.1:8000/cache-status](http://127.0.0.1:8000/cache-status)
- **Protocol:** HTTP GET (Returns HTML5)
- **Features:**
  - Modern dark-mode UI with JetBrains Mono typography.
  - Key performance indicators: Total Regions (13), Cached in Memory/Disk, Pending, and TTL countdown.
  - Table of all 13 registered regions showing:
    - Administrative Level (Continent / Country / State / City).
    - Status Badge: `Memory` (Green), `Disk Cached` (Cyan), or `Pending` (Amber).
    - Environmental Health Score (EHI) with color-coded badges.
    - Sub-indicators: 💨 Air Quality, 🌿 Vegetation, 💧 Water, 🌡️ Climate.
    - Historical sample size (e.g., `181 rows`).
    - Cache age timer (e.g., `12m 45s ago`).
  - Interactive **"🔄 Refresh Now"** button and automatic polling every 10 seconds.
  - Quick-action buttons to jump directly to the raw JSON API or the main GeoVision AI dashboard.

### Regional Status JSON API (`/api/regions/status`)
- **URL:** [http://127.0.0.1:8000/api/regions/status](http://127.0.0.1:8000/api/regions/status)
- **Protocol:** HTTP GET (Returns JSON)
- **Response Time:** `< 60 ms`
- **Output Schema:**
  ```json
  {
    "status": "ok",
    "total_regions": 13,
    "cached_regions": 13,
    "pending_regions": 0,
    "ttl_seconds": 3600,
    "date": "2026-09-30",
    "regions": [
      {
        "region_id": "MUM-IN",
        "name": "Mumbai, India",
        "level": "city",
        "status": "cached_disk",
        "cached": true,
        "ehi": 40.7,
        "air_quality": 71.3,
        "vegetation": 21.4,
        "water": 53.0,
        "climate": 3.5,
        "records": 181,
        "age_seconds": 1658,
        "expires_in_seconds": 1942,
        "cache_file": "ehi_MUM-IN_180_2026-09-30.pkl"
      }
    ]
  }
  ```

### System Cache Diagnostics (`/api/cache/status`)
- **URL:** [http://127.0.0.1:8000/api/cache/status](http://127.0.0.1:8000/api/cache/status)
- **Protocol:** HTTP GET (Returns JSON)
- Provides diagnostic details regarding active in-memory keys, disk file inventory in `data/cache/`, and current TTL configuration.

### Cache Flush Endpoint (`/api/cache/clear`)
- **URL:** [http://127.0.0.1:8000/api/cache/clear](http://127.0.0.1:8000/api/cache/clear)
- **Protocol:** HTTP POST (Returns JSON)
- **Action:**
  1. Clears all entries from `service._CACHE`.
  2. Unlinks and deletes all `.pkl` files in `data/cache/`.
  3. Returns total entries and files purged:
     ```json
     {
       "status": "cleared",
       "entries_removed": 27
     }
     ```
  Subsequent queries will re-trigger fresh satellite ingestion from Planetary Computer.

---

## 7. Edge Cases & Technical Solutions

### DataFrame Truthiness Ambiguity
**Problem:** In Python, evaluating `if series:` when `series` is a pandas `DataFrame` raises:
```
ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```
Furthermore, `series[-1]` attempts a column lookup rather than accessing the last row.

**Solution:** The status extractor `_extract_series_snapshot()` safely handles DataFrames and lists:
```python
def _extract_series_snapshot(series_obj):
    if series_obj is None:
        return {}, 0
    try:
        if hasattr(series_obj, "iloc"):  # pandas DataFrame
            n = len(series_obj)
            if n == 0:
                return {}, 0
            row = series_obj.iloc[-1].to_dict()
            return row, n
        elif isinstance(series_obj, list):
            n = len(series_obj)
            if n == 0:
                return {}, 0
            row = series_obj[-1] if isinstance(series_obj[-1], dict) else {}
            return row, n
    except Exception:
        pass
    return {}, 0
```

### Safe Float & NaN/Inf JSON Serialization
**Problem:** Satellite indices (such as water or snow indices over arid areas) can produce IEEE 754 `NaN` or `Inf` floating point values. FastAPI's JSON encoder crashes when serializing these:
```
ValueError: Out of range float values are not JSON compliant
```

**Solution:** All numerical fields in the status endpoint pass through `_safe_float()`:
```python
def _safe_float(val, digits=1):
    if val is None:
        return None
    try:
        import math
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, digits)
    except (ValueError, TypeError):
        return None
```
In the broader service layer, `_sanitize()` recursively traverses data structures and converts invalid float tokens into `None` (which serializes safely as `null` in JSON).

### Atomic Disk Writes
**Problem:** If the server process is killed or reloaded while `pickle.dump()` is writing to disk, the destination `.pkl` file becomes truncated and corrupted. Future server startups would fail when unpickling.

**Solution:** Writes are directed to a sibling `.tmp` file and committed atomically via filesystem rename:
```python
tmp_file = disk_file.with_suffix(".tmp")
with open(tmp_file, "wb") as f:
    pickle.dump(value, f)
tmp_file.replace(disk_file)  # Atomic rename on POSIX and modern Windows NTFS
```

### Historical Snapshot Recovery
**Problem:** If a server starts on `2026-10-01` but the last ingestion ran on `2026-09-30`, an exact search for `ehi_{rid}_180_2026-10-01.pkl` would report the region as `pending`, even though high-quality snapshot data is available on disk.

**Solution:** The status lookup checks for today's file first. If absent, it queries `CACHE_DIR.glob(f"ehi_{safe_rid}_*.pkl")` and selects the file with the most recent modification time (`max(matches, key=lambda p: p.stat().st_mtime)`), serving valid metrics instantly.

---

## 8. Configuration & Tuning

All cache parameters can be customized via environment variables in `.env` or system environment configurations:

| Parameter | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| **Ingestion Cadence / TTL** | `INGESTION_INTERVAL_MIN` | `60` | In-memory cache expiration window in minutes (60m = 3600s). |
| **Cache Storage Directory** | `CACHE_DIR` (in `config.py`) | `data/cache` | Path where persistent `.pkl` files are stored. |
| **Spatial Resolution** | `DEFAULT_SCALE_M` | `250` | Resolution (in metres) used when extracting zonal statistics. |
| **Forecast Horizon** | `FORECAST_HORIZON_DAYS` | `14` | Number of days forecast by Deep Learning models. |

### Adjusting TTL
To change the cache duration to 2 hours, set in `.env`:
```bash
INGESTION_INTERVAL_MIN=120
```

### Forcing a Complete Cache Refresh
To discard existing caches and pull fresh satellite data:
```bash
# Using curl:
curl -X POST http://127.0.0.1:8000/api/cache/clear

# Or manually via PowerShell:
Remove-Item -Path "data\cache\*.pkl" -Force
```
After clearing, the next request for any region will automatically execute the live ingestion pipeline and re-populate the cache tiers.
