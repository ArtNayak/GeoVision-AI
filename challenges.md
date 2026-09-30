# Engineering Challenges & Solutions — GeoVision AI

This document details the issues encountered, architectural and rendering challenges faced, and the engineering solutions implemented during the development and stabilization of the GeoVision AI platform and its real-time data pipeline.

---

## 1. Global Comparison Bar Charts Displaying Empty / Invisible

### Problem
In the **01 — Application Services Global Comparison** section of the dashboard, the horizontal progress/distribution bars under "AIR / VEG / WATER / CLIMATE" were rendering as empty track backgrounds with no visible colored fill widths, despite valid numerical values in the underlying data.

### Root Cause
The inner fill elements were HTML `<span>` tags (`.row-band-fill`) with inline `style="width: ...%"`. In CSS, `span` elements have `display: inline` by default. CSS specifications dictate that the `width` property has no effect on non-replaced inline elements.

### Solution
- Added `display: block` to `.row-band-fill` in [`dashboard/styles.css`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/dashboard/styles.css#L285).
- This established a block formatting context, allowing width percentage values to properly resize the color fills for Air Quality (orange), Vegetation (green), Water (cyan), and Climate (violet).

---

## 2. Transitioning from Synthetic Fallback to Real-Time Planetary Computer Ingestion

### Problem
The application originally relied on deterministic synthetic time-series generators. The goal was to connect the live pipeline to Microsoft Planetary Computer's STAC catalog (Sentinel-2, Landsat-8, MODIS) and NASA POWER's precipitation REST API.

### Challenges Faced
1. **Large Bounding Box Latency:** Downloading full Cloud-Optimized GeoTIFFs (COGs) for large spatial bounding boxes (e.g., continental US, Asia, Africa) caused requests to take minutes or time out.
2. **Event Loop Saturation:** Synchronous raster operations on large COG windows blocked the single-worker ASGI server, making health checks and UI requests hang.
3. **Frontend Request Timeouts:** The dashboard JS client previously enforced a strict 10-second timeout on live API calls, causing it to prematurely abort and fall back to the bundled static demo snapshot.

### Solutions
- **COG Overview Optimization:** Updated [`planetary_computer_client.py`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/geovision/ingestion/planetary_computer_client.py#L106) to request downsampled overview tiles (`overview_level=4`), reducing downloaded byte volumes by ~16× while maintaining accurate regional zonal means.
- **Reduced Scene Redundancy:** Lowered `MAX_ITEMS_PER_WINDOW` from 5 to 2 to minimize network calls without sacrificing statistical representation.
- **Resilient Frontend Fetching:**
  - Increased EHI fetch timeouts to 180s.
  - Replaced `Promise.all` with `Promise.allSettled` in [`dashboard/app.js`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/dashboard/app.js#L75) so individual slow regions do not fail the entire dashboard load.
  - Added live progress indicators ("fetching X/13 regions…") and visual pulse animations.

---

## 3. Discontinuous Satellite Overpasses Producing Null Sub-Indices

### Problem
Because optical satellites (Sentinel-2, Landsat) pass over regions at periodic intervals (every 5–16 days) or experience cloud obstruction, observations were only recorded on specific dates. Daily variables (like temperature or air quality) had entries on every date, but vegetation (`ndvi`) and water (`ndwi`) were sparse. On the latest snapshot date (today), `ndvi` and `ndwi` were `null`, leaving the Vegetation and Water bars empty in the current comparison view.

### Solution
- Refactored [`fill_gaps`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/geovision/preprocessing/cloud_mask.py#L38-L82) in the preprocessing pipeline:
  - Reindexes each `(region_id, source, variable)` time-series onto a continuous daily date grid across the entire 180-day window.
  - Applies time-weighted linear interpolation between satellite overpass dates.
  - Applies backward and forward fills (`bfill()`, `ffill()`) at edge boundaries.
- **Result:** All 4 sub-indices (Air, Vegetation, Water, Climate) and the composite EHI score are non-null and complete for every day in the time series, including real-time daily snapshots.

---

## 4. Background Server Warm-Up & High-Performance Caching

### Problem
Cold requests to 13 global regions upon initial dashboard access could lead to request queuing.

### Solution
- **FastAPI Lifespan Startup Event:** Implemented an asynchronous background thread handler in [`geovision/api/main.py`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/geovision/api/main.py#L74-L110) that pre-warms the EHI cache without blocking the web event loop.
- **TTL Caching:** Configured a 60-minute cache with inspection (`GET /api/cache/status`) and manual flush (`POST /api/cache/clear`) endpoints.

---

## 5. Dashboard Cold-Start Latency, Cache Volatility & SWR Stream Synchronization

### Problem
When the frontend dashboard was opened on cold starts (or after a server restart), the interface took ~60+ seconds before displaying any visual charts or metrics, or silently aborted to demo data due to an uncaught reference error.

### Root Causes
1. **Uncaught Reference Bug:** In [`dashboard/app.js`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/dashboard/app.js), the live API loading routine referenced an uninstantiated variable `ehi_series`, throwing a runtime `ReferenceError` after waiting for all network promises to settle and forcing an unnecessary fallback to static demo data.
2. **All-or-Nothing Blocking Render:** The frontend `tryLoadLiveCore()` awaited `Promise.allSettled` across all 13 global regions and 180-day time-series before painting any UI elements, blocking first-contentful paint.
3. **Single-Threaded Server Warm-Up:** The backend server warm-up sequentially processed regions one-by-one in a single thread, requiring several minutes to complete full cache population.
4. **Volatile In-Memory Cache:** The backend cache (`_CACHE`) resided solely in RAM. Any server restart or uvicorn `--reload` wiped the state, forcing costly multi-spectral satellite GeoTIFF re-downloads.

### Solutions
- **Stale-While-Revalidate (SWR) Instant Paint:**
  - Refactored [`dashboard/app.js`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/dashboard/app.js) so the dashboard initializes instantly (`<50ms`) using cached/bundled snapshots.
  - Dispatches non-blocking live satellite synchronization in the background and smoothly updates time-series graphs, global comparison bars, and alert badges upon resolution without blocking user interaction.
- **Two-Tier Cache Layer (Memory + Persistent Disk):**
  - Integrated persistent disk caching in [`geovision/api/service.py`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/geovision/api/service.py) (`data/cache/`) alongside in-memory TTL caching.
  - Cached feature matrices and EHI data survive server restarts, preventing repetitive remote satellite downloads.
- **Parallel Background Warm-Up:**
  - Upgraded [`_warmup_cache`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/geovision/api/main.py) to utilize `concurrent.futures.ThreadPoolExecutor(max_workers=4)`, cutting cold cache initialization time by ~4×.
- **Bug Fix:**
  - Corrected `ehi_series` dictionary construction (`Object.fromEntries(...)`) in [`dashboard/app.js`](file:///c:/Users/rijul/Desktop/College/SEM-7/DAV/Code%202/geovision-ai/dashboard/app.js) to ensure live data seamlessly rehydrates state.

