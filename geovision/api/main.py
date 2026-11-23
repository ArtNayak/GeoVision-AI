"""
FastAPI application -- presentation layer (Stage 7) + deployment surface
(Stage 8). Serves the JSON API the dashboard consumes, and the dashboard's
static files themselves, so the whole platform can be run with a single
command: `uvicorn geovision.api.main:app --reload`.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from geovision.api.routers import alerts, environment, models

DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "dashboard"

app = FastAPI(
    title="GeoVision AI",
    description=(
        "Global near real-time environmental intelligence and forecasting "
        "platform -- ingestion, ML/DL analytics, Environmental Health "
        "Index fusion, and alerting behind one API.\n\n"
        "On startup the server automatically warms the EHI cache for all "
        "configured regions in a background thread so the dashboard loads "
        "instantly after the initial warm-up completes (~10-15 min first run, "
        "instant on subsequent requests within the 60-min TTL)."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(environment.router)
app.include_router(alerts.router)
app.include_router(models.router)


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "service": "geovision-ai"}


@app.post("/api/cache/clear", tags=["system"])
def clear_cache():
    """Flush both in-memory and persistent disk TTL caches so next API call re-ingests fresh data."""
    from geovision.api import service
    count = service.clear_all_caches()
    return {"status": "cleared", "entries_removed": count}


@app.get("/api/cache/status", tags=["system"])
def cache_status():
    """Show which regions are currently cached in memory and on disk."""
    import time
    from geovision.api import service
    from geovision.config import CACHE_DIR, REGION_REGISTRY
    now = time.time()
    ttl = service._CACHE_TTL_SECONDS
    entries = []
    for key, (ts, val) in service._CACHE.items():
        age = now - ts
        records = len(val) if hasattr(val, "__len__") else None
        entries.append({
            "key": key,
            "tier": "memory",
            "age_seconds": round(age),
            "expires_in_seconds": round(ttl - age),
            "records": records,
        })
    disk_files = [p.name for p in CACHE_DIR.glob("*.pkl")] if CACHE_DIR.exists() else []
    return {
        "status": "healthy",
        "total_registered_regions": len(REGION_REGISTRY),
        "memory_cached_entries": len(entries),
        "disk_cached_files": len(disk_files),
        "ttl_seconds": ttl,
        "regions_status_url": "/api/regions/status",
        "cache_status_dashboard_url": "/cache-status",
        "entries": entries,
        "disk_files": disk_files,
    }


def _extract_series_snapshot(series_obj):
    """Safely extracts latest row dictionary and record count from DataFrame or list."""
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


def _safe_float(val, digits=1):
    """Safely cast and round float, filtering NaN/Inf."""
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


@app.get("/api/regions/status", tags=["system"])
def regions_status():
    """Fast status endpoint for ALL 13 regions — reads memory + disk cache
    instantly, never triggers heavy ingestion. Responds in <100ms always."""
    import time
    import pickle
    import re
    from datetime import date
    from geovision.api import service
    from geovision.config import CACHE_DIR, REGION_REGISTRY

    now = time.time()
    ttl = service._CACHE_TTL_SECONDS
    today = date.today().isoformat()
    rows = []

    for rid, meta in REGION_REGISTRY.items():
        safe_rid = re.sub(r"[^a-zA-Z0-9_.-]", "_", rid)
        cache_key = f"ehi:{rid}:180:{today}"

        # 1. Check in-memory cache
        hit = service._CACHE.get(cache_key)
        if hit and (now - hit[0] < ttl):
            ts, series = hit
            age = round(now - ts)
            latest, count = _extract_series_snapshot(series)
            rows.append({
                "region_id": rid,
                "name": meta["name"],
                "level": meta["level"],
                "status": "cached_memory",
                "cached": True,
                "ehi": _safe_float(latest.get("ehi")),
                "air_quality": _safe_float(latest.get("air_quality")),
                "vegetation": _safe_float(latest.get("vegetation")),
                "water": _safe_float(latest.get("water")),
                "climate": _safe_float(latest.get("climate")),
                "records": count,
                "age_seconds": age,
                "expires_in_seconds": max(0, round(ttl - age)),
            })
            continue

        # 2. Check disk cache (fast pickle read, ~1ms)
        disk_file = None
        mtime = None
        if CACHE_DIR.exists():
            today_file = CACHE_DIR / f"ehi_{safe_rid}_180_{today}.pkl"
            if today_file.exists():
                disk_file = today_file
                mtime = today_file.stat().st_mtime
            else:
                matches = list(CACHE_DIR.glob(f"ehi_{safe_rid}_*.pkl"))
                if matches:
                    best = max(matches, key=lambda p: p.stat().st_mtime)
                    disk_file = best
                    mtime = best.stat().st_mtime

        if disk_file and disk_file.exists():
            try:
                with open(disk_file, "rb") as f:
                    series = pickle.load(f)
                latest, count = _extract_series_snapshot(series)
                age = round(now - mtime)
                # Keep in memory cache so future calls are even faster
                if count > 0:
                    service._CACHE[cache_key] = (mtime, series)
                rows.append({
                    "region_id": rid,
                    "name": meta["name"],
                    "level": meta["level"],
                    "status": "cached_disk",
                    "cached": True,
                    "ehi": _safe_float(latest.get("ehi")),
                    "air_quality": _safe_float(latest.get("air_quality")),
                    "vegetation": _safe_float(latest.get("vegetation")),
                    "water": _safe_float(latest.get("water")),
                    "climate": _safe_float(latest.get("climate")),
                    "records": count,
                    "age_seconds": age,
                    "expires_in_seconds": max(0, round(ttl - age)),
                    "cache_file": disk_file.name,
                })
                continue
            except Exception:
                pass

        # 3. Not cached anywhere yet — pending initial run
        rows.append({
            "region_id": rid,
            "name": meta["name"],
            "level": meta["level"],
            "status": "pending",
            "cached": False,
            "ehi": None,
            "air_quality": None,
            "vegetation": None,
            "water": None,
            "climate": None,
            "records": None,
            "age_seconds": None,
            "expires_in_seconds": None,
        })

    cached_count = sum(1 for r in rows if r["cached"])
    return {
        "status": "ok",
        "total_regions": len(rows),
        "cached_regions": cached_count,
        "pending_regions": len(rows) - cached_count,
        "ttl_seconds": ttl,
        "date": today,
        "regions": sorted(rows, key=lambda r: (not r["cached"], r["name"])),
    }


@app.get("/cache-status", tags=["system"])
def cache_status_dashboard():
    """Visual HTML status dashboard displaying real-time cache state for all 13 regions."""
    from fastapi.responses import HTMLResponse
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GeoVision AI — Regional Cache Status</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --surface: #121827;
      --surface-border: #1f293d;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0ea5e9;
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.12);
      --warning: #f59e0b;
      --warning-bg: rgba(245, 158, 11, 0.12);
      --disk-cyan: #06b6d4;
      --disk-bg: rgba(6, 182, 212, 0.12);
      --card-gradient: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem 1.5rem;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 75%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .brand { display: flex; align-items: center; gap: 0.75rem; }
    .brand-icon {
      width: 40px; height: 40px;
      background: linear-gradient(135deg, #38bdf8, #10b981);
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
      font-size: 1.25rem;
    }
    .brand h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px; }
    .brand p { font-size: 0.85rem; color: var(--text-muted); }
    .header-actions { display: flex; gap: 0.75rem; align-items: center; }
    .btn {
      font-family: inherit; font-size: 0.85rem; font-weight: 500;
      padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid var(--surface-border);
      background: var(--surface); color: var(--text); cursor: pointer;
      text-decoration: none; display: inline-flex; align-items: center; gap: 0.4rem;
      transition: all 0.2s ease;
    }
    .btn:hover { background: #1e293b; border-color: var(--primary); }
    .btn-primary {
      background: var(--primary); color: #0f172a; border-color: var(--primary); font-weight: 600;
    }
    .btn-primary:hover { background: var(--primary-hover); }
    
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2rem;
    }
    .stat-card {
      background: var(--card-gradient);
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      padding: 1.25rem;
      backdrop-filter: blur(10px);
    }
    .stat-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 0.4rem; }
    .stat-value { font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .stat-desc { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem; }

    .card {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .card-header {
      padding: 1rem 1.5rem;
      border-bottom: 1px solid var(--surface-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(30, 41, 59, 0.3);
    }
    .card-title { font-size: 1.05rem; font-weight: 600; }
    .last-updated { font-size: 0.8rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }
    th {
      padding: 0.9rem 1.25rem;
      background: rgba(15, 23, 42, 0.6);
      color: var(--text-muted);
      font-weight: 500;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--surface-border);
    }
    td {
      padding: 0.9rem 1.25rem;
      border-bottom: 1px solid rgba(31, 41, 61, 0.6);
      vertical-align: middle;
    }
    tr:hover td { background: rgba(56, 189, 248, 0.03); }
    
    .badge {
      display: inline-flex; align-items: center; gap: 0.35rem;
      padding: 0.25rem 0.6rem; border-radius: 9999px;
      font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    }
    .badge-memory { background: var(--success-bg); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-disk { background: var(--disk-bg); color: var(--disk-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-pending { background: var(--warning-bg); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-level {
      background: rgba(148, 163, 184, 0.1); color: var(--text-muted);
      border: 1px solid rgba(148, 163, 184, 0.2); font-size: 0.7rem;
    }
    
    .score-badge {
      font-family: 'JetBrains Mono', monospace; font-weight: 700;
      padding: 0.25rem 0.6rem; border-radius: 6px; display: inline-block;
    }
    .score-good { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .score-med { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .score-low { background: rgba(239, 68, 68, 0.15); color: #f87171; }
    
    .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
    .indicator-pills { display: flex; gap: 0.4rem; font-size: 0.75rem; }
    .indicator-pill {
      background: rgba(15, 23, 42, 0.8);
      padding: 0.2rem 0.45rem;
      border-radius: 4px;
      border: 1px solid var(--surface-border);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="brand-icon">🌐</div>
        <div>
          <h1>Regional Cache Status</h1>
          <p>GeoVision AI Environmental Intelligence Platform</p>
        </div>
      </div>
      <div class="header-actions">
        <button class="btn btn-primary" onclick="loadStatus()">🔄 Refresh Now</button>
        <a href="/api/regions/status" target="_blank" class="btn">📄 Raw JSON</a>
        <a href="/" class="btn">📊 Main Dashboard</a>
      </div>
    </header>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Total Regions</div>
        <div class="stat-value" id="stat-total" style="color: var(--primary);">13</div>
        <div class="stat-desc">Continents, countries, states, cities</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Ready / Cached</div>
        <div class="stat-value" id="stat-cached" style="color: var(--success);">-</div>
        <div class="stat-desc">Available for immediate zero-wait query</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Pending Ingestion</div>
        <div class="stat-value" id="stat-pending" style="color: var(--warning);">-</div>
        <div class="stat-desc">Will compute on first dashboard access</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Cache TTL</div>
        <div class="stat-value" id="stat-ttl" style="color: #a78bfa;">60m</div>
        <div class="stat-desc">Automatic persistent disk retention</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <span class="card-title">All Registered Regions</span>
        <span class="last-updated" id="last-updated">Checking...</span>
      </div>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Region</th>
              <th>Level</th>
              <th>Status</th>
              <th>EHI Health Score</th>
              <th>Indicators (Air / Veg / Water / Climate)</th>
              <th>Records</th>
              <th>Cache Age</th>
            </tr>
          </thead>
          <tbody id="regions-tbody">
            <tr><td colspan="7" style="text-align: center; padding: 2rem; color: var(--text-muted);">Loading cache status...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    async function loadStatus() {
      const tbody = document.getElementById('regions-tbody');
      try {
        const res = await fetch('/api/regions/status');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        
        document.getElementById('stat-total').textContent = data.total_regions;
        document.getElementById('stat-cached').textContent = data.cached_regions;
        document.getElementById('stat-pending').textContent = data.pending_regions;
        document.getElementById('stat-ttl').textContent = Math.round(data.ttl_seconds / 60) + 'm';
        document.getElementById('last-updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();

        tbody.innerHTML = data.regions.map(r => {
          let statusBadge = '';
          if (r.status === 'cached_memory') {
            statusBadge = '<span class="badge badge-memory">● Memory</span>';
          } else if (r.status === 'cached_disk') {
            statusBadge = '<span class="badge badge-disk">● Disk Cached</span>';
          } else {
            statusBadge = '<span class="badge badge-pending">○ Pending</span>';
          }

          let scoreBadge = '<span style="color: var(--text-muted);">-</span>';
          if (r.ehi !== null && r.ehi !== undefined) {
            let cls = r.ehi >= 60 ? 'score-good' : (r.ehi >= 40 ? 'score-med' : 'score-low');
            scoreBadge = `<span class="score-badge ${cls}">${r.ehi.toFixed(1)}</span>`;
          }

          let indicators = '<span style="color: var(--text-muted);">-</span>';
          if (r.cached) {
            indicators = `
              <div class="indicator-pills">
                <span class="indicator-pill" title="Air Quality">💨 ${r.air_quality ?? '-'}</span>
                <span class="indicator-pill" title="Vegetation">🌿 ${r.vegetation ?? '-'}</span>
                <span class="indicator-pill" title="Water">💧 ${r.water ?? '-'}</span>
                <span class="indicator-pill" title="Climate">🌡️ ${r.climate ?? '-'}</span>
              </div>
            `;
          }

          let age = '-';
          if (r.age_seconds !== null && r.age_seconds !== undefined) {
            let m = Math.floor(r.age_seconds / 60);
            let s = r.age_seconds % 60;
            age = m > 0 ? `${m}m ${s}s ago` : `${s}s ago`;
          }

          return `
            <tr>
              <td>
                <div style="font-weight: 600;">${r.name}</div>
                <div class="mono" style="color: var(--text-muted); font-size: 0.75rem;">${r.region_id}</div>
              </td>
              <td><span class="badge badge-level">${r.level}</span></td>
              <td>${statusBadge}</td>
              <td>${scoreBadge}</td>
              <td>${indicators}</td>
              <td class="mono">${r.records ? r.records + ' rows' : '-'}</td>
              <td class="mono" style="color: var(--text-muted);">${age}</td>
            </tr>
          `;
        }).join('');
      } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: #f87171; padding: 2rem;">Failed to load cache status: ${err.message}</td></tr>`;
      }
    }

    loadStatus();
    // Auto-refresh every 10 seconds
    setInterval(loadStatus, 10000);
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


def _warmup_cache():
    """Pre-warm EHI cache for all regions in parallel using a thread pool.
    Called once on server startup -- keeps the event loop free."""
    import logging
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from geovision.api import service
    from geovision.config import REGION_REGISTRY

    log = logging.getLogger("geovision.warmup")
    regions = list(REGION_REGISTRY.keys())
    log.info("Parallel cache warm-up started for %d regions", len(regions))

    def warm_region(rid: str):
        t0 = time.time()
        try:
            service.ehi_series(rid, days=180)
            log.info("  [%s] warm in %.1fs", rid, time.time() - t0)
            return rid, True, time.time() - t0
        except Exception as exc:
            log.warning("  [%s] warm failed: %s", rid, exc)
            return rid, False, 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(warm_region, rid) for rid in regions]
        for f in as_completed(futures):
            f.result()

    log.info("Parallel cache warm-up complete.")


@app.on_event("startup")
async def startup_warmup():
    """Cache warm-up is intentionally disabled at startup.

    The disk cache (data/cache/*.pkl) already persists EHI data between
    server restarts, so /api/regions/status and the dashboard load
    instantly from disk on the first request.

    To force a full re-ingestion from the live data sources, call:
        POST /api/cache/clear
    which flushes memory+disk and the next request per region will re-fetch.
    """
    import logging
    logging.getLogger("geovision.warmup").info(
        "Startup warm-up skipped — disk cache will be used on first request."
    )


if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
