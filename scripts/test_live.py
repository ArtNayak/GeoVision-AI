"""Quick smoke-test: hit the running API and print EHI rows for Mumbai."""
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"

print("=== GeoVision AI — Live Data Smoke Test ===\n")

# Health
h = requests.get(f"{BASE}/api/health", timeout=5)
print("Health:", h.json())

# Regions
regions = requests.get(f"{BASE}/api/regions", timeout=5).json()
print(f"Regions registered: {len(regions)}\n")

# EHI — Mumbai (small bbox → fast COG windowed read)
print("Fetching EHI for Mumbai (live Open-Meteo + MODIS, 15 days)...")
t0 = time.time()
r = requests.get(f"{BASE}/api/ehi", params={"region_id": "MUM-IN", "days": 15}, timeout=180)
elapsed = time.time() - t0
print(f"HTTP {r.status_code}  ({elapsed:.1f}s)\n")

if r.status_code != 200:
    print("ERROR:", r.text[:600])
    sys.exit(1)

series = r.json().get("series", [])
print(f"{len(series)} daily rows returned\n")
for row in series:
    print(
        f"  {row.get('date')}  "
        f"EHI={row.get('ehi')}  "
        f"AQ={row.get('air_quality')}  "
        f"Veg={row.get('vegetation')}  "
        f"Water={row.get('water')}  "
        f"Climate={row.get('climate')}"
    )

print("\n=== Done ===")
