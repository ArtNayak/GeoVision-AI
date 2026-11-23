"""Central configuration for GeoVision AI (no API keys needed: all data sources are free)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"


@dataclass(frozen=True)
class Settings:
    # Ingestion cadence
    ingestion_interval_minutes: int = int(os.getenv("INGESTION_INTERVAL_MIN", "60"))

    # Forecast horizon (days) produced by the LSTM / Transformer models
    forecast_horizon_days: int = int(os.getenv("FORECAST_HORIZON_DAYS", "14"))

    # Environmental Health Index sub-index weights (must sum to 1.0)
    ehi_weights: dict = field(
        default_factory=lambda: {
            "air_quality": 0.30,
            "vegetation": 0.25,
            "water": 0.25,
            "climate": 0.20,
        }
    )

    # Alert thresholds (sub-index 0-100 scale, lower = worse)
    alert_thresholds: dict = field(
        default_factory=lambda: {
            "air_quality": 40.0,
            "vegetation": 35.0,
            "water": 35.0,
            "climate": 40.0,
            "composite": 45.0,
        }
    )


settings = Settings()

# Regions available for country / state / city / continent comparison in the
# dashboard. Coordinates are representative centroids; bbox is
# [min_lon, min_lat, max_lon, max_lat].
REGION_REGISTRY = {
    "IN": {"name": "India", "level": "country", "lat": 22.35, "lon": 78.66,
           "bbox": [68.1, 6.5, 97.4, 35.5]},
    "US": {"name": "United States", "level": "country", "lat": 39.83, "lon": -98.58,
           "bbox": [-124.8, 24.4, -66.9, 49.4]},
    "BR": {"name": "Brazil", "level": "country", "lat": -14.24, "lon": -51.93,
           "bbox": [-73.9, -33.7, -34.8, 5.3]},
    "AU": {"name": "Australia", "level": "country", "lat": -25.27, "lon": 133.78,
           "bbox": [113.3, -43.6, 153.6, -10.7]},
    "KE": {"name": "Kenya", "level": "country", "lat": -0.02, "lon": 37.91,
           "bbox": [33.9, -4.7, 41.9, 4.6]},
    "ID": {"name": "Indonesia", "level": "country", "lat": -0.79, "lon": 113.92,
           "bbox": [95.0, -11.0, 141.0, 6.0]},
    "MH-IN": {"name": "Maharashtra, India", "level": "state", "lat": 19.75, "lon": 75.71,
              "bbox": [72.6, 15.6, 80.9, 22.1]},
    "CA-US": {"name": "California, USA", "level": "state", "lat": 36.78, "lon": -119.42,
              "bbox": [-124.4, 32.5, -114.1, 42.0]},
    "MUM-IN": {"name": "Mumbai, India", "level": "city", "lat": 19.076, "lon": 72.877,
               "bbox": [72.77, 18.89, 72.98, 19.27]},
    "SP-BR": {"name": "São Paulo, Brazil", "level": "city", "lat": -23.55, "lon": -46.63,
              "bbox": [-46.83, -23.73, -46.36, -23.36]},
    "AS": {"name": "Asia", "level": "continent", "lat": 34.0, "lon": 100.0,
           "bbox": [26.0, -11.0, 180.0, 77.0]},
    "AF": {"name": "Africa", "level": "continent", "lat": 1.0, "lon": 20.0,
           "bbox": [-17.0, -35.0, 51.0, 37.0]},
    "SA": {"name": "South America", "level": "continent", "lat": -15.0, "lon": -60.0,
           "bbox": [-81.0, -56.0, -34.0, 13.0]},
}
