"""Common interface every ingestion client implements, so the pipeline
(geovision.ingestion.pipeline) can treat Open-Meteo and MODIS uniformly."""
from __future__ import annotations

import abc
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

from geovision.config import CACHE_DIR


@dataclass(frozen=True)
class Observation:
    """One normalized observation record produced by any ingestion client.

    This is the common currency the preprocessing and feature-engineering
    stages operate on, regardless of which upstream API produced it.
    """

    region_id: str
    date: date
    source: str            # e.g. "open_meteo", "modis"
    variable: str           # e.g. "ndvi", "temperature_2m", "precipitation", "pm25"
    value: float
    unit: str
    quality_flag: str = "ok"   # "ok" | "cloud_masked" | "interpolated" | "missing"
    meta: dict[str, Any] | None = None


class BaseIngestionClient(abc.ABC):
    """Abstract base class for a data-source client."""

    source_name: str = "base"

    @abc.abstractmethod
    def is_configured(self) -> bool:
        """Whether credentials/config needed to hit the live API are present."""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch(self, region_id: str, start: date, end: date) -> list[Observation]:
        """Fetch observations for a region between start and end (inclusive).

        Implementations raise `IngestionUnavailableError` if the live API cannot be
        reached rather than silently returning nothing; the pipeline lets it propagate
        (there is no synthetic fallback).
        """
        raise NotImplementedError


class IngestionUnavailableError(RuntimeError):
    """Raised when a live upstream API is unreachable or unconfigured."""


def get_json(url: str, params: dict, timeout: int = 60, tries: int = 5, ttl: int = 3600) -> dict:
    """GET + parse JSON. Responses are cached on disk for `ttl` seconds (so re-runs after a crash,
    or several dashboard loads, don't re-hit the free APIs). HTTP 429 (rate limit) waits and
    retries; any other failure retries briefly. Raises IngestionUnavailableError -- never
    returns fake data -- if the API still can't be reached."""
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True, default=str)).encode()).hexdigest()
    cache_file = CACHE_DIR / "http" / f"{key}.json"
    try:
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < ttl:
            return json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    last: Exception | None = None
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:  # "try again in one minute": back off, don't fail
                last = RuntimeError("429 rate limited")
                time.sleep(25 * (attempt + 1))
                continue
            if 400 <= resp.status_code < 500:  # bad request / no data here: retrying won't help
                raise IngestionUnavailableError(f"{url}: HTTP {resp.status_code} {resp.text[:120]!r}")
            resp.raise_for_status()
            data = resp.json()
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.replace(cache_file)
            except OSError:
                pass
            return data
        except IngestionUnavailableError:
            raise
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(2)
    raise IngestionUnavailableError(f"{url} failed after {tries} tries: {last}") from last


def sample_points(region: dict, n: int = 2) -> list[tuple[float, float]]:
    """Centroid + an n x n grid inside the region's bbox (at 1/4 and 3/4 for n=2). Sampling
    several points instead of only the centroid gives a region average (a country centroid
    can be a desert or a city block). Ocean points are dropped later by each client."""
    x0, y0, x1, y1 = region["bbox"]
    fracs = [(2 * i + 1) / (2 * n) for i in range(n)]
    grid = [(round(y0 + f * (y1 - y0), 3), round(x0 + g * (x1 - x0), 3)) for f in fracs for g in fracs]
    return [(region["lat"], region["lon"])] + grid
