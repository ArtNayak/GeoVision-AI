"""
Automated alerting -- Stage 8 of the methodology.

Two complementary triggers, matching the report's disaster-alerting
application service:

1. Threshold alerts: a sub-index or the composite EHI drops below its
   configured floor (`settings.alert_thresholds`), signalling sustained
   poor conditions (e.g. chronic poor air quality).
2. Rapid-decline alerts: a sub-index falls sharply over a short window
   even if still above the absolute floor, catching fast-onset events
   (flood, wildfire smoke, sudden drought) that the literature's
   flood/wildfire papers (DeepSAR 2025; Wildfire Susceptibility 2022)
   emphasize need near real-time detection, not just static thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from geovision.config import REGION_REGISTRY, settings

RAPID_DECLINE_WINDOW_DAYS = 7
RAPID_DECLINE_DROP_POINTS = 15.0  # sub-index points lost within the window


@dataclass
class Alert:
    region_id: str
    date: date
    severity: str          # "watch" | "warning" | "critical"
    trigger: str            # "threshold" | "rapid_decline"
    indicator: str          # e.g. "air_quality", "ehi"
    value: float
    message: str
    context: dict = field(default_factory=dict)


def _label(indicator: str) -> str:
    return "EHI" if indicator == "ehi" else indicator.replace("_", " ").title()


def _severity_for_threshold(value: float, floor: float) -> str:
    if value < floor * 0.5:
        return "critical"
    if value < floor * 0.8:
        return "warning"
    return "watch"


def check_thresholds(latest: pd.DataFrame) -> list[Alert]:
    """Emit threshold alerts for the most recent row per region."""
    alerts: list[Alert] = []
    for _, row in latest.iterrows():
        for indicator, floor in settings.alert_thresholds.items():
            column = "ehi" if indicator == "composite" else indicator
            if column not in row or pd.isna(row[column]):
                continue
            value = float(row[column])
            if value < floor:
                region_name = REGION_REGISTRY.get(row["region_id"], {}).get("name", row["region_id"])
                alerts.append(
                    Alert(
                        region_id=row["region_id"],
                        date=row["date"].date() if hasattr(row["date"], "date") else row["date"],
                        severity=_severity_for_threshold(value, floor),
                        trigger="threshold",
                        indicator=indicator,
                        value=value,
                        message=(
                            f"{_label(indicator)} at {value:.1f} "
                            f"is below the {floor:.0f} threshold for {region_name}."
                        ),
                    )
                )
    return alerts


def check_rapid_decline(history: pd.DataFrame) -> list[Alert]:
    """Emit rapid-decline alerts by comparing each region's most recent
    sub-index reading against its value `RAPID_DECLINE_WINDOW_DAYS` ago."""
    alerts: list[Alert] = []
    indicators = [k for k in settings.alert_thresholds if k != "composite"] + ["ehi"]

    for region_id, group in history.groupby("region_id"):
        group = group.sort_values("date")
        if len(group) < 2:
            continue
        latest_row = group.iloc[-1]
        cutoff = latest_row["date"] - pd.Timedelta(days=RAPID_DECLINE_WINDOW_DAYS)
        prior_candidates = group[group["date"] <= cutoff]
        if prior_candidates.empty:
            continue
        prior_row = prior_candidates.iloc[-1]

        for indicator in indicators:
            if indicator not in group.columns:
                continue
            latest_val, prior_val = latest_row[indicator], prior_row[indicator]
            if pd.isna(latest_val) or pd.isna(prior_val):
                continue
            drop = float(prior_val) - float(latest_val)
            if drop >= RAPID_DECLINE_DROP_POINTS:
                region_name = REGION_REGISTRY.get(region_id, {}).get("name", region_id)
                alerts.append(
                    Alert(
                        region_id=region_id,
                        date=latest_row["date"].date() if hasattr(latest_row["date"], "date") else latest_row["date"],
                        severity="critical" if drop >= RAPID_DECLINE_DROP_POINTS * 1.5 else "warning",
                        trigger="rapid_decline",
                        indicator=indicator,
                        value=float(latest_val),
                        message=(
                            f"{_label(indicator)} for {region_name} dropped "
                            f"{drop:.1f} points in {RAPID_DECLINE_WINDOW_DAYS} days."
                        ),
                        context={"prior_value": float(prior_val),
                                 "window_days": RAPID_DECLINE_WINDOW_DAYS},
                    )
                )
    return alerts


def generate_alerts(ehi_history: pd.DataFrame) -> list[Alert]:
    """Full Stage-8 pipeline over an EHI history spanning one or more
    regions: threshold + rapid-decline alerts, most severe first."""
    if ehi_history.empty:
        return []
    latest = (
        ehi_history.sort_values("date").groupby("region_id", as_index=False).tail(1)
    )
    alerts = check_thresholds(latest) + check_rapid_decline(ehi_history)
    severity_rank = {"critical": 0, "warning": 1, "watch": 2}
    return sorted(alerts, key=lambda a: severity_rank.get(a.severity, 9))
