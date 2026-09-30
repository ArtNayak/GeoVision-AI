from datetime import date, timedelta

import pandas as pd
import pytest

from geovision.alerting.alerts import check_rapid_decline, check_thresholds, generate_alerts


def _row(region_id, d, ehi, air, veg, water, climate):
    return {
        "region_id": region_id, "date": pd.Timestamp(d), "ehi": ehi,
        "air_quality": air, "vegetation": veg, "water": water, "climate": climate,
    }


def test_check_thresholds_flags_low_subindex():
    latest = pd.DataFrame([_row("IN", date.today(), 70, 20, 80, 80, 80)])
    alerts = check_thresholds(latest)
    assert any(a.indicator == "air_quality" and a.severity in {"watch", "warning", "critical"} for a in alerts)


def test_check_thresholds_no_alert_when_healthy():
    latest = pd.DataFrame([_row("IN", date.today(), 90, 90, 90, 90, 90)])
    alerts = check_thresholds(latest)
    assert alerts == []


def test_check_rapid_decline_detects_sharp_drop():
    today = date.today()
    history = pd.DataFrame([
        _row("IN", today - timedelta(days=10), 80, 80, 80, 80, 80),
        _row("IN", today, 80, 55, 80, 80, 80),  # air quality dropped 25 points
    ])
    alerts = check_rapid_decline(history)
    assert any(a.trigger == "rapid_decline" and a.indicator == "air_quality" for a in alerts)


def test_generate_alerts_sorts_by_severity():
    today = date.today()
    history = pd.DataFrame([
        _row("A", today, 90, 90, 90, 90, 90),      # healthy, no alert
        _row("B", today, 10, 10, 10, 10, 10),      # critical across the board
    ])
    alerts = generate_alerts(history)
    assert alerts[0].severity == "critical"


def test_generate_alerts_empty_input():
    assert generate_alerts(pd.DataFrame()) == []
