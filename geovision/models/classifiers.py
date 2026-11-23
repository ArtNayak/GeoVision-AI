"""
Environmental classification models -- Random Forest and XGBoost, per the
literature survey's finding that RF/SVM dominate GEE-native classification
(Amani et al. 2020; Alshehri et al. 2025) while gradient-boosted trees
(XGBoost) offer a competitive, deployable alternative to deep nets for
tabular environmental indicators.

Since no field-verified land-cover/condition labels ship with this
platform, `derive_condition_labels` builds a literature-informed proxy
label (Healthy / Stressed / Degraded) from NDVI, PM2.5 and precipitation
z-scores so the classifiers are trainable end-to-end out of the box. In
production this should be replaced with ground-truth labels (e.g. ESA
WorldCover transitions, in-situ survey data, or expert-annotated tiles).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from geovision.models.evaluate import ClassificationMetrics, classification_metrics

import os
N_JOBS = int(os.getenv("GEO_N_JOBS", "4"))  # run_demo sets 1 per worker to avoid CPU oversubscription
CLASS_LABELS = ["degraded", "stressed", "healthy"]
LEAD_DAYS = 7  # the classifiers predict the condition one week AHEAD
ModelType = Literal["random_forest", "xgboost"]

FEATURE_COLUMNS = [
    "ndvi", "soil_moisture", "temperature_2m", "precipitation_total", "pm25", "no2",
    "ndvi_anomaly", "pm25_anomaly", "precipitation_total_anomaly",
    "pca_1", "pca_2", "pca_3",
]
# Today's region-relative condition (known on day t, computed with the same rule as the target):
# gives a POOLED model the information "what is normal for THIS region" and its persistence.
NOW_FEATURES = ["now_code", "now_deg", "now_str", "now_hea", "now_roll7", "now_lag7"]
ALL_FEATURES = FEATURE_COLUMNS + NOW_FEATURES


def derive_condition_labels(features: pd.DataFrame) -> pd.Series:
    """Rule-based proxy label combining vegetation health (NDVI z-score),
    air quality (PM2.5 z-score) and moisture deficit (precipitation
    z-score) into a 3-class environmental condition target."""
    def zscore(s: pd.Series) -> pd.Series:
        std = s.std() or 1.0
        return (s - s.mean()) / std

    score = (
        zscore(features["ndvi"])
        - zscore(features["pm25"])
        + zscore(features["precipitation_total"]).clip(lower=-1.5, upper=1.5) * 0.5
    )
    return pd.cut(score, bins=[-np.inf, -0.5, 0.5, np.inf], labels=CLASS_LABELS)


@dataclass
class TrainedClassifier:
    model_type: ModelType
    model: object
    metrics: ClassificationMetrics
    feature_importances: dict[str, float]


class EnvironmentalClassifier:
    """Common fit/predict interface over Random Forest and XGBoost so the
    training script can loop over both and compare, per the methodology's
    'parallel training of ML classifiers' stage."""

    def __init__(self, model_type: ModelType = "random_forest", **kwargs):
        self.model_type = model_type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.pop("n_estimators", 300),
                max_depth=kwargs.pop("max_depth", None),
                random_state=42,
                n_jobs=N_JOBS,
                **kwargs,
            )
        elif model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=kwargs.pop("n_estimators", 300),
                max_depth=kwargs.pop("max_depth", 6),
                learning_rate=kwargs.pop("learning_rate", 0.1),
                objective="multi:softprob",
                num_class=len(CLASS_LABELS),
                eval_metric="mlogloss",
                seed=42,
                n_jobs=N_JOBS,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnvironmentalClassifier":
        y_codes = pd.Categorical(y, categories=CLASS_LABELS).codes
        self.model.fit(X, y_codes)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        codes = self.model.predict(X)
        return np.array(CLASS_LABELS)[codes]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return {}
        return dict(sorted(zip(feature_names, map(float, importances)),
                            key=lambda kv: -kv[1]))


def _prepare(features: pd.DataFrame, test_size: float):
    """X, y (condition LEAD_DAYS ahead), naive 'today's condition' labels and the chronological split."""
    working = features.copy()
    for col in FEATURE_COLUMNS:
        if col not in working.columns:
            working[col] = 0.0
    working = working.dropna(subset=["ndvi", "pm25", "precipitation_total"]).sort_values("date")

    # Target = condition LEAD_DAYS ahead, from today's features. (Predicting today's
    # rule-derived label from the same columns would just re-learn the rule.)
    now_labels = derive_condition_labels(working)
    code = pd.Series(pd.Categorical(now_labels, categories=CLASS_LABELS).codes, index=working.index, dtype=float)
    working = working.assign(
        now_code=code, now_deg=(code == 0).astype(float), now_str=(code == 1).astype(float),
        now_hea=(code == 2).astype(float), now_roll7=code.rolling(7, min_periods=1).mean(),
        now_lag7=code.shift(7).fillna(code))
    labels = now_labels.shift(-LEAD_DAYS)
    valid_mask = labels.notna()
    working, labels, now_labels = working[valid_mask], labels[valid_mask], now_labels[valid_mask]
    X = working[ALL_FEATURES].fillna(0.0)
    split = int(len(X) * (1 - test_size))  # chronological split: train on the past, test on the future
    # standardise the raw features per region with TRAINING-portion statistics (so pooled models
    # see comparable scales); the condition indicators stay as they are
    mu, sd = X.iloc[:split].mean(), X.iloc[:split].std().replace(0, 1)
    raw = FEATURE_COLUMNS
    X = X.copy()
    X[raw] = (X[raw] - mu[raw]) / sd[raw]
    return X, labels, now_labels, split


def _evaluate_classifier(model_type, clf, X_test, y_test, baseline_accuracy) -> TrainedClassifier:
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    y_test_codes = pd.Categorical(y_test, categories=CLASS_LABELS).codes
    y_pred_codes = pd.Categorical(y_pred, categories=CLASS_LABELS).codes
    metrics = classification_metrics(y_test_codes, y_pred_codes, y_proba)
    metrics.baseline_accuracy = baseline_accuracy
    return TrainedClassifier(model_type=model_type, model=clf, metrics=metrics,
                             feature_importances=clf.feature_importances(ALL_FEATURES))


def train_and_compare(
    features: pd.DataFrame,
    model_types: list[ModelType] = ("random_forest", "xgboost"),
    test_size: float = 0.25,
) -> dict[ModelType, TrainedClassifier]:
    """Train each requested model type on the same chronological split and return
    comparable metrics (single region)."""
    X, y, now_labels, split = _prepare(features, test_size)
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    baseline_accuracy = float((now_labels.iloc[split:].to_numpy() == y_test.to_numpy()).mean())
    return {mt: _evaluate_classifier(mt, EnvironmentalClassifier(model_type=mt).fit(X_train, y_train),
                                     X_test, y_test, baseline_accuracy) for mt in model_types}


def train_and_compare_pooled(
    features_by_region: dict[str, pd.DataFrame],
    model_types: list[ModelType] = ("random_forest", "xgboost"),
    test_size: float = 0.25,
) -> dict[str, dict[ModelType, TrainedClassifier]]:
    """ONE classifier per model type trained on the first 75 % of EVERY region (about 13x the
    data), evaluated on each region's last 25 %. Labels are region-relative (z-scores within the
    region), so pooling learns the shared dynamics; the per-region chronological test sets stay
    untouched by training."""
    prepared = {rid: _prepare(f, test_size) for rid, f in features_by_region.items()}
    X_train = pd.concat([X.iloc[:sp] for X, _, _, sp in prepared.values()])
    y_train = pd.concat([y.iloc[:sp] for _, y, _, sp in prepared.values()])
    out: dict[str, dict[ModelType, TrainedClassifier]] = {rid: {} for rid in prepared}
    for mt in model_types:
        clf = EnvironmentalClassifier(model_type=mt).fit(X_train, y_train)
        for rid, (X, y, now, sp) in prepared.items():
            base = float((now.iloc[sp:].to_numpy() == y.iloc[sp:].to_numpy()).mean())
            out[rid][mt] = _evaluate_classifier(mt, clf, X.iloc[sp:], y.iloc[sp:], base)
    return out
