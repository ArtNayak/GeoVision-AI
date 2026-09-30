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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from geovision.models.evaluate import ClassificationMetrics, classification_metrics

CLASS_LABELS = ["degraded", "stressed", "healthy"]
ModelType = Literal["random_forest", "xgboost"]

FEATURE_COLUMNS = [
    "ndvi", "ndwi", "temperature_2m", "precipitation_total", "pm25", "no2",
    "ndvi_anomaly", "pm25_anomaly", "precipitation_total_anomaly",
    "pca_1", "pca_2", "pca_3",
]


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
                n_jobs=-1,
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
                n_jobs=-1,
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


def train_and_compare(
    features: pd.DataFrame,
    model_types: list[ModelType] = ("random_forest", "xgboost"),
    test_size: float = 0.25,
) -> dict[ModelType, TrainedClassifier]:
    """Train each requested model type on the same split and return
    comparable metrics -- this is what produces the RF-vs-XGBoost
    comparison referenced in the objectives."""
    working = features.copy()
    for col in FEATURE_COLUMNS:
        if col not in working.columns:
            working[col] = 0.0
    working = working.dropna(subset=["ndvi", "pm25", "precipitation_total"])

    labels = derive_condition_labels(working)
    # Drop rows where label derivation produced NaN (edge of pd.cut bins),
    # otherwise train_test_split(stratify=y) raises ValueError.
    valid_mask = labels.notna()
    working = working[valid_mask]
    labels = labels[valid_mask]

    X = working[FEATURE_COLUMNS].fillna(0.0)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    results: dict[ModelType, TrainedClassifier] = {}
    for model_type in model_types:
        clf = EnvironmentalClassifier(model_type=model_type).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        y_test_codes = pd.Categorical(y_test, categories=CLASS_LABELS).codes
        y_pred_codes = pd.Categorical(y_pred, categories=CLASS_LABELS).codes
        metrics = classification_metrics(y_test_codes, y_pred_codes, y_proba)
        results[model_type] = TrainedClassifier(
            model_type=model_type,
            model=clf,
            metrics=metrics,
            feature_importances=clf.feature_importances(FEATURE_COLUMNS),
        )
    return results
