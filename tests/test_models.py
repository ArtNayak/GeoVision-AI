from datetime import date, timedelta

import numpy as np
import pytest

from geovision.features.spectral_indices import build_feature_matrix
from geovision.index.health_index import compute_ehi
from geovision.ingestion.pipeline import IngestionPipeline
from geovision.models.classifiers import CLASS_LABELS, derive_condition_labels, train_and_compare
from geovision.models.evaluate import classification_metrics, regression_metrics
from geovision.models.forecasting import SequenceForecaster, build_sequences
from geovision.preprocessing.cloud_mask import run_preprocessing


@pytest.fixture(scope="module")
def ehi_features():
    pipeline = IngestionPipeline()
    end = date.today()
    start = end - timedelta(days=150)
    raw = pipeline.run("IN", start, end)
    clean = run_preprocessing(raw)
    features = build_feature_matrix(clean)
    return compute_ehi(features)


def test_derive_condition_labels_uses_expected_classes(ehi_features):
    labels = derive_condition_labels(ehi_features)
    assert set(labels.dropna().unique()).issubset(set(CLASS_LABELS))


def test_train_and_compare_classifiers(ehi_features):
    results = train_and_compare(ehi_features, model_types=["random_forest", "xgboost"])
    assert set(results.keys()) == {"random_forest", "xgboost"}
    for result in results.values():
        assert 0.0 <= result.metrics.accuracy <= 1.0
        assert -1.0 <= result.metrics.kappa <= 1.0
        assert result.feature_importances  # non-empty


def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m.rmse == pytest.approx(0.0, abs=1e-9)
    assert m.mae == pytest.approx(0.0, abs=1e-9)
    assert m.r2 == pytest.approx(1.0, abs=1e-6)


def test_classification_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 1, 0])
    m = classification_metrics(y, y)
    assert m.accuracy == 1.0
    assert m.kappa == pytest.approx(1.0)


def test_build_sequences_shapes():
    series = np.arange(20, dtype=np.float32)
    X, y = build_sequences(series, lookback=5, horizon=2)
    assert X.shape == (14, 5)
    assert y.shape == (14, 2)
    assert list(X[0]) == [0, 1, 2, 3, 4]
    assert list(y[0]) == [5, 6]


@pytest.mark.parametrize("arch", ["lstm", "cnn", "transformer"])
def test_sequence_forecaster_trains_and_predicts(arch):
    # small synthetic sine wave, kept short + low-epoch so the test is fast
    series = np.sin(np.linspace(0, 20, 80)) * 10 + 50
    forecaster = SequenceForecaster(arch=arch, lookback=10, horizon=3, epochs=3).fit(series)
    forecast = forecaster.forecast_next()
    assert forecast.shape == (3,)
    metrics = forecaster.evaluate()
    assert metrics.rmse >= 0


def test_sequence_forecaster_raises_on_insufficient_data():
    series = np.arange(10, dtype=np.float32)
    with pytest.raises(ValueError):
        SequenceForecaster(arch="lstm", lookback=10, horizon=5, epochs=1).fit(series)


def test_pooled_forecaster_returns_shrunk_per_region_results():
    rng = np.random.default_rng(0)
    t = np.arange(260)
    series = {f"R{i}": (np.sin(t / 12.0 + i) * 5 + 50 + rng.normal(0, 0.5, len(t))).astype(np.float32) for i in range(3)}
    from geovision.models.forecasting import train_and_compare_pooled
    out = train_and_compare_pooled(series, architectures=["lstm", "cnn"], horizon=5, epochs=3, seeds=(1,))
    assert set(out) == set(series)
    for res in out.values():
        assert set(res) == {"lstm", "cnn", "ensemble"}
        m = res["lstm"].metrics
        assert 0.0 <= m.shrink <= 1.0 and m.baseline_rmse > 0  # shrinkage weight and naive baseline reported
        assert res["lstm"].forecast.shape == (5,)
