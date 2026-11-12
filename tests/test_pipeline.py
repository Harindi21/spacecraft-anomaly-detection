import numpy as np

from src.data import synthetic_telemetry
from src.model import get_forecaster
from src.pipeline import detect
from src.evaluate import evaluate


def test_detect_end_to_end_synthetic():
    series, labels = synthetic_telemetry(n_steps=4000, n_channels=5,
                                         n_anomalies=6, seed=0)
    split = int(len(series) * 0.4)
    train, test, test_labels = series[:split], series[split:], labels[split:]

    forecaster = get_forecaster("ar", window=10)
    mask, err, thresholds = detect(train, test, forecaster, window=10)

    # mask is point-level over the test stream
    assert mask.shape == (len(test),)
    assert err.shape[0] == len(test) - 10
    assert len(thresholds) == test.shape[1]

    res = evaluate(mask, test_labels)
    # the AR baseline should catch at least one injected anomaly segment
    assert res["point_adjusted"]["recall"] > 0.0
    # and must not degenerate into flagging almost everything
    assert mask.mean() < 0.3


def test_detect_quiet_on_clean_stream():
    # no injected anomalies -> detector should stay mostly quiet
    series, _ = synthetic_telemetry(n_steps=2000, n_channels=3,
                                    n_anomalies=0, seed=3)
    split = int(len(series) * 0.5)
    train, test = series[:split], series[split:]
    forecaster = get_forecaster("ar", window=10)
    mask, _, _ = detect(train, test, forecaster, window=10)
    assert mask.mean() < 0.1
