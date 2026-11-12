import numpy as np

from src.data import synthetic_telemetry


def test_shapes_and_label_range():
    series, labels = synthetic_telemetry(n_steps=2000, n_channels=4, seed=1)
    assert series.shape == (2000, 4)
    assert labels.shape == (2000,)
    assert set(np.unique(labels)).issubset({0, 1})


def test_some_anomalies_injected():
    _, labels = synthetic_telemetry(n_steps=4000, n_channels=5, n_anomalies=6, seed=0)
    # at least a few anomalous points, but a small minority of the stream
    frac = labels.mean()
    assert 0.0 < frac < 0.2


def test_seed_is_reproducible():
    a, la = synthetic_telemetry(seed=7)
    b, lb = synthetic_telemetry(seed=7)
    assert np.array_equal(a, b)
    assert np.array_equal(la, lb)


def test_different_seeds_differ():
    a, _ = synthetic_telemetry(seed=1)
    b, _ = synthetic_telemetry(seed=2)
    assert not np.array_equal(a, b)
