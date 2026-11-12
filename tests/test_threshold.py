import numpy as np

from src.threshold import _sequences, smooth, nonparametric_threshold


def test_sequences_finds_contiguous_runs():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert _sequences(mask) == [(1, 2), (5, 5), (7, 9)]


def test_sequences_handles_trailing_run():
    mask = np.array([0, 0, 1, 1], dtype=bool)
    assert _sequences(mask) == [(2, 3)]


def test_smooth_preserves_length_and_constant():
    e = np.full(50, 3.0)
    s = smooth(e, window=5)
    assert s.shape == e.shape
    # EWMA of a constant signal stays at that constant
    assert np.allclose(s, 3.0)


def test_threshold_flags_injected_spikes():
    rng = np.random.default_rng(0)
    e = np.abs(rng.normal(0, 1.0, 1000))
    spikes = [200, 500, 800]
    for i in spikes:
        e[i] = 25.0
    thr, mask = nonparametric_threshold(e)
    # threshold sits above normal noise and below the spikes
    assert 3.0 < thr < 25.0
    for i in spikes:
        assert mask[i], f"spike at {i} should be flagged"
    # and it does not flag a large share of the normal points
    assert mask.mean() < 0.05


def test_threshold_no_anomaly_is_quiet():
    # smooth, low-variance error -> very few or no flags
    e = np.full(500, 0.5) + 1e-3 * np.sin(np.arange(500))
    thr, mask = nonparametric_threshold(e)
    assert mask.sum() <= 5
