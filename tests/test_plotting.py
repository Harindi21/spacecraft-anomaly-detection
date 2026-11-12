import numpy as np

from src.plotting import _runs, plot_detection


def test_runs_pairs():
    assert _runs([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]


def test_plot_writes_valid_svg(tmp_path):
    n, window = 300, 10
    t = np.linspace(0, 20, n)
    signal = np.sin(t)
    labels = np.zeros(n, dtype=bool); labels[150:170] = True
    pred = np.zeros(n, dtype=bool); pred[152:168] = True
    error = np.abs(np.random.default_rng(0).normal(size=n - window))

    out = tmp_path / "fig.svg"
    svg = plot_detection(signal, labels, pred, error=error, threshold=2.0,
                         window=window, path=str(out), title="t", channel_name="c0")

    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "polyline" in svg
    assert out.exists() and out.stat().st_size > 0


def test_plot_single_panel_without_error(tmp_path):
    n = 100
    signal = np.random.default_rng(1).normal(size=n)
    labels = np.zeros(n, dtype=bool)
    pred = np.zeros(n, dtype=bool)
    out = tmp_path / "fig.svg"
    svg = plot_detection(signal, labels, pred, path=str(out))
    assert "threshold" not in svg  # no error panel -> no threshold label
    assert out.exists()
