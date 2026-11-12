import numpy as np

from src.model import make_windows, ARForecaster, get_forecaster


def test_make_windows_shapes():
    series = np.arange(20, dtype=float).reshape(10, 2)
    X, y = make_windows(series, window=3)
    assert X.shape == (7, 3, 2)
    assert y.shape == (7, 2)
    # first target is the step right after the first window
    assert np.array_equal(y[0], series[3])
    assert np.array_equal(X[0], series[0:3])


def test_ar_forecaster_predict_shape():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(size=(300, 3)), axis=0)
    f = ARForecaster(window=10).fit(series)
    pred = f.predict(series)
    assert pred.shape == (len(series) - 10, 3)


def test_ar_forecaster_learns_linear_signal():
    # a perfectly predictable ramp: AR should forecast it with tiny error
    t = np.linspace(0, 10, 400)
    series = np.stack([t, 2 * t + 1], axis=1)
    f = ARForecaster(window=5).fit(series)
    pred = f.predict(series)
    actual = series[5:]
    mae = np.abs(pred - actual).mean()
    assert mae < 1e-3


def test_get_forecaster_ar_fallback():
    # 'ar' is explicit and must never require TensorFlow
    f = get_forecaster("ar", window=8)
    assert isinstance(f, ARForecaster)
    assert f.window == 8
