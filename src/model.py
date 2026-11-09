"""Forecasting models for telemetry.

The headline model is an LSTM (Keras) that predicts the next telemetry values
from a window of recent history, exactly as in NASA's Telemanom. Because heavy
deep-learning frameworks are not always installed, a dependency-free linear
autoregressive (AR) forecaster is included so the full pipeline runs end to end
with NumPy alone (useful for a quick demo or CI). Both expose the same
fit / predict interface, so the rest of the pipeline does not change.
"""
from __future__ import annotations
import numpy as np


def make_windows(series, window):
    """series: (T, C) -> X: (N, window, C), y: (N, C) predicting the next step."""
    series = np.asarray(series, dtype=float)
    T, C = series.shape
    N = T - window
    X = np.stack([series[i:i + window] for i in range(N)])
    y = series[window:window + N]
    return X, y


class ARForecaster:
    """Linear autoregressive forecaster solved in closed form (NumPy lstsq).

    Trains instantly, no external dependencies. Predicts all channels jointly
    from the flattened window of recent history.
    """

    def __init__(self, window=10):
        self.window = window
        self.W = None

    def fit(self, series):
        X, y = make_windows(series, self.window)
        A = X.reshape(len(X), -1)
        A = np.hstack([A, np.ones((len(A), 1))])  # bias term
        self.W, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, series):
        X, _ = make_windows(series, self.window)
        A = X.reshape(len(X), -1)
        A = np.hstack([A, np.ones((len(A), 1))])
        return A @ self.W  # (N, C), aligned to series[window:]


class LSTMForecaster:
    """Keras LSTM forecaster (used for the real SMAP/MSL data).

    Requires TensorFlow:  pip install tensorflow
    """

    def __init__(self, window=250, units=80, dropout=0.3, epochs=20, batch=64):
        self.window, self.units, self.dropout = window, units, dropout
        self.epochs, self.batch, self.model = epochs, batch, None

    def fit(self, series):
        from tensorflow import keras
        from tensorflow.keras import layers
        X, y = make_windows(series, self.window)
        C = series.shape[1]
        self.model = keras.Sequential([
            layers.Input((self.window, C)),
            layers.LSTM(self.units, return_sequences=True),
            layers.Dropout(self.dropout),
            layers.LSTM(self.units),
            layers.Dropout(self.dropout),
            layers.Dense(C),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch,
                       validation_split=0.1, verbose=2)
        return self

    def predict(self, series):
        X, _ = make_windows(series, self.window)
        return self.model.predict(X, verbose=0)


def get_forecaster(kind="auto", **kw):
    """kind: 'lstm', 'ar', or 'auto' (LSTM if TensorFlow is installed else AR)."""
    if kind == "ar":
        return ARForecaster(**kw)
    if kind == "lstm":
        return LSTMForecaster(**kw)
    import importlib.util
    if importlib.util.find_spec("tensorflow") is not None:
        return LSTMForecaster(**kw)
    print("[model] TensorFlow not found - using the NumPy AR baseline.")
    return ARForecaster(window=kw.get("window", 10))
