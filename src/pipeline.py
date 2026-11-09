"""End-to-end detection: forecast -> error -> per-channel dynamic threshold.

Train the forecaster on (nominal) training telemetry, run it on the test
telemetry, turn the prediction error into a per-channel anomaly mask with the
nonparametric dynamic threshold, and OR the channels together into a single
point-level anomaly flag.
"""
from __future__ import annotations
import numpy as np
from .model import make_windows
from .threshold import nonparametric_threshold, smooth


def detect(train, test, forecaster, window):
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    C = test.shape[1]

    forecaster.fit(train)
    pred = forecaster.predict(test)          # (T-window, C)
    actual = test[window:]                    # aligned to predictions
    err = np.abs(actual - pred)               # (T-window, C)

    mask = np.zeros(len(test), dtype=bool)
    thresholds = []
    for c in range(C):
        se = smooth(err[:, c])
        thr, ch_mask = nonparametric_threshold(se)
        thresholds.append(thr)
        mask[window:][ch_mask] = True         # first `window` steps have no prediction
    return mask, err, thresholds
