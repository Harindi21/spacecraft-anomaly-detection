"""Data loading: real NASA SMAP/MSL telemetry, plus a synthetic generator.

The real benchmark is the SMAP (Soil Moisture Active Passive) and MSL (Mars
Science Laboratory / Curiosity rover) telemetry released with NASA JPL's
Telemanom work. A convenient copy is on Kaggle:

    patrickfleith/nasa-anomaly-detection-dataset-smap-msl

Download it (see README) and point `load_smap_msl` at the folder. Until then,
`synthetic_telemetry` produces labelled, anomaly-injected signals so the whole
pipeline can be run and demonstrated immediately.
"""
from __future__ import annotations
import os
import numpy as np


def synthetic_telemetry(n_steps=4000, n_channels=5, n_anomalies=6, seed=0):
    """Return (series (T,C), labels (T,) in {0,1}).

    Each channel is a smooth, periodic signal with noise; anomalies are short
    bursts (level shifts / spikes / dropouts) injected at random positions.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    series = np.zeros((n_steps, n_channels))
    for c in range(n_channels):
        period = rng.uniform(120, 400)
        phase = rng.uniform(0, 2 * np.pi)
        trend = rng.uniform(-1, 1) * (t / n_steps)
        series[:, c] = (np.sin(2 * np.pi * t / period + phase)
                        + 0.4 * np.sin(2 * np.pi * t / (period / 3) + phase)
                        + trend
                        + rng.normal(0, 0.05, n_steps))

    labels = np.zeros(n_steps, dtype=int)
    for _ in range(n_anomalies):
        start = rng.integers(int(n_steps * 0.3), n_steps - 60)
        length = rng.integers(15, 45)
        ch = rng.integers(0, n_channels)
        kind = rng.integers(0, 3)
        sl = slice(start, start + length)
        if kind == 0:      # level shift
            series[sl, ch] += rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
        elif kind == 1:    # spikes
            series[sl, ch] += rng.normal(0, 1.5, length)
        else:              # dropout / flatline
            series[sl, ch] = series[start, ch]
        labels[sl] = 1
    return series, labels


def load_smap_msl(root, channel):
    """Load one channel of the real benchmark.

    Expects the Telemanom layout:
        <root>/train/<channel>.npy   and   <root>/test/<channel>.npy
    plus a labelled_anomalies.csv mapping channels to anomalous ranges in test.
    Returns (train (Tr,C), test (Te,C), test_labels (Te,) or None).
    """
    train = np.load(os.path.join(root, "train", f"{channel}.npy"))
    test = np.load(os.path.join(root, "test", f"{channel}.npy"))
    if train.ndim == 1:
        train, test = train[:, None], test[:, None]

    labels = None
    csv = os.path.join(root, "labeled_anomalies.csv")
    if not os.path.exists(csv):
        csv = os.path.join(root, "labelled_anomalies.csv")
    if os.path.exists(csv):
        import pandas as pd, ast
        df = pd.read_csv(csv)
        row = df[df["chan_id"] == channel]
        if len(row):
            labels = np.zeros(len(test), dtype=int)
            for s, e in ast.literal_eval(row.iloc[0]["anomaly_sequences"]):
                labels[s:e + 1] = 1
    return train, test, labels
