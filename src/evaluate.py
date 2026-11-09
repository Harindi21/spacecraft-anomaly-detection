"""Evaluation metrics (pure NumPy), with point-adjusted scoring.

Point adjustment is the standard convention for time-series anomaly detection
(and is used in the Telemanom line of work): if any point inside a true
anomalous segment is flagged, the whole segment counts as detected. We report
both raw and point-adjusted precision / recall / F1.
"""
from __future__ import annotations
import numpy as np


def _segments(labels):
    segs, start = [], None
    for i, v in enumerate(labels):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i - 1)); start = None
    if start is not None:
        segs.append((start, len(labels) - 1))
    return segs


def point_adjust(pred, labels):
    """Flag the whole true segment if any point in it was predicted."""
    pred = np.asarray(pred).astype(bool).copy()
    for s, e in _segments(np.asarray(labels).astype(int)):
        if pred[s:e + 1].any():
            pred[s:e + 1] = True
    return pred


def prf(pred, labels):
    pred = np.asarray(pred).astype(bool)
    labels = np.asarray(labels).astype(bool)
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def evaluate(pred, labels):
    return {"raw": prf(pred, labels),
            "point_adjusted": prf(point_adjust(pred, labels), labels)}
