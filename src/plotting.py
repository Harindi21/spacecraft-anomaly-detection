"""Lightweight, dependency-free plotting (renders straight to SVG).

Matplotlib is not always installed, and a detection plot is just a few lines and
shaded regions, so this module builds the SVG by hand. The result opens in any
browser or Markdown viewer and is small enough to commit to the repo.

Two stacked panels:
  1. the telemetry signal, with ground-truth anomaly spans shaded and the
     detector's flagged spans overlaid in red;
  2. the smoothed prediction error, with the per-channel threshold line.
"""
from __future__ import annotations
import numpy as np

# colour palette (readable on a white background)
_SIGNAL = "#1f3a5f"      # navy signal line
_ERROR = "#2d6a9f"       # blue error line
_TRUE = "#9fb3c8"        # ground-truth anomaly band
_PRED = "#e2574c"        # detector flags
_THRESH = "#e2574c"      # threshold line
_AXIS = "#8a94a6"
_GRID = "#e6e9ee"

_W = 1040
_ML, _MR = 70, 24        # left / right margins
_PANEL_H = 250
_GAP = 70


def _runs(mask):
    """Contiguous True runs as (start, end_inclusive) index pairs."""
    mask = np.asarray(mask).astype(bool)
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _xmap(n):
    span = _W - _ML - _MR
    return lambda i: _ML + (i / max(n - 1, 1)) * span


def _ymap(top, vmin, vmax):
    rng = (vmax - vmin) or 1.0
    return lambda v: top + _PANEL_H - (v - vmin) / rng * _PANEL_H


def _polyline(values, x, y, stride):
    pts = []
    for i in range(0, len(values), stride):
        v = values[i]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        pts.append(f"{x(i):.1f},{y(v):.1f}")
    return " ".join(pts)


def _bands(runs, x, top, colour, opacity):
    out = []
    for s, e in runs:
        x0, x1 = x(s), x(e)
        w = max(x1 - x0, 1.2)
        out.append(f'<rect x="{x0:.1f}" y="{top:.1f}" width="{w:.1f}" '
                   f'height="{_PANEL_H:.1f}" fill="{colour}" opacity="{opacity}"/>')
    return "\n".join(out)


def plot_detection(signal, labels, pred_mask, error=None, threshold=None,
                   window=0, path="detection.svg", title="Anomaly detection",
                   channel_name="channel"):
    """Render a detection figure to `path` (SVG) and return the SVG string.

    signal, labels, pred_mask : 1-D arrays of length T (the test stream).
    error                     : 1-D array of length T-window (optional 2nd panel).
    threshold                 : scalar threshold for the error panel (optional).
    window                    : offset so the error aligns under the signal.
    """
    signal = np.asarray(signal, dtype=float)
    labels = np.asarray(labels).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)
    n = len(signal)
    stride = max(1, n // 2000)

    two = error is not None
    height = _GAP + _PANEL_H + (_GAP + _PANEL_H if two else 0) + 50
    x = _xmap(n)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" '
           f'height="{height}" viewBox="0 0 {_W} {height}" '
           f'font-family="Segoe UI, Helvetica, Arial, sans-serif">']
    svg.append(f'<rect width="{_W}" height="{height}" fill="white"/>')
    svg.append(f'<text x="{_ML}" y="34" font-size="20" font-weight="600" '
               f'fill="#1a2533">{title}</text>')

    # ---- panel 1: signal ----
    top1 = _GAP
    vmin, vmax = float(signal.min()), float(signal.max())
    pad = (vmax - vmin) * 0.08 or 1.0
    y1 = _ymap(top1, vmin - pad, vmax + pad)
    svg.append(f'<rect x="{_ML}" y="{top1}" width="{_W-_ML-_MR}" '
               f'height="{_PANEL_H}" fill="none" stroke="{_GRID}"/>')
    svg.append(_bands(_runs(labels), x, top1, _TRUE, 0.55))
    svg.append(_bands(_runs(pred_mask), x, top1, _PRED, 0.22))
    svg.append(f'<polyline fill="none" stroke="{_SIGNAL}" stroke-width="1.1" '
               f'points="{_polyline(signal, x, y1, stride)}"/>')
    # red bar above flagged spans
    for s, e in _runs(pred_mask):
        x0 = x(s); w = max(x(e) - x0, 1.2)
        svg.append(f'<rect x="{x0:.1f}" y="{top1-6:.1f}" width="{w:.1f}" '
                   f'height="4" fill="{_PRED}"/>')
    svg.append(f'<text x="{_ML}" y="{top1-14}" font-size="13" fill="{_AXIS}">'
               f'telemetry ({channel_name})</text>')

    # ---- panel 2: error ----
    if two:
        err = np.asarray(error, dtype=float)
        top2 = _GAP + _PANEL_H + _GAP
        xe = _xmap(n)

        def xe_err(i):  # error index -> global time index (i + window)
            return xe(i + window)

        emax = float(err.max()) or 1.0
        y2 = _ymap(top2, 0.0, emax * 1.1)
        svg.append(f'<rect x="{_ML}" y="{top2}" width="{_W-_ML-_MR}" '
                   f'height="{_PANEL_H}" fill="none" stroke="{_GRID}"/>')
        epts = " ".join(f"{xe_err(i):.1f},{y2(err[i]):.1f}"
                        for i in range(0, len(err), stride))
        svg.append(f'<polyline fill="none" stroke="{_ERROR}" stroke-width="1.0" '
                   f'points="{epts}"/>')
        if threshold is not None:
            ty = y2(min(threshold, emax * 1.1))
            svg.append(f'<line x1="{_ML}" y1="{ty:.1f}" x2="{_W-_MR}" '
                       f'y2="{ty:.1f}" stroke="{_THRESH}" stroke-width="1.3" '
                       f'stroke-dasharray="6 4"/>')
            svg.append(f'<text x="{_W-_MR-4:.1f}" y="{ty-5:.1f}" font-size="12" '
                       f'text-anchor="end" fill="{_THRESH}">threshold</text>')
        svg.append(f'<text x="{_ML}" y="{top2-14}" font-size="13" fill="{_AXIS}">'
                   f'smoothed prediction error</text>')

    # ---- legend ----
    ly = height - 24
    legend = [(_TRUE, 0.55, "true anomaly"), (_PRED, 0.5, "detected")]
    lx = _ML
    for colour, op, text in legend:
        svg.append(f'<rect x="{lx}" y="{ly-11}" width="16" height="12" '
                   f'fill="{colour}" opacity="{op}"/>')
        svg.append(f'<text x="{lx+22}" y="{ly}" font-size="13" '
                   f'fill="#3a4454">{text}</text>')
        lx += 22 + 9 * len(text) + 28

    svg.append("</svg>")
    out = "\n".join(s for s in svg if s)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    return out
