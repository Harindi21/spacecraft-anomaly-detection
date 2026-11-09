"""Nonparametric dynamic thresholding for telemetry anomaly detection.

Implements the error-thresholding idea from NASA JPL's Telemanom
(Hundman et al., KDD 2018): instead of a hand-set cutoff, the threshold for
each channel is chosen automatically from the distribution of smoothed
prediction errors, then borderline anomalous sequences are pruned to limit
false positives.

Pure NumPy, so it runs anywhere with no heavy dependencies.
"""
from __future__ import annotations
import numpy as np


def _sequences(mask: np.ndarray):
    """Return list of (start, end_inclusive) index ranges where mask is True."""
    seqs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            seqs.append((start, i - 1))
            start = None
    if start is not None:
        seqs.append((start, len(mask) - 1))
    return seqs


def nonparametric_threshold(errors, z_min=2.5, z_max=12.0, z_step=0.5, prune_p=0.13):
    """Pick a threshold on the error signal and return (threshold, anomaly_mask).

    The chosen z maximises the Telemanom objective:
        ( dMu/Mu + dSigma/Sigma ) / ( |E_a| + |E_seq|**2 )
    where removing the points above the threshold should sharply drop the mean
    and std of the remaining errors while flagging as few points/sequences as
    possible. Flagged sequences whose peak error is not materially above the
    next-highest error are then pruned (controlled by `prune_p`).
    """
    e = np.asarray(errors, dtype=float)
    mu, sd = e.mean(), e.std() + 1e-12

    best = None  # (score, threshold)
    z = z_min
    while z <= z_max + 1e-9:
        thr = mu + z * sd
        above = e > thr
        if above.any():
            kept = e[~above]
            d_mu = (mu - kept.mean()) / mu
            d_sd = (sd - (kept.std() + 1e-12)) / sd
            n_a = int(above.sum())
            n_seq = len(_sequences(above))
            score = (d_mu + d_sd) / (n_a + n_seq ** 2)
            if best is None or score > best[0]:
                best = (score, thr)
        z += z_step

    thr = best[1] if best is not None else mu + z_max * sd
    mask = e > thr
    mask = _prune(e, mask, prune_p)
    return thr, mask


def _prune(e, mask, prune_p):
    """Drop flagged sequences whose peak is within `prune_p` of the next peak."""
    seqs = _sequences(mask)
    if not seqs:
        return mask
    peaks = sorted(((e[s:t + 1].max(), s, t) for s, t in seqs), reverse=True)
    # baseline = highest *normal* error
    normal_max = e[~mask].max() if (~mask).any() else 0.0
    kept = np.zeros_like(mask)
    prev = peaks[0][0]
    for peak, s, t in peaks:
        ref = max(prev, normal_max, 1e-12)
        if (prev - peak) / ref <= prune_p:
            kept[s:t + 1] = True
            prev = peak
        else:
            break  # once a gap is too large, lower peaks are treated as normal
    return kept


def smooth(errors, window=None):
    """Exponentially-weighted smoothing of the raw error signal."""
    e = np.asarray(errors, dtype=float)
    if window is None:
        window = max(5, len(e) // 100)
    alpha = 2.0 / (window + 1.0)
    out = np.empty_like(e)
    acc = e[0]
    for i, v in enumerate(e):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out
