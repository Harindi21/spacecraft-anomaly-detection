import numpy as np

from src.evaluate import _segments, point_adjust, prf, evaluate


def test_segments():
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    assert _segments(labels) == [(1, 2), (4, 4), (7, 7)]


def test_point_adjust_fills_detected_segment():
    labels = np.array([0, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0])  # one hit inside the true segment
    adj = point_adjust(pred, labels)
    assert adj.tolist() == [False, True, True, True, False]


def test_point_adjust_leaves_undetected_segment():
    labels = np.array([0, 1, 1, 1, 0])
    pred = np.array([0, 0, 0, 0, 0])  # no hit
    adj = point_adjust(pred, labels)
    assert adj.sum() == 0


def test_prf_known_counts():
    labels = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    m = prf(pred, labels)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert abs(m["f1"] - 0.5) < 1e-9


def test_prf_empty_prediction_is_zero_not_error():
    labels = np.array([0, 1, 0])
    pred = np.array([0, 0, 0])
    m = prf(pred, labels)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_evaluate_reports_both_views():
    labels = np.array([0, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0])
    res = evaluate(pred, labels)
    assert set(res) == {"raw", "point_adjusted"}
    # point adjustment can only help recall here
    assert res["point_adjusted"]["recall"] >= res["raw"]["recall"]
    assert res["point_adjusted"]["recall"] == 1.0
