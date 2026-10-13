"""Unit tests for eval/point_metrics.py on synthetic cases with known answers."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "point_metrics", Path(__file__).resolve().parents[2] / "eval" / "point_metrics.py"
)
pm = importlib.util.module_from_spec(SPEC)
sys.modules["point_metrics"] = pm  # needed for @dataclass in a module loaded by path
SPEC.loader.exec_module(pm)

GT = np.array([[100.0, 100.0], [300.0, 200.0], [500.0, 400.0]])


def counts(pred, gt=GT, radius=10.0):
    pred = np.asarray(pred, dtype=float)
    if pred.size == 0:
        pred = pred.reshape(0, 2)
    c = pm.frame_counts(pred, gt, radius)
    return c.tp, c.fp, c.fn


def test_perfect():
    assert counts(GT) == (3, 0, 0)
    assert pm.precision_recall_f1(pm.Counts(3, 0, 0)) == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_all_miss():
    far = GT + 100.0
    assert counts(far) == (0, 3, 3)
    assert pm.precision_recall_f1(pm.Counts(0, 3, 3)) == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_offset_9px_matches_11px_does_not():
    assert counts(GT + [9.0, 0.0]) == (3, 0, 0)
    assert counts(GT + [11.0, 0.0]) == (0, 3, 3)


def test_exactly_radius_matches():
    assert counts([[106.0, 108.0]], GT[:1]) == (1, 0, 0)  # 6-8-10 triangle, distance 10.0


def test_diagonal_offset_uses_euclidean_distance():
    # (7, 7) is about 9.9 px away, (8, 8) about 11.3 px
    assert counts(GT[:1] + [7.0, 7.0], GT[:1]) == (1, 0, 0)
    assert counts(GT[:1] + [8.0, 8.0], GT[:1]) == (0, 1, 1)


def test_duplicates_count_once():
    pred = [[100.0, 100.0], [102.0, 100.0], [98.0, 101.0]]
    assert counts(pred, GT[:1]) == (1, 2, 0)


def test_nearest_prediction_wins_the_gt_point():
    pred = [[108.0, 100.0], [103.0, 100.0]]  # 8 px and 3 px from the GT point
    assert pm.match_points(np.array(pred), GT[:1], 10.0) == [(1, 0)]


def test_greedy_nearest_first_is_not_optimal_assignment():
    # Documents the agreed rule. Optimal assignment would match both predictions
    # (p0-g1 at 6 px, p1-g0 at 5 px); nearest-first takes p0-g0 (4 px) and leaves p1 unmatched.
    gt = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred = np.array([[4.0, 0.0], [-5.0, 0.0]])
    assert pm.match_points(pred, gt, 10.0) == [(0, 0)]


def test_one_to_one_never_reuses_a_point():
    gt = np.array([[0.0, 0.0], [5.0, 0.0]])
    pred = np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    m = pm.match_points(pred, gt, 10.0)
    assert len(m) == 2
    assert len({p for p, _ in m}) == 2 and len({g for _, g in m}) == 2


def test_empty_predictions():
    assert counts(np.zeros((0, 2))) == (0, 0, 3)
    assert pm.precision_recall_f1(pm.Counts(0, 0, 3)) == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_empty_ground_truth():
    assert counts(GT, np.zeros((0, 2))) == (0, 3, 0)
    assert pm.precision_recall_f1(pm.Counts(0, 3, 0))["precision"] == 0.0


def test_both_empty_is_perfect():
    assert counts(np.zeros((0, 2)), np.zeros((0, 2))) == (0, 0, 0)
    assert pm.precision_recall_f1(pm.Counts(0, 0, 0))["f1"] == 1.0


def test_f1_known_value():
    r = pm.precision_recall_f1(pm.Counts(tp=6, fp=2, fn=4))
    assert r["precision"] == pytest.approx(0.75)
    assert r["recall"] == pytest.approx(0.6)
    assert r["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_bad_radius_and_shape():
    with pytest.raises(ValueError):
        pm.match_points(GT, GT, 0.0)
    with pytest.raises(ValueError):
        pm.match_points(np.zeros((3, 1)), GT, 10.0)


def test_extra_columns_are_ignored_for_matching():
    pred = np.hstack([GT, np.full((3, 1), 0.9)])
    assert counts(pred) == (3, 0, 0)


def test_score_threshold_filters_low_scores():
    pred = {"0": np.array([[100.0, 100.0, 0.9], [300.0, 200.0, 0.3], [900.0, 900.0, 0.2]])}
    gt = {"0": GT[:2]}
    low = pm.evaluate(pred, gt, 10.0, score_thr=0.0)
    assert (low["tp"], low["fp"], low["fn"]) == (2, 1, 0)
    high = pm.evaluate(pred, gt, 10.0, score_thr=0.5)
    assert (high["tp"], high["fp"], high["fn"]) == (1, 0, 1)
    edge = pm.evaluate(pred, gt, 10.0, score_thr=0.3)  # score == thr is kept
    assert (edge["tp"], edge["fp"], edge["fn"]) == (2, 0, 0)


def test_counts_are_summed_over_frames():
    gt = {"0": GT, "1": GT[:1]}
    pred = {"0": np.hstack([GT, np.ones((3, 1))]), "1": np.array([[500.0, 500.0, 1.0]])}
    r = pm.evaluate(pred, gt, 10.0)
    assert (r["tp"], r["fp"], r["fn"], r["n_frames"]) == (3, 1, 1, 2)
    assert r["precision"] == pytest.approx(3 / 4)
    assert r["recall"] == pytest.approx(3 / 4)


def test_missing_prediction_frame_is_all_fn_and_extra_frame_ignored():
    gt = {"0": GT}
    pred = {"5": np.array([[1.0, 1.0, 1.0]])}
    r = pm.evaluate(pred, gt, 10.0)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 3)
    assert r["pred_frames_not_in_gt"] == 1


def test_loads_contract_points_file(tmp_path):
    doc = {"frames": {"0": [[1.0, 2.0, 0.5]], "1": []}}
    path = tmp_path / "p.json"
    path.write_text(json.dumps(doc))
    frames = pm.load_points_json(path)
    assert frames["0"].shape == (1, 3) and frames["1"].shape == (0, 3)


def test_sample_contract_file_scores_perfectly_against_itself():
    sample = Path(__file__).resolve().parents[2] / "docs" / "samples" / "clip_011_points.json"
    frames = pm.load_points_json(sample)
    r = pm.evaluate(frames, frames, 10.0)
    assert r["f1"] == 1.0 and r["fp"] == 0 and r["fn"] == 0
