"""Tests for eval/analyse_id_switches.py on tiny hand-made fixtures (CPU only)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parents[2] / "eval"
SPEC = importlib.util.spec_from_file_location(
    "analyse_id_switches", EVAL / "analyse_id_switches.py")
ais = importlib.util.module_from_spec(SPEC)
sys.modules["analyse_id_switches"] = ais
SPEC.loader.exec_module(ais)


def m(t, gt, trk, iou=0.8):
    return {"t": t, "gt": gt, "trk": trk, "iou": iou}


def test_iou_of_identical_boxes_is_one_and_disjoint_is_zero():
    a = np.array([[0.0, 0.0, 20.0, 20.0]])
    b = np.array([[0.0, 0.0, 20.0, 20.0], [100.0, 100.0, 20.0, 20.0]])
    assert np.allclose(ais.iou_matrix(a, b), [[1.0, 0.0]])


def test_iou_of_two_20px_boxes_7px_apart_is_below_half():
    a = np.array([[0.0, 0.0, 20.0, 20.0]])
    b = np.array([[7.0, 0.0, 20.0, 20.0]])
    assert abs(ais.iou_matrix(a, b)[0, 0] - 260 / 540) < 1e-9


def test_a_stable_track_has_no_switch():
    assert ais.find_switches([m(t, 1, 7) for t in range(1, 6)]) == []


def test_new_id_on_the_next_frame_is_new_id_direct():
    sw = ais.find_switches([m(1, 1, 7), m(2, 1, 7), m(3, 1, 8)])
    assert [s["cls"] for s in sw] == ["new_id_direct"]


def test_a_b_a_is_a_flip_back():
    sw = ais.find_switches([m(1, 1, 7), m(2, 1, 8), m(3, 1, 7)])
    assert [s["cls"] for s in sw] == ["new_id_direct", "flip_back"]


def test_new_id_after_unmatched_frames_is_after_gap_and_counts_as_a_switch():
    sw = ais.find_switches([m(1, 1, 7), m(10, 1, 8)])
    assert [(s["cls"], s["gap"]) for s in sw] == [("after_gap", 9)]


def test_two_people_exchanging_ids_are_real_swaps():
    sw = ais.find_switches([m(1, 1, 7), m(1, 2, 8), m(2, 1, 8), m(2, 2, 7)])
    assert [s["cls"] for s in sw] == ["real_swap", "real_swap"]


def test_clear_match_prefers_the_previous_frame_id():
    # two tracks both overlap the person at frame 2; the one used at frame 1 must win
    box = np.array([[0.0, 0.0, 20.0, 20.0]])
    gt = {1: (np.array([1]), box), 2: (np.array([1]), box)}
    trk = {1: (np.array([7]), box), 2: (np.array([8, 7]), np.vstack([box, box + 1]))}
    got = ais.clear_match(gt, trk, 0.5)
    assert [(x["t"], x["trk"]) for x in got] == [(1, 7), (2, 7)]


def test_identity_stats_counts_tracks_per_person_and_purity():
    box = np.zeros((1, 4))
    gt = {1: (np.array([1]), box), 2: (np.array([1]), box)}
    trk = {1: (np.array([7]), box), 2: (np.array([8]), box)}
    s = ais.identity_stats([m(1, 1, 7), m(2, 1, 8)], gt, trk)
    assert s["gt_identities"] == 1 and s["predicted_ids"] == 2
    assert s["tracks_per_person_mean"] == 2.0 and s["purity_matched_weighted"] == 1.0
