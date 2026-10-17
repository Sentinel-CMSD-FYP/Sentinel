"""Tests for eval/trackeval_eval.py: MOT conversion and TrackEval sanity checks."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval"
sys.path.insert(0, str(EVAL))  # the oracle helper imports point_metrics
SPEC = importlib.util.spec_from_file_location("trackeval_eval", EVAL / "trackeval_eval.py")
te = importlib.util.module_from_spec(SPEC)
sys.modules["trackeval_eval"] = te
SPEC.loader.exec_module(te)

TRACKEVAL = ROOT / "ml" / "third_party" / "uav-dot-track" / "eval" / "TrackEval"
needs_trackeval = pytest.mark.skipif(not (TRACKEVAL / "trackeval").exists(),
                                     reason="TrackEval clone not available (git-ignored)")


def synthetic_gt(n_people=4, n_frames=30):
    """People 60 px apart, drifting slowly; GT as {frame_id: (points, ids)}."""
    gt = {}
    for f in range(n_frames):
        pts = np.array([[100.0 + 60 * i + 0.5 * f, 200.0 + 40 * (i % 2)] for i in range(n_people)])
        gt[f] = (pts, np.arange(1, n_people + 1))
    return gt


def test_gt_rows_are_one_based_20px_boxes_with_the_active_flag():
    rows = te.gt_rows({0: (np.array([[50.0, 30.0]]), np.array([7]))})
    assert rows == [[1, 7, 40.0, 20.0, 20, 20, 1, -1, -1, -1]]


def doc_with(frames):
    return {"frames": frames}


def test_tracks_rows_use_the_same_box_convention_and_can_filter_short_ids():
    doc = doc_with({"0": [[1, 50.0, 30.0, 0.9], [2, 90.0, 30.0, 0.8]],
                    "1": [[1, 51.0, 30.0, 0.9]]})
    rows = te.tracks_rows(doc)
    assert rows[0] == [1, 1, 40.0, 20.0, 20, 20, 0.9, -1, -1, -1]
    assert len(rows) == 3
    kept = te.tracks_rows(doc, min_len=2)  # id 2 appears in one frame only
    assert {r[1] for r in kept} == {1} and len(kept) == 2


def test_mot_files_and_seqinfo_layout(tmp_path):
    gt = synthetic_gt(2, 5)
    te.write_gt(tmp_path, "007", te.gt_rows(gt), 5)
    base = tmp_path / "gt" / te.BENCH / "007"
    lines = (base / "gt" / "gt.txt").read_text().splitlines()
    assert len(lines) == 10 and lines[0].startswith("1,1,90")  # frame 1, id 1, x 100 - 10
    assert (base / "seqinfo.ini").read_text() == "[Sequence]\nseqlength = 5"
    seqmap = te.write_seqmap(tmp_path, ["007", "008"])
    assert seqmap.read_text() == "name\n007\n008\n"


def test_sanity_variants_have_the_intended_properties():
    gt = {1: synthetic_gt(6, 20)}
    v = te.sanity_variants(gt, seed=0)
    assert set(v) == {"gt_as_prediction", "consistent_relabel", "shuffled_ids",
                      "half_detections_dropped"}
    pts, ids = gt[1][5]
    spts, sids = v["shuffled_ids"][1][5]
    np.testing.assert_array_equal(spts, pts)           # same boxes
    assert sorted(sids) == sorted(ids)                  # same ids, permuted among the objects
    rpts, rids = v["consistent_relabel"][1][5]
    np.testing.assert_array_equal(rids, ids + 1000)
    hpts, hids = v["half_detections_dropped"][1][5]
    assert len(hids) == len(ids) // 2 and set(hids) <= set(ids)


@pytest.mark.skipif(not (ROOT / "data" / "dronecrowd").exists()
                    or not (TRACKEVAL / "trackeval").exists(),
                    reason="needs the dataset and the authors' repo")
def test_our_gt_conversion_matches_the_authors_gt_file_on_a_real_clip():
    from sentinel_ml.localization import data as D

    cfg = D.load_config(ROOT / "ml" / "configs" / "dronecrowd_data.yaml")
    cfg = {**cfg, "root": str(ROOT / cfg["root"])}
    ours = te.gt_rows(D.clip_gt_frames(cfg, "test", 11, list(range(60))))
    gt_file = (ROOT / "ml" / "third_party" / "uav-dot-track" / "eval" / "gt" / "dronecrowd"
               / "MOT17-test" / "011" / "gt" / "gt.txt")
    authors = np.loadtxt(gt_file, delimiter=",")
    authors = authors[authors[:, 0] <= 60]
    assert len(ours) == len(authors)  # the same rows (same people in the same frames)
    a = {(int(r[0]), int(r[1])): r[2:6] for r in authors}
    diff = np.array([np.array(r[2:6]) - a[(int(r[0]), int(r[1]))] for r in ours])
    assert np.abs(diff[:, 2:]).max() == 0  # the boxes are 20x20 in both
    close = (np.abs(diff[:, 0]) <= 0.5) & (np.abs(diff[:, 1]) <= 0.5)
    assert close.mean() > 0.9  # the rest are heads at the frame border (clipped boxes)
    assert np.abs(diff[:, :2]).max() <= 10.0


@needs_trackeval
@pytest.mark.parametrize("threshold", [0.5])
def test_trackeval_sanity_checks_behave(tmp_path, threshold):
    trackeval = te.load_trackeval(TRACKEVAL)
    res = te.run_sanity(trackeval, {1: synthetic_gt(6, 30), 2: synthetic_gt(5, 30)}, tmp_path,
                        threshold)
    perfect = res["gt_as_prediction"]["mean_over_clips"]
    assert all(perfect[k] == pytest.approx(1.0) for k in ("HOTA", "IDF1", "MOTA"))
    assert res["gt_as_prediction"]["totals"]["IDSW"] == 0
    assert res["consistent_relabel"]["mean_over_clips"] == perfect  # names do not matter
    shuffled = res["shuffled_ids"]
    assert shuffled["mean_over_clips"]["DetA"] > 0.99      # boxes are perfect ...
    assert shuffled["mean_over_clips"]["IDF1"] < 0.3       # ... but the identities are not
    assert shuffled["totals"]["IDSW"] > 50
    half = res["half_detections_dropped"]["mean_over_clips"]
    assert half["MOTA"] == pytest.approx(0.5, abs=0.1)
    assert half["IDF1"] == pytest.approx(2 * 0.5 / 1.5, abs=0.1)


@needs_trackeval
def test_the_authors_threshold_zero_hides_id_errors(tmp_path):
    trackeval = te.load_trackeval(TRACKEVAL)
    gt = {1: synthetic_gt(6, 30)}
    zero = te.run_sanity(trackeval, gt, tmp_path / "a", 0.0)["shuffled_ids"]
    half = te.run_sanity(trackeval, gt, tmp_path / "b", 0.5)["shuffled_ids"]
    # with threshold 0.0 every pair of boxes counts as a match, so shuffled ids look fine
    assert zero["mean_over_clips"]["IDF1"] > 0.9 > 0.3 > half["mean_over_clips"]["IDF1"]
    assert zero["totals"]["IDSW"] < half["totals"]["IDSW"] / 5


def test_oracle_gives_true_ids_to_matched_detections_and_fresh_ids_to_false_ones():
    gt = {0: (np.array([[100.0, 100.0], [300.0, 100.0]]), np.array([5, 6])),
          1: (np.array([[101.0, 100.0], [301.0, 100.0]]), np.array([5, 6]))}
    doc = {"frames": {"0": [[103.0, 100.0, 0.9], [300.0, 104.0, 0.8], [700.0, 50.0, 0.9]],
                      "1": [[101.0, 101.0, 0.9], [301.0, 100.0, 0.1]]}}
    rows = te.oracle_rows(doc, gt, score_thr=0.5)
    by_frame = {}
    for r in rows:
        by_frame.setdefault(int(r[0]), []).append(int(r[1]))
    assert by_frame[1][:2] == [5, 6] and by_frame[1][2] >= 10**6   # true ids, then a fresh id
    assert by_frame[2] == [5]                                      # the 0.1 detection is dropped
    ids = [r[1] for r in rows]
    assert len(set(i for i in ids if i >= 10**6)) == 1             # one false detection, one id


def test_oracle_with_perfect_detections_equals_the_ground_truth_rows():
    gt = synthetic_gt(3, 5)
    doc = {"frames": {str(f): [[x, y, 0.9] for x, y in pts] for f, (pts, _) in gt.items()}}
    rows = te.oracle_rows(doc, gt, score_thr=0.5)
    assert [r[:6] for r in rows] == [r[:6] for r in te.gt_rows(gt)]
