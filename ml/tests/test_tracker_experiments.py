"""Tests for eval/tracker_experiments.py: scoring a grid of configs and the selection rule."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from test_diagnose_tracking import make_clip

from sentinel_ml.tracking import TrackerConfig

EVAL = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL))
SPEC = importlib.util.spec_from_file_location(
    "tracker_experiments", EVAL / "tracker_experiments.py")
te = importlib.util.module_from_spec(SPEC)
sys.modules["tracker_experiments"] = te
SPEC.loader.exec_module(te)

BASE = TrackerConfig()
HEADER = {"schema_version": "1.0", "clip_id": "clip_999", "fps": 25, "frame_stride": 1,
          "altitude_m": 60.0, "altitude_source": "assumed_constant"}


def clip(n_frames, **kw):
    """Synthetic points document (with the contract header) and its GT."""
    doc, gt = make_clip(n_frames, **kw)
    return {**HEADER, **doc}, gt


def result(thr, hits, age, err, cov=0.95):
    return {"score_thr": thr, "min_hits": hits, "max_age": age, "mean_err_pct": err,
            "mean_err_pct_min_len": err, "ids_per_person": 1.0, "mean_coverage": cov,
            "mean_purity": 0.97, "max_age_s": None, "spurious_ids": 0,
            "extra_ids_for_tracked": 0}


def test_perfect_clip_scores_zero_error_full_coverage():
    doc, gt = clip(30)
    r = te.evaluate_config({1: doc}, {1: gt}, BASE, 45)
    assert r["mean_err_pct"] == 0.0 and r["mean_coverage"] == 1.0
    assert r["ids_per_person"] == 1.0 and r["spurious_ids"] == 0 and r["clips"] == 1


def test_a_larger_max_age_bridges_a_long_dropout():
    doc, gt = clip(40, missing={2: set(range(10, 20))})  # 10 missed frames
    short = te.evaluate_config({1: doc}, {1: gt}, replace(BASE, max_age=5), 45)
    long = te.evaluate_config({1: doc}, {1: gt}, replace(BASE, max_age=15), 45)
    assert short["mean_err_pct"] == pytest.approx(100 / 3) and short["extra_ids_for_tracked"] == 1
    assert long["mean_err_pct"] == 0.0 and long["extra_ids_for_tracked"] == 0


def test_a_higher_score_thr_drops_low_score_detections():
    doc, gt = clip(30)
    for rows in doc["frames"].values():
        rows[0][2] = 0.3  # person 1 is only ever seen at score 0.3
    low = te.evaluate_config({1: doc}, {1: gt}, replace(BASE, score_thr=0.2), 45)
    high = te.evaluate_config({1: doc}, {1: gt}, replace(BASE, score_thr=0.5), 45)
    assert low["mean_coverage"] == 1.0
    assert high["mean_coverage"] == pytest.approx(2 / 3)  # person 1 is no longer tracked


def test_selection_ignores_configs_below_the_coverage_floor():
    rs = [result(0.5, 3, 5, 5.0, cov=0.60), result(0.2, 3, 5, 40.0, cov=0.95)]
    assert te.select(rs, 0.90, 1.0, 1)["score_thr"] == 0.2
    assert te.select([result(0.5, 3, 5, 5.0, cov=0.6)], 0.90, 1.0, 1) is None


def test_step1_ties_prefer_the_threshold_closest_to_0_20_then_the_lower():
    rs = [result(0.10, 3, 5, 50.0), result(0.15, 3, 5, 50.4), result(0.25, 3, 5, 50.6),
          result(0.30, 3, 5, 52.0)]
    assert te.select(rs, 0.9, 1.0, 1)["score_thr"] == 0.15  # |0.15-0.2| == |0.25-0.2|; lower wins
    far = [result(0.10, 3, 5, 50.0), result(0.35, 3, 5, 50.5)]
    assert te.select(far, 0.9, 1.0, 1)["score_thr"] == 0.10  # closer to 0.2 than 0.35 is


def test_step2_ties_prefer_the_smaller_min_hits_then_max_age():
    rs = [result(0.2, 8, 5, 30.0), result(0.2, 5, 15, 30.5), result(0.2, 5, 10, 30.9),
          result(0.2, 12, 25, 30.0)]
    chosen = te.select(rs, 0.9, 1.0, 2)
    assert (chosen["min_hits"], chosen["max_age"]) == (5, 10)


def test_a_clearly_better_config_wins_over_the_tie_break():
    rs = [result(0.2, 3, 5, 60.0), result(0.2, 12, 25, 20.0)]
    assert te.select(rs, 0.9, 1.0, 2)["min_hits"] == 12


def test_test_split_paths_are_refused():
    te.check_val_only(Path("data/outputs/points/val"))
    with pytest.raises(SystemExit, match="val-only"):
        te.check_val_only(Path("data/outputs/points/test"))


def test_selected_config_is_written_and_the_baseline_is_untouched(tmp_path):
    base = tmp_path / "tracker.yaml"
    base.write_text(yaml.safe_dump({"points_dir": "x", "tracker": {
        "max_age": 5, "min_hits": 3, "score_thr": 0.2, "size_px": 20}}))
    before = base.read_text()
    out = tmp_path / "selected.yaml"
    te.write_selected_config(base, result(0.3, 8, 15, 10.0), out)
    cfg = yaml.safe_load(out.read_text())
    assert cfg["tracker"] == {"max_age": 15, "min_hits": 8, "score_thr": 0.3, "size_px": 20}
    assert base.read_text() == before and "Frozen" in out.read_text()


def test_outputs_and_table(tmp_path):
    rs = [result(0.2, 3, 5, 10.0), result(0.5, 3, 5, 5.0, cov=0.5)]
    chosen = rs[0]
    te.write_results(rs, chosen, {"note": "x"}, tmp_path)
    assert (tmp_path / "results.json").exists()
    assert "score_thr" in (tmp_path / "results.csv").read_text()
    table = te.format_table(rs, chosen, 0.9)
    assert "selected" in table and "coverage below floor" in table


def result3(age_s, err, cov=0.95, purity=0.97):
    r = result(0.35, 3, int(age_s * 25), err, cov)
    r.update(max_age_s=age_s, mean_purity=purity)
    return r


def test_step3_purity_floor_stops_a_long_max_age_that_merges_people():
    rs = [result3(1.0, 54.0), result3(2.0, 40.0), result3(4.0, 20.0, purity=0.90)]
    assert te.select(rs, 0.90, 1.0, 3, min_purity=0.935)["max_age_s"] == 2.0
    assert te.select(rs, 0.90, 1.0, 3, min_purity=None)["max_age_s"] == 4.0  # without the floor


def test_step3_ties_go_to_the_smaller_max_age_s():
    rs = [result3(3.0, 30.0), result3(1.5, 30.8), result3(4.0, 29.9)]
    assert te.select(rs, 0.90, 1.0, 3, min_purity=0.935)["max_age_s"] == 1.5


def test_step3_returns_none_when_no_config_meets_both_floors():
    assert te.select([result3(2.0, 10.0, purity=0.5)], 0.90, 1.0, 3, min_purity=0.935) is None


def test_evaluate_config_reports_purity_and_accepts_seconds():
    doc, gt = clip(40, missing={2: set(range(10, 30))})  # a 20-frame dropout
    five = te.evaluate_config({1: doc}, {1: gt}, replace(BASE, max_age=5), 45)
    two_s = te.evaluate_config({1: doc}, {1: gt}, replace(BASE, max_age_s=2.0), 45)  # 50 frames
    assert five["mean_purity"] == 1.0 and two_s["mean_purity"] == 1.0
    assert two_s["mean_err_pct"] == 0.0 and five["mean_err_pct"] > 0.0


def test_floors_for_step_use_the_step3_overrides_only_for_step3():
    sel = {"min_coverage": 0.90, "step3_min_purity": 0.935, "step3_min_coverage": 0.88}
    assert te.floors_for_step(sel, 1) == (0.90, None)
    assert te.floors_for_step(sel, 2) == (0.90, None)
    assert te.floors_for_step(sel, 3) == (0.88, 0.935)
    no_override = {"min_coverage": 0.90, "step3_min_purity": 0.935}
    assert te.floors_for_step(no_override, 3) == (0.90, 0.935)


def test_the_relaxed_step3_floor_selects_3s_from_the_recorded_val_numbers():
    recorded = [(1.0, 54.69, 0.9041, 0.955), (1.5, 45.79, 0.8955, 0.948),
                (2.0, 43.10, 0.8925, 0.948), (3.0, 40.39, 0.8897, 0.9463),
                (4.0, 39.43, 0.8873, 0.9453)]
    rows = [result3(age, err, cov=cov, purity=pur) for age, err, cov, pur in recorded]
    assert te.select(rows, 0.90, 1.0, 3, 0.935)["max_age_s"] == 1.0   # the rule as first written
    assert te.select(rows, 0.88, 1.0, 3, 0.935)["max_age_s"] == 3.0   # post-hoc floor 0.88


def metrics(err, cov=0.9, purity=0.95):
    return {"mean_err_pct": err, "mean_coverage": cov, "mean_purity": purity}


def test_altitude_is_turned_on_only_with_a_big_enough_gain_and_the_floors_held():
    off = metrics(40.0)
    assert te.select_altitude(off, metrics(38.9), 0.88, 0.935, 1.0)["use_altitude"] is True
    assert te.select_altitude(off, metrics(39.2), 0.88, 0.935, 1.0)["use_altitude"] is False  # 0.8
    worse = te.select_altitude(off, metrics(45.0), 0.88, 0.935, 1.0)
    assert worse["use_altitude"] is False
    low_cov = te.select_altitude(off, metrics(30.0, cov=0.85), 0.88, 0.935, 1.0)
    assert low_cov["use_altitude"] is False and low_cov["floors_ok"] is False
    low_pur = te.select_altitude(off, metrics(30.0, purity=0.90), 0.88, 0.935, 1.0)
    assert low_pur["use_altitude"] is False and low_pur["floors_ok"] is False
    assert te.select_altitude(off, metrics(38.0), 0.88, 0.935, 1.0)["gain_points"] == 2.0


def jump_clip():
    """One person standing still, then stepping 30 px sideways at frame 10 (GT follows)."""
    import numpy as np

    doc, _ = clip(1)
    frames, gt = {}, {}
    for f in range(30):
        x = 300.0 if f < 10 else 330.0
        frames[str(f)] = [[x, 300.0, 0.9]]
        gt[f] = (np.array([[x, 300.0]]), np.array([1]))
    doc["frames"] = frames
    return doc, gt


def test_per_clip_altitude_changes_the_result_through_evaluate_config():
    doc, gt = jump_clip()
    cfg = replace(BASE, use_altitude=True)
    plain = te.evaluate_config({1: doc}, {1: gt}, replace(BASE, use_altitude=False), 45)
    wide = te.evaluate_config({1: doc}, {1: gt}, cfg, 45, altitude_by_clip={1: 50.0})  # gate 40
    narrow = te.evaluate_config({1: doc}, {1: gt}, cfg, 45, altitude_by_clip={1: 100.0})  # gate 20
    assert plain["extra_ids_for_tracked"] == 1 and narrow["extra_ids_for_tracked"] == 1
    assert wide["extra_ids_for_tracked"] == 0 and wide["mean_err_pct"] < plain["mean_err_pct"]


def test_candidate_config_records_the_altitude_decision(tmp_path):
    base = tmp_path / "base.yaml"
    tracker = {"score_thr": 0.35, "min_hits": 3, "max_age_s": 3.0}
    base.write_text(yaml.safe_dump({"tracker": tracker}))
    out = tmp_path / "candidate.yaml"
    te.write_candidate_config(base, True, out)
    loaded = yaml.safe_load(out.read_text())
    assert loaded["tracker"] == {"score_thr": 0.35, "min_hits": 3, "max_age_s": 3.0,
                                 "use_altitude": True}
    assert "Candidate" in out.read_text()
