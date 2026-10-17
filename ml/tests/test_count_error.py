"""Tests for eval/count_error.py with small synthetic cases and known answers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EVAL = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL))  # count_error does `import point_metrics`
SPEC = importlib.util.spec_from_file_location("count_error", EVAL / "count_error.py")
ce = importlib.util.module_from_spec(SPEC)
sys.modules["count_error"] = ce
SPEC.loader.exec_module(ce)

CUT = [100, 200]


def row(clip, gt, pred, fid=0):
    return {"clip": clip, "frame_id": fid, "gt": gt, "pred": pred}


def test_bin_edges():
    assert [ce.bin_of(n, CUT) for n in (1, 99, 100, 200, 201, 455)] == [
        "sparse", "sparse", "moderate", "moderate", "dense", "dense"]


def test_frame_rows_count_predictions_at_threshold():
    gt = {1: {"0": np.zeros((4, 2)), "1": np.zeros((2, 2))}}
    pred = {1: {"0": np.array([[0, 0, 0.9], [0, 0, 0.3], [0, 0, 0.1]]), "1": np.zeros((0, 3))}}
    rows = ce.frame_rows(pred, gt, 0.3)
    assert [(r["gt"], r["pred"]) for r in rows] == [(4, 2), (2, 0)]  # 0.3 is kept, 0.1 not


def test_frame_rows_missing_clip_raises():
    with pytest.raises(FileNotFoundError):
        ce.frame_rows({}, {1: {"0": np.zeros((1, 2))}}, 0.2)


def test_stats_known_values():
    s = ce.stats([row(1, 100, 90), row(1, 100, 120)])  # -10% and +20%
    assert s["frames"] == 2 and s["clips"] == 1
    assert s["mean_abs_error_pct"] == pytest.approx(15.0)
    assert s["median_abs_error_pct"] == pytest.approx(15.0)
    assert s["mean_bias_pct"] == pytest.approx(5.0)


def test_under_counting_gives_negative_bias():
    assert ce.stats([row(1, 100, 50)])["mean_bias_pct"] == pytest.approx(-50.0)


def test_zero_gt_frames_are_skipped_and_counted():
    s = ce.stats([row(1, 0, 5), row(1, 10, 10)])
    assert s["frames"] == 1 and s["skipped_gt0"] == 1 and s["mean_abs_error_pct"] == 0.0
    assert ce.stats([row(1, 0, 5)]) == {"frames": 0, "clips": 0, "skipped_gt0": 1}


def test_report_splits_frames_into_bins_by_gt_count():
    rows = [row(1, 50, 50), row(1, 150, 150), row(2, 250, 250), row(2, 260, 260)]
    rep = ce.report(rows, CUT, None)
    assert [rep["by_bin"][b]["frames"] for b in ce.BINS] == [1, 1, 2]
    assert "by_lighting" not in rep


def test_full_report_has_all_clips_primary_and_day_only_secondary():
    rows = [row(1, 100, 100), row(2, 100, 50), row(3, 100, 100)]
    tags = {1: "day", 2: "night", 3: "dusk"}
    res = ce.full_report(rows, CUT, tags)
    assert res["all_clips"]["overall"]["clips"] == 3  # nothing is dropped
    assert res["day_only"]["overall"]["clips"] == 1
    assert res["all_clips"]["by_lighting"]["night"]["mean_bias_pct"] == pytest.approx(-50.0)
    assert res["all_clips"]["clips_per_lighting"] == {"day": [1], "dusk": [3], "night": [2]}


def test_without_tags_there_is_no_day_only_view():
    assert "day_only" not in ce.full_report([row(1, 100, 100)], CUT, None)


def test_load_tags_ignores_unfilled_template_and_checks_values(tmp_path):
    path = tmp_path / "tags.yaml"
    path.write_text(yaml.safe_dump({"val": {1: None, 2: "day"}, "test": {11: "night"}}))
    assert ce.load_tags(path, "val") is None            # not fully filled in
    assert ce.load_tags(path, "test") == {11: "night"}
    assert ce.load_tags(tmp_path / "missing.yaml", "val") is None
    path.write_text(yaml.safe_dump({"val": {1: "sunny"}}))
    with pytest.raises(ValueError, match="sunny"):
        ce.load_tags(path, "val")


def test_test_split_is_refused_when_results_already_exist(tmp_path, monkeypatch):
    cfg = {"data_config": "unused", "lighting_tags": "unused", "threshold": 0.2,
           "bin_cutoffs": CUT, "provisional_targets_pct": {}, "out_dir": str(tmp_path / "out")}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    done = tmp_path / "out" / "test"
    done.mkdir(parents=True)
    (done / "count_error.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["count_error.py", "--config", str(cfg_path),
                                      "--split", "test"])
    with pytest.raises(SystemExit, match="once"):
        ce.main()


def test_outputs_are_written(tmp_path):
    res = ce.full_report([row(1, 100, 90), row(1, 250, 250)], CUT, {1: "day"})
    paths = ce.write_outputs(res, {"label": "x"}, tmp_path)
    assert {p.name for p in paths} == {"count_error.json", "count_error.csv"}
    assert "all_clips,overall" in (tmp_path / "count_error.csv").read_text()
