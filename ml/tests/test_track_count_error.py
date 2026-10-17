"""Tests for eval/track_count_error.py with synthetic tracks and GT of known size."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EVAL = Path(__file__).resolve().parents[2] / "eval"
SPEC = importlib.util.spec_from_file_location("track_count_error", EVAL / "track_count_error.py")
tc = importlib.util.module_from_spec(SPEC)
sys.modules["track_count_error"] = tc
SPEC.loader.exec_module(tc)


def gt_frames(n_people, n_frames=10):
    """GT with ``n_people`` ids present in every frame."""
    pts = np.zeros((n_people, 2))
    ids = np.arange(1, n_people + 1)
    return {f: (pts, ids) for f in range(n_frames)}


def tracks_doc(spans):
    """Tracks document from ``{track_id: (first_frame, last_frame)}`` (inclusive)."""
    n = max(last for _, last in spans.values()) + 1
    frames = {str(f): [[tid, 1.0, 1.0, 0.9] for tid, (a, b) in spans.items() if a <= f <= b]
              for f in range(n)}
    return {"frames": frames}


def test_exact_count_is_zero_error():
    row = tc.clip_row(7, tracks_doc({1: (0, 9), 2: (0, 9), 3: (0, 9)}), gt_frames(3), 45)
    assert row["gt_people"] == 3 and row["ids_all"] == 3 and row["err_pct_all"] == 0.0


def test_over_counting_and_under_counting_are_both_positive_errors():
    over = tc.clip_row(1, tracks_doc({i: (0, 9) for i in range(1, 7)}), gt_frames(4), 45)
    under = tc.clip_row(1, tracks_doc({i: (0, 9) for i in range(1, 3)}), gt_frames(4), 45)
    assert over["err_pct_all"] == pytest.approx(50.0) and over["ratio_all"] == pytest.approx(1.5)
    assert under["err_pct_all"] == pytest.approx(50.0) and under["ratio_all"] == pytest.approx(0.5)


def test_min_len_filter_counts_only_long_tracks():
    spans = {1: (0, 49), 2: (0, 44), 3: (10, 60), 4: (5, 7)}  # lengths 50, 45, 51, 3
    row = tc.clip_row(1, tracks_doc(spans), gt_frames(2, n_frames=61), 45)
    assert row["ids_all"] == 4 and row["ids_min_len"] == 3  # 45 frames is kept, 3 is dropped
    row2 = tc.clip_row(1, tracks_doc(spans), gt_frames(2, n_frames=61), 46)
    assert row2["ids_min_len"] == 2


def test_gt_people_are_distinct_ids_not_the_per_frame_count():
    gt = {0: (np.zeros((2, 2)), np.array([1, 2])), 1: (np.zeros((2, 2)), np.array([2, 3]))}
    row = tc.clip_row(1, tracks_doc({1: (0, 1)}), gt, 45)
    assert row["gt_people"] == 3 and row["mean_gt_per_frame"] == 2.0


def test_aggregate_mean_median_max():
    rows = [{"err_pct_all": e, "err_pct_min_len": e / 2, "ratio_all": 1 + e / 100,
             "ratio_min_len": 1.0} for e in (10.0, 20.0, 60.0)]
    a = tc.aggregate(rows)
    assert a["clips"] == 3
    assert a["all"]["mean_err_pct"] == pytest.approx(30.0)
    assert a["all"]["median_err_pct"] == pytest.approx(20.0)
    assert a["all"]["max_err_pct"] == pytest.approx(60.0)
    assert a["min_len"]["mean_err_pct"] == pytest.approx(15.0)
    assert tc.aggregate([]) == {"clips": 0}


def test_summary_primary_is_all_clips_and_day_only_is_secondary():
    rows = [{"clip": c, "err_pct_all": e, "err_pct_min_len": e, "ratio_all": 1.0,
             "ratio_min_len": 1.0} for c, e in ((1, 10.0), (2, 90.0), (3, 20.0))]
    s = tc.summarise(rows, {1: "day", 2: "night", 3: "day"})
    assert s["all_clips"]["clips"] == 3  # no clip dropped
    assert s["day_only"]["clips"] == 2 and s["day_only"]["all"]["mean_err_pct"] == 15.0
    assert s["by_lighting"]["night"]["clips"] == 1 and s["by_lighting"]["dusk"] == {"clips": 0}
    assert "day_only" not in tc.summarise(rows, None)


def test_load_tags_needs_a_filled_in_file(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({"val": {1: None, 2: "day"}, "test": {11: "dusk"}}))
    assert tc.load_tags(path, "val") is None
    assert tc.load_tags(path, "test") == {11: "dusk"}
    assert tc.load_tags(tmp_path / "none.yaml", "val") is None
    path.write_text(yaml.safe_dump({"val": {1: "bright"}}))
    with pytest.raises(ValueError, match="bright"):
        tc.load_tags(path, "val")


def test_report_text_and_outputs(tmp_path):
    rows = [tc.clip_row(7, tracks_doc({1: (0, 9), 2: (0, 9), 3: (0, 9), 4: (2, 3)}),
                        gt_frames(3), 45)]
    summary = tc.summarise(rows, {7: "day"})
    text = tc.format_report(rows, summary, 25.0)
    assert "ALL CLIPS (primary)" in text and "not pass or fail" in text
    paths = tc.write_outputs(rows, summary, {"label": "x"}, tmp_path)
    assert {p.name for p in paths} == {"track_count_error.json", "track_count_error.csv"}
    assert "gt_people" in (tmp_path / "track_count_error.csv").read_text()


def test_test_split_is_refused_when_results_already_exist(tmp_path, monkeypatch):
    cfg = {"data_config": "unused", "lighting_tags": "unused", "tracks_dir": "unused",
           "min_len_frames": 45, "provisional_target_pct": 25, "out_dir": str(tmp_path / "out")}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    done = tmp_path / "out" / "test"
    done.mkdir(parents=True)
    (done / "track_count_error.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["track_count_error.py", "--config", str(cfg_path),
                                      "--split", "test",
                                      "--tracker-config", "unused.yaml"])
    with pytest.raises(SystemExit, match="once"):
        tc.main()


def test_label_comes_from_the_tracker_config_including_cmc_and_altitude_status():
    base = {"score_thr": 0.35, "min_hits": 3, "max_age": 25}
    label = tc.describe_tracker(base)
    assert "score_thr 0.35" in label and "min_hits 3" in label and "max_age 25" in label
    assert "no CMC" in label and "no altitude scaling" in label
    on = tc.describe_tracker({**base, "use_cmc": True, "use_altitude": True})
    assert "CMC on" in on and "altitude-scaled gate" in on and "no CMC" not in on
