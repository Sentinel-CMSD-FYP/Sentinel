"""Tests for eval/density_dmcount/count_error.py with small synthetic cases and known answers."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "density_dmcount" / "count_error.py"
SPEC = importlib.util.spec_from_file_location("density_count_error", SCRIPT)
ce = importlib.util.module_from_spec(SPEC)
sys.modules["density_count_error"] = ce
SPEC.loader.exec_module(ce)

CUT = [100, 200]


def row(clip, gt, pred, fid=0):
    return {"clip": clip, "frame_id": fid, "gt": gt, "pred": pred}


def write_csv(path, text):
    path.write_text(text)
    return path


def test_bin_edges():
    assert [ce.bin_of(n, CUT) for n in (1, 99, 100, 200, 201, 455)] == [
        "sparse", "sparse", "moderate", "moderate", "dense", "dense"]


def test_stats_known_values():
    s = ce.stats([row(1, 100, 90.0), row(2, 100, 120.0)], min_gt=10)  # -10 and +20
    assert s["frames"] == 2 and s["clips"] == 2 and s["mean_gt"] == 100
    assert s["mae"] == pytest.approx(15.0)
    assert s["rmse"] == pytest.approx((250.0) ** 0.5)  # sqrt((100 + 400) / 2)
    assert s["mean_abs_error_pct"] == pytest.approx(15.0)
    assert s["median_abs_error_pct"] == pytest.approx(15.0)
    assert s["mean_bias_pct"] == pytest.approx(5.0)


def test_undercounting_gives_negative_bias():
    assert ce.stats([row(1, 200, 150.5)], 10)["mean_bias_pct"] == pytest.approx(-24.75)


def test_frames_below_min_gt_stay_in_mae_but_not_in_pct():
    s = ce.stats([row(1, 5, 9.0), row(1, 100, 100.0)], min_gt=10)
    assert s["frames"] == 2 and s["pct_frames"] == 1
    assert s["mae"] == pytest.approx(2.0)                  # (4 + 0) / 2
    assert s["mean_abs_error_pct"] == 0.0                  # only the GT 100 frame


def test_zero_gt_never_enters_pct_even_with_min_gt_zero():
    s = ce.stats([row(1, 0, 3.0)], min_gt=0)
    assert s["pct_frames"] == 0 and "mean_abs_error_pct" not in s and s["mae"] == 3.0
    assert ce.stats([], 10) == {"frames": 0}


def test_report_splits_frames_into_bins_by_gt_count():
    rows = [row(1, 50, 50), row(1, 150, 150), row(2, 250, 250), row(2, 260, 200)]
    rep = ce.report(rows, CUT, 10, None)
    assert [rep["by_bin"][b]["frames"] for b in ce.BINS] == [1, 1, 2]
    assert rep["by_bin"]["dense"]["mae"] == pytest.approx(30.0)
    assert "by_lighting" not in rep


def test_full_report_has_all_clips_primary_and_day_only_secondary():
    rows = [row(1, 100, 100), row(2, 100, 50), row(3, 100, 100)]
    tags = {1: "day", 2: "night", 3: "dusk"}
    res = ce.full_report(rows, CUT, 10, tags)
    assert res["all_clips"]["overall"]["clips"] == 3     # nothing is dropped
    assert res["day_only"]["overall"]["clips"] == 1
    assert res["all_clips"]["by_lighting"]["night"]["mean_bias_pct"] == pytest.approx(-50.0)
    assert res["all_clips"]["clips_per_lighting"] == {"day": [1], "dusk": [3], "night": [2]}
    assert res["day_only"]["by_lighting"]["night"] == {"frames": 0}
    assert "day_only" not in ce.full_report(rows, CUT, 10, None)


def test_full_report_refuses_untagged_clip():
    with pytest.raises(ValueError, match="no lighting tag"):
        ce.full_report([row(1, 100, 100), row(9, 100, 100)], CUT, 10, {1: "day"})


def test_load_preds_and_join(tmp_path):
    p = write_csv(tmp_path / "p.csv", "clip,frame_id,pred\n11,0,95.5\n11,1,101\n")
    preds = ce.load_preds(p)
    assert preds == {(11, 0): 95.5, (11, 1): 101.0}
    rows = ce.join([{"clip": 11, "frame_id": 0, "gt": 100}, {"clip": 11, "frame_id": 1, "gt": 99}],
                   preds)
    assert [r["pred"] for r in rows] == [95.5, 101.0]


def test_join_refuses_missing_or_extra_frames():
    gt = [{"clip": 11, "frame_id": 0, "gt": 100}]
    with pytest.raises(ValueError, match="1 GT frames without prediction"):
        ce.join(gt, {})
    with pytest.raises(ValueError, match="1 predictions without GT frame"):
        ce.join(gt, {(11, 0): 1.0, (11, 5): 1.0})


@pytest.mark.parametrize("text, match", [
    ("clip,frame,pred\n1,0,3\n", "needs columns"),
    ("clip,frame_id,pred\n1,0,3\n1,0,4\n", "duplicate"),
    ("clip,frame_id,pred\n1,0,nan\n", "prediction nan"),
])
def test_load_preds_rejects_bad_files(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        ce.load_preds(write_csv(tmp_path / "p.csv", text))


def test_load_tags_ignores_unfilled_template_and_checks_values(tmp_path):
    path = tmp_path / "tags.yaml"
    path.write_text(yaml.safe_dump({"val": {1: None, 2: "day"}, "test": {11: "night"}}))
    assert ce.load_tags(path, "val") is None
    assert ce.load_tags(path, "test") == {11: "night"}
    assert ce.load_tags(tmp_path / "missing.yaml", "val") is None
    path.write_text(yaml.safe_dump({"val": {1: "sunny"}}))
    with pytest.raises(ValueError, match="sunny"):
        ce.load_tags(path, "val")


def test_markdown_table_shows_bins_and_provisional_targets():
    res = ce.full_report([row(1, 50, 40), row(1, 250, 250)], CUT, 10, {1: "day"})
    md = ce.markdown(res, CUT, {"sparse": 15, "dense": 25})
    assert "**all clips**" in md and "**day only**" in md
    assert "| sparse (GT < 100) | 1 | 1 | 50.0 | 10.00 | 10.00 | 20.0% | 20.0% | -20.0% | <= 15% |" in md
    assert "moderate" not in md                          # empty bins are left out


def test_outputs_are_written(tmp_path):
    rows = [row(1, 100, 90), row(1, 250, 250, fid=1)]
    res = ce.full_report(rows, CUT, 10, {1: "day"})
    meta = {"label": "x", "bin_cutoffs": CUT}
    paths = ce.write_outputs(res, rows, meta, ce.markdown(res, CUT, {}), tmp_path)
    assert {p.name for p in paths} == {"count_error.json", "count_error.csv", "count_error.md",
                                       "frames.csv"}
    assert "all_clips,overall,2,1,175.0,5.0" in (tmp_path / "count_error.csv").read_text()
    assert "1,1,250,250,dense" in (tmp_path / "frames.csv").read_text()


def test_test_split_is_refused_when_results_already_exist(tmp_path, monkeypatch):
    cfg = {"count_error": {"out_dir": str(tmp_path / "out")}}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    done = tmp_path / "out" / "run1" / "test"
    done.mkdir(parents=True)
    (done / "count_error.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["count_error.py", "--config", str(cfg_path), "--split",
                                      "test", "--preds", "unused.csv", "--name", "run1"])
    with pytest.raises(SystemExit, match="once"):
        ce.main()
