"""Tests for eval/compare_runs.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"
SPEC = importlib.util.spec_from_file_location("compare_runs", EVAL / "compare_runs.py")
cr = importlib.util.module_from_spec(SPEC)
sys.modules["compare_runs"] = cr
SPEC.loader.exec_module(cr)


def count_doc(clips, ids_all, err):
    return {"per_clip": [{"clip": c, "gt_people": 10, "ids_all": ids_all, "err_pct_all": err,
                          "ids_min_len": 9, "err_pct_min_len": err / 2} for c in clips]}


def te_doc(clips, idf1_a, idf1_b):
    def block(idf1):
        return {"per_clip": {f"{c:03d}": {"IDF1": idf1, "HOTA": 0.3, "MOTA": 0.2, "IDSW": 5}
                             for c in clips}}
    return {"iou_0.5": {"runA": block(idf1_a), "runB": block(idf1_b)}}


def test_rows_join_count_error_and_trackeval_metrics_per_clip():
    rows = cr.compare(count_doc([11, 15], 20, 100.0), count_doc([11, 15], 14, 40.0),
                      te_doc([11, 15], 0.4, 0.5), "runA", "runB")
    assert [r["clip"] for r in rows] == [11, 15]
    r = rows[0]
    assert (r["ids_a"], r["err_pct_a"], r["ids_b"], r["err_pct_b"]) == (20, 100.0, 14, 40.0)
    assert (r["idf1_a"], r["idf1_b"], r["idsw_a"]) == (0.4, 0.5, 5)
    assert r["err_pct_min45_a"] == 50.0


def test_different_clip_sets_are_refused():
    with pytest.raises(ValueError, match="different clips"):
        cr.compare(count_doc([11], 1, 1.0), count_doc([12], 1, 1.0), te_doc([11, 12], 0.1, 0.1),
                   "runA", "runB")


def test_means_and_table():
    rows = cr.compare(count_doc([11, 15], 20, 100.0), count_doc([11, 15], 14, 40.0),
                      te_doc([11, 15], 0.4, 0.5), "runA", "runB")
    m = cr.means(rows)
    assert m["err_pct_a"] == 100.0 and m["idf1_b"] == 0.5
    table = cr.format_table(rows, "A", "B")
    assert table.splitlines()[0].startswith("clip") and table.splitlines()[-1].startswith("mean")
    assert len(table.splitlines()) == 4
