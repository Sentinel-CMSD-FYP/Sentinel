"""Tests for eval/threshold_sweep.py on the tiny fixture dataset from the loader tests."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from test_localization_data import TEST_DIR, TRAIN_DIR, write_frame

from sentinel_ml.localization import data as D

EVAL = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL))  # threshold_sweep does `import point_metrics`
SPEC = importlib.util.spec_from_file_location("threshold_sweep", EVAL / "threshold_sweep.py")
ts = importlib.util.module_from_spec(SPEC)
sys.modules["threshold_sweep"] = ts
SPEC.loader.exec_module(ts)


@pytest.fixture()
def val_ds(tmp_path: Path) -> D.DroneCrowdDataset:
    """Val clips 1 and 2, three frames each. GT per frame: (10*clip, 5*n) and (20, 30)."""
    train = tmp_path / TRAIN_DIR
    for clip in (1, 2, 3):
        for n in (1, 2, 3):
            write_frame(train, clip, n, np.array([[10.0 * clip, 5.0 * n, 1], [20.0, 30.0, 2]]))
    write_frame(tmp_path / TEST_DIR, 11, 1, np.array([[7.0, 9.0, 1]]))
    manifest = {"train": [3], "val": [1, 2], "test": [11]}
    return D.DroneCrowdDataset(tmp_path, manifest, "val", TRAIN_DIR, TEST_DIR)


def predictions(gt, extra_score=0.3):
    """Exact hits at score 0.9 plus one far-away false point per frame at ``extra_score``."""
    return {clip: {fid: np.vstack([np.hstack([pts, np.full((len(pts), 1), 0.9)]),
                                   [[500.0, 500.0, extra_score]]])
                   for fid, pts in frames.items()}
            for clip, frames in gt.items()}


def test_gt_by_clip_uses_zero_based_frame_ids(val_ds):
    gt = ts.gt_by_clip(val_ds)
    assert sorted(gt) == [1, 2] and sorted(gt[1]) == ["0", "1", "2"]
    np.testing.assert_allclose(gt[1]["0"], [[10.0, 5.0], [20.0, 30.0]])


def test_sweep_counts_match_hand_computation(val_ds):
    gt = ts.gt_by_clip(val_ds)
    rows = ts.sweep(predictions(gt), gt, 10.0, [0.1, 0.5, 0.95])
    n_frames, n_gt = 6, 12
    low, mid, high = rows
    assert (low["tp"], low["fp"], low["fn"]) == (n_gt, n_frames, 0)     # extra points kept
    assert (mid["tp"], mid["fp"], mid["fn"]) == (n_gt, 0, 0)            # 0.3 < 0.5 < 0.9
    assert (high["tp"], high["fp"], high["fn"]) == (0, 0, n_gt)         # everything dropped
    assert mid["f1"] == 1.0 and high["f1"] == 0.0
    assert low["precision"] == pytest.approx(12 / 18) and low["recall"] == 1.0


def test_sweep_reports_per_clip_and_totals_add_up(val_ds):
    gt = ts.gt_by_clip(val_ds)
    row = ts.sweep(predictions(gt), gt, 10.0, [0.1])[0]
    assert sorted(row["per_clip"]) == [1, 2]
    assert sum(c["tp"] for c in row["per_clip"].values()) == row["tp"]
    assert sum(c["fp"] for c in row["per_clip"].values()) == row["fp"]


def test_sweep_raises_when_a_clips_predictions_are_missing(val_ds):
    gt = ts.gt_by_clip(val_ds)
    pred = predictions(gt)
    del pred[2]
    with pytest.raises(FileNotFoundError, match="2"):
        ts.sweep(pred, gt, 10.0, [0.5])


def test_test_split_allows_one_threshold_only():
    ts.check_split_rule("test", [0.4])
    with pytest.raises(SystemExit, match="exactly one"):
        ts.check_split_rule("test", [0.3, 0.4])
    ts.check_split_rule("val", [0.1, 0.2, 0.3])


def test_outputs_are_written_and_readable(val_ds, tmp_path):
    gt = ts.gt_by_clip(val_ds)
    rows = ts.sweep(predictions(gt), gt, 10.0, [0.1, 0.5])
    meta = {"label": "fixture", "radius_px": 10.0}
    paths = ts.write_outputs(rows, meta, tmp_path / "out")
    assert {p.name for p in paths} >= {"sweep.json", "sweep.csv"}
    doc = json.loads((tmp_path / "out" / "sweep.json").read_text())
    assert doc["label"] == "fixture" and len(doc["rows"]) == 2
    with (tmp_path / "out" / "sweep.csv").open() as f:
        table = list(csv.reader(f))
    assert table[0] == ["threshold", "tp", "fp", "fn", "precision", "recall", "f1"]
    assert float(table[2][6]) == 1.0  # F1 at 0.5


def test_format_table_has_one_line_per_threshold(val_ds):
    gt = ts.gt_by_clip(val_ds)
    rows = ts.sweep(predictions(gt), gt, 10.0, [0.1, 0.5])
    assert len(ts.format_table(rows).splitlines()) == 3
