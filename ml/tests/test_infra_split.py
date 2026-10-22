"""Tests for eval/infrastructure_yolo26/split_scenes.py on small fake label sets."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "infrastructure_yolo26" / "split_scenes.py"
SPEC = importlib.util.spec_from_file_location("infra_split_scenes", SCRIPT)
split = importlib.util.module_from_spec(SPEC)
sys.modules["infra_split_scenes"] = split
SPEC.loader.exec_module(split)


def write_labels(folder: Path, files: dict[str, list[int]]) -> None:
    for stem, ids in files.items():
        (folder / f"{stem}.txt").write_text("".join(f"{i} 0 0 1 0 1 1\n" for i in ids))


def test_counts_and_group_stats(tmp_path):
    write_labels(tmp_path, {"img001150": [0, 2, 2], "img002150": [], "img003001": [3]})
    counts = split.load_counts(tmp_path, 4)
    assert counts["img002150"].tolist() == [0, 0, 0, 0]
    stats = split.group_stats([[1, 2], [3]], counts)
    assert stats.tolist() == [[2, 1, 0, 2, 0], [1, 0, 0, 0, 1]]


def test_group_stats_rejects_bad_groups(tmp_path):
    write_labels(tmp_path, {"img001150": [0], "img005150": [1]})
    counts = split.load_counts(tmp_path, 2)
    with pytest.raises(ValueError, match="more than one"):
        split.group_stats([[1], [1, 5]], counts)
    with pytest.raises(ValueError, match="no group"):
        split.group_stats([[1]], counts)


def test_pick_val_keeps_groups_whole_and_meets_constraints():
    groups = [[1, 2], [3], [4], [5, 6], [7], [8], [9], [10]]
    # images, class 0 (rare), class 1
    stats = np.array([[2, 0, 4], [1, 0, 2], [1, 1, 2], [2, 0, 4],
                      [1, 0, 2], [1, 1, 2], [1, 0, 2], [1, 0, 2]])
    val = split.pick_val(groups, stats, night=[7], frac=0.2, seed=0, tries=500)
    assert 4 in val  # the only night group
    assert stats[val, 1].sum() >= 1  # rare class present
    assert val == split.pick_val(groups, stats, night=[7], frac=0.2, seed=0, tries=500)


def test_pick_val_raises_when_impossible():
    stats = np.array([[1, 1], [1, 0]])
    with pytest.raises(ValueError):
        split.pick_val([[1], [2]], stats, night=[2], frac=0.2, seed=0, tries=50)
