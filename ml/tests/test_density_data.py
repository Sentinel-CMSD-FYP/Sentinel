"""Tests for sentinel_ml.density.data on tiny fake DroneCrowd layouts (local and Kaggle nesting)."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_localization_data import write_frame

from sentinel_ml.density import data as DD

MANIFEST = {"train": [1, 2], "val": [3], "test": [11]}


def make_root(tmp_path: Path, train_dir: str, test_dir: str) -> Path:
    for clip in (1, 2, 3):
        for n in range(1, 11):
            write_frame(tmp_path / train_dir, clip, n, np.array([[float(n), 2.0, 1]]))
    for n in range(1, 4):
        write_frame(tmp_path / test_dir, 11, n, np.array([[7.0, 9.0, 1], [8.0, 9.0, 2]]))
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    return tmp_path


@pytest.mark.parametrize("train_dir,test_dir", [
    ("train_data-002/train_data", "test_data-001/test_data"),  # local copy
    ("train_data/train_data", "test_data/test_data"),          # Kaggle copy
])
def test_find_split_dirs_handles_both_layouts(tmp_path, train_dir, test_dir):
    root = make_root(tmp_path, train_dir, test_dir)
    assert DD.find_split_dirs(root) == {"train": train_dir, "test": test_dir}


def test_find_split_dirs_ignores_val_data_and_raises_when_missing(tmp_path):
    write_frame(tmp_path / "val_data/val_data", 11, 1, np.array([[1.0, 1.0, 1]]))
    with pytest.raises(FileNotFoundError):
        DD.find_split_dirs(tmp_path)


def test_load_split_uses_manifest_and_stride(tmp_path):
    root = make_root(tmp_path, "train_data/train_data", "test_data/test_data")
    train = DD.load_split("train", root, root / "manifest.json", frame_stride=5)
    assert [(r.clip, r.frame_id) for r in train.refs] == [(1, 0), (1, 5), (2, 0), (2, 5)]
    val = DD.load_split("val", root, root / "manifest.json")
    assert {r.clip for r in val.refs} == {3} and len(val) == 10
    s = DD.load_split("test", root, root / "manifest.json")[0]
    assert s.points.shape == (2, 2) and s.image.shape[2] == 3


def test_load_split_from_config_reads_strides(tmp_path):
    root = make_root(tmp_path, "train_data/train_data", "test_data/test_data")
    cfg = {"root": "ignored", "manifest": "manifest.json", "frame_stride": {"train": 2}}
    assert len(DD.load_split_from_config(cfg, "train", root=root, repo=root)) == 10
    assert len(DD.load_split_from_config(cfg, "test", root=root, repo=root)) == 3


def test_load_split_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError):
        DD.load_split("val_data", tmp_path, tmp_path / "manifest.json")
