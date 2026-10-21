"""Tests for eval/compute_camera_shifts.py (path building and the missing-images message)."""

import importlib.util
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"
SPEC = importlib.util.spec_from_file_location("compute_camera_shifts",
                                              EVAL / "compute_camera_shifts.py")
cs = importlib.util.module_from_spec(SPEC)
sys.modules["compute_camera_shifts"] = cs
SPEC.loader.exec_module(cs)

TRAIN, TEST = "train_data-002/train_data", "test_data-001/test_data"


def make_cfg(root):
    return {"root": str(root), "train_dir": TRAIN, "test_dir": TEST}


def test_paths_use_one_based_file_numbers_and_the_right_folder(tmp_path):
    (tmp_path / TRAIN / "images").mkdir(parents=True)
    (tmp_path / TEST / "images").mkdir(parents=True)
    for folder, clip in ((TRAIN, 6), (TEST, 11)):
        for n in (1, 2):
            (tmp_path / folder / "images" / f"img{clip:03d}{n:03d}.jpg").write_bytes(b"x")
    val = cs.image_paths_for_clip(make_cfg(tmp_path), "val", 6, [0, 1])
    assert val[0].name == "img006001.jpg" and val[1].name == "img006002.jpg"
    assert "train_data" in str(val[0])
    test = cs.image_paths_for_clip(make_cfg(tmp_path), "test", 11, [0, 1])
    assert "test_data" in str(test[0]) and test[1].name == "img011002.jpg"


def test_missing_images_give_a_usb_hint(tmp_path):
    with pytest.raises(SystemExit, match="USB"):
        cs.image_paths_for_clip(make_cfg(tmp_path), "val", 6, [0, 1])
