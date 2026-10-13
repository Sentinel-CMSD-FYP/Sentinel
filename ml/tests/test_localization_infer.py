"""Tests for sentinel_ml.localization.infer: numpy helpers, contract output, restricted load."""

import collections
import json
import pathlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel_ml.localization import data as D
from sentinel_ml.localization import infer as I  # noqa: E741

W, H = 64, 48
CFG = {"fps": 25, "frame_stride": 1, "altitude_m": 60.0, "altitude_source": "assumed_constant"}


def test_preprocess_shape_dtype_and_normalisation():
    img = np.full((H, W, 3), 255, np.uint8)
    x = I.preprocess(img, (32, 24), [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    assert x.shape == (3, 24, 32) and x.dtype == np.float32
    np.testing.assert_allclose(x, 2.0, atol=1e-5)  # (1.0 - 0.5) / 0.25


def test_preprocess_uses_given_channel_stats_per_channel():
    img = np.zeros((H, W, 3), np.uint8)
    x = I.preprocess(img, (W, H), [0.1, 0.2, 0.3], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(x[:, 0, 0], [-0.1, -0.2, -0.3], atol=1e-6)


def test_net_to_original_stretches_y_back_by_1080_over_1088():
    out = I.net_to_original(np.array([[960.0, 544.0], [0.0, 1088.0]]), (1920, 1080), (1920, 1088))
    np.testing.assert_allclose(out, [[960.0, 540.0], [0.0, 1080.0]])


def test_net_to_original_empty():
    assert I.net_to_original(np.zeros((0, 2)), (1920, 1080), (1920, 1088)).shape == (0, 2)


def test_sigmoid_known_values():
    np.testing.assert_allclose(I.sigmoid(np.array([0.0, 100.0, -100.0])), [0.5, 1.0, 0.0],
                               atol=1e-9)


def test_build_doc_matches_contract_header_and_rounds():
    doc = I.build_doc("clip_011", (1920, 1080),
                      {0: np.array([[10.123, 20.987, 0.123456]]), 2: np.zeros((0, 3))}, CFG)
    assert doc["schema_version"] == "1.0" and doc["clip_id"] == "clip_011"
    assert doc["frame_size"] == [1920, 1080] and doc["fps"] == 25
    assert doc["frame_stride"] == 1 and doc["altitude_source"] == "assumed_constant"
    assert doc["frames"] == {"0": [[10.1, 21.0, 0.1235]], "2": []}
    I.check_doc(doc)


def test_check_doc_rejects_out_of_range_rows():
    base = I.build_doc("c", (100, 50), {0: np.array([[10.0, 10.0, 0.5]])}, CFG)
    I.check_doc(base)
    for bad in ([[101.0, 10.0, 0.5]], [[10.0, 51.0, 0.5]], [[10.0, 10.0, 1.5]],
                [[-1.0, 10.0, 0.5]]):
        doc = dict(base, frames={"0": bad})
        with pytest.raises(ValueError):
            I.check_doc(doc)


def test_write_doc_is_valid_json_and_round_trips(tmp_path):
    frames = {0: np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 0.25]]), 1: np.zeros((0, 3))}
    doc = I.build_doc("clip_001", (W, H), frames, CFG)
    path = tmp_path / "sub" / "clip_001_points.json"
    I.write_doc(doc, path)
    assert json.loads(path.read_text()) == doc


def make_dataset(tmp_path: Path):
    folder = tmp_path / "train_data-002" / "train_data"
    (folder / "images").mkdir(parents=True)
    for clip in (1, 2):
        for n in (1, 2, 3):
            cv2.imwrite(str(folder / "images" / f"img{clip:03d}{n:03d}.jpg"),
                        np.zeros((H, W, 3), np.uint8))
    (tmp_path / "test_data-001" / "test_data" / "images").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "test_data-001/test_data/images/img011001.jpg"),
                np.zeros((H, W, 3), np.uint8))
    manifest = {"train": [1], "val": [2], "test": [11]}
    return D.DroneCrowdDataset(tmp_path, manifest, "val", "train_data-002/train_data",
                               "test_data-001/test_data")


def fake_detector(image_rgb):
    """Two points per frame, in original pixels, with scores in [0, 1]."""
    return np.array([[5.0, 6.0, 0.9], [W - 1.0, H - 1.0, 0.1]])


def test_run_split_writes_one_contract_file_per_clip(tmp_path):
    ds = make_dataset(tmp_path)
    paths = I.run_split(fake_detector, ds, tmp_path / "out", CFG, log=lambda *_: None)
    assert [p.name for p in paths] == ["clip_002_points.json"]
    doc = json.loads(paths[0].read_text())
    assert doc["clip_id"] == "clip_002" and doc["frame_size"] == [W, H]
    assert list(doc["frames"]) == ["0", "1", "2"]  # 0-based frame ids
    assert all(len(rows) == 2 for rows in doc["frames"].values())
    I.check_doc(doc)


def test_run_split_skips_existing_clips_and_respects_max_frames(tmp_path):
    ds = make_dataset(tmp_path)
    out = tmp_path / "out"
    first = I.run_split(fake_detector, ds, out, CFG, max_frames=2, log=lambda *_: None)
    assert list(json.loads(first[0].read_text())["frames"]) == ["0", "1"]
    again = I.run_split(fake_detector, ds, out, CFG, log=lambda *_: None)
    assert again == []  # resume: nothing rewritten


def test_scores_are_kept_not_thresholded(tmp_path):
    ds = make_dataset(tmp_path)
    paths = I.run_split(fake_detector, ds, tmp_path / "o", CFG, log=lambda *_: None)
    scores = [r[2] for rows in json.loads(paths[0].read_text())["frames"].values() for r in rows]
    assert sorted(set(scores)) == [0.1, 0.9]


# ---------- restricted checkpoint loading (needs torch, which CI installs) ----------

torch = pytest.importorskip("torch")


def test_restricted_load_accepts_allowed_types(tmp_path):
    obj = {"hp": collections.defaultdict(list, a=[1]), "w": torch.zeros(2), "n": 3}
    path = tmp_path / "ok.ckpt"
    torch.save(obj, path)
    with I.restricted_torch_load():
        loaded = torch.load(path, map_location="cpu")
    assert loaded["n"] == 3 and isinstance(loaded["hp"], collections.defaultdict)
    assert loaded["w"].shape == (2,)


def test_restricted_load_blocks_unlisted_types(tmp_path):
    path = tmp_path / "bad.ckpt"
    torch.save({"p": pathlib.PurePosixPath("/tmp/x")}, path)
    with I.restricted_torch_load(), pytest.raises(Exception, match="blocked"):
        torch.load(path, map_location="cpu")


def test_restricted_load_restores_torch_load():
    original = torch.load
    with I.restricted_torch_load():
        assert torch.load is not original
    assert torch.load is original
