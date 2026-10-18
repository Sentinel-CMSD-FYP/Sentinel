"""Tests for sentinel_ml.localization.data on a tiny synthetic DroneCrowd-shaped fixture."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import scipy.io as sio

from sentinel_ml.localization import data as D

W, H = 64, 48
TRAIN_DIR, TEST_DIR = "train_data-002/train_data", "test_data-001/test_data"


def write_frame(folder: Path, clip: int, n: int, points: np.ndarray) -> None:
    """Write one image + GT file named like the real dataset (n is the 1-based file number)."""
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "ground_truth").mkdir(parents=True, exist_ok=True)
    name = f"img{clip:03d}{n:03d}"
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :, 0] = 200  # blue channel in BGR -> red-ish check on RGB conversion below
    cv2.imwrite(str(folder / "images" / f"{name}.jpg"), img)
    rec = np.zeros((1, 1), dtype=[("location", "O"), ("number", "O")])
    rec[0, 0]["location"] = points
    rec[0, 0]["number"] = np.array([[len(points)]], dtype=np.uint8)
    info = np.empty((1, 1), dtype=object)
    info[0, 0] = rec
    sio.savemat(str(folder / "ground_truth" / f"GT_{name}.mat"), {"image_info": info})


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    train = tmp_path / TRAIN_DIR
    test = tmp_path / TEST_DIR
    for clip in (1, 2, 3, 4):
        for n in (1, 2, 3):
            pts = np.array([[10.0 * clip, 5.0 * n, 1], [20.0, 30.0, 2]])
            write_frame(train, clip, n, pts)
    for clip in (11, 12):
        for n in (1, 2):
            write_frame(test, clip, n, np.array([[7.0, 9.0, 1]]))
    (tmp_path / "trainlist.txt").write_text("00001\r\n00002\r\n00003\r\n00004")
    (tmp_path / "testlist.txt").write_text("00011\n00012\n")
    return tmp_path


@pytest.fixture()
def manifest(root: Path) -> dict:
    return D.build_manifest(D.read_clip_list(root / "trainlist.txt"),
                            D.read_clip_list(root / "testlist.txt"), n_val=1, seed=0)


def make(root, manifest, split, **kw):
    return D.DroneCrowdDataset(root, manifest, split, TRAIN_DIR, TEST_DIR, **kw)


def test_read_clip_list_handles_crlf_and_missing_final_newline(root):
    assert D.read_clip_list(root / "trainlist.txt") == [1, 2, 3, 4]


def test_manifest_splits_are_disjoint_and_cover_official_clips(manifest):
    assert sorted(manifest["train"] + manifest["val"]) == [1, 2, 3, 4]
    assert manifest["test"] == [11, 12]
    D.check_manifest(manifest)


def test_val_choice_is_reproducible_and_seed_dependent():
    clips = list(range(1, 83))
    a = D.choose_val_clips(clips, 10, seed=0)
    assert a == D.choose_val_clips(clips, 10, seed=0)
    assert len(a) == 10 and set(a) <= set(clips)
    assert a != D.choose_val_clips(clips, 10, seed=1)


def test_overlapping_manifest_is_rejected():
    bad = {"train": [1, 2], "val": [2], "test": [11]}
    with pytest.raises(ValueError, match="overlap"):
        D.check_manifest(bad)


def test_bad_n_val_is_rejected():
    with pytest.raises(ValueError):
        D.choose_val_clips([1, 2, 3], 0, 0)
    with pytest.raises(ValueError):
        D.choose_val_clips([1, 2, 3], 3, 0)


def test_val_comes_from_train_folder_and_test_from_test_folder(root, manifest):
    val = make(root, manifest, "val")
    assert {r.clip for r in val.refs} == set(manifest["val"])
    assert all("train_data" in str(r.image_path) for r in val.refs)
    test = make(root, manifest, "test")
    assert {r.clip for r in test.refs} == {11, 12}
    assert all("test_data" in str(r.image_path) for r in test.refs)


def test_train_and_val_share_no_clips(root, manifest):
    tr = {r.clip for r in make(root, manifest, "train").refs}
    va = {r.clip for r in make(root, manifest, "val").refs}
    assert tr and va and not tr & va


def test_sample_fields_and_zero_based_frame_id(root, manifest):
    ds = make(root, manifest, "test")
    s = ds[0]
    assert (s.ref.clip_id, s.ref.frame_id) == ("clip_011", 0)  # file img011001 -> frame 0
    assert ds[1].ref.frame_id == 1
    assert s.image.shape == (H, W, 3) and s.image.dtype == np.uint8
    assert s.points.shape == (1, 2) and s.points.dtype == np.float32
    assert s.person_ids.tolist() == [1]
    np.testing.assert_allclose(s.points[0], [7.0, 9.0])  # x = column, y = row, unchanged
    assert s.scale == (1.0, 1.0)


def test_image_is_rgb(root, manifest):
    s = make(root, manifest, "test")[0]
    # file was written as BGR (200, 0, 0), so RGB must have the 200 in the last channel
    assert s.image[0, 0, 2] > 150 and s.image[0, 0, 0] < 50


def test_resize_keeps_points_in_original_pixels(root, manifest):
    s = make(root, manifest, "test", image_size=(W * 2, H * 3))[0]
    assert s.image.shape == (H * 3, W * 2, 3)
    assert s.scale == (2.0, 3.0)
    np.testing.assert_allclose(s.points[0], [7.0, 9.0])  # NOT rescaled
    np.testing.assert_allclose(D.to_original_pixels(s.points * np.array(s.scale), s.scale),
                               s.points)


def test_frame_stride_keeps_every_nth_frame(root, manifest):
    ds = make(root, manifest, "train", frame_stride=2)
    ids = sorted({r.frame_id for r in ds.refs})
    assert ids == [0, 2]  # of frames 0, 1, 2


def test_gt_ids_and_order_are_read(root, manifest):
    ds = make(root, manifest, "train")
    s = next(x for x in (ds[i] for i in range(len(ds))) if x.ref.clip == 1 and x.ref.frame_id == 0)
    assert s.person_ids.tolist() == [1, 2]
    np.testing.assert_allclose(s.points, [[10.0, 5.0], [20.0, 30.0]])


def test_empty_split_raises(root, manifest):
    bad = dict(manifest, val=[99], train=[1, 2, 3, 4], test=[11, 12])
    with pytest.raises(FileNotFoundError):
        make(root, bad, "val")


def test_unknown_split_and_bad_stride(root, manifest):
    with pytest.raises(ValueError):
        make(root, manifest, "dev")
    with pytest.raises(ValueError):
        make(root, manifest, "train", frame_stride=0)


def test_draw_points_returns_bgr_image_with_dots(root, manifest):
    s = make(root, manifest, "test")[0]
    out = D.draw_points(s)
    assert out.shape == s.image.shape
    assert (out[9, 7] == [0, 0, 255]).all()  # red dot (BGR) at x=7, y=9


REAL = Path(__file__).resolve().parents[2] / "data" / "dronecrowd"


@pytest.mark.skipif(not REAL.exists(), reason="DroneCrowd not on this machine")
def test_real_clip_011_first_frame_has_211_points():
    pts, ids = D.load_gt(REAL / TEST_DIR / "ground_truth" / "GT_img011001.mat")
    assert pts.shape == (211, 2) and ids.shape == (211,)
