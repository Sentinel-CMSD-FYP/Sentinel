"""Tests for eval/render_demo.py (clip choice) and render_trails.render_video (10 s video)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL))
SPEC = importlib.util.spec_from_file_location("render_demo", EVAL / "render_demo.py")
rd = importlib.util.module_from_spec(SPEC)
sys.modules["render_demo"] = rd
SPEC.loader.exec_module(rd)

import render_trails as rt  # noqa: E402

IDF1 = {c: c / 100 for c in range(1, 31)}  # clip 30 is best, clip 1 is worst


def test_pick_takes_the_highest_and_lowest_idf1_and_random_from_the_rest():
    g = rd.pick_clips(IDF1, best=4, worst=4, n_random=4, seed=0)
    assert [c for c, k in g.items() if k == "best"] == [27, 28, 29, 30]
    assert [c for c, k in g.items() if k == "worst"] == [1, 2, 3, 4]
    rand = [c for c, k in g.items() if k == "random"]
    assert len(rand) == 4 and all(5 <= c <= 26 for c in rand)
    assert len(g) == 12


def test_pick_is_reproducible_and_changes_with_the_seed():
    a = rd.pick_clips(IDF1, 4, 4, 4, seed=0)
    assert a == rd.pick_clips(IDF1, 4, 4, 4, seed=0)
    assert a != rd.pick_clips(IDF1, 4, 4, 4, seed=1)


def test_pick_breaks_ties_by_clip_number_and_rejects_too_many():
    tied = {c: 0.5 for c in range(1, 11)}
    g = rd.pick_clips(tied, best=2, worst=0, n_random=0, seed=0)
    assert list(g) == [1, 2]
    with pytest.raises(ValueError):
        rd.pick_clips(tied, 5, 5, 5, seed=0)


def test_render_video_writes_ten_seconds_at_the_clip_fps(tmp_path):
    refs = {}
    for i in range(30):
        p = tmp_path / f"f{i}.jpg"
        cv2.imwrite(str(p), np.zeros((48, 64, 3), np.uint8))
        refs[i] = SimpleNamespace(image_path=p)
    frames = {str(i): [[1, 10.0 + i, 20.0, 0.9]] for i in range(30)}
    doc = {"clip_id": "clip_001", "fps": 2, "frame_stride": 1, "frame_size": [64, 48],
           "frames": frames}
    out = tmp_path / "v" / "clip_001.mp4"
    n = rt.render_video(doc, refs, start=0, seconds=10.0, trail=5, width=64, out_path=out)
    assert n == 20 and out.exists() and out.stat().st_size > 0   # 10 s * 2 fps
    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 20
    cap.release()
