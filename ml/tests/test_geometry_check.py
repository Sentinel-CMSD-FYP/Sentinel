"""eval/pipeline/geometry_check.py on fake clips with known answers."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel_ml.geometry import NadirCamera

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "pipeline" / "geometry_check.py"
SPEC = importlib.util.spec_from_file_location("geometry_check", SCRIPT)
gc = importlib.util.module_from_spec(SPEC)
sys.modules["geometry_check"] = gc
SPEC.loader.exec_module(gc)


def test_segment_metres_and_implied_altitude():
    gsd = NadirCamera(60, 70, 1920).gsd
    px = 4.5 / gsd / 1.5  # the car looks 1.5x too short, so the drone was 1.5x higher
    (row,) = gc.segment_table([{"label": "car", "p1": [0, 0], "p2": [px, 0]}], gsd, 60, 4.5)
    assert row["m"] == pytest.approx(3.0, abs=0.01)
    assert row["implied_altitude_m"] == pytest.approx(90, abs=0.1)
    (own,) = gc.segment_table([{"label": "pitch", "p1": [0, 0], "p2": [0, px], "known_m": 3}],
                              gsd, 60, 4.5)
    assert own["ratio"] == pytest.approx(1.0, abs=0.01)


def test_track_speeds_over_one_second_windows():
    fps, gsd = 25.0, 0.05
    # track 1 walks 1 px per frame (1.25 m/s), track 2 stands, track 3 is shorter than 1 s
    frames = {str(f): [[1, 100 + f, 50, 0.9], [2, 300, 300, 0.9]] for f in range(60)}
    for f in range(10):
        frames[str(f)].append([3, 0, f, 0.9])
    v = gc.track_speeds(frames, fps, gsd)
    assert sorted(v) == pytest.approx([0.0, 1.25])


def test_flow_speeds_keep_valid_cells_and_convert_px():
    doc = {"frames": {"0": {"speed": [[1.0, 2.0]], "valid": [[True, False]]},
                      "25": {"speed": [[3.0, 4.0]], "valid": [[True, True]]}}}
    assert gc.flow_speeds(doc, 2.0).tolist() == [2.0, 6.0, 8.0]


def test_check_end_to_end_on_a_fake_clip(tmp_path):
    w, h, fps = 640, 360, 25.0
    cam = {"altitude_m": 50.0, "hfov_deg": 60.0, "altitude_source": "assumed"}
    gsd = NadirCamera(50, 60, w).gsd
    clip = tmp_path / "c"
    (clip / "layers").mkdir(parents=True)
    (clip / "clip.json").write_text(json.dumps({
        "clip_id": "c", "fps": fps, "frame_size": [w, h], "n_frames": 30, "camera": cam,
        "layers": []}))
    step = 1.2 / (gsd * fps)  # px per frame for 1.2 m/s
    (clip / "layers" / "tracks.json").write_text(json.dumps({"frames": {
        str(f): [[7, 10 + step * f, 100, 0.9]] for f in range(30)}}))
    (clip / "layers" / "flow.json").write_text(json.dumps({"units": "m_per_s", "frames": {
        "0": {"speed": [[1.2]], "valid": [[True]]}}}))
    out = cv2.VideoWriter(str(clip / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(3):
        out.write(np.zeros((h, w, 3), np.uint8))
    out.release()

    lines = {"frame": 2, "known_m": 4.5,
             "segments": [{"label": "car", "p1": [10, 10], "p2": [10 + 4.5 / gsd, 10]}]}
    r, rows = gc.check(clip, lines)
    assert r["median_implied_altitude_m"] == pytest.approx(50, abs=0.2)
    assert r["tracks"]["moving"]["median"] == pytest.approx(1.2, abs=0.01)
    assert r["flow"]["cells"]["median"] == pytest.approx(1.2)
    assert "| car |" in gc.markdown(r)
    assert gc.draw(gc.read_frame(clip / "video.mp4", 2), rows, lines["segments"]).shape == (h, w, 3)
    assert math.isclose(r["gsd_m_per_px"], gsd, abs_tol=1e-5)


def test_a_clip_without_camera_is_refused(tmp_path):
    (tmp_path / "clip.json").write_text(json.dumps({"clip_id": "c", "frame_size": [10, 10]}))
    with pytest.raises(SystemExit, match="no clip.json camera block"):
        gc.check(tmp_path, {"segments": []})
