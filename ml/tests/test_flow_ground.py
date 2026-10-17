import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel_ml.flow.export import ExportConfig, export_flow
from sentinel_ml.flow.ground_flow import get_flow
from sentinel_ml.geometry import gsd

ALT, HFOV = 60.0, 84.0


def texture(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (h, w)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 2.0).astype(np.uint8)


def shifted(img, dx):
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def interior(a, margin=0.25):
    h, w = a.shape
    return a[int(h * margin) : h - int(h * margin), int(w * margin) : w - int(w * margin)]


def test_get_flow_returns_metres_per_second_on_a_regular_grid():
    base = texture()
    # 3 px over 2 frames is 1.5 px/frame; at 25 fps and this GSD that is a known speed.
    g = get_flow(base, shifted(base, 3), altitude_m=ALT, hfov_deg=HFOV, fps=25.0,
                 frames_apart=2, rows=3, cols=4)
    expected = 1.5 * gsd(ALT, HFOV, 320) * 25
    assert g.vx.shape == g.vy.shape == g.speed.shape == g.speed_var.shape == (3, 4)
    assert g.gsd == pytest.approx(gsd(ALT, HFOV, 320))
    assert np.median(interior(g.vx)) == pytest.approx(expected, rel=0.15)
    assert np.median(interior(g.vy)) == pytest.approx(0.0, abs=0.1 * expected)
    assert np.median(interior(g.speed)) == pytest.approx(expected, rel=0.15)


def test_the_same_motion_from_twice_the_height_is_twice_as_fast_in_metres():
    base = texture()
    low = get_flow(base, shifted(base, 2), altitude_m=30, hfov_deg=HFOV, fps=25.0, rows=3, cols=4)
    high = get_flow(base, shifted(base, 2), altitude_m=60, hfov_deg=HFOV, fps=25.0, rows=3, cols=4)
    assert np.median(high.speed) == pytest.approx(2 * np.median(low.speed), rel=1e-6)


def test_speed_variance_is_in_metres_squared_per_second_squared():
    base = texture()
    a = get_flow(base, shifted(base, 2), altitude_m=30, hfov_deg=HFOV, fps=25.0, rows=3, cols=4)
    b = get_flow(base, shifted(base, 2), altitude_m=60, hfov_deg=HFOV, fps=25.0, rows=3, cols=4)
    np.testing.assert_allclose(b.speed_var, 4 * a.speed_var, rtol=1e-4, atol=1e-9)


def test_stabilise_removes_a_whole_frame_shift_but_the_default_keeps_it():
    base = texture()
    kept = get_flow(base, shifted(base, 3), altitude_m=ALT, hfov_deg=HFOV, fps=25.0, rows=3, cols=4)
    gone = get_flow(base, shifted(base, 3), altitude_m=ALT, hfov_deg=HFOV, fps=25.0, rows=3,
                    cols=4, stabilise=True)
    assert np.median(interior(kept.vx)) > 1.0
    assert abs(np.median(interior(gone.vx))) < 0.1 * np.median(interior(kept.vx))


@pytest.mark.parametrize("kwargs", [{"frames_apart": 0}, {"altitude_m": 0}, {"hfov_deg": 200}])
def test_get_flow_rejects_bad_inputs(kwargs):
    base = texture()
    args = {"altitude_m": ALT, "hfov_deg": HFOV, "fps": 25.0, **kwargs}
    with pytest.raises(ValueError):
        get_flow(base, base, **args)


def write_folder(path: Path, n=30, dx=2):
    path.mkdir()
    base = texture()
    for i in range(n):
        frame = cv2.cvtColor(shifted(base, dx * i), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(path / f"img{i:04d}.png"), frame)
    return path


def test_exporter_writes_metres_per_second_when_given_a_camera(tmp_path):
    src = write_folder(tmp_path / "f")
    out = tmp_path / "clip"
    cfg = ExportConfig(rows=3, cols=4, every=10, gap=5, min_texture=0.5, stabilise=False,
                       altitude_m=ALT, hfov_deg=HFOV)
    s = export_flow(src, out, fps=10.0, config=cfg)

    layer = json.loads((out / "layers" / "flow.json").read_text())
    clip = json.loads((out / "clip.json").read_text())
    g = gsd(ALT, HFOV, 320)
    assert layer["units"] == clip["layers"][0]["units"] == "m_per_s"
    assert s["units"] == "m_per_s" and s["gsd_m_per_px"] == pytest.approx(g)

    entry = layer["frames"]["10"]
    valid = np.array(entry["valid"], bool)
    # 2 px per frame at 10 fps, converted with this camera
    assert np.median(np.array(entry["u"])[valid]) == pytest.approx(2 * g * 10, rel=0.15)

    made = clip["produced_by"]
    assert made["altitude_m"] == ALT and made["hfov_deg"] == HFOV
    assert made["ground_sampling_m_per_px"] == pytest.approx(g)


def test_exporter_keeps_pixels_per_frame_without_a_camera(tmp_path):
    out = tmp_path / "clip"
    export_flow(write_folder(tmp_path / "f"), out, fps=10.0, config=ExportConfig(
        rows=3, cols=4, every=10, gap=5, min_texture=0.5, stabilise=False))
    layer = json.loads((out / "layers" / "flow.json").read_text())
    clip = json.loads((out / "clip.json").read_text())
    assert layer["units"] == "px_per_frame"
    assert clip["produced_by"]["ground_sampling_m_per_px"] is None


def test_gsd_uses_the_width_of_the_written_video_after_shrinking(tmp_path):
    out = tmp_path / "clip"
    cfg = ExportConfig(rows=2, cols=2, every=10, gap=5, min_texture=0.5, stabilise=False,
                       max_width=160, altitude_m=ALT, hfov_deg=HFOV)
    export_flow(write_folder(tmp_path / "f"), out, fps=10.0, config=cfg)
    clip = json.loads((out / "clip.json").read_text())
    width = clip["frame_size"][0]
    assert width == 160
    assert clip["produced_by"]["ground_sampling_m_per_px"] == pytest.approx(gsd(ALT, HFOV, width))


@pytest.mark.parametrize("kwargs", [{"altitude_m": 60.0}, {"hfov_deg": 84.0},
                                    {"altitude_m": 0.0, "hfov_deg": 84.0},
                                    {"altitude_m": 60.0, "hfov_deg": 180.0}])
def test_the_camera_must_be_complete_and_sensible(kwargs):
    with pytest.raises(ValueError):
        ExportConfig(**kwargs)
