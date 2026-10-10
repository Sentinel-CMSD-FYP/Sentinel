from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel_ml.flow import FarnebackConfig, FarnebackFlow, FlowField, pool_regions

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "flow_farneback.yaml"


def textured_frame(h=256, w=256, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (h, w)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 2.0).astype(np.uint8)


def shifted(frame, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def interior(a, margin=0.2):
    h, w = a.shape
    my, mx = int(h * margin), int(w * margin)
    return a[my:h - my, mx:w - mx]


@pytest.mark.parametrize("downscale, dx, dy", [(1.0, 3, -2), (0.5, 6, 4)])
def test_recovers_translation_in_original_pixels(downscale, dx, dy):
    frame = textured_frame()
    flow = FarnebackFlow(FarnebackConfig(downscale=downscale)).compute(
        frame, shifted(frame, dx, dy)
    )
    assert np.median(interior(flow.u)) == pytest.approx(dx, abs=0.5)
    assert np.median(interior(flow.v)) == pytest.approx(dy, abs=0.5)


def test_static_scene_has_no_motion():
    frame = textured_frame()
    flow = FarnebackFlow(FarnebackConfig(downscale=1.0)).compute(frame, frame)
    # Farnebäck leaves a small residual near borders even on identical frames.
    assert interior(flow.magnitude).max() < 0.1


def test_accepts_bgr_and_downscales():
    frame = cv2.cvtColor(textured_frame(200, 320), cv2.COLOR_GRAY2BGR)
    flow = FarnebackFlow(FarnebackConfig(downscale=0.25)).compute(frame, frame)
    assert flow.shape == (50, 80)


def test_rejects_mismatched_frames():
    with pytest.raises(ValueError):
        FarnebackFlow().compute(textured_frame(64, 64), textured_frame(64, 80))


def test_update_streams_and_resets():
    model = FarnebackFlow(FarnebackConfig(downscale=1.0))
    frame = textured_frame()
    assert model.update(frame) is None
    flow = model.update(shifted(frame, 2, 0))
    assert np.median(interior(flow.u)) == pytest.approx(2, abs=0.5)
    model.reset()
    assert model.update(frame) is None


def test_pool_regions_localises_motion():
    u = np.zeros((40, 40), np.float32)
    u[:, 20:] = 2.0  # right half moves right at 2 px/frame
    summary = pool_regions(FlowField(u, np.zeros_like(u)), rows=2, cols=2)
    np.testing.assert_allclose(summary.mean_u, [[0, 2], [0, 2]])
    np.testing.assert_allclose(summary.mean_speed, [[0, 2], [0, 2]])
    np.testing.assert_allclose(summary.speed_var, 0)


def test_pool_regions_rejects_bad_grid():
    flow = FlowField(np.zeros((4, 4)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        pool_regions(flow, rows=5, cols=1)


def test_to_velocity_scales_to_metres_per_second():
    flow = FlowField(np.full((2, 2), 4.0), np.zeros((2, 2)))
    vel = flow.to_velocity(metres_per_pixel=0.05, fps=25)
    np.testing.assert_allclose(vel.u, 5.0)


def test_config_from_yaml():
    cfg = FarnebackConfig.from_yaml(CONFIG)
    assert cfg.downscale == 0.25
    assert cfg.params.winsize == 15


def test_config_rejects_bad_downscale():
    with pytest.raises(ValueError):
        FarnebackConfig(downscale=0)
