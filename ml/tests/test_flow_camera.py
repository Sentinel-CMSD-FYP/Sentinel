import math

import cv2
import numpy as np
import pytest

from sentinel_ml.flow import FarnebackConfig, FarnebackFlow, FlowField
from sentinel_ml.flow.camera import CameraMotion, fit_camera_motion, remove_camera_motion

W, H = 640, 360
GH, GW = 90, 160  # a flow grid at quarter resolution


def grid():
    x = ((np.arange(GW) + 0.5) / GW - 0.5) * W
    y = ((np.arange(GH) + 0.5) / GH - 0.5) * H
    return np.meshgrid(x, y)


def camera_flow(tx=0.0, ty=0.0, zoom=0.0, rot=0.0):
    """Flow of a camera that shifts, zooms about the centre and rotates by small amounts."""
    x, y = grid()
    return FlowField(tx + zoom * x - rot * y, ty + rot * x + zoom * y)


def add_crowd(flow, cx, cy, size, du, dv):
    """A patch of people moving differently from the camera."""
    x, y = grid()
    patch = (abs(x - cx) < size) & (abs(y - cy) < size)
    u, v = flow.u.copy(), flow.v.copy()
    u[patch] += du
    v[patch] += dv
    return FlowField(u, v), patch


def test_recovers_shift_zoom_and_rotation():
    cam = fit_camera_motion(camera_flow(tx=0.3, ty=-0.2, zoom=0.004, rot=0.002), (W, H))
    assert (cam.tx, cam.ty) == pytest.approx((0.3, -0.2), abs=1e-3)
    assert cam.zoom == pytest.approx(0.004, abs=1e-5)
    assert cam.rotation == pytest.approx(0.002, abs=1e-5)
    assert cam.inlier_fraction == pytest.approx(1.0)
    assert cam.explained == pytest.approx(1.0, abs=1e-6)


def test_removing_pure_camera_motion_leaves_nothing():
    flow = camera_flow(tx=0.5, ty=0.1, zoom=0.01, rot=-0.003)
    clean, _ = remove_camera_motion(flow, (W, H))
    assert np.abs(clean.u).max() < 1e-3 and np.abs(clean.v).max() < 1e-3


def test_a_moving_crowd_survives_and_does_not_bias_the_fit():
    base = camera_flow(tx=0.2, ty=0.0, zoom=0.005)
    flow, patch = add_crowd(base, cx=40, cy=-20, size=60, du=1.5, dv=-0.8)  # ~12% of the frame
    clean, cam = remove_camera_motion(flow, (W, H))

    assert cam.zoom == pytest.approx(0.005, abs=3e-4)  # the crowd barely moved the estimate
    assert np.median(clean.u[patch]) == pytest.approx(1.5, abs=0.1)
    assert np.median(clean.v[patch]) == pytest.approx(-0.8, abs=0.1)
    assert np.abs(clean.u[~patch]).mean() < 0.02  # background is flat again
    assert 0.8 < cam.inlier_fraction < 0.95  # it knew the patch was not camera


def test_a_plain_median_would_fail_where_the_fit_succeeds():
    """Zoom makes the edges move opposite ways, so subtracting one shift cannot fix it."""
    flow = camera_flow(zoom=0.01)
    after_median = FlowField(flow.u - np.median(flow.u), flow.v - np.median(flow.v))
    clean, _ = remove_camera_motion(flow, (W, H))
    assert np.abs(after_median.u).max() > 1.0
    assert np.abs(clean.u).max() < 1e-3


@pytest.mark.parametrize("share", [0.2, 0.35, 0.45])
def test_survives_a_large_coherent_crowd_covering_up_to_half_the_frame(share):
    """People walking together over a big part of the frame must not be read as camera motion."""
    x, _ = grid()
    flow = camera_flow(zoom=0.01)
    crowd = x < (-W / 2 + share * W)
    flow = FlowField(flow.u + 2.0 * crowd, flow.v)
    clean, cam = remove_camera_motion(flow, (W, H))
    assert cam.zoom == pytest.approx(0.01, abs=2e-4)
    assert (cam.tx, cam.ty) == pytest.approx((0.0, 0.0), abs=0.05)
    assert np.median(clean.u[crowd]) == pytest.approx(2.0, abs=0.1)  # the crowd is kept
    assert cam.inlier_fraction == pytest.approx(1 - share, abs=0.08)  # and it knows how much


def test_two_opposite_halves_are_flagged_as_ambiguous():
    x, _ = grid()
    flow = FlowField(np.where(x < 0, 2.0, -2.0).astype(np.float32), np.zeros((GH, GW), np.float32))
    cam = fit_camera_motion(flow, (W, H))
    assert cam.inlier_fraction < 0.65
    assert not cam.reliable(min_inliers=0.65)


def test_pure_noise_is_harmless():
    """No structure to fit: the estimate stays near zero, so removing it changes nothing."""
    rng = np.random.default_rng(0)
    noise = FlowField(rng.normal(0, 2, (GH, GW)), rng.normal(0, 2, (GH, GW)))
    cam = fit_camera_motion(noise, (W, H))
    assert abs(cam.zoom) < 1e-3 and abs(cam.tx) < 0.2 and abs(cam.ty) < 0.2
    assert cam.explained < 0.05


def test_a_still_camera_gives_a_zero_motion():
    cam = fit_camera_motion(FlowField(np.zeros((GH, GW)), np.zeros((GH, GW))), (W, H))
    assert (cam.tx, cam.ty, cam.zoom, cam.rotation) == (0.0, 0.0, 0.0, 0.0)
    assert cam.explained == 1.0


def test_human_units():
    cam = CameraMotion(tx=0.5, ty=-1.0, a=0.002, b=0.0, c=0.0, d=0.002,
                       inlier_fraction=0.9, explained=0.99)
    p = cam.per_second(gap=5, fps=25)
    assert p["shift_x_px_per_frame"] == pytest.approx(0.1)
    assert p["shift_y_px_per_frame"] == pytest.approx(-0.2)
    assert p["zoom_percent_per_s"] == pytest.approx(0.002 * 5 * 100)  # 0.002 per 5 frames
    assert p["rotation_deg_per_s"] == 0


def test_field_matches_the_model_on_any_grid_size():
    cam = CameraMotion(0.1, 0.2, 0.003, 0.0, 0.0, 0.003, 1.0, 1.0)
    u, v = cam.field((30, 50), (W, H))
    assert u.shape == v.shape == (30, 50)
    assert u[15, 25] == pytest.approx(0.1 + 0.003 * ((25.5 / 50 - 0.5) * W), abs=1e-9)


def test_real_images_zoomed_by_a_known_amount():
    """End to end: Farnebäck on a texture zoomed 1.5% about its centre, then the fit."""
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (H, W)).astype(np.float32), (0, 0), 2.0)
    base = base.astype(np.uint8)
    scale = 1.015
    m = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
    zoomed = cv2.warpAffine(base, m, (W, H), borderMode=cv2.BORDER_REFLECT)

    flow = FarnebackFlow(FarnebackConfig(downscale=1.0)).compute(base, zoomed)
    clean, cam = remove_camera_motion(flow, (W, H))
    assert cam.zoom == pytest.approx(scale - 1, abs=0.003)
    assert math.degrees(cam.rotation) == pytest.approx(0, abs=0.2)

    # Judge the middle of the frame: Farnebäck is unreliable in the border strip that
    # warpAffine fills by reflection, and that is not what this test is about.
    mid = (slice(H // 5, -H // 5), slice(W // 5, -W // 5))
    assert np.abs(flow.u[mid]).mean() > 3 * np.abs(clean.u[mid]).mean()  # the zoom is gone
