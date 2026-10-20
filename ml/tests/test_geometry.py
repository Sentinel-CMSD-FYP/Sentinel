import math

import cv2
import numpy as np
import pytest

from sentinel_ml.geometry import (
    NadirCamera,
    flow_to_ms,
    fov_from_focal_length,
    ground_width_m,
    gsd,
    gsd_from_focal_px,
    hfov_from_dfov,
    m_to_px,
    ms_to_flow,
    px_to_m,
)

ALT, HFOV, W = 60.0, 84.0, 1280  # the worked example used when this module was designed


def test_ground_width_and_gsd_worked_example():
    assert ground_width_m(ALT, HFOV) == pytest.approx(2 * 60 * math.tan(math.radians(42)))
    assert ground_width_m(ALT, HFOV) == pytest.approx(108.0, abs=0.05)
    assert gsd(ALT, HFOV, W) == pytest.approx(0.0844, abs=0.0001)


def test_gsd_agrees_with_the_pinhole_focal_length_route():
    """Two independent routes to the same number: FOV and image width, or focal length in px."""
    fx = (W / 2) / math.tan(math.radians(HFOV) / 2)
    assert gsd_from_focal_px(ALT, fx) == pytest.approx(gsd(ALT, HFOV, W), rel=1e-12)


def test_gsd_scales_with_altitude_and_inversely_with_resolution():
    base = gsd(ALT, HFOV, W)
    assert gsd(2 * ALT, HFOV, W) == pytest.approx(2 * base)
    assert gsd(ALT, HFOV, 2 * W) == pytest.approx(base / 2)


def test_walking_speed_is_about_half_a_pixel_per_frame():
    """The sanity check from the sprint: 1.2 m/s should be ~0.57 px/frame at 25 fps."""
    px_per_frame = ms_to_flow(1.2, gsd(ALT, HFOV, W), fps=25)
    assert px_per_frame == pytest.approx(0.568, abs=0.005)


def test_conversions_round_trip():
    g = gsd(ALT, HFOV, W)
    assert m_to_px(px_to_m(100.0, g), g) == pytest.approx(100.0)
    assert ms_to_flow(flow_to_ms(0.7, g, 25), g, 25) == pytest.approx(0.7)
    a = np.array([[0.1, -0.4], [2.0, 0.0]])
    np.testing.assert_allclose(ms_to_flow(flow_to_ms(a, g, 30), g, 30), a)


def test_same_walker_looks_slower_from_higher_up():
    low = ms_to_flow(1.2, gsd(60, HFOV, W), 25)
    high = ms_to_flow(1.2, gsd(120, HFOV, W), 25)
    assert high == pytest.approx(low / 2)


def test_a_car_of_known_length_measures_correctly_in_a_synthetic_top_down_image():
    """Draw a 4.5 m x 1.8 m car as the camera at 60 m would see it, then measure it back."""
    g = gsd(ALT, HFOV, W)
    length_px, width_px = round(4.5 / g), round(1.8 / g)
    img = np.zeros((720, W), np.uint8)
    cv2.rectangle(img, (400, 300), (400 + length_px - 1, 300 + width_px - 1), 255, -1)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(img))
    assert px_to_m(w, g) == pytest.approx(4.5, rel=0.02)
    assert px_to_m(h, g) == pytest.approx(1.8, rel=0.03)


def test_field_of_view_helpers():
    assert fov_from_focal_length(focal_mm=18, sensor_mm=36) == pytest.approx(90.0)
    # A square sensor splits a diagonal FOV by sqrt(2) in the tangent.
    h = hfov_from_dfov(90.0, 1000, 1000)
    assert math.tan(math.radians(h) / 2) == pytest.approx(1 / math.sqrt(2))
    # 16:9: the horizontal FOV is narrower than the diagonal one.
    assert hfov_from_dfov(84.0, 1920, 1080) == pytest.approx(76.1, abs=0.2)
    assert hfov_from_dfov(84.0, 1920, 1080) < 84.0


def test_nadir_camera_wraps_the_same_formulas():
    cam = NadirCamera(altitude_m=ALT, hfov_deg=HFOV, image_width=W)
    assert cam.gsd == pytest.approx(gsd(ALT, HFOV, W))
    assert cam.ground_width_m == pytest.approx(ground_width_m(ALT, HFOV))
    assert cam.to_m(50) == pytest.approx(50 * cam.gsd)
    assert cam.flow_to_ms(1.0, fps=25) == pytest.approx(cam.gsd * 25)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_non_positive_or_non_finite_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        gsd(bad, HFOV, W)
    with pytest.raises(ValueError):
        gsd(ALT, HFOV, bad)
    with pytest.raises(ValueError):
        flow_to_ms(1.0, 0.08, bad)
    with pytest.raises(ValueError):
        gsd_from_focal_px(ALT, bad)


@pytest.mark.parametrize("bad", [0, 180, 200, -10, float("nan")])
def test_field_of_view_must_be_between_0_and_180(bad):
    with pytest.raises(ValueError):
        ground_width_m(ALT, bad)
    with pytest.raises(ValueError):
        NadirCamera(ALT, bad, W)
