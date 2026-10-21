"""Tests for camera-motion compensation: the flow-in / shift-out interface and the tracker."""

import cv2
import numpy as np
import pytest

from sentinel_ml.flow.farneback import FarnebackConfig
from sentinel_ml.tracking import PointTracker, TrackerConfig
from sentinel_ml.tracking.cmc import (
    CameraShift,
    FarnebackProvider,
    FlowInput,
    camera_shifts_for_clip,
    estimate_camera_shift,
)
from sentinel_ml.tracking.kalman import ConstantVelocityKF
from sentinel_ml.tracking.run_tracker import track_clip

H, W = 60, 120  # a flow grid
SIZE = (1000, 600)


def const_flow(du, dv, downscale=0.25, in_original=True):
    return FlowInput(np.full((H, W), du, np.float32), np.full((H, W), dv, np.float32), downscale,
                     in_original)


def test_flow_input_validation():
    u = np.zeros((4, 5), np.float32)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="downscale"):
            FlowInput(u, u, bad)
    with pytest.raises(ValueError, match="same shape"):
        FlowInput(u, np.zeros((4, 6), np.float32), 0.5)
    with pytest.raises(ValueError):
        FlowInput(np.zeros(5), np.zeros(5), 0.5)


def test_vectors_in_original_pixels_are_used_as_they_are():
    u, v = const_flow(3.0, -2.0, in_original=True).vectors_in_frame_pixels()
    assert (u[0, 0], v[0, 0]) == (3.0, -2.0)


def test_vectors_in_grid_pixels_are_scaled_by_the_explicit_downscale():
    u, v = const_flow(1.0, -0.5, downscale=0.25, in_original=False).vectors_in_frame_pixels()
    assert (u[0, 0], v[0, 0]) == (4.0, -2.0)  # 1 grid px at a quarter size = 4 frame px
    u2, _ = const_flow(1.5, 0.0, downscale=0.5, in_original=False).vectors_in_frame_pixels()
    assert u2[0, 0] == 3.0


def test_uniform_flow_gives_that_shift_with_zero_spread():
    s = estimate_camera_shift(const_flow(3.0, -2.0))
    assert (s.dx, s.dy) == (3.0, -2.0) and s.spread == 0.0 and s.cells == H * W


def test_moving_people_do_not_move_the_median_estimate():
    f = const_flow(2.0, 1.0)
    u, v = f.u.copy(), f.v.copy()
    u[10:30, 20:60], v[10:30, 20:60] = 15.0, -9.0  # a crowd moving fast, about 11% of the grid
    s = estimate_camera_shift(FlowInput(u, v, 0.25, True))
    assert (s.dx, s.dy) == (2.0, 1.0)


def test_excluded_regions_are_left_out_and_too_few_cells_give_no_estimate():
    f = const_flow(2.0, 1.0)
    u, v = f.u.copy(), f.v.copy()
    u[20:40, 40:80] = 50.0  # a block that would change the median if it were large enough
    flow = FlowInput(u, v, 0.25, True)
    centre = np.array([[60 / 0.25, 30 / 0.25]])  # grid cell (60, 30) in frame pixels
    with_mask = estimate_camera_shift(flow, exclude_xy=centre, exclude_radius_px=200.0)
    assert (with_mask.dx, with_mask.dy) == (2.0, 1.0)
    none_left = estimate_camera_shift(flow, exclude_xy=centre, exclude_radius_px=100000.0)
    assert (none_left.dx, none_left.dy, none_left.cells) == (0.0, 0.0, 0)


def test_non_finite_flow_is_ignored():
    f = const_flow(1.0, 1.0)
    u = f.u.copy()
    u[:5, :] = np.nan
    s = estimate_camera_shift(FlowInput(u, f.v, 0.25, True))
    assert (s.dx, s.dy) == (1.0, 1.0) and s.cells == (H - 5) * W


def textured(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 2.0)


def test_farneback_provider_recovers_the_shift_with_the_right_sign():
    base = textured(300, 520)
    # the view moves 6 px right and 4 px down, so static scene content moves 6 px left, 4 px up
    first = cv2.merge([base[0:270, 0:480].astype(np.uint8)] * 3)
    second = cv2.merge([base[4:274, 6:486].astype(np.uint8)] * 3)
    provider = FarnebackProvider(FarnebackConfig(downscale=0.5))
    flow = provider(first, second)
    s = estimate_camera_shift(flow)
    assert abs(s.dx - (-6.0)) < 1.0 and abs(s.dy - (-4.0)) < 1.0
    assert flow.downscale == 0.5 and flow.in_original_pixels is True


def test_camera_shifts_for_clip_keys_are_the_later_frames(tmp_path):
    paths = {}
    for k in (3, 4, 5):
        p = tmp_path / f"f{k}.png"
        cv2.imwrite(str(p), np.zeros((20, 30, 3), np.uint8))
        paths[k] = p
    calls = []

    def provider(prev, cur):
        calls.append(1)
        return const_flow(2.0, 0.0)

    shifts = camera_shifts_for_clip(paths, provider)
    assert sorted(shifts) == [4, 5] and len(calls) == 2  # none for the first frame
    assert shifts[4] == CameraShift(2.0, 0.0, 0.0, H * W)
    missing = {0: tmp_path / "missing.png", 1: tmp_path / "missing.png"}
    with pytest.raises(FileNotFoundError):
        camera_shifts_for_clip(missing, provider)


def test_kalman_shift_moves_the_position_only():
    kf = ConstantVelocityKF((100.0, 50.0), 10.0, 10000.0, 1.0, 0.5, 1.0)
    for k in range(1, 6):
        kf.predict()
        kf.update((100.0 + 2 * k, 50.0))
    velocity = kf.velocity
    p_before = kf.P.copy()
    x0, y0 = kf.xy
    kf.shift(30.0, -4.0)
    assert kf.xy == (x0 + 30.0, y0 - 4.0) and kf.velocity == velocity
    np.testing.assert_array_equal(kf.P, p_before)


def pts(*xy):
    return np.array([[x, y, 1.0] for x, y in xy], dtype=float).reshape(-1, 3)


def jump_frames(n_before=10, n_after=6, jump=30.0):
    """Still people, then the whole view jumps: every head moves +jump px in x at one frame."""
    base = [(200.0, 200.0), (500.0, 300.0), (800.0, 400.0)]
    return ([pts(*base)] * n_before
            + [pts(*[(x + jump, y) for x, y in base])] * n_after)


def ids_from(frames, cfg, shifts, first_frame):
    tracker = PointTracker(SIZE, cfg)
    out = [tracker.update(f, shifts.get(k)) for k, f in enumerate(frames)]
    return {r[0] for rows in out[first_frame:] for r in rows}


def test_a_camera_jump_breaks_tracks_without_cmc_and_not_with_it():
    frames = jump_frames()
    shifts = {10: (30.0, 0.0)}
    on = ids_from(frames, TrackerConfig(use_cmc=True), shifts, 10)
    off = ids_from(frames, TrackerConfig(use_cmc=False), shifts, 10)  # shifts offered, ignored
    assert on == {1, 2, 3}  # the same three tracks carry on
    assert not off <= {1, 2, 3}  # new ids appear when the 30 px jump exceeds the 20 px gate


def test_the_predicted_position_moves_by_exactly_the_shift():
    tracker = PointTracker(SIZE, TrackerConfig(use_cmc=True))
    tracker.update(pts((400.0, 300.0)))  # birth
    x0, y0 = tracker.tracks[0].xy
    tracker.update(pts(), (12.0, -5.0))  # no detection: only prediction and shift
    x1, y1 = tracker.tracks[0].xy
    assert (x1 - x0, y1 - y0) == pytest.approx((12.0, -5.0))  # velocity is 0: the full shift
    off = PointTracker(SIZE, TrackerConfig(use_cmc=False))
    off.update(pts((400.0, 300.0)))
    off.update(pts(), (12.0, -5.0))
    assert off.tracks[0].xy == (x0, y0)  # toggle off: the shift is ignored


def test_a_shift_that_pushes_a_track_out_of_the_frame_kills_it():
    tracker = PointTracker(SIZE, TrackerConfig(use_cmc=True))
    tracker.update(pts((985.0, 300.0)))
    tracker.update(pts(), (30.0, 0.0))  # 1015 is beyond the frame width
    assert tracker.tracks == []


def doc_with(frames):
    return {"schema_version": "1.0", "clip_id": "clip_999", "frame_size": list(SIZE), "fps": 25,
            "frame_stride": 1, "altitude_m": 60.0, "altitude_source": "assumed_constant",
            "frames": frames}


def test_track_clip_needs_shifts_when_cmc_is_on_and_passes_them_per_frame():
    base = [(200.0, 200.0), (500.0, 300.0)]
    frames = {str(k): [[x + (30.0 if k >= 5 else 0.0), y, 0.9] for x, y in base]
              for k in range(10)}
    with pytest.raises(ValueError, match="camera_shifts"):
        track_clip(doc_with(frames), TrackerConfig(use_cmc=True))
    on = track_clip(doc_with(frames), TrackerConfig(use_cmc=True), camera_shifts={5: (30.0, 0.0)})
    off = track_clip(doc_with(frames), TrackerConfig(use_cmc=False))
    ids_on = {r[0] for k in range(5, 10) for r in on["frames"][str(k)]}
    ids_off = {r[0] for k in range(5, 10) for r in off["frames"][str(k)]}
    assert ids_on == {1, 2} and ids_off != {1, 2}


def test_shift_documents_round_trip():
    from sentinel_ml.tracking.cmc import load_shifts, shifts_to_doc

    shifts = {1: CameraShift(0.5, -0.25, 0.1, 100), 3: CameraShift(-2.0, 4.0, 0.2, 90)}
    doc = shifts_to_doc("clip_007", shifts, 0.25)
    assert doc["clip_id"] == "clip_007" and doc["flow_downscale"] == 0.25
    assert load_shifts(doc) == {1: (0.5, -0.25), 3: (-2.0, 4.0)}
