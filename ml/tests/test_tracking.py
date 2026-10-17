"""Tests for the point tracker on synthetic moving points with known identities."""

import json
from pathlib import Path

import numpy as np
import pytest

from sentinel_ml.tracking import PointTracker, TrackerConfig
from sentinel_ml.tracking.kalman import ConstantVelocityKF
from sentinel_ml.tracking.run_tracker import check_tracks_doc, track_clip, write_tracks

SIZE = (1000, 600)


def pts(*xy, score=1.0):
    return np.array([[x, y, score] for x, y in xy], dtype=float).reshape(-1, 3)


def run(frames, cfg=None, size=SIZE):
    """Feed frames to a fresh tracker; return the output rows of every frame."""
    tracker = PointTracker(size, cfg)
    return [tracker.update(f) for f in frames], tracker


def ids_in(out):
    return {row[0] for rows in out for row in rows}


def test_kalman_learns_constant_velocity():
    kf = ConstantVelocityKF((0.0, 0.0), 10.0, 10000.0, 1.0, 0.5, 1.0)
    for k in range(1, 15):
        kf.predict()
        kf.update((3.0 * k, -2.0 * k))
    vx, vy = kf.velocity
    assert vx == pytest.approx(3.0, abs=0.3) and vy == pytest.approx(-2.0, abs=0.3)
    x, y = kf.predict()
    assert x == pytest.approx(3.0 * 15, abs=1.5) and y == pytest.approx(-2.0 * 15, abs=1.5)


def test_single_moving_object_keeps_one_id_from_the_first_frame():
    frames = [pts((100 + 3 * k, 200 + 2 * k)) for k in range(12)]
    out, _ = run(frames)
    assert ids_in(out) == {1}
    assert all(len(rows) == 1 for rows in out)  # confirmed at birth, reported every frame
    assert out[5][0][1:3] == (115.0, 210.0)  # the matched detection's own position


def test_new_object_is_reported_only_after_min_hits():
    base = [(100.0, 100.0)]
    frames = [pts(*base)] + [pts(*base, (500, 300))] * 4  # B appears in frame 1
    out, _ = run(frames)
    b_frames = [k for k, rows in enumerate(out) if any(r[0] == 2 for r in rows)]
    assert b_frames == [3, 4]  # hits in frames 1, 2, 3 -> confirmed in frame 3


def test_two_objects_keep_separate_stable_ids():
    frames = [pts((100 + 4 * k, 100), (800 - 4 * k, 400)) for k in range(15)]
    out, _ = run(frames)
    assert ids_in(out) == {1, 2}
    for rows in out:
        by_id = {r[0]: r for r in rows}
        assert by_id[1][2] == 100.0 and by_id[2][2] == 400.0  # id 1 stays on the top object


def test_gap_of_max_age_frames_keeps_the_id_but_longer_gap_starts_a_new_one():
    cfg = TrackerConfig(max_age=5, min_hits=3)
    seen = [pts((300, 300))] * 3
    for gap, expect_same in ((5, True), (6, False)):
        out, _ = run(seen + [pts()] * gap + [pts((300, 300))] * 4, cfg)
        ids_after = {r[0] for rows in out[3 + gap:] for r in rows}
        assert (ids_after == {1}) is expect_same


def test_ids_are_unique_and_never_reused():
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(80):
        n = int(rng.integers(0, 6))
        frames.append(pts(*[(float(rng.uniform(50, 950)), float(rng.uniform(50, 550)))
                            for _ in range(n)]))
    cfg = TrackerConfig()
    out, tracker = run(frames, cfg)
    last_seen: dict[int, int] = {}
    for k, rows in enumerate(out):
        assert len({r[0] for r in rows}) == len(rows)
        for r in rows:
            tid = r[0]
            assert 1 <= tid < tracker._next_id
            if tid in last_seen:  # an id must not come back after it could have died
                assert k - last_seen[tid] <= cfg.max_age + 1
            last_seen[tid] = k


def test_jump_beyond_the_gate_starts_a_new_track_inside_the_gate_does_not():
    for jump, same in ((15.0, True), (25.0, False)):
        frames = [pts((300, 300))] * 3 + [pts((300 + jump, 300))] * 4
        out, _ = run(frames, TrackerConfig(size_px=20))
        assert (ids_in(out[3:]) == {1}) is same


def test_nearest_detection_gets_the_existing_track():
    frames = [pts((300, 300)), pts((302, 300), (306, 300))]
    out, _ = run(frames)
    assert out[1] == [(1, 302.0, 300.0, 1.0)]  # the nearer point keeps id 1
    assert ids_in(out) == {1}  # the farther one is a new, unconfirmed track


def test_low_score_points_are_ignored():
    cfg = TrackerConfig(score_thr=0.2)
    out, tracker = run([pts((100, 100), score=0.1), pts((100, 100), score=0.5)], cfg)
    assert out[0] == []  # nothing above the threshold, tracker not started
    assert [r[0] for r in out[1]] == [1]
    assert tracker._next_id == 2


def test_score_equal_to_threshold_is_kept():
    out, _ = run([pts((100, 100), score=0.2)], TrackerConfig(score_thr=0.2))
    assert out[0][0][0] == 1


def test_track_leaving_the_frame_dies_and_a_return_gets_a_new_id():
    size = (200, 100)
    cfg = TrackerConfig(size_px=20)
    frames = [pts((150 + 10 * k, 50)) for k in range(4)]  # 150, 160, 170, 180
    frames += [pts()] * 12 + [pts((100, 50))] * 4
    out, _ = run(frames, cfg, size)
    assert ids_in(out[:4]) == {1}
    assert 1 not in ids_in(out[4:])


def test_empty_frames_are_fine():
    out, _ = run([pts(), pts(), pts((50, 50)), pts((50, 50))])
    assert out[0] == [] and out[1] == []
    assert [r[0] for r in out[3]] == [1]


def test_one_to_one_matching_two_close_detections_do_not_share_a_track():
    frames = [pts((300, 300))] * 2 + [pts((299, 300), (301, 300))] * 3
    out, _ = run(frames)
    for rows in out:
        assert len({r[0] for r in rows}) == len(rows)


def points_doc(frames):
    return {"schema_version": "1.0", "clip_id": "clip_999", "frame_size": list(SIZE),
            "fps": 25, "frame_stride": 1, "altitude_m": 60.0,
            "altitude_source": "assumed_constant", "frames": frames}


def test_track_clip_output_follows_the_contract_and_copies_the_header():
    frames = {str(k): [[100.0 + 3 * k, 200.0, 0.9], [700.0, 300.0 - 2 * k, 0.8]] for k in range(8)}
    frames["3"].append([50.0, 50.0, 0.05])  # below the threshold, must be ignored
    doc = track_clip(points_doc(frames), TrackerConfig())
    check_tracks_doc(doc)
    assert doc["clip_id"] == "clip_999" and doc["fps"] == 25 and doc["frame_stride"] == 1
    assert doc["altitude_m"] == 60.0 and doc["altitude_source"] == "assumed_constant"
    assert list(doc["frames"]) == [str(k) for k in range(8)]
    assert {r[0] for rows in doc["frames"].values() for r in rows} == {1, 2}


def test_frames_are_processed_in_numeric_order():
    frames = {str(k): [[100.0 + 3 * k, 200.0, 0.9]] for k in (10, 2, 1, 0)}
    doc = track_clip(points_doc(frames), TrackerConfig())
    assert list(doc["frames"]) == ["0", "1", "2", "10"]


def test_check_tracks_doc_rejects_bad_documents():
    good = track_clip(points_doc({"0": [[10.0, 10.0, 0.9]]}), TrackerConfig())
    check_tracks_doc(good)
    for bad_rows in ([[1, 10.0, 10.0, 0.9], [1, 20.0, 20.0, 0.9]], [[1, 9999.0, 10.0, 0.9]],
                     [[1, 10.0, 10.0, 1.5]], [[1.5, 10.0, 10.0, 0.9]]):
        with pytest.raises(ValueError):
            check_tracks_doc(dict(good, frames={"0": bad_rows}))
    with pytest.raises(ValueError):
        check_tracks_doc({k: v for k, v in good.items() if k != "fps"})


def test_write_tracks_round_trips(tmp_path):
    doc = track_clip(points_doc({"0": [[10.0, 10.0, 0.9]], "1": []}), TrackerConfig())
    path = tmp_path / "out" / "clip_999_tracks.json"
    write_tracks(doc, path)
    assert json.loads(path.read_text()) == doc


def test_heads_within_the_border_margin_never_get_a_confirmed_track():
    size = (200, 100)
    frames = [pts((100, 95), (100, 50))] * 8  # y=95 is within 10 px of the bottom edge
    out, _ = run(frames, TrackerConfig(border_margin_px=10), size)
    later = [r for rows in out[1:] for r in rows]
    assert {r[2] for r in later} == {50.0}  # only the head away from the border is reported
    out0, _ = run(frames, TrackerConfig(border_margin_px=0), size)
    assert {r[2] for rows in out0 for r in rows} == {95.0, 50.0}  # margin 0 tracks both


def test_contract_sample_points_give_stable_ids():
    sample = Path(__file__).resolve().parents[2] / "docs" / "samples" / "clip_011_points.json"
    cfg = TrackerConfig(border_margin_px=0)  # one head sits at y=1071, 9 px from the bottom
    doc = track_clip(json.loads(sample.read_text()), cfg)
    check_tracks_doc(doc)
    ids = sorted({r[0] for rows in doc["frames"].values() for r in rows})
    assert len(ids) == 12  # 11 people in every frame + 1 who appears mid-way
    first = {tuple(r[1:3]): r[0] for r in doc["frames"]["35"]}
    last = {tuple(r[1:3]): r[0] for r in doc["frames"]["39"]}
    assert len(first) == 11 and set(first.values()) <= set(last.values())  # same ids kept
    assert len(last) == 12  # the person who appears in frame 37 is confirmed by frame 39


def test_contract_sample_with_the_authors_border_rule_drops_the_head_at_the_edge():
    sample = Path(__file__).resolve().parents[2] / "docs" / "samples" / "clip_011_points.json"
    doc = track_clip(json.loads(sample.read_text()), TrackerConfig())  # margin 10 px
    ys = [r[2] for k in ("36", "37", "38", "39") for r in doc["frames"][k]]
    assert max(ys) < 1070  # the head at y=1071 is never reported after the first frame
