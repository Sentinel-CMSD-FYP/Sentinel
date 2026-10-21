"""Tests for eval/diagnose_tracking.py on small synthetic clips with known answers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

import sentinel_ml.tracking.tracker as T
from sentinel_ml.tracking import PointTracker, TrackerConfig

EVAL = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL))  # the script does `import point_metrics`
SPEC = importlib.util.spec_from_file_location("diagnose_tracking", EVAL / "diagnose_tracking.py")
dg = importlib.util.module_from_spec(SPEC)
sys.modules["diagnose_tracking"] = dg
SPEC.loader.exec_module(dg)

SIZE = (1000, 600)
CFG = TrackerConfig()
PEOPLE = {1: (200.0, 200.0), 2: (500.0, 300.0), 3: (800.0, 400.0)}  # still people, GT ids


def make_clip(n_frames, missing=None, extra_blip=None):
    """Points and GT for still people. ``missing`` = {gt_id: set(frames)} B does not detect."""
    missing = missing or {}
    frames, gt = {}, {}
    for f in range(n_frames):
        rows, xy, ids = [], [], []
        for gid, (x, y) in PEOPLE.items():
            xy.append((x, y))
            ids.append(gid)
            if f not in missing.get(gid, set()):
                rows.append([x, y, 0.9])
        if extra_blip and f in extra_blip:
            rows.append([100.0, 500.0, 0.9])  # a detection with no GT person
        frames[str(f)] = rows
        gt[f] = (np.array(xy), np.array(ids))
    doc = {"frame_size": list(SIZE), "frames": frames}
    return doc, gt


def test_perfect_detections_give_one_id_per_person():
    doc, gt = make_clip(30)
    r = dg.analyse_clip(doc, gt, CFG, None)
    assert r["gt_unique_ids"] == 3 and r["output_ids"] == 3
    assert r["identity"]["spurious_ids"] == 0 and r["identity"]["extra_ids_for_tracked"] == 0
    assert r["end"]["end_cause"] == {"alive_at_end": 3}
    assert r["count_error_pct"]["no_filter"] == 0.0
    assert r["detection"]["all"]["detected"] == r["detection"]["all"]["person_frames"]


def test_a_dropout_longer_than_max_age_creates_one_extra_id_and_an_expired_track():
    doc, gt = make_clip(40, missing={2: set(range(10, 20))})  # 10 missed frames > max_age 5
    r = dg.analyse_clip(doc, gt, CFG, None)
    assert r["output_ids"] == 4  # person 2 gets a second id
    assert r["identity"]["extra_ids_for_tracked"] == 1
    assert r["end"]["end_cause"]["expired"] == 1
    assert r["detection"]["all"]["dropouts_longer_than_max_age"] == 1
    assert r["detection"]["all"]["dropouts_up_to_max_age"] == 0
    # the track expired although a detection reappeared later: none was near it at the miss
    assert r["end"]["expired_first_miss_nearest_detection"] == {"farther": 1}


def test_a_short_dropout_is_bridged_and_counted_as_a_short_dropout():
    doc, gt = make_clip(40, missing={2: set(range(10, 13))})
    r = dg.analyse_clip(doc, gt, CFG, None)
    assert r["output_ids"] == 3
    assert r["detection"]["all"]["dropouts_up_to_max_age"] == 1
    assert r["detection"]["all"]["dropouts_longer_than_max_age"] == 0


def test_a_blip_without_a_gt_person_is_a_spurious_id():
    doc, gt = make_clip(40, extra_blip=set(range(10, 16)))  # a 6-frame false detection
    r = dg.analyse_clip(doc, gt, CFG, None)
    assert r["identity"]["spurious_ids"] == 1
    assert r["identity"]["real_ids"] == 3
    assert r["output_ids"] == 4
    assert r["identity"]["spurious_shorter_than_min_len"] == 1


def test_short_track_links_find_predecessor_and_successor():
    tracks = {1: [(f, 100.0, 100.0) for f in range(0, 10)],
              2: [(f, 105.0, 100.0) for f in range(12, 20)],      # starts 2 frames after id 1 ends
              3: [(f, 900.0, 500.0) for f in range(0, 5)]}        # far from everything
    links = dg.short_track_links(tracks, min_len=45)
    assert links["short_tracks"] == 3
    assert links["with_predecessor"] == 1 and links["with_successor"] == 1
    assert links["isolated"] == 1


def test_gt_motion_reports_a_static_scene_and_a_global_shift():
    _, gt = make_clip(10)
    m = dg.gt_motion(gt)
    assert m["step_median_px"] == 0.0 and m["common_mode_max_px"] == 0.0
    shifted = {f: (xy + np.array([3.0 * f, 0.0]), ids) for f, (xy, ids) in gt.items()}
    m2 = dg.gt_motion(shifted)
    assert abs(m2["step_median_px"] - 3.0) < 1e-9 and abs(m2["common_mode_median_px"] - 3.0) < 1e-9


def test_border_stats_count_heads_near_the_edge():
    gt = {0: (np.array([[5.0, 100.0], [500.0, 300.0], [500.0, 596.0]]), np.array([1, 2, 3]))}
    pts = {0: np.array([[5.0, 100.0, 0.9], [500.0, 300.0, 0.9]])}
    b = dg.border_stats(pts, gt, SIZE, 10, 0.2)
    assert b["gt_within_margin"] == 2 and b["gt_people_touching"] == 2
    assert b["detections"] == 2 and b["detections_within_margin"] == 1


def test_count_error_percent():
    assert dg.count_error(150, 100) == 50.0 and dg.count_error(50, 100) == 50.0


def test_recording_restores_the_tracker_methods():
    before = (PointTracker.update, PointTracker._new_track, T.Track.predict, T.Track.mark_missed)
    doc, _ = make_clip(5)
    dg.run_recorded(doc, CFG)
    after = (PointTracker.update, PointTracker._new_track, T.Track.predict, T.Track.mark_missed)
    assert before == after


def test_recording_does_not_change_the_tracker_output():
    doc, _ = make_clip(20, missing={1: set(range(5, 9))})
    recorded, _ = dg.run_recorded(doc, CFG)
    plain = PointTracker(SIZE, CFG)
    for f in range(20):
        rows = plain.update(np.array(doc["frames"][str(f)], float).reshape(-1, 3))
        assert rows == recorded[f]


def test_camera_translation_estimates_a_known_shift(tmp_path):
    import cv2

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (400, 700)).astype(np.float32), (0, 0), 2.0)
    paths = []
    for k in range(8):  # the view moves 4 px right and 3 px down per frame -> 5 px translation
        crop = base[3 * k:3 * k + 270, 4 * k:4 * k + 480].astype(np.uint8)
        path = tmp_path / f"f{k}.png"
        cv2.imwrite(str(path), cv2.merge([crop, crop, crop]))
        paths.append(path)
    est = dg.camera_translation(paths)
    assert est["frames_pairs"] == 7
    assert abs(est["median_px"] - 5.0) < 0.6


def test_camera_translation_of_a_still_camera_is_about_zero(tmp_path):
    import cv2

    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (270, 480)).astype(np.float32), (0, 0), 2.0)
    paths = []
    for k in range(5):
        path = tmp_path / f"s{k}.png"
        cv2.imwrite(str(path), cv2.merge([base.astype(np.uint8)] * 3))
        paths.append(path)
    assert dg.camera_translation(paths)["max_px"] < 0.2


def test_purity_is_one_for_clean_tracks_and_drops_when_a_track_jumps_to_another_person():
    doc, gt = make_clip(30)
    assert dg.analyse_clip(doc, gt, CFG, None)["identity"]["purity"] == 1.0
    # person 1 (A) is seen in frames 0-9 only; person 2 (B) stands 15 px away (inside the 20 px
    # gate) and is seen in frames 10-29 only, so the track that followed A is handed to B
    frames, gt2 = {}, {}
    for f in range(30):
        if f < 10:
            frames[str(f)] = [[200.0, 200.0, 0.9]]
            gt2[f] = (np.array([[200.0, 200.0]]), np.array([1]))
        else:
            frames[str(f)] = [[215.0, 200.0, 0.9]]
            gt2[f] = (np.array([[215.0, 200.0]]), np.array([2]))
    r = dg.analyse_clip({"frame_size": list(SIZE), "frames": frames}, gt2, CFG, None)
    assert r["output_ids"] == 1  # one id covers both people
    assert abs(r["identity"]["purity"] - 20 / 30) < 1e-9  # 20 of its 30 rows are on person 2
