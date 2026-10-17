"""Tests for seconds/altitude units (resolve_config) and the DroneCrowd altitude proxy."""

import numpy as np
import pytest
import yaml

from sentinel_ml.tracking import PointTracker, TrackerConfig
from sentinel_ml.tracking.altitude import (
    PROXY_ALTITUDE_M,
    altitude_from_size_tags,
    load_size_tags,
)
from sentinel_ml.tracking.run_tracker import check_tracks_doc, track_clip
from sentinel_ml.tracking.tracker import resolve_config

SIZE = [1000, 600]


def doc_with(frames, fps=25, stride=1, altitude=60.0):
    return {"schema_version": "1.0", "clip_id": "clip_999", "frame_size": SIZE, "fps": fps,
            "frame_stride": stride, "altitude_m": altitude, "altitude_source": "assumed_constant",
            "frames": frames}


def ids(doc, frame_range):
    return {r[0] for k in map(str, frame_range) for r in doc["frames"][k]}


def test_defaults_resolve_to_themselves():
    cfg = TrackerConfig()
    assert resolve_config(cfg, 25, 1, 60.0) == cfg  # both toggles are off by default


def test_max_age_in_seconds_becomes_frames_using_fps_and_stride():
    cfg = TrackerConfig(max_age_s=1.0)
    assert resolve_config(cfg, 25, 1, None).max_age == 25
    assert resolve_config(cfg, 25, 5, None).max_age == 5      # 5 processed frames span 1 s
    assert resolve_config(cfg, 30, 1, None).max_age == 30
    assert resolve_config(TrackerConfig(max_age_s=0.01), 25, 1, None).max_age == 1  # at least 1
    assert resolve_config(cfg, 25, 1, None).max_age_s is None  # applied once


def test_seconds_without_fps_is_an_error():
    with pytest.raises(ValueError, match="fps"):
        resolve_config(TrackerConfig(max_age_s=1.0), None, 1, None)


def test_altitude_scales_the_gate_with_the_authors_formula():
    cfg = TrackerConfig(use_altitude=True)  # size_px 20 at 100 m
    assert resolve_config(cfg, 25, 1, 100.0).size_px == 20
    assert resolve_config(cfg, 25, 1, 50.0).size_px == 40
    assert resolve_config(cfg, 25, 1, 60.0).size_px == 33   # int(100 / 60 * 20)
    assert resolve_config(cfg, 25, 1, 100.0).use_altitude is False


def test_altitude_off_ignores_the_altitude():
    assert resolve_config(TrackerConfig(), 25, 1, 50.0).size_px == 20


def test_altitude_needs_a_positive_altitude():
    for alt in (None, 0.0, -5.0):
        with pytest.raises(ValueError, match="altitude"):
            resolve_config(TrackerConfig(use_altitude=True), 25, 1, alt)


def test_tracker_refuses_an_unresolved_config():
    for cfg in (TrackerConfig(use_altitude=True), TrackerConfig(max_age_s=1.0)):
        with pytest.raises(ValueError, match="resolve_config"):
            PointTracker((100, 100), cfg)


def test_altitude_changes_the_gate_through_track_clip():
    frames = {str(k): [[300.0, 300.0, 0.9]] for k in range(3)}
    frames.update({str(k): [[330.0, 300.0, 0.9]] for k in range(3, 7)})  # a 30 px jump
    off = track_clip(doc_with(frames), TrackerConfig(use_altitude=False))
    alt_cfg = TrackerConfig(use_altitude=True)
    on_far = track_clip(doc_with(frames), alt_cfg, altitude_m=50.0)     # gate 40 px
    on_high = track_clip(doc_with(frames), alt_cfg, altitude_m=100.0)   # gate 20 px
    assert ids(off, range(3, 7)) == {2}            # 30 px > 20 px gate: a new track
    assert ids(on_far, range(3, 7)) == {1}         # 30 px < 40 px gate: same track
    assert ids(on_high, range(3, 7)) == {2}
    assert on_far["altitude_used_m"] == 50.0 and "altitude_used_m" not in off
    check_tracks_doc(on_far)  # the extra key does not break the contract check


def test_header_altitude_is_used_when_there_is_no_override():
    frames = {str(k): [[300.0, 300.0, 0.9]] for k in range(5)}
    d = track_clip(doc_with(frames, altitude=60.0), TrackerConfig(use_altitude=True))
    assert d["altitude_used_m"] == 60.0


def test_max_age_seconds_through_track_clip_bridges_exactly_one_second():
    seen = {str(k): [[300.0, 300.0, 0.9]] for k in range(3)}
    for gap, same in ((25, True), (26, False)):
        frames = dict(seen)
        frames.update({str(k): [] for k in range(3, 3 + gap)})
        frames.update({str(k): [[300.0, 300.0, 0.9]] for k in range(3 + gap, 3 + gap + 4)})
        d = track_clip(doc_with(frames), TrackerConfig(max_age_s=1.0))
        assert (ids(d, range(3 + gap, 3 + gap + 4)) == {1}) is same


def test_proxy_altitudes_and_tag_validation(tmp_path):
    assert altitude_from_size_tags({1: "large", 2: "small"}) == {1: 50.0, 2: 100.0}
    assert PROXY_ALTITUDE_M == {"large": 50.0, "small": 100.0}
    with pytest.raises(ValueError, match="medium"):
        altitude_from_size_tags({1: "medium"})
    path = tmp_path / "tags.yaml"
    path.write_text(yaml.safe_dump({"val": {1: None, 2: "small"}, "test": {11: "large"}}))
    assert load_size_tags(path, "val") is None        # not fully filled in
    assert load_size_tags(path, "test") == {11: "large"}
    assert load_size_tags(tmp_path / "none.yaml", "test") is None


def test_the_committed_test_tags_cover_the_manifest_test_clips():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    tags = load_size_tags(root / "data" / "manifests" / "object_size_tags.yaml", "test")
    manifest = json.loads((root / "data" / "manifests" / "dronecrowd_split.json").read_text())
    assert sorted(tags) == sorted(manifest["test"])
    assert sum(t == "large" for t in tags.values()) == 12
    assert np.isfinite(sum(altitude_from_size_tags(tags).values()))
