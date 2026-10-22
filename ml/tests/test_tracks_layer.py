"""Model E as the ``tracks`` pipeline module on the ``points`` module's rows (step 24)."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from sentinel_ml.localization.layer import PointsLayerConfig, PointsModule
from sentinel_ml.pipeline import Module, build_modules, registry, run_clip
from sentinel_ml.tracking.layer import TracksLayerConfig, TracksModule
from sentinel_ml.tracking.run_tracker import HEADER_KEYS, check_tracks_doc, track_clip
from sentinel_ml.tracking.tracker import TrackerConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "ml" / "configs"
SIZE = (240, 320)  # (h, w) of the fake frames
E2 = TrackerConfig(**yaml.safe_load((CONFIGS / "tracker_e2_candidate.yaml").read_text())["tracker"])


def check_layer():
    path = ROOT / "docs" / "layers" / "check_layer.py"
    spec = importlib.util.spec_from_file_location("layers_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_layer


class Walkers:
    """A detector whose heads walk 2 px per call; head 3 shows up only from call 4 on."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        t, self.calls = self.calls, self.calls + 1
        rows = [[50.0 + 2 * t, 60.0, 0.9], [200.0, 100.0 + 2 * t, 0.8], [150.0, 150.0, 0.1]]
        if t >= 4:
            rows.append([100.0, 200.0, 0.7])
        return np.array(rows)


def frames(n: int):
    return ((i, np.zeros((*SIZE, 3), np.uint8)) for i in range(n))


def run(n: int, every: int = 1, tracker: TrackerConfig = E2, fps: float | None = 25.0):
    points = PointsModule(Walkers(), PointsLayerConfig(every=every))
    tracks = TracksModule(points, tracker, fps=fps)
    return points, tracks, run_clip(frames(n), [points, tracks], "c1")


def test_config_from_yaml_matches_the_defaults():
    assert TracksLayerConfig.from_yaml(CONFIGS / "tracks_layer.yaml") == TracksLayerConfig()


def test_layer_follows_the_tracks_contract():
    points, tracks, layers = run(10)
    assert isinstance(tracks, Module)
    doc = layers["tracks"]
    assert doc["type"] == "tracks" and doc["layer"] == "tracks" and doc["clip_id"] == "c1"
    assert doc["frame_size"] == [320, 240] and doc["max_age_frames"] == 1
    assert doc["camera_motion_compensated"] is False
    assert list(doc["frames"]) == [str(i) for i in range(10)]
    assert check_layer()(doc, n_frames=10) == []


def test_ids_stay_stable_and_new_heads_need_min_hits():
    _, _, layers = run(10)
    frames_ = layers["tracks"]["frames"]
    assert frames_["0"] == [[1, 50.0, 60.0, 0.9], [2, 200.0, 100.0, 0.8]]  # score 0.1 < 0.35
    assert frames_["9"] == [[1, 68.0, 60.0, 0.9], [2, 200.0, 118.0, 0.8], [3, 100.0, 200.0, 0.7]]
    first_seen = min(int(f) for f, rows in frames_.items() if any(r[0] == 3 for r in rows))
    assert first_seen == 4 + E2.min_hits - 1  # appears at call 4, confirmed on its 3rd hit
    assert {r[0] for rows in frames_.values() for r in rows} == {1, 2, 3}


def test_same_tracks_as_the_offline_tracker_on_the_points_layer():
    _, _, layers = run(12)
    doc = layers["points"]
    header = {"schema_version": 1, "clip_id": "c1", "frame_size": doc["frame_size"], "fps": 25.0,
              "frame_stride": 1, "altitude_m": None, "altitude_source": None}
    offline = track_clip({**header, "frames": doc["frames"]}, E2)
    assert set(HEADER_KEYS) <= set(offline)
    check_tracks_doc(offline)
    assert layers["tracks"]["frames"] == offline["frames"]


def test_tracks_only_frames_points_ran_on_and_max_age_s_uses_fps():
    points, tracks, layers = run(10, every=5)
    assert list(layers["tracks"]["frames"]) == ["0", "5"]
    assert tracks.tracker_cfg.max_age == 15  # 3.0 s * 25 fps / every 5
    assert TracksModule(PointsModule(Walkers()), E2, fps=25.0).tracker_cfg.max_age == 75
    with pytest.raises(ValueError, match="fps"):
        TracksModule(points, E2, fps=None)


def test_cmc_and_altitude_are_refused():
    points = PointsModule(Walkers())
    for kw in ({"use_cmc": True}, {"use_altitude": True}):
        with pytest.raises(ValueError, match="use_cmc nor use_altitude"):
            TracksModule(points, replace(E2, **kw), fps=25.0)


def test_tracks_must_run_after_points():
    points = PointsModule(Walkers())
    tracks = TracksModule(points, E2, fps=25.0)
    with pytest.raises(ValueError, match="needs \\['points'\\] to run before it"):
        run_clip(frames(1), [tracks, points], "c1")
    with pytest.raises(ValueError, match="needs \\['points'\\]"):
        run_clip(frames(1), [tracks], "c1")


def test_build_modules_adds_points_once_and_first(monkeypatch):
    made = []

    def fake_points(device):
        made.append(PointsModule(Walkers()))
        return made[-1]

    monkeypatch.setitem(registry.BUILDERS, "points", fake_points)
    modules = build_modules(["tracks"], fps=25.0)
    assert [m.name for m in modules] == ["points", "tracks"] and len(made) == 1
    assert modules[1].points is made[0] and modules[1].tracker_cfg.max_age == 75
    modules = build_modules(["tracks", "points"], fps=25.0)
    assert [m.name for m in modules] == ["points", "tracks"] and len(made) == 2
    with pytest.raises(KeyError, match="available: density, flow, infrastructure, points, tracks"):
        build_modules(["nope"])
