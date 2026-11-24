"""Model B as the ``points`` pipeline module (step 23)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sentinel_ml.localization.layer import PointsLayerConfig, PointsModule
from sentinel_ml.pipeline import Module, run_clip

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ml" / "configs" / "points_layer.yaml"
CHECKPOINT = ROOT / "ml/third_party/uav-dot/checkpoints/dot_pd_dronecrowd_51.00.ckpt"
SIZE = (240, 320)  # (h, w) of the fake frames


def check_layer():
    path = ROOT / "docs" / "layers" / "check_layer.py"
    spec = importlib.util.spec_from_file_location("layers_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_layer


class FakeDetector:
    """Returns fixed rows and remembers the frames it got (RGB, as DotDetector expects)."""

    def __init__(self, rows) -> None:
        self.rows = np.asarray(rows, np.float64).reshape(-1, 3)
        self.seen: list[np.ndarray] = []

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        self.seen.append(rgb)
        return self.rows


def frames(n: int):
    return ((i, np.zeros((*SIZE, 3), np.uint8)) for i in range(n))


def test_config_from_yaml_matches_the_defaults():
    assert PointsLayerConfig.from_yaml(CONFIG) == PointsLayerConfig()


@pytest.mark.parametrize("kw", [{"every": 0}, {"max_age_frames": 0}, {"score_thr": 1.5}])
def test_bad_config_is_refused(kw):
    with pytest.raises(ValueError):
        PointsLayerConfig(**kw)


def test_layer_follows_the_points_contract():
    det = FakeDetector([[10.0, 20.0, 0.9], [30.0, 40.0, 0.5]])
    module = PointsModule(det)
    assert isinstance(module, Module)
    doc = run_clip(frames(3), [module], "c1")["points"]
    assert doc["type"] == "points" and doc["layer"] == "points" and doc["clip_id"] == "c1"
    assert doc["frame_size"] == [320, 240] and doc["max_age_frames"] == 1
    assert list(doc["frames"]) == ["0", "1", "2"]  # every frame by default
    assert doc["frames"]["0"] == [[10.0, 20.0, 0.9], [30.0, 40.0, 0.5]]
    assert check_layer()(doc, n_frames=3) == []


def test_points_below_the_threshold_are_dropped():
    det = FakeDetector([[1, 1, 0.19], [2, 2, 0.2], [3, 3, 0.7]])
    doc = run_clip(frames(1), [PointsModule(det, PointsLayerConfig(score_thr=0.2))], "c1")["points"]
    assert [r[2] for r in doc["frames"]["0"]] == [0.2, 0.7]


def test_runs_only_every_kth_frame():
    det = FakeDetector([[1, 1, 0.9]])
    doc = run_clip(frames(11), [PointsModule(det, PointsLayerConfig(every=5))], "c1")["points"]
    assert list(doc["frames"]) == ["0", "5", "10"] and len(det.seen) == 3


def test_frame_is_passed_as_rgb():
    det = FakeDetector([])
    bgr = np.zeros((*SIZE, 3), np.uint8)
    bgr[..., 0] = 255  # blue in OpenCV order
    PointsModule(det).process(0, bgr)
    assert det.seen[0][0, 0].tolist() == [0, 0, 255]


def test_empty_frame_stays_an_empty_sample():
    doc = run_clip(frames(2), [PointsModule(FakeDetector([]))], "c1")["points"]
    assert doc["frames"] == {"0": [], "1": []}


def test_rounding_keeps_border_points_inside_the_frame():
    det = FakeDetector([[319.97, 239.99, 0.523456], [-0.01, 5.04, 1.0]])
    doc = run_clip(frames(1), [PointsModule(det)], "c1")["points"]
    assert doc["frames"]["0"] == [[319.9, 239.9, 0.5235], [0.0, 5.0, 1.0]]
    assert check_layer()(doc) == []


def test_frame_size_change_is_refused():
    module = PointsModule(FakeDetector([]))
    module.process(0, np.zeros((*SIZE, 3), np.uint8))
    with pytest.raises(ValueError, match="earlier frames"):
        module.process(1, np.zeros((10, 10, 3), np.uint8))


def test_layer_without_samples_is_refused():
    with pytest.raises(ValueError, match="every=5"):
        PointsModule(FakeDetector([]), PointsLayerConfig(every=5)).layer("c1")


def test_real_model_from_the_registry_runs_on_cpu():
    pytest.importorskip("segmentation_models_pytorch")
    pytest.importorskip("pytorch_lightning")
    if not CHECKPOINT.is_file():
        pytest.skip("uav-dot checkpoint not on this machine")
    from sentinel_ml.pipeline import build_module

    with pytest.raises(TypeError, match="no_such_key"):
        build_module("points", device="cpu", no_such_key=1)
    frame = np.full((1080, 1920, 3), 90, np.uint8)
    doc = run_clip([(0, frame)], [build_module("points", device="cpu")], "c1")["points"]
    assert doc["frame_size"] == [1920, 1080] and list(doc["frames"]) == ["0"]
    assert check_layer()(doc) == []
