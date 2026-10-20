import numpy as np
import pytest

from sentinel_ml.pipeline import Module, available, build_module, run_clip
from sentinel_ml.pipeline.registry import CONFIGS, REPO


class Recorder:
    """A module that remembers which frames it saw and returns them as a points layer."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.seen: list[int] = []

    def process(self, frame_id: int, frame: np.ndarray) -> None:
        self.seen.append(frame_id)

    def layer(self, clip_id: str) -> dict:
        return {"clip_id": clip_id, "layer": self.name, "frames": list(self.seen)}


def frames(n: int):
    return ((i, np.zeros((4, 6, 3), np.uint8)) for i in range(n))


def test_every_module_sees_every_frame_once_in_order():
    a, b = Recorder("a"), Recorder("b")
    layers = run_clip(frames(5), [a, b], "c1")
    assert a.seen == b.seen == [0, 1, 2, 3, 4]
    assert list(layers) == ["a", "b"]
    assert layers["a"] == {"clip_id": "c1", "layer": "a", "frames": [0, 1, 2, 3, 4]}


def test_names_must_be_unique_and_present():
    with pytest.raises(ValueError, match="unique"):
        run_clip(frames(1), [Recorder("a"), Recorder("a")], "c1")
    with pytest.raises(ValueError, match="no modules"):
        run_clip(frames(1), [], "c1")


def test_recorder_and_real_adapters_follow_the_interface():
    assert isinstance(Recorder("a"), Module)
    pytest.importorskip("torch")
    from sentinel_ml.density import DensityModule

    assert isinstance(DensityModule(model=None), Module)


def test_registry_lists_a_and_c_and_rejects_unknown_names():
    assert available() == ["density", "infrastructure"]
    with pytest.raises(KeyError, match="available: density, infrastructure"):
        build_module("flow")


def test_registry_finds_the_repo_configs():
    assert (REPO / "ml" / "pyproject.toml").is_file()
    for name in ("density_dmcount.yaml", "density_layer.yaml", "infrastructure_yolo26.yaml"):
        assert (CONFIGS / name).is_file()


def test_density_module_from_the_registry_runs_on_cpu():
    pytest.importorskip("torch")
    if not (REPO / "data" / "weights" / "dmcount" / "finetune_e4.pth").is_file():
        pytest.skip("fine tuned DM-Count weights not on this machine")
    module = build_module("density", device="cpu")
    layers = run_clip(((i, np.zeros((240, 320, 3), np.uint8)) for i in range(30)), [module], "c1")
    doc = layers["density"]
    assert doc["type"] == "grid" and doc["frame_size"] == [320, 240]
    assert list(doc["frames"]) == ["0", "25"]  # every 25th frame (density_layer.yaml)


def test_infrastructure_module_from_the_registry_runs_on_cpu():
    pytest.importorskip("ultralytics")
    if not (REPO / "data" / "weights" / "yolo26" / "model_c_best.pt").is_file():
        pytest.skip("fine tuned Model C weights not on this machine")
    module = build_module("infrastructure", device="cpu")
    layers = run_clip(((i, np.zeros((240, 320, 3), np.uint8)) for i in range(130)), [module], "c1")
    doc = layers["infrastructure"]
    assert doc["type"] == "polygons" and doc["frame_size"] == [320, 240]
    assert list(doc["frames"]) == ["0", "125"]  # every 125th frame (infrastructure_yolo26.yaml)


def test_registry_overrides_replace_model_config_fields():
    pytest.importorskip("torch")
    with pytest.raises(TypeError):
        build_module("density", device="cpu", no_such_field=1)
    if not (REPO / "data" / "weights" / "dmcount" / "finetune_e4.pth").is_file():
        pytest.skip("fine tuned DM-Count weights not on this machine")
    module = build_module("density", device="cpu", half=True)
    assert module.model.config.half and not module.model.half  # fp16 is CUDA only
