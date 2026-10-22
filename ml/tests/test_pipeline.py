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


def test_run_clip_reports_seconds_per_module():
    times = {}
    run_clip(frames(3), [Recorder("a"), Recorder("b")], "c1", times)
    assert sorted(times) == ["a", "b"] and all(t >= 0 for t in times.values())


def test_names_must_be_unique_and_present():
    with pytest.raises(ValueError, match="unique"):
        run_clip(frames(1), [Recorder("a"), Recorder("a")], "c1")
    with pytest.raises(ValueError, match="no modules"):
        run_clip(frames(1), [], "c1")


class Breaks(Recorder):
    """Raises on frame ``at`` (or in ``layer`` when ``at`` is None)."""

    def __init__(self, name: str, at: int | None, needs: tuple[str, ...] = ()) -> None:
        super().__init__(name)
        self.at, self.needs = at, needs

    def process(self, frame_id: int, frame: np.ndarray) -> None:
        if frame_id == self.at:
            raise RuntimeError("boom")
        super().process(frame_id, frame)

    def layer(self, clip_id: str) -> dict:
        if self.at is None:
            raise RuntimeError("no layer")
        return super().layer(clip_id)


def test_a_failing_module_stops_the_run_without_errors():
    with pytest.raises(RuntimeError, match="boom"):
        run_clip(frames(3), [Recorder("a"), Breaks("b", 1)], "c1")


def test_with_errors_a_failing_module_is_dropped_and_the_others_finish():
    a, b, needs_b = Recorder("a"), Breaks("b", 2), Breaks("c", 99, needs=("b",))
    errors, times = {}, {}
    layers = run_clip(frames(5), [b, needs_b, a], "c1", times, errors)
    assert list(layers) == ["a"] and a.seen == [0, 1, 2, 3, 4]
    assert b.seen == [0, 1] and needs_b.seen == [0, 1]  # no frames after the failure
    assert errors == {"b": "process (frame 2) failed: RuntimeError: boom",
                      "c": "needs ['b'], which failed"}
    assert sorted(times) == ["a", "b", "c"]


def test_with_errors_a_failing_layer_drops_only_that_layer_and_its_dependents():
    errors = {}
    layers = run_clip(frames(2), [Breaks("b", None), Breaks("c", 99, ("b",)), Recorder("a")],
                      "c1", errors=errors)
    assert list(layers) == ["a"]
    assert errors == {"b": "layer failed: RuntimeError: no layer",
                      "c": "needs ['b'], which failed"}


def test_build_modules_notes_a_builder_that_fails_and_skips_what_needs_it(monkeypatch):
    from sentinel_ml.pipeline import build_modules, registry

    def missing(device, **kw):
        raise FileNotFoundError("no checkpoint")

    monkeypatch.setitem(registry.BUILDERS, "points", missing)
    monkeypatch.setitem(registry.BUILDERS, "a", lambda device: Recorder("a"))
    with pytest.raises(FileNotFoundError):
        build_modules(["tracks", "a"])
    errors = {}
    built = build_modules(["tracks", "a"], errors=errors)
    assert [m.name for m in built] == ["a"]
    assert errors == {"points": "build failed: FileNotFoundError: no checkpoint",
                      "tracks": "needs ['points'], which failed"}
    with pytest.raises(KeyError, match="unknown module 'nope'"):
        build_modules(["nope"], errors={})


def test_recorder_and_real_adapters_follow_the_interface():
    assert isinstance(Recorder("a"), Module)
    pytest.importorskip("torch")
    from sentinel_ml.density import DensityModule

    assert isinstance(DensityModule(model=None), Module)


def test_registry_lists_the_modules_and_rejects_unknown_names():
    assert available() == ["density", "flow", "infrastructure", "points", "tracks"]
    with pytest.raises(KeyError, match="available: density, flow, infrastructure, points, tracks"):
        build_module("nope")


def test_registry_finds_the_repo_configs():
    assert (REPO / "ml" / "pyproject.toml").is_file()
    for name in ("density_dmcount.yaml", "density_layer.yaml", "infrastructure_yolo26.yaml",
                 "flow_layer.yaml", "infer_b.yaml", "points_layer.yaml"):
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
    made = module.produced_by
    assert made["model"] == "dmcount fp32" and made["weights"] == "finetune_e4.pth"
    assert made["weights_md5"] == "725694ba" + made["weights_md5"][8:] and made["every"] == 25
    assert made["device"] == "cpu" and module.default_visible is True


def test_infrastructure_module_from_the_registry_runs_on_cpu():
    pytest.importorskip("ultralytics")
    if not (REPO / "data" / "weights" / "yolo26" / "model_c_best.pt").is_file():
        pytest.skip("fine tuned Model C weights not on this machine")
    module = build_module("infrastructure", device="cpu")
    layers = run_clip(((i, np.zeros((240, 320, 3), np.uint8)) for i in range(130)), [module], "c1")
    doc = layers["infrastructure"]
    assert doc["type"] == "polygons" and doc["frame_size"] == [320, 240]
    assert list(doc["frames"]) == ["0", "125"]  # every 125th frame (infrastructure_yolo26.yaml)
    assert module.produced_by["weights_md5"].startswith("12213bdea884")
    assert module.produced_by["model"] == "yolo26s-seg fp32"


def test_flow_from_the_registry_says_what_made_it_and_starts_hidden():
    pytest.importorskip("cv2")
    module = build_module("flow")
    assert module.produced_by == {"model": "farneback", "every": 25, "device": "cpu",
                                  "units": "px_per_frame"}
    assert module.default_visible is False


def test_registry_overrides_replace_model_config_fields():
    pytest.importorskip("torch")
    with pytest.raises(TypeError):
        build_module("density", device="cpu", no_such_field=1)
    if not (REPO / "data" / "weights" / "dmcount" / "finetune_e4.pth").is_file():
        pytest.skip("fine tuned DM-Count weights not on this machine")
    module = build_module("density", device="cpu", half=True)
    assert module.model.config.half and not module.model.half  # fp16 is CUDA only
