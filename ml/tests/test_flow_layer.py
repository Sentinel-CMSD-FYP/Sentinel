import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel_ml.flow.layer import FlowLayerConfig, FlowModule, load_flow_config
from sentinel_ml.geometry import NadirCamera
from sentinel_ml.pipeline import Module, build_module, run_clip
from sentinel_ml.pipeline.registry import CONFIGS

CHECKER = Path(__file__).resolve().parents[2] / "docs" / "layers" / "check_layer.py"


def check_layer(doc, n_frames):
    if not CHECKER.is_file():
        pytest.skip("docs/layers/check_layer.py is not available here")
    spec = importlib.util.spec_from_file_location("check_layer", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_layer(doc, n_frames)


def sliding(n, dx=2, h=240, w=320):
    """``(frame_id, BGR frame)`` of a texture sliding right by ``dx`` px per frame."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 2.0)
    base = base.astype(np.uint8)
    for i in range(n):
        m = np.float32([[1, 0, dx * i], [0, 1, 0]])
        g = cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        yield i, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def median_u(entry):
    valid = np.array(entry["valid"], bool)
    return float(np.median(np.array(entry["u"])[valid]))


def test_config_file_holds_the_chosen_settings():
    cfg, fb = load_flow_config(CONFIGS / "flow_layer.yaml")
    assert (cfg.rows, cfg.cols, cfg.every, cfg.gap) == (6, 10, 25, 5)
    assert cfg.stabilise and cfg.altitude_m is None
    assert fb.downscale == 0.5 and fb.params.winsize == 9


def test_registry_module_writes_a_px_per_frame_grid_layer():
    # The whole frame slides, which is camera motion, so removal is off to see the raw flow.
    module = build_module("flow", stabilise=False)
    assert isinstance(module, Module)
    doc = run_clip(sliding(60), [module], "c1")["flow"]
    assert check_layer(doc, 60) == []
    assert list(doc["frames"]) == ["0", "25", "50"]  # every 25, each needs the frame 5 later
    assert doc["units"] == "px_per_frame" and doc["fields"] == ["u", "v", "speed"]
    assert doc["max_age_frames"] == 25 and doc["grid"] == {"rows": 6, "cols": 10}
    assert doc["frame_size"] == [320, 240]
    assert median_u(doc["frames"]["25"]) == pytest.approx(2.0, abs=0.4)


def test_camera_motion_is_removed_by_default():
    doc = run_clip(sliding(30), [build_module("flow")], "c1")["flow"]
    assert median_u(doc["frames"]["0"]) == pytest.approx(0.0, abs=0.1)


def test_metres_per_second_with_a_known_camera():
    px = FlowModule(FlowLayerConfig(every=10, stabilise=False))
    ms = FlowModule(FlowLayerConfig(every=10, stabilise=False, altitude_m=50, hfov_deg=80),
                    fps=25)
    px_doc = run_clip(sliding(20), [px], "c1")["flow"]
    ms_doc = run_clip(sliding(20), [ms], "c1")["flow"]
    gsd = NadirCamera(50, 80, 320).gsd  # metres per pixel at the frame's width
    assert ms_doc["units"] == "m_per_s" and ms.gsd_m == pytest.approx(gsd)
    want = np.array(px_doc["frames"]["0"]["u"]) * gsd * 25
    assert np.array(ms_doc["frames"]["0"]["u"]) == pytest.approx(want, abs=1e-3)


def test_build_modules_passes_the_clip_camera_and_fps_to_flow():
    from sentinel_ml.pipeline import build_modules

    camera = {"altitude_m": 50.0, "hfov_deg": 80.0, "altitude_source": "assumed"}
    (ms,) = build_modules(["flow"], fps=25.0, camera=camera)
    assert ms.units == "m_per_s" and ms.produced_by["units"] == "m_per_s"
    (px,) = build_modules(["flow"], fps=25.0)
    assert px.units == "px_per_frame" and px.produced_by["units"] == "px_per_frame"


def test_fps_is_required_only_when_it_is_used():
    FlowModule(FlowLayerConfig(every=5))
    with pytest.raises(ValueError, match="fps"):
        FlowModule(FlowLayerConfig(every=None))
    with pytest.raises(ValueError, match="fps"):
        FlowModule(FlowLayerConfig(every=5, altitude_m=50, hfov_deg=80))
    assert FlowModule(FlowLayerConfig(every=None), fps=12.4).every == 12


def test_frame_size_change_and_empty_runs_are_refused():
    module = FlowModule(FlowLayerConfig(every=5))
    with pytest.raises(ValueError, match="no frames"):
        module.layer("c1")
    module.process(0, np.zeros((240, 320, 3), np.uint8))
    with pytest.raises(ValueError, match="earlier frames"):
        module.process(1, np.zeros((120, 160, 3), np.uint8))
