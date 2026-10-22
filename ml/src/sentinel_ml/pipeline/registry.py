"""Module names to builders, so callers (backend, scripts) pick modules by name.

Each builder loads its model with the configs in ``ml/configs/`` and resolves relative weight
paths against the repo root, so it works from any working directory. Imports happen inside the
builders: only the modules that are asked for pull in torch or ultralytics.

To add a model, write a class that follows ``Module`` next to the model (as
``sentinel_ml.density.layer.DensityModule``) and add one builder here. A module that needs
another module's output also gets an entry in ``NEEDS``; use ``build_modules`` to build such sets.

Every built module also gets ``produced_by`` (model, weights file and short md5, sample rate,
device), which the backend writes into clip.json, and ``default_visible`` (``SHOWN``).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from sentinel_ml.pipeline.module import Module

REPO = Path(__file__).resolve().parents[4]  # ml/src/sentinel_ml/pipeline -> repo root
CONFIGS = REPO / "ml" / "configs"

logger = logging.getLogger(__name__)


def _from_repo(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else REPO / p)


def _about(module: Module, model: str, every: int, device, weights: str | None = None) -> Module:
    """Note on ``module`` what produced its layer; ``weights_md5`` tells checkpoints apart."""
    info = {"model": model, "every": every, "device": str(device)}
    if weights is not None:
        with open(weights, "rb") as f:
            md5 = hashlib.file_digest(f, "md5").hexdigest()
        info.update(weights=Path(weights).name, weights_md5=md5[:12])
    module.produced_by = info
    return module


def _density(device: str | None, **overrides) -> Module:
    from sentinel_ml.density import DensityLayerConfig, DensityModule, DMCount, DMCountConfig

    cfg = DMCountConfig.from_yaml(CONFIGS / "density_dmcount.yaml")
    cfg = replace(cfg, device=device or cfg.device, **overrides)
    cfg = replace(cfg, weights=_from_repo(cfg.weights))
    model = DMCount(cfg)
    m = DensityModule(model, DensityLayerConfig.from_yaml(CONFIGS / "density_layer.yaml"))
    precision = "fp16" if model.half else "fp32"
    return _about(m, f"dmcount {precision}", m.cfg.every, model.device, cfg.weights)


def _infrastructure(device: str | None, **overrides) -> Module:
    from sentinel_ml.infrastructure import InfraConfig, InfraModel, InfraModule

    cfg = InfraConfig.from_yaml(CONFIGS / "infrastructure_yolo26.yaml")
    cfg = replace(cfg, device=device or cfg.device, **overrides)
    cfg = replace(cfg, weights=_from_repo(cfg.weights))
    model = InfraModel(cfg)
    precision = "fp16" if model.half else "fp32"
    return _about(InfraModule(model), f"yolo26s-seg {precision}", cfg.every, model._kw["device"],
                  cfg.weights)


def _flow(device: str | None, fps: float | None = None, camera: dict | None = None,
          **overrides) -> Module:
    """Flow in px per frame, or in m/s when ``camera`` (clip.json) gives altitude and FOV."""
    from sentinel_ml.flow.layer import FlowModule, load_flow_config

    cfg, farneback = load_flow_config(CONFIGS / "flow_layer.yaml")
    if camera:
        overrides = {"altitude_m": camera["altitude_m"], "hfov_deg": camera["hfov_deg"],
                     **overrides}
    m = FlowModule(replace(cfg, **overrides), farneback, fps)  # OpenCV on the CPU; no device
    _about(m, "farneback", m.every, "cpu")
    m.produced_by["units"] = m.units
    return m


def _points(device: str | None, **overrides) -> Module:
    import torch
    import yaml

    from sentinel_ml.localization.infer import DotDetector
    from sentinel_ml.localization.layer import PointsLayerConfig, PointsModule

    cfg = yaml.safe_load((CONFIGS / "infer_b.yaml").read_text())
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise TypeError(f"unknown points model settings {sorted(unknown)}")
    cfg.update(overrides)
    for key in ("repo_dir", "checkpoint"):
        cfg[key] = _from_repo(cfg[key])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = DotDetector(cfg, device)
    m = PointsModule(detector, PointsLayerConfig.from_yaml(CONFIGS / "points_layer.yaml"))
    return _about(m, "uav-dot", m.cfg.every, detector.device, cfg["checkpoint"])


def _tracks(device: str | None, points: Module, fps: float | None, **overrides) -> Module:
    import yaml

    from sentinel_ml.tracking.layer import TracksLayerConfig, TracksModule
    from sentinel_ml.tracking.tracker import TrackerConfig

    cfg = yaml.safe_load((CONFIGS / "tracker_e2_candidate.yaml").read_text())["tracker"]
    tracker = replace(TrackerConfig(**cfg), **overrides)  # pure numpy; no device
    m = TracksModule(points, tracker, TracksLayerConfig.from_yaml(CONFIGS / "tracks_layer.yaml"),
                     fps)
    return _about(m, "point tracker (tracker_e2_candidate)", points.cfg.every, "cpu")


BUILDERS: dict[str, Callable[..., Module]] = {
    "density": _density,
    "flow": _flow,
    "infrastructure": _infrastructure,
    "points": _points,
    "tracks": _tracks,
}
NEEDS: dict[str, tuple[str, ...]] = {"tracks": ("points",)}  # passed to the builder by name
USES_FPS = {"flow", "tracks"}  # builders that take the clip's fps
USES_CAMERA = {"flow"}  # builders that take the clip.json "camera" block (altitude, FOV)
# Layers switched on when a clip opens. Points stay off (tracks already mark the people) and so
# does flow (on DroneCrowd people barely move, so its arrows are mostly noise there).
SHOWN = {"density", "infrastructure", "tracks"}


def available() -> list[str]:
    return sorted(BUILDERS)


def build_modules(names: list[str], device: str | None = None, fps: float | None = None,
                  errors: dict[str, str] | None = None,
                  camera: dict | None = None) -> list[Module]:
    """The modules called ``names`` plus the ones they need, each built once.

    A needed module comes before the module that needs it (``run_clip`` checks this), and is
    added even if not asked for, so asking for ``tracks`` also gives the ``points`` layer.
    ``fps`` is the clip's frame rate, passed to the builders in ``USES_FPS``, and ``camera``
    the clip.json ``camera`` block (``altitude_m``, ``hfov_deg``), passed to ``USES_CAMERA``.

    Without ``errors`` a builder's exception (missing weights, missing package) is raised. With
    it, that module and the ones needing it are left out and ``errors[name]`` says why. An
    unknown name always raises.
    """
    built: dict[str, Module] = {}
    failed: dict[str, str] = {}

    def add(name: str) -> None:
        if name in built or name in failed:
            return
        if name not in BUILDERS:
            raise KeyError(f"unknown module {name!r}; available: {', '.join(available())}")
        for dep in NEEDS.get(name, ()):
            add(dep)
        lost = [dep for dep in NEEDS.get(name, ()) if dep in failed]
        if lost:
            failed[name] = f"needs {lost}, which failed"
            return
        kwargs = {dep: built[dep] for dep in NEEDS.get(name, ())}
        if name in USES_FPS:
            kwargs["fps"] = fps
        if name in USES_CAMERA:
            kwargs["camera"] = camera
        try:
            built[name] = _shown(name, BUILDERS[name](device, **kwargs))
        except Exception as e:
            if errors is None:
                raise
            logger.exception("module %r could not be built", name)
            failed[name] = f"build failed: {type(e).__name__}: {e}"

    for name in names:
        add(name)
    if errors is not None:
        errors.update(failed)
    return list(built.values())


def build_module(name: str, device: str | None = None, **overrides) -> Module:
    """The module called ``name``; ``device`` ("cpu", "cuda") overrides its config.

    ``overrides`` replace fields of the model config, e.g. ``weights="/kaggle/input/.../best.pth"``
    or ``half=False`` (an unknown field raises TypeError). A module listed in ``NEEDS`` also
    takes those modules (and ``fps`` if in ``USES_FPS``) here, e.g.
    ``build_module("tracks", points=points_module, fps=25.0)``.
    """
    if name not in BUILDERS:
        raise KeyError(f"unknown module {name!r}; available: {', '.join(available())}")
    return _shown(name, BUILDERS[name](device, **overrides))


def _shown(name: str, module: Module) -> Module:
    module.default_visible = name in SHOWN
    return module
