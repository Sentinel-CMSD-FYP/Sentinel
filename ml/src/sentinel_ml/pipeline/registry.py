"""Module names to builders, so callers (backend, scripts) pick modules by name.

Each builder loads its model with the configs in ``ml/configs/`` and resolves relative weight
paths against the repo root, so it works from any working directory. Imports happen inside the
builders: only the modules that are asked for pull in torch or ultralytics.

To add a model, write a class that follows ``Module`` next to the model (as
``sentinel_ml.density.layer.DensityModule``) and add one builder here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from sentinel_ml.pipeline.module import Module

REPO = Path(__file__).resolve().parents[4]  # ml/src/sentinel_ml/pipeline -> repo root
CONFIGS = REPO / "ml" / "configs"


def _from_repo(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else REPO / p)


def _density(device: str | None) -> Module:
    from sentinel_ml.density import DensityLayerConfig, DensityModule, DMCount, DMCountConfig

    cfg = DMCountConfig.from_yaml(CONFIGS / "density_dmcount.yaml")
    cfg = replace(cfg, weights=_from_repo(cfg.weights), device=device or cfg.device)
    return DensityModule(DMCount(cfg), DensityLayerConfig.from_yaml(CONFIGS / "density_layer.yaml"))


def _infrastructure(device: str | None) -> Module:
    from sentinel_ml.infrastructure import InfraConfig, InfraModel, InfraModule

    cfg = InfraConfig.from_yaml(CONFIGS / "infrastructure_yolo26.yaml")
    cfg = replace(cfg, weights=_from_repo(cfg.weights), device=device or cfg.device)
    return InfraModule(InfraModel(cfg))


BUILDERS: dict[str, Callable[[str | None], Module]] = {
    "density": _density,
    "infrastructure": _infrastructure,
}


def available() -> list[str]:
    return sorted(BUILDERS)


def build_module(name: str, device: str | None = None) -> Module:
    """The module called ``name``; ``device`` ("cpu", "cuda") overrides its config."""
    if name not in BUILDERS:
        raise KeyError(f"unknown module {name!r}; available: {', '.join(available())}")
    return BUILDERS[name](device)
