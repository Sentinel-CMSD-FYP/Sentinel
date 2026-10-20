"""Run perception modules on a stored clip and add their layers to it.

The clip's ``video.mp4`` is decoded once and every frame goes to every module (the common
interface ``sentinel_ml.pipeline.Module``). Each finished layer is added with the team's layer
tool ``docs/layers/add_layer.py``, which checks it against the layer contract and the clip and
updates ``clip.json``; the clips API then serves it like any other layer.

``sentinel_ml`` (and torch for the models) is imported only here, so the API itself runs
without it.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import tempfile
import time
from functools import cache
from pathlib import Path
from types import ModuleType

from django.conf import settings

from apps.clips.store import ClipStore, NotFound

logger = logging.getLogger(__name__)


class RunError(Exception):
    """A problem the user can fix; the message says how."""


@cache
def layer_tool() -> ModuleType:
    """``docs/layers/add_layer.py`` (a script, not a package, so it is loaded by path)."""
    path = Path(settings.REPO_ROOT) / "docs" / "layers" / "add_layer.py"
    spec = importlib.util.spec_from_file_location("add_layer", path)
    if spec is None or not path.is_file():
        raise RunError(f"layer tool not found: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_modules(names: list[str], device: str | None = None) -> list:
    try:
        from sentinel_ml.pipeline import available, build_module
    except ImportError as e:
        raise RunError(
            "sentinel_ml is not installed in this environment. Use the repo .venv, which has "
            'it (pip install -e "ml[density,infra]")'
        ) from e
    unknown = sorted(set(names) - set(available()))
    if unknown:
        raise RunError(f"unknown modules {unknown}; available: {', '.join(available())}")
    return [build_module(name, device) for name in names]


def run_models(store: ClipStore, clip_id: str, modules: list, replace: bool = False) -> list[str]:
    """Run ``modules`` over the clip's video and add each layer; returns one line per layer."""
    from sentinel_ml.flow.sources import open_frames
    from sentinel_ml.pipeline import run_clip

    try:
        clip = store.get_clip(clip_id)
        video = store.video_path(clip_id)
    except NotFound as e:
        raise RunError(str(e)) from e
    clip_dir = video.parent

    declared = {entry["layer"] for entry in clip["layers"]}
    taken = sorted(m.name for m in modules if m.name in declared)
    if taken and not replace:  # refuse before spending minutes on the models
        raise RunError(f"clip {clip_id!r} already has layers {taken}; use --replace")

    try:
        frames, _, size = open_frames(video)
    except SystemExit as e:  # open_frames reports a bad file this way
        raise RunError(str(e)) from e
    if list(size) != clip["frame_size"]:
        raise RunError(f"video is {list(size)} but clip.json says frame_size {clip['frame_size']}")

    start = time.perf_counter()
    # zip stops at n_frames, so a video longer than clip.json says adds no out of range frames
    layers = run_clip(zip(range(clip["n_frames"]), frames, strict=False), modules,
                      clip["clip_id"])
    logger.info("ran %s on %s in %.1f s", list(layers), clip_id, time.perf_counter() - start)

    tool = layer_tool()
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, doc in layers.items():
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            try:
                out.append(tool.add_layer(clip_dir, path, replace=replace))
            except tool.LayerError as e:
                raise RunError(f"layer {name!r} was refused: {e}") from e
    return out
