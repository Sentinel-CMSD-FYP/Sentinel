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
from datetime import date
from functools import cache
from pathlib import Path
from types import ModuleType

from django.conf import settings

from apps.clips.store import ClipStore, NotFound

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 100  # frames between progress lines


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


def build_modules(names: list[str], device: str | None = None, fps: float | None = None,
                  errors: dict[str, str] | None = None, camera: dict | None = None) -> list:
    """The asked modules plus the ones they need (tracks needs points), each built once.

    ``["all"]`` means every registered module. ``fps`` is the clip's frame rate from clip.json
    (tracks turns its 3 s miss limit into frames), ``camera`` its ``camera`` block (flow in m/s
    when it is there). With ``errors``, a module that cannot be
    built (missing weights or package) is left out and noted there instead of stopping the
    others; an unknown name is always refused.
    """
    from sentinel_ml.pipeline import build_modules as build

    return build(module_names(names), device, fps, errors, camera)


def module_names(names: list[str]) -> list[str]:
    """``names`` checked against the registry, with ``["all"]`` meaning every module."""
    try:
        from sentinel_ml.pipeline import available
    except ImportError as e:
        raise RunError(
            "sentinel_ml is not installed in this environment. Use the repo .venv, which has "
            'it (pip install -e "ml[density,infra]")'
        ) from e
    if names == ["all"]:
        return available()
    unknown = sorted(set(names) - set(available()))
    if unknown or not names:
        raise RunError(f"unknown modules {unknown}; available: {', '.join(available())}")
    return names


def run_models(store: ClipStore, clip_id: str, modules: list, replace: bool = False,
               errors: dict[str, str] | None = None, progress=None) -> list[str]:
    """Run ``modules`` over the clip's video and add each layer; returns one line per layer.

    Without ``errors`` the first failing module (or refused layer) stops the run with RunError.
    With it, a failing module is noted there and the other layers are still added; checks on
    the clip itself (missing, wrong size, layers taken) always raise. ``progress(done, total)``
    is called with each progress line (a run started from the API shows it in its status).
    """
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
    times: dict[str, float] = {}
    decoded = 0
    total = clip["n_frames"]

    def counted():
        nonlocal decoded
        # zip stops at n_frames, so a video longer than clip.json says adds no out of range frames
        for item in zip(range(total), frames, strict=False):
            yield item
            decoded += 1
            if decoded % PROGRESS_EVERY == 0 and decoded < total:
                spent = time.perf_counter() - start
                logger.info("%s: %d/%d frames, %.1f frames/s, about %.0f s left", clip_id,
                            decoded, total, decoded / spent, (total - decoded) * spent / decoded)
                if progress:
                    progress(decoded, total)

    layers = run_clip(counted(), modules, clip["clip_id"], times, errors)
    logger.info("ran %s on %s in %.1f s (%s)", list(layers), clip_id, time.perf_counter() - start,
                ", ".join(f"{n} {t:.1f} s" for n, t in times.items()))

    out = add_layers(clip_dir, layers, modules, replace, errors)
    if errors:
        logger.error("modules failed on %s: %s", clip_id, errors)
    added = [m for m in modules if m.name in layers and m.name not in (errors or {})]
    if added:
        record_producers(clip_dir, added, times, decoded)
    return out


def add_layers(clip_dir: Path, layers: dict[str, dict], modules: list, replace: bool = False,
               errors: dict[str, str] | None = None) -> list[str]:
    """Add each finished layer with ``add_layer``; returns one line per layer added.

    A refused layer raises RunError, or with ``errors`` is noted there and skipped.
    """
    tool = layer_tool()
    visible = {m.name: getattr(m, "default_visible", None) for m in modules}
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, doc in layers.items():
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            try:
                out.append(tool.add_layer(clip_dir, path, replace=replace, visible=visible[name]))
            except tool.LayerError as e:
                if errors is None:
                    raise RunError(f"layer {name!r} was refused: {e}") from e
                errors[name] = f"layer refused: {e}"
    return out


def record_producers(clip_dir: Path, modules: list, times: dict[str, float], frames: int) -> None:
    """Write what made each layer into clip.json ``produced_by``, keyed by layer name.

    Entries of layers this run did not touch are kept. ``flow/export.py`` writes one flat entry
    for its flow layer; that one is kept under ``flow``.
    """
    path = clip_dir / "clip.json"
    clip = json.loads(path.read_text(encoding="utf-8"))
    made = clip.get("produced_by")
    if not isinstance(made, dict):
        made = {}
    elif "model" in made:
        made = {"flow": made}
    today = date.today().isoformat()
    for m in modules:
        made[m.name] = {**getattr(m, "produced_by", {}), "seconds": round(times[m.name], 1),
                        "frames": frames, "date": today}
    clip["produced_by"] = made
    tmp = path.with_name(path.name + ".tmp")  # as add_layer writes it: never half written
    tmp.write_text(json.dumps(clip, indent=2), encoding="utf-8")
    tmp.replace(path)
