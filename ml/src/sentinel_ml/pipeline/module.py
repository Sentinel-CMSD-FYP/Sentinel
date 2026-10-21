"""The one interface every perception model follows, and the runner that drives them.

A module turns frames into one layer of the layer contract (``docs/layers/layers.md``, plus the
``polygons`` type in ``docs/layers_density_infrastructure.md``). The runner decodes a clip once
and hands every frame to every module, in frame order; each module decides itself which frames
it runs its model on (``every``) and keeps whatever state it needs (previous frame for flow,
tracks for tracking). After the last frame, ``layer(clip_id)`` returns the finished layer.

A module that works on another module's output (tracks on points) lists it in an optional
``needs`` tuple and holds that module itself; ``registry.build_modules`` wires this and puts the
needed module first, so per frame it has already run when the dependent one reads it.

Two more optional attributes are read by the backend: ``produced_by`` (a dict saying which
model and weights made the layer, written into clip.json) and ``default_visible`` (whether the
viewer shows the layer when the clip opens). The registry sets both.

Live mode (``sentinel_ml.pipeline.live``) also reads ``samples`` (the samples so far, keyed by
frame id) and ``wants(frame_id)`` (whether ``process`` does any work on that frame).

    modules = [build_module("density"), build_module("infrastructure")]
    layers = run_clip(frames, modules, "seq011")   # {"density": {...}, "infrastructure": {...}}
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Module(Protocol):
    """Frames in, one layer out.

    ``name`` is the layer name (lowercase letters, digits, underscores) and becomes the file
    ``layers/<name>.json``. ``process`` gets every frame of the clip with its 0 based id, as an
    OpenCV BGR uint8 array. ``layer`` returns the layer as a dict ready for ``json.dump``.
    """

    name: str

    def process(self, frame_id: int, frame: np.ndarray) -> None: ...

    def layer(self, clip_id: str) -> dict: ...


def run_clip(frames: Iterable[tuple[int, np.ndarray]], modules: list[Module],
             clip_id: str, times: dict[str, float] | None = None,
             errors: dict[str, str] | None = None) -> dict[str, dict]:
    """Feed ``(frame_id, BGR frame)`` pairs to every module, then collect their layers by name.

    If ``times`` is given, it gets each module's wall seconds (``process`` plus ``layer``; the
    shared decode is not counted).

    Without ``errors`` an exception in any module stops the run. With it, a failing module is
    dropped (no more frames, no layer) and ``errors[name]`` says why; the modules that need it
    are dropped too, and the others finish normally.
    """
    names = [m.name for m in modules]
    if not names:
        raise ValueError("no modules to run")
    if len(set(names)) != len(names):
        raise ValueError(f"module names must be unique, got {names}")
    for i, m in enumerate(modules):
        missing = [n for n in getattr(m, "needs", ()) if n not in names[:i]]
        if missing:
            raise ValueError(f"module {m.name!r} needs {missing} to run before it")
    spent = dict.fromkeys(names, 0.0)
    failed: dict[str, str] = {}

    def call(m: Module, step: str, fn, *args):
        t = time.perf_counter()
        try:
            return fn(*args)
        except Exception as e:
            if errors is None:
                raise
            logger.exception("module %r failed in %s", m.name, step)
            failed[m.name] = f"{step} failed: {type(e).__name__}: {e}"
            for d in modules:  # in order, so a chain of needs is dropped in one pass
                if d.name not in failed and set(getattr(d, "needs", ())) & failed.keys():
                    failed[d.name] = f"needs {sorted(set(d.needs) & failed.keys())}, which failed"
        finally:
            spent[m.name] += time.perf_counter() - t

    for fid, frame in frames:
        for m in modules:
            if m.name not in failed:
                call(m, f"process (frame {fid})", m.process, fid, frame)
    layers = {}
    for m in modules:
        if m.name not in failed:
            doc = call(m, "layer", m.layer, clip_id)
            if m.name not in failed:
                layers[m.name] = doc
    if times is not None:
        times.update(spent)
    if errors is not None:
        errors.update(failed)
    return layers
