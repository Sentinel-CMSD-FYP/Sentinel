"""The one interface every perception model follows, and the runner that drives them.

A module turns frames into one layer of the layer contract (``docs/layers/layers.md``, plus the
``polygons`` type in ``docs/layers_density_infrastructure.md``). The runner decodes a clip once
and hands every frame to every module, in frame order; each module decides itself which frames
it runs its model on (``every``) and keeps whatever state it needs (previous frame for flow,
tracks for tracking). After the last frame, ``layer(clip_id)`` returns the finished layer.

    modules = [build_module("density"), build_module("infrastructure")]
    layers = run_clip(frames, modules, "seq011")   # {"density": {...}, "infrastructure": {...}}
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import numpy as np


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
             clip_id: str) -> dict[str, dict]:
    """Feed ``(frame_id, BGR frame)`` pairs to every module, then collect their layers by name."""
    names = [m.name for m in modules]
    if not names:
        raise ValueError("no modules to run")
    if len(set(names)) != len(names):
        raise ValueError(f"module names must be unique, got {names}")
    for fid, frame in frames:
        for m in modules:
            m.process(fid, frame)
    return {m.name: m.layer(clip_id) for m in modules}
