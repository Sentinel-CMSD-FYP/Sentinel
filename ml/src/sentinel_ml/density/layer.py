"""DM-Count output as a ``grid`` layer (layer contract, ``docs/layers/layers.md`` on main).

The native density map (8 px cells) is summed into a coarse ``rows x cols`` zone grid, so each
zone's value is the predicted number of people in it and the zones sum to the frame count.

    layer = density_layer(frames, DMCount(...), "seq011", DensityLayerConfig())
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from sentinel_ml.density.dmcount import DOWNSAMPLE, DensityMap, DMCount


@dataclass(frozen=True)
class DensityLayerConfig:
    rows: int = 9  # zone grid laid over the frame (9 x 16 = 120 px zones at 1920x1080)
    cols: int = 16
    every: int = 25  # run on frames whose id is a multiple of this (25 = 1 Hz at 25 fps)
    max_age_frames: int = 25  # how long one sample stays on screen
    decimals: int = 3  # rounding of the people per zone values in the JSON

    def __post_init__(self) -> None:
        if min(self.rows, self.cols, self.every, self.max_age_frames) < 1:
            raise ValueError("rows, cols, every and max_age_frames must be at least 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> DensityLayerConfig:
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))


def zone_grid(dm: DensityMap, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum the density map into ``rows x cols`` zones.

    Each map cell goes to the zone holding its centre pixel, so nothing is lost or counted twice:
    the zones sum to ``dm.count``. Returns ``(people per zone, valid)``; a zone is valid when at
    least one map cell lies in it (the map stops ~8 px short of the bottom of a 1080 frame).
    """
    h, w = dm.density.shape
    H, W = dm.frame_shape
    zr = np.minimum(((np.arange(h) + 0.5) * DOWNSAMPLE * rows / H).astype(int), rows - 1)
    zc = np.minimum(((np.arange(w) + 0.5) * DOWNSAMPLE * cols / W).astype(int), cols - 1)
    idx = (zr[:, None] * cols + zc[None, :]).ravel()
    n = rows * cols
    people = np.bincount(idx, weights=dm.density.ravel().astype(np.float64), minlength=n)
    valid = np.bincount(idx, minlength=n) > 0
    return people.reshape(rows, cols), valid.reshape(rows, cols)


class DensityModule:
    """Model A behind the common interface (``sentinel_ml.pipeline.Module``).

    ``process`` runs DM-Count only on frame ids that are a multiple of ``cfg.every``.
    """

    name = "density"

    def __init__(self, model: DMCount, cfg: DensityLayerConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or DensityLayerConfig()
        self._frame_size: list[int] | None = None
        self._frames: dict[str, dict] = {}

    def wants(self, frame_id: int) -> bool:
        return frame_id % self.cfg.every == 0

    @property
    def samples(self) -> dict:
        """The samples so far, keyed by frame id in the order made (read by live mode)."""
        return self._frames

    def process(self, frame_id: int, frame: np.ndarray) -> None:
        cfg = self.cfg
        if not self.wants(frame_id):
            return
        size = [frame.shape[1], frame.shape[0]]
        if self._frame_size is None:
            self._frame_size = size
        elif size != self._frame_size:
            raise ValueError(f"frame {frame_id} is {size}, earlier frames are {self._frame_size}")
        people, valid = zone_grid(self.model.predict(frame), cfg.rows, cfg.cols)
        self._frames[str(frame_id)] = {"valid": valid.astype(int).tolist(),
                                       "density": np.round(people, cfg.decimals).tolist()}

    def layer(self, clip_id: str) -> dict:
        cfg = self.cfg
        if self._frame_size is None:
            raise ValueError(f"no frame id is a multiple of every={cfg.every}")
        return {"schema_version": 1, "clip_id": clip_id, "layer": self.name, "type": "grid",
                "frame_size": self._frame_size, "grid": {"rows": cfg.rows, "cols": cfg.cols},
                "units": "people_per_cell", "fields": ["density"],
                "max_age_frames": cfg.max_age_frames, "frames": self._frames}


def density_layer(frames: Iterable[tuple[int, np.ndarray]], model: DMCount, clip_id: str,
                  cfg: DensityLayerConfig | None = None) -> dict:
    """Run DM-Count on ``(frame_id, BGR frame)`` pairs and return the grid layer as a dict.

    Only frames whose id is a multiple of ``cfg.every`` are run; ids are 0 based clip indices.
    """
    module = DensityModule(model, cfg)
    for fid, frame in frames:
        module.process(fid, frame)
    return module.layer(clip_id)
