"""Model B output as a ``points`` layer (layer contract, ``docs/layers/layers.md``).

Each sampled frame gets the head points the detector finds at or above ``score_thr``, as
``[x, y, score]`` rows in original frame pixels (x = column, y = row).

    module = PointsModule(DotDetector(cfg, "cpu"), PointsLayerConfig())
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from sentinel_ml.localization.infer import Detector


@dataclass(frozen=True)
class PointsLayerConfig:
    score_thr: float = 0.2  # B's operating threshold, chosen on val (docs/project-log.md)
    every: int = 1  # run on frames whose id is a multiple of this (1 = every frame)
    max_age_frames: int = 1  # how long one sample stays on screen
    xy_decimals: int = 1  # rounding in the JSON, as in B's points files
    score_decimals: int = 4

    def __post_init__(self) -> None:
        if min(self.every, self.max_age_frames) < 1:
            raise ValueError("every and max_age_frames must be at least 1")
        if not 0.0 <= self.score_thr <= 1.0:
            raise ValueError("score_thr must be between 0 and 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> PointsLayerConfig:
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))


class PointsModule:
    """Model B behind the common interface (``sentinel_ml.pipeline.Module``).

    ``detector`` maps one RGB frame to (N, 3) ``(x, y, score)``, as ``DotDetector`` does.
    """

    name = "points"

    def __init__(self, detector: Detector, cfg: PointsLayerConfig | None = None) -> None:
        self.detector = detector
        self.cfg = cfg or PointsLayerConfig()
        self._frame_size: list[int] | None = None
        self._frames: dict[str, list[list[float]]] = {}

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
        h, w = frame.shape[:2]
        if self._frame_size is None:
            self._frame_size = [w, h]
        elif [w, h] != self._frame_size:
            raise ValueError(f"frame {frame_id} is {[w, h]}, earlier frames are {self._frame_size}")
        pts = np.asarray(self.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), np.float64)
        pts = pts.reshape(-1, 3)
        pts = pts[pts[:, 2] >= cfg.score_thr]
        # The contract wants 0 <= x < w; keep rounding from pushing a border point onto w.
        step = 10.0 ** -cfg.xy_decimals
        xy = np.clip(np.round(pts[:, :2], cfg.xy_decimals), 0.0, [w - step, h - step])
        score = np.round(pts[:, 2], cfg.score_decimals)
        self._frames[str(frame_id)] = [[float(x), float(y), float(s)]
                                       for (x, y), s in zip(xy, score, strict=True)]

    def rows(self, frame_id: int) -> list[list[float]] | None:
        """The ``[x, y, score]`` rows of a processed frame, or None if B did not run on it."""
        return self._frames.get(str(frame_id))

    def layer(self, clip_id: str) -> dict:
        if self._frame_size is None:
            raise ValueError(f"no frame id is a multiple of every={self.cfg.every}")
        return {"schema_version": 1, "clip_id": clip_id, "layer": self.name, "type": "points",
                "frame_size": self._frame_size, "max_age_frames": self.cfg.max_age_frames,
                "frames": self._frames}
