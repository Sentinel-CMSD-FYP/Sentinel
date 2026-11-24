"""Model E output as a ``tracks`` layer (layer contract, ``docs/layers/layers.md``).

E links Model B's points over time, so the module does not run a detector: it reads the rows
the ``points`` module produced for the same frame (``needs = ("points",)``; the registry gives
it that module and ``run_clip`` runs points first). Each frame B ran on gets the confirmed
tracks matched in it, as ``[track_id, x, y, score]`` rows in original frame pixels (the matched
detection's position, as ``run_tracker.track_clip`` writes them).

    module = TracksModule(points_module, TrackerConfig(**cfg["tracker"]), fps=25.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from sentinel_ml.tracking.tracker import PointTracker, TrackerConfig, resolve_config

if TYPE_CHECKING:
    from sentinel_ml.localization.layer import PointsModule


@dataclass(frozen=True)
class TracksLayerConfig:
    max_age_frames: int = 1  # how long one sample stays on screen
    xy_decimals: int = 1  # rounding in the JSON, as in run_tracker's tracks files
    score_decimals: int = 4

    def __post_init__(self) -> None:
        if self.max_age_frames < 1:
            raise ValueError("max_age_frames must be at least 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TracksLayerConfig:
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))


class TracksModule:
    """Model E behind the common interface (``sentinel_ml.pipeline.Module``).

    ``tracker_cfg`` may give the miss limit in seconds (``max_age_s``); it is resolved with
    ``fps`` and the points module's ``every``. Camera motion compensation and the altitude gate
    need inputs the pipeline does not pass yet, so they are refused here.
    """

    name = "tracks"
    needs = ("points",)

    def __init__(self, points: PointsModule, tracker_cfg: TrackerConfig,
                 cfg: TracksLayerConfig | None = None, fps: float | None = None) -> None:
        if tracker_cfg.use_cmc or tracker_cfg.use_altitude:
            raise ValueError("the tracks module supports neither use_cmc nor use_altitude yet")
        self.points = points
        self.tracker_cfg = resolve_config(tracker_cfg, fps, points.cfg.every, None)
        self.cfg = cfg or TracksLayerConfig()
        self._tracker: PointTracker | None = None
        self._frame_size: list[int] | None = None
        self._frames: dict[str, list[list[float]]] = {}

    def wants(self, frame_id: int) -> bool:
        return self.points.wants(frame_id)

    @property
    def samples(self) -> dict:
        """The samples so far, keyed by frame id in the order made (read by live mode)."""
        return self._frames

    def process(self, frame_id: int, frame: np.ndarray) -> None:
        rows = self.points.rows(frame_id)
        if rows is None:  # B did not run on this frame
            return
        if self._tracker is None:
            h, w = frame.shape[:2]
            self._frame_size = [w, h]
            self._tracker = PointTracker((w, h), self.tracker_cfg)
        cfg = self.cfg
        self._frames[str(frame_id)] = [
            [tid, round(x, cfg.xy_decimals), round(y, cfg.xy_decimals),
             round(s, cfg.score_decimals)]
            for tid, x, y, s in self._tracker.update(np.asarray(rows, float).reshape(-1, 3))]

    def layer(self, clip_id: str) -> dict:
        if self._frame_size is None:
            raise ValueError("the points module produced no frames to track")
        return {"schema_version": 1, "clip_id": clip_id, "layer": self.name, "type": "tracks",
                "frame_size": self._frame_size, "max_age_frames": self.cfg.max_age_frames,
                "camera_motion_compensated": self.tracker_cfg.use_cmc,
                "frames": self._frames}
