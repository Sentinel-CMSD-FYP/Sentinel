"""Farnebäck dense optical flow for the live perception loop.

Chosen over a learned model (RAFT) on cost: it runs on CPU inside the latency
budget once frames are downscaled. RAFT-Small is used for evaluation only.

Input frames must already be registered to the drone's reference frame
(ORB + RANSAC, see ``sentinel_ml.geometry``), so any remaining motion is crowd
motion rather than drone drift. Flow is only computed while the drone holds
position; call ``FarnebackFlow.reset()`` when it goes into transit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class FarnebackParams:
    """Arguments passed straight to ``cv2.calcOpticalFlowFarneback``."""

    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


@dataclass(frozen=True)
class FarnebackConfig:
    # Fraction of the original resolution flow is computed at. 0.25 takes a
    # 1920x1080 DroneCrowd frame down to 480x270.
    downscale: float = 0.25
    params: FarnebackParams = field(default_factory=FarnebackParams)

    def __post_init__(self) -> None:
        if not 0.0 < self.downscale <= 1.0:
            raise ValueError(f"downscale must be in (0, 1], got {self.downscale}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> FarnebackConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        params = FarnebackParams(**raw.get("params", {}))
        return cls(downscale=raw.get("downscale", cls.downscale), params=params)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FlowField:
    """Dense flow between two frames.

    ``u`` and ``v`` are sampled on the downscaled grid but expressed in
    original-frame pixels per frame, so callers never need to know the
    downscale factor.
    """

    u: np.ndarray
    v: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.u.shape

    @property
    def magnitude(self) -> np.ndarray:
        return np.hypot(self.u, self.v)

    @property
    def angle(self) -> np.ndarray:
        """Direction in radians, image convention (x right, y down)."""
        return np.arctan2(self.v, self.u)

    def to_velocity(self, metres_per_pixel: float, fps: float) -> FlowField:
        """Convert from pixels per frame to metres per second.

        A single scale is only valid for a near-nadir camera; for oblique views
        project through the ground homography in ``sentinel_ml.geometry``.
        """
        k = metres_per_pixel * fps
        return FlowField(self.u * k, self.v * k)


@dataclass(frozen=True)
class RegionSummary:
    """Per-region flow statistics on a ``rows x cols`` grid over the frame."""

    mean_u: np.ndarray
    mean_v: np.ndarray
    mean_speed: np.ndarray
    speed_var: np.ndarray


def pool_regions(flow: FlowField, rows: int, cols: int) -> RegionSummary:
    """Summarise a flow field over a regular grid of image regions (FR2)."""
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    h, w = flow.shape
    if rows > h or cols > w:
        raise ValueError(f"grid {rows}x{cols} is finer than the flow field {h}x{w}")

    y_edges = np.linspace(0, h, rows + 1).astype(int)
    x_edges = np.linspace(0, w, cols + 1).astype(int)
    speed = flow.magnitude
    out = {k: np.empty((rows, cols), np.float32) for k in RegionSummary.__dataclass_fields__}

    for r in range(rows):
        ys = slice(y_edges[r], y_edges[r + 1])
        for c in range(cols):
            xs = slice(x_edges[c], x_edges[c + 1])
            out["mean_u"][r, c] = flow.u[ys, xs].mean()
            out["mean_v"][r, c] = flow.v[ys, xs].mean()
            out["mean_speed"][r, c] = speed[ys, xs].mean()
            out["speed_var"][r, c] = speed[ys, xs].var()

    return RegionSummary(**out)


class FarnebackFlow:
    """Farnebäck optical flow with downscaling and optional frame streaming."""

    def __init__(self, config: FarnebackConfig | None = None) -> None:
        self.config = config or FarnebackConfig()
        self._prev: np.ndarray | None = None

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """Convert a BGR or grayscale frame to downscaled grayscale."""
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim != 2:
            raise ValueError(f"expected a 2D or 3D frame, got shape {frame.shape}")
        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        s = self.config.downscale
        if s < 1.0:
            h, w = frame.shape
            size = (max(1, round(w * s)), max(1, round(h * s)))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return frame

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        """Flow from ``prev`` to ``curr``. Both frames must be the same size."""
        if prev.shape[:2] != curr.shape[:2]:
            raise ValueError(f"frame sizes differ: {prev.shape[:2]} vs {curr.shape[:2]}")
        return self._flow(self.prepare(prev), self.prepare(curr))

    def update(self, frame: np.ndarray) -> FlowField | None:
        """Feed the next frame of a stream; returns ``None`` for the first one."""
        curr = self.prepare(frame)
        prev, self._prev = self._prev, curr
        if prev is None:
            return None
        if prev.shape != curr.shape:
            raise ValueError(f"frame size changed mid-stream: {prev.shape} vs {curr.shape}")
        return self._flow(prev, curr)

    def reset(self) -> None:
        """Drop stream state, e.g. when the drone leaves observe state."""
        self._prev = None

    def _flow(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        p = self.config.params
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            p.pyr_scale, p.levels, p.winsize, p.iterations, p.poly_n, p.poly_sigma, p.flags,
        )
        # Rescale vectors from downscaled pixels back to original-frame pixels.
        k = 1.0 / self.config.downscale
        return FlowField(flow[..., 0] * k, flow[..., 1] * k)
