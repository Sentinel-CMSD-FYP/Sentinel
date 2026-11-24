"""Farnebäck flow as a ``grid`` layer, one frame at a time (Model D behind the common interface).

Every ``every`` frames, flow is computed between that frame and the one ``gap`` frames later,
the camera's own shift, zoom and rotation are removed (``camera.py``), and the flow is averaged
over a ``rows x cols`` grid and divided by ``gap``: pixels per frame, or metres per second when
the camera's altitude and field of view are known (``sentinel_ml.geometry``). A sample is stored
under the first frame of its pair, so it appears ``gap`` frames after that frame was read.

    module = FlowModule(FlowLayerConfig(every=25), FarnebackConfig(downscale=0.5))
    layers = run_clip(frames, [module], "seq011")   # {"flow": {...}}

``export.py`` writes a whole clip folder with this module; the backend runs it with the others.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from sentinel_ml.geometry import NadirCamera

from .camera import CameraMotion, remove_camera_motion
from .farneback import FarnebackConfig, FarnebackFlow, FarnebackParams, pool_regions


@dataclass(frozen=True)
class FlowLayerConfig:
    rows: int = 6
    cols: int = 10
    every: int | None = None  # frames between samples; None means about one per second
    gap: int = 5  # frames between the two images of each flow pair
    # A cell with less image texture than this has no reliable flow, so it is marked invalid
    # (a gap) and not drawn. Texture is the mean image gradient in grey levels per pixel.
    min_texture: float = 1.5
    # Subtract the camera's own shift, zoom and rotation, fitted to the flow (see camera.py).
    # A sample where too little of the frame agreed on one camera motion is left out entirely:
    # a gap, not a guess.
    stabilise: bool = True
    min_inliers: float = 0.5
    # Keep only cells whose centre is in the middle `centre_fraction` of the frame (1.0 = all).
    # Other cells are marked invalid. Useful where the fit is weakest, at the edges of a zoom.
    centre_fraction: float = 1.0
    # With both set, flow is written in metres per second instead of pixels per frame, using
    # the ground sampling distance of a straight-down camera (see sentinel_ml.geometry). Neither
    # is known for DroneCrowd or YouTube clips, so these are assumptions the caller owns.
    altitude_m: float | None = None
    hfov_deg: float | None = None

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be positive")
        if self.gap < 1:
            raise ValueError("gap must be at least 1 frame")
        if self.every is not None and self.every < 1:
            raise ValueError("every must be at least 1 frame")
        if not 0.0 < self.centre_fraction <= 1.0:
            raise ValueError("centre_fraction must be in (0, 1]")
        if not 0.0 <= self.min_inliers <= 1.0:
            raise ValueError("min_inliers must be in [0, 1]")
        if (self.altitude_m is None) != (self.hfov_deg is None):
            raise ValueError("altitude_m and hfov_deg must be given together")
        if self.altitude_m is not None:
            NadirCamera(self.altitude_m, self.hfov_deg, 1)  # validates the ranges


def load_flow_config(path: str | Path) -> tuple[FlowLayerConfig, FarnebackConfig]:
    """The layer settings and the ``farneback:`` block of a config such as ``flow_layer.yaml``."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    fb = raw.pop("farneback", None) or {}
    farneback = FarnebackConfig(downscale=fb.get("downscale", FarnebackConfig.downscale),
                                params=FarnebackParams(**fb.get("params", {})))
    return FlowLayerConfig(**raw), farneback


def centre_mask(rows: int, cols: int, fraction: float) -> np.ndarray:
    """True for cells whose centre lies inside the middle ``fraction`` of the frame."""
    cy = (np.arange(rows) + 0.5) / rows - 0.5
    cx = (np.arange(cols) + 0.5) / cols - 0.5
    half = fraction / 2 + 1e-9
    return (np.abs(cy)[:, None] <= half) & (np.abs(cx)[None, :] <= half)


def cell_texture(gray: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean gradient magnitude per grid cell, in grey levels per pixel."""
    g = gray.astype(np.float32)
    # The Sobel kernel has a gain of 8 on a unit gradient; divide it out.
    mag = (np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))) / 8.0
    h, w = mag.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    out = np.empty((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = mag[ys[r] : ys[r + 1], xs[c] : xs[c + 1]].mean()
    return out


class FlowModule:
    """Model D behind the common interface (``sentinel_ml.pipeline.Module``).

    ``fps`` is needed only when ``every`` is None (about one sample per second) or when the
    layer is in m/s. Besides the layer, a run leaves what export.py reports: ``camera`` (the
    motion removed at each sample and whether it was trusted), ``flow_ms``, ``speeds`` and
    ``speeds_raw`` (mean px per frame after and before camera removal).
    """

    name = "flow"

    def __init__(self, cfg: FlowLayerConfig | None = None,
                 farneback: FarnebackConfig | None = None, fps: float | None = None) -> None:
        self.cfg = cfg or FlowLayerConfig()
        self.farneback = farneback or FarnebackConfig()
        if fps is None and (self.cfg.every is None or self.cfg.altitude_m is not None):
            raise ValueError("fps is needed when every is None or the layer is in m/s")
        self.fps = fps
        self.every = self.cfg.every or max(1, round(fps))
        self.model = FarnebackFlow(self.farneback)
        self._inside = centre_mask(self.cfg.rows, self.cfg.cols, self.cfg.centre_fraction)
        self._frame_size: list[int] | None = None
        self._to_units = 1.0  # px per frame -> layer units, set on the first frame
        self.gsd_m: float | None = None
        self._pending: dict[int, np.ndarray] = {}
        self._frames: dict[str, dict] = {}
        self.camera: dict[str, tuple[CameraMotion, bool]] = {}
        self.flow_ms: list[float] = []
        self.speeds: list[float] = []
        self.speeds_raw: list[float] = []

    @property
    def units(self) -> str:
        return "m_per_s" if self.cfg.altitude_m is not None else "px_per_frame"

    def wants(self, frame_id: int) -> bool:
        """A sample's first frame, or the frame ``gap`` after it that completes the pair."""
        start = frame_id - self.cfg.gap
        return frame_id % self.every == 0 or (start >= 0 and start % self.every == 0)

    @property
    def samples(self) -> dict:
        """The samples so far, keyed by frame id in the order made (read by live mode)."""
        return self._frames

    def process(self, frame_id: int, frame: np.ndarray) -> None:
        cfg = self.cfg
        size = [frame.shape[1], frame.shape[0]]
        if self._frame_size is None:
            self._frame_size = size
            if cfg.altitude_m is not None:
                # Ground sampling distance of the pixels flow is measured in.
                self.gsd_m = NadirCamera(cfg.altitude_m, cfg.hfov_deg, size[0]).gsd
                self._to_units = self.gsd_m * self.fps
        elif size != self._frame_size:
            raise ValueError(f"frame {frame_id} is {size}, earlier frames are {self._frame_size}")

        if frame_id % self.every == 0:
            self._pending[frame_id] = frame
        prev = self._pending.pop(frame_id - cfg.gap, None)
        for old in [k for k in self._pending if k < frame_id - cfg.gap]:
            del self._pending[old]  # its pair frame never came (live mode skipped it)
        if prev is None:
            return

        t0 = time.perf_counter()
        flow = self.model.compute(prev, frame)
        self.flow_ms.append((time.perf_counter() - t0) * 1000)
        k = 1.0 / cfg.gap  # pixels over the gap -> pixels per frame
        at = str(frame_id - cfg.gap)

        if cfg.stabilise:
            self.speeds_raw.append(float(flow.magnitude.mean()) * k)
            flow, cam = remove_camera_motion(flow, tuple(size))
            reliable = cam.reliable(cfg.min_inliers)
            self.camera[at] = (cam, reliable)
            if not reliable:
                return  # camera motion is ambiguous here; say nothing for this sample

        pooled = pool_regions(flow, cfg.rows, cfg.cols)
        valid = cell_texture(self.model.prepare(prev), cfg.rows, cfg.cols) >= cfg.min_texture
        valid &= self._inside
        to = self._to_units
        self._frames[at] = {
            "valid": valid.astype(int).tolist(),
            "u": np.round(pooled.mean_u * k * to, 4).tolist(),
            "v": np.round(pooled.mean_v * k * to, 4).tolist(),
            "speed": np.round(pooled.mean_speed * k * to, 4).tolist(),
        }
        self.speeds.append(float(pooled.mean_speed[valid].mean() * k) if valid.any() else 0.0)

    def layer(self, clip_id: str) -> dict:
        if self._frame_size is None:
            raise ValueError("no frames were processed")
        cfg = self.cfg
        return {"schema_version": 1, "clip_id": clip_id, "layer": self.name, "type": "grid",
                "frame_size": self._frame_size, "grid": {"rows": cfg.rows, "cols": cfg.cols},
                "units": self.units, "fields": ["u", "v", "speed"],
                "max_age_frames": self.every, "frames": self._frames}

