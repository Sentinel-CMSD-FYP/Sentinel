"""Camera-motion compensation (CMC) behind a small interface: flow field in, camera shift out.

A flow model gives a dense flow field between two consecutive frames. This module turns it into
one number pair, the **camera shift** ``(dx, dy)``: how far a static point on the ground moved in
the image, in original-frame pixels. The tracker adds that shift to its predicted track positions.

The interface is deliberately small so another flow model can be swapped in without touching the
CMC or the tracker:

* :class:`FlowInput` is the input: ``u``, ``v`` on a (possibly downscaled) grid, the ``downscale``
  of that grid (grid size / frame size, e.g. 0.25), and which pixels the vectors are in.
* :func:`estimate_camera_shift` is the estimator (robust median over the grid).
* a *flow provider* is any callable ``(prev_frame, frame) -> FlowInput``; ``FarnebackProvider``
  wraps Mahid's ``sentinel_ml.flow.farneback`` model.

Units: pixels. Coordinates: x = column, y = row. Sign: flow goes from the previous frame to the
current one, so a scene that moves right by 3 px gives ``dx = +3``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowInput:
    """Dense flow between two consecutive frames, as any flow model can deliver it.

    Attributes:
        u, v: (h, w) horizontal and vertical flow on the flow grid.
        downscale: grid size divided by frame size (0.25 = the grid is a quarter of the frame in
            each direction). Explicit, because it decides how to map the grid back to the frame.
        in_original_pixels: True if the vectors are already in original-frame pixels per frame
            (the convention of ``sentinel_ml.flow.farneback.FlowField``); False if they are in
            grid pixels (raw OpenCV output), in which case they are divided by ``downscale``.
    """

    u: np.ndarray
    v: np.ndarray
    downscale: float
    in_original_pixels: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.downscale <= 1.0:
            raise ValueError(f"downscale must be in (0, 1], got {self.downscale}")
        if self.u.shape != self.v.shape or self.u.ndim != 2:
            raise ValueError(f"u and v must be 2D arrays of the same shape, got "
                             f"{self.u.shape} and {self.v.shape}")

    def vectors_in_frame_pixels(self) -> tuple[np.ndarray, np.ndarray]:
        """``u`` and ``v`` in original-frame pixels per frame."""
        if self.in_original_pixels:
            return self.u, self.v
        k = 1.0 / self.downscale
        return self.u * k, self.v * k


@dataclass(frozen=True)
class CameraShift:
    """Estimated camera shift between two frames, in original-frame pixels.

    ``spread`` is the median absolute deviation of the flow vectors around the shift (a sign of
    how consistent the background motion is); ``cells`` is the number of grid cells used.
    """

    dx: float
    dy: float
    spread: float = 0.0
    cells: int = 0


def estimate_camera_shift(flow: FlowInput, exclude_xy: np.ndarray | None = None,
                          exclude_radius_px: float = 20.0, min_cells: int = 100) -> CameraShift:
    """Robust camera shift from a dense flow field: the median flow over the grid.

    The median ignores moving people as long as they cover less than half of the grid. Grid
    cells within ``exclude_radius_px`` (original pixels) of any point in ``exclude_xy`` can be left
    out, for example the detected heads. With fewer than ``min_cells`` cells left the shift is
    zero (no estimate).
    """
    u, v = flow.vectors_in_frame_pixels()
    h, w = u.shape
    keep = np.ones((h, w), bool)
    if exclude_xy is not None and len(exclude_xy):
        r = max(1, round(exclude_radius_px * flow.downscale))
        for x, y in np.asarray(exclude_xy, dtype=float).reshape(-1, 2):
            cx, cy = round(x * flow.downscale), round(y * flow.downscale)
            keep[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1] = False
    finite = np.isfinite(u) & np.isfinite(v)
    keep &= finite
    if int(keep.sum()) < min_cells:
        return CameraShift(0.0, 0.0, 0.0, int(keep.sum()))
    du, dv = float(np.median(u[keep])), float(np.median(v[keep]))
    spread = float(np.median(np.hypot(u[keep] - du, v[keep] - dv)))
    return CameraShift(du, dv, spread, int(keep.sum()))


# A flow provider turns two frames into a FlowInput. Any model can be wrapped this way.
FlowProvider = Callable[[np.ndarray, np.ndarray], FlowInput]


class FarnebackProvider:
    """Flow provider backed by ``sentinel_ml.flow.farneback.FarnebackFlow`` (Mahid's module).

    Its ``FlowField`` is in original-frame pixels but does not store the downscale, so the
    downscale of the model's config is passed on explicitly.
    """

    def __init__(self, config=None) -> None:
        from sentinel_ml.flow.farneback import FarnebackConfig, FarnebackFlow

        self._model = FarnebackFlow(config or FarnebackConfig())
        self.downscale = self._model.config.downscale

    def __call__(self, prev_frame: np.ndarray, frame: np.ndarray) -> FlowInput:
        field = self._model.compute(prev_frame, frame)
        return FlowInput(field.u, field.v, self.downscale, in_original_pixels=True)


def camera_shifts_for_clip(image_paths: dict[int, Path], provider: FlowProvider
                           ) -> dict[int, CameraShift]:
    """Camera shift from each processed frame to the next one, keyed by the *later* frame id.

    ``image_paths`` maps frame_id to the image file, in processed order. The first frame has no
    previous frame and gets no entry (the tracker treats it as zero shift).
    """
    shifts: dict[int, CameraShift] = {}
    prev = None
    for fid in sorted(image_paths):
        frame = cv2.imread(str(image_paths[fid]))
        if frame is None:
            raise FileNotFoundError(f"cannot read image {image_paths[fid]}")
        if prev is not None:
            shifts[fid] = estimate_camera_shift(provider(prev, frame))
        prev = frame
    return shifts


def shifts_to_doc(clip_id: str, shifts: dict[int, CameraShift], downscale: float) -> dict:
    """JSON-ready document of a clip's camera shifts (rounded to 0.001 px)."""
    return {"clip_id": clip_id, "flow_downscale": downscale,
            "frames": {str(f): [round(s.dx, 3), round(s.dy, 3), round(s.spread, 3), s.cells]
                       for f, s in sorted(shifts.items())}}


def load_shifts(doc: dict) -> dict[int, tuple[float, float]]:
    """``{frame_id: (dx, dy)}`` from a document written by :func:`shifts_to_doc`."""
    return {int(f): (float(v[0]), float(v[1])) for f, v in doc["frames"].items()}
