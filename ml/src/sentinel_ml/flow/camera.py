"""Estimate and remove whole-frame camera motion (shift, zoom, rotation) from a flow field.

A drone that sways, drifts or zooms moves every pixel at once, and that shows up as flow
everywhere. The proposal's fix is ORB + RANSAC registration before flow; this is a cheaper
stand-in that works on the flow we already have: fit a global affine motion to the field,
ignoring the pixels that move differently (the crowd), and subtract it.

It assumes the crowd covers less than about half the frame: measured on made-up fields, the fit
is exact for a coherent crowd of up to ~45% of the frame, and beyond that it cannot tell crowd
from camera and locks onto the crowd. It cannot remove parallax (tall objects shift differently
from the ground), so it is not a replacement for registration.

Known blind spot: from flow alone, a crowd converging on (or spreading from) the whole frame
cannot be told apart from the camera zooming out (or in), so the fit removes it. Convergence
over a few neighbouring cells, which is what the Tier 2 features measure, is left alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .farneback import FlowField


@dataclass(frozen=True)
class CameraMotion:
    """``u = tx + a*x + b*y`` and ``v = ty + c*x + d*y``.

    x and y are original-frame pixels from the frame centre; u and v are the same units as
    the flow they were fitted to (pixels over the frame gap).
    """

    tx: float
    ty: float
    a: float
    b: float
    c: float
    d: float
    inlier_fraction: float  # share of the frame that agreed with the fit
    explained: float  # share of the flow's energy the fit accounts for

    @property
    def zoom(self) -> float:
        """Fractional scale change per flow pair; positive means zooming in."""
        return (self.a + self.d) / 2

    @property
    def rotation(self) -> float:
        """Radians per flow pair, counter-clockwise positive on screen."""
        return (self.c - self.b) / 2

    def reliable(self, min_inliers: float = 0.3) -> bool:
        """False when too little of the frame agreed, so the fit cannot be trusted."""
        return self.inlier_fraction >= min_inliers

    def field(
        self, shape: tuple[int, int], frame_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = _grid(shape, frame_size)
        return self.tx + self.a * x + self.b * y, self.ty + self.c * x + self.d * y

    def per_second(self, gap: int, fps: float) -> dict[str, float]:
        """Human units: shift in px/frame, zoom in %/s, rotation in degrees/s."""
        k = fps / gap
        return {
            "shift_x_px_per_frame": self.tx / gap,
            "shift_y_px_per_frame": self.ty / gap,
            "zoom_percent_per_s": self.zoom * k * 100,
            "rotation_deg_per_s": math.degrees(self.rotation) * k,
            "inlier_fraction": self.inlier_fraction,
            "explained": self.explained,
        }


def _grid(shape: tuple[int, int], frame_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Frame-pixel coordinates (from the centre) of every cell of a flow grid."""
    gh, gw = shape
    w, h = frame_size
    x = ((np.arange(gw) + 0.5) / gw - 0.5) * w
    y = ((np.arange(gh) + 0.5) / gh - 0.5) * h
    return np.meshgrid(x, y)


def _least_median_of_squares(
    design: np.ndarray, u: np.ndarray, v: np.ndarray, trials: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    """Affine motion that minimises the *median* squared residual over random 3-point fits.

    Unlike least squares, this survives up to about half the frame moving differently, so a
    crowd cannot skew the starting point. Returns the two coefficient vectors and the
    residual scale (a robust standard deviation per coordinate).
    """
    n = len(u)
    probe = rng.choice(n, min(n, 4000), replace=False)  # points the median is judged on
    best = (np.inf, np.zeros(3), np.zeros(3))
    for _ in range(trials):
        tri = rng.choice(n, 3, replace=False)
        a = design[tri]
        if abs(np.linalg.det(a)) < 1e-6 * np.abs(a).max() ** 3:
            continue  # three nearly collinear points do not pin down a motion
        pu, pv = np.linalg.solve(a, u[tri]), np.linalg.solve(a, v[tri])
        d = design[probe]
        med = float(np.median((u[probe] - d @ pu) ** 2 + (v[probe] - d @ pv) ** 2))
        if med < best[0]:
            best = (med, pu, pv)
    # Median of squared 2-D residual magnitude is 2 ln 2 sigma^2 for Gaussian noise.
    return best[1], best[2], math.sqrt(best[0] / (2 * math.log(2)))


def fit_camera_motion(
    flow: FlowField,
    frame_size: tuple[int, int],
    *,
    keep: float = 3.0,
    trials: int = 300,
    max_points: int = 60_000,
    seed: int = 0,
) -> CameraMotion:
    """Robustly fit a global affine motion to ``flow``.

    Starts from a least-median-of-squares fit, which tolerates a crowd covering up to about
    half the frame, then refines with least squares on the points within ``keep`` robust
    standard deviations of it. ``inlier_fraction`` is the share of the frame that agreed:
    low means the camera motion is ambiguous because much of the frame moved differently.
    """
    gh, gw = flow.shape
    stride = max(1, math.ceil(math.sqrt(gh * gw / max_points)))
    gx, gy = _grid(flow.shape, frame_size)
    x, y = gx[::stride, ::stride].ravel(), gy[::stride, ::stride].ravel()
    u, v = flow.u[::stride, ::stride].ravel(), flow.v[::stride, ::stride].ravel()
    design = np.column_stack([np.ones_like(x), x, y])

    pu, pv, sigma = _least_median_of_squares(design, u, v, trials, np.random.default_rng(seed))
    sigma = max(sigma, 1e-3)
    for _ in range(3):  # refine on the agreeing points
        use = np.hypot(u - design @ pu, v - design @ pv) < keep * sigma
        if use.sum() < 3:
            break
        pu = np.linalg.lstsq(design[use], u[use], rcond=None)[0]
        pv = np.linalg.lstsq(design[use], v[use], rcond=None)[0]

    resid = np.hypot(u - design @ pu, v - design @ pv)
    use = resid < keep * sigma
    energy = float((u**2 + v**2).sum())
    explained = 1.0 - float((resid**2).sum()) / energy if energy > 0 else 1.0
    return CameraMotion(
        tx=float(pu[0]), ty=float(pv[0]), a=float(pu[1]), b=float(pu[2]),
        c=float(pv[1]), d=float(pv[2]),
        inlier_fraction=float(use.mean()), explained=explained,
    )


def remove_camera_motion(
    flow: FlowField, frame_size: tuple[int, int], **fit_kwargs: float
) -> tuple[FlowField, CameraMotion]:
    """The flow with the fitted camera motion subtracted, and the motion that was removed."""
    cam = fit_camera_motion(flow, frame_size, **fit_kwargs)
    cu, cv = cam.field(flow.shape, frame_size)
    return FlowField(flow.u - cu, flow.v - cv), cam
