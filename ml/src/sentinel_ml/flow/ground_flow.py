"""Flow between two frames as a regular grid in metres per second.

This is the one-call form of the sprint's "get_flow(frame_a, frame_b, altitude)": run
Farnebäck, pool onto a ``rows x cols`` grid, and convert pixels per frame to metres per second
with the ground sampling distance from ``sentinel_ml.geometry``.

The frames are assumed registered (see ``FarnebackFlow``). Set ``stabilise=True`` to subtract
a fitted camera shift/zoom/rotation instead; it is off here because it also removes any
motion of the whole frame, which a registered pipeline wants to keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel_ml.geometry import NadirCamera

from .camera import remove_camera_motion
from .farneback import FarnebackConfig, FarnebackFlow, FarnebackParams, pool_regions


@dataclass(frozen=True)
class GroundFlow:
    """Per-cell flow in metres per second; every array is ``(rows, cols)``."""

    vx: np.ndarray  # mean flow to the right
    vy: np.ndarray  # mean flow downward (image convention)
    speed: np.ndarray  # mean of the speeds, which does not cancel when people go different ways
    speed_var: np.ndarray  # variance of the speeds, in (m/s)^2
    gsd: float  # metres per pixel that was used


def get_flow(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    *,
    altitude_m: float,
    hfov_deg: float,
    fps: float,
    frames_apart: int = 1,
    rows: int = 6,
    cols: int = 10,
    stabilise: bool = False,
    model: FarnebackFlow | None = None,
) -> GroundFlow:
    """Flow from ``frame_a`` to ``frame_b``, which are ``frames_apart`` video frames apart."""
    if frames_apart < 1:
        raise ValueError("frames_apart must be at least 1")
    h, w = frame_a.shape[:2]
    camera = NadirCamera(altitude_m=altitude_m, hfov_deg=hfov_deg, image_width=w)

    # Defaults follow the DroneCrowd evaluation: downscale 0.5, window 9.
    flow_model = model or FarnebackFlow(
        FarnebackConfig(downscale=0.5, params=FarnebackParams(winsize=9))
    )
    flow = flow_model.compute(frame_a, frame_b)
    if stabilise:
        flow, _ = remove_camera_motion(flow, (w, h))

    pooled = pool_regions(flow, rows, cols)
    k = camera.flow_to_ms(1.0 / frames_apart, fps)  # pixels over the gap -> metres per second
    return GroundFlow(
        vx=pooled.mean_u * k,
        vy=pooled.mean_v * k,
        speed=pooled.mean_speed * k,
        speed_var=pooled.speed_var * k * k,
        gsd=camera.gsd,
    )
