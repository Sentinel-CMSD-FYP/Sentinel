"""Pixels to metres for a camera looking straight down at flat ground.

A pixel covers a different amount of ground depending on how high the drone is, so the same
walker looks twice as slow at double the altitude. The ground sampling distance (GSD) is the
number of metres one pixel covers; with it, distances and flow can be stated in metres, which
is what the risk thresholds use (normal walking is about 1.2 m/s).

For a straight-down pinhole camera::

    ground width = 2 * altitude * tan(horizontal FOV / 2)
    GSD          = ground width / image width in pixels

Assumptions, all of which the caller must own:

* the camera points straight down (an angled camera needs a ground homography instead, because
  the GSD then changes across the image);
* the ground is flat, and altitude is height above that ground;
* there is no lens distortion;
* the field of view belongs to the *same pixels* the measurement was made in. A crop, a resize
  that keeps the field of view (fine) or a zoom (the FOV changes) must be reflected in the inputs.

No dataset used so far carries altitude or field of view, so callers working with those must
pass assumed values and say so wherever the results are shown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Number = float | np.ndarray


def _positive(name: str, value: float) -> float:
    if not value > 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a positive number, got {value}")
    return float(value)


def _fov(name: str, deg: float) -> float:
    if not 0 < deg < 180 or not math.isfinite(deg):
        raise ValueError(f"{name} must be between 0 and 180 degrees, got {deg}")
    return float(deg)


def hfov_from_dfov(dfov_deg: float, width_px: int, height_px: int) -> float:
    """Horizontal field of view from the diagonal one that spec sheets usually quote."""
    _fov("dfov_deg", dfov_deg)
    _positive("width_px", width_px)
    _positive("height_px", height_px)
    t = math.tan(math.radians(dfov_deg) / 2) * width_px / math.hypot(width_px, height_px)
    return math.degrees(2 * math.atan(t))


def fov_from_focal_length(focal_mm: float, sensor_mm: float) -> float:
    """Field of view across a sensor dimension (use the sensor width for the horizontal FOV)."""
    _positive("focal_mm", focal_mm)
    _positive("sensor_mm", sensor_mm)
    return math.degrees(2 * math.atan(sensor_mm / (2 * focal_mm)))


def ground_width_m(altitude_m: float, hfov_deg: float) -> float:
    """Width of ground seen across the image."""
    half_fov = math.radians(_fov("hfov_deg", hfov_deg)) / 2
    return 2 * _positive("altitude_m", altitude_m) * math.tan(half_fov)


def gsd(altitude_m: float, hfov_deg: float, image_width_px: int) -> float:
    """Metres of ground per pixel."""
    return ground_width_m(altitude_m, hfov_deg) / _positive("image_width_px", image_width_px)


def gsd_from_focal_px(altitude_m: float, focal_px: float) -> float:
    """Metres per pixel from a calibrated focal length in pixels (the camera matrix's fx)."""
    return _positive("altitude_m", altitude_m) / _positive("focal_px", focal_px)


def px_to_m(px: Number, gsd_m_per_px: float) -> Number:
    return px * _positive("gsd_m_per_px", gsd_m_per_px)


def m_to_px(metres: Number, gsd_m_per_px: float) -> Number:
    return metres / _positive("gsd_m_per_px", gsd_m_per_px)


def flow_to_ms(px_per_frame: Number, gsd_m_per_px: float, fps: float) -> Number:
    """Pixels per frame to metres per second."""
    return px_per_frame * _positive("gsd_m_per_px", gsd_m_per_px) * _positive("fps", fps)


def ms_to_flow(metres_per_s: Number, gsd_m_per_px: float, fps: float) -> Number:
    """Metres per second to pixels per frame."""
    return metres_per_s / (_positive("gsd_m_per_px", gsd_m_per_px) * _positive("fps", fps))


@dataclass(frozen=True)
class NadirCamera:
    """A straight-down camera at a known height: the inputs to every conversion above."""

    altitude_m: float
    hfov_deg: float
    image_width: int

    def __post_init__(self) -> None:
        _positive("altitude_m", self.altitude_m)
        _fov("hfov_deg", self.hfov_deg)
        _positive("image_width", self.image_width)

    @property
    def gsd(self) -> float:
        return gsd(self.altitude_m, self.hfov_deg, self.image_width)

    @property
    def ground_width_m(self) -> float:
        return ground_width_m(self.altitude_m, self.hfov_deg)

    def to_m(self, px: Number) -> Number:
        return px_to_m(px, self.gsd)

    def flow_to_ms(self, px_per_frame: Number, fps: float) -> Number:
        return flow_to_ms(px_per_frame, self.gsd, fps)
