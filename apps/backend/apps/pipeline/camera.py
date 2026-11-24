"""The clip.json ``camera`` block: what flow needs to report metres per second.

Example::

    "camera": {"model": "DJI Mini 3", "altitude_m": 60.0, "hfov_deg": 69.71,
               "altitude_source": "stated", "dfov_deg": 82.1, "sensor_aspect": "4:3"}

Altitude (height above the ground, straight down camera) and the horizontal field of view are
what ``sentinel_ml.geometry`` needs. A spec sheet usually quotes the diagonal FOV of the photo
sensor, so a diagonal value is converted with the sensor's shape, not the video's: a 16:9 video
from a 4:3 sensor keeps the full width and crops top and bottom, so its horizontal FOV is the
sensor's. ``altitude_source`` says how much to trust the altitude: ``telemetry`` (flight log),
``stated`` (the pilot or the clip's description) or ``assumed`` (a guess, e.g. DroneCrowd).
"""

from __future__ import annotations

import math

ALTITUDE_SOURCES = ("assumed", "stated", "telemetry")
SENSOR_ASPECTS = {"4:3": (4, 3), "3:2": (3, 2), "16:9": (16, 9)}
MODEL_LENGTH = 100


def camera_block(altitude_m: float, *, hfov_deg: float | None = None,
                 dfov_deg: float | None = None, sensor_aspect: str = "4:3",
                 model: str | None = None, altitude_source: str = "assumed") -> dict:
    """The checked ``camera`` block; give exactly one of ``hfov_deg`` and ``dfov_deg``."""
    from sentinel_ml.geometry import hfov_from_dfov

    if not (math.isfinite(altitude_m) and 0 < altitude_m <= 1000):
        raise ValueError(f"altitude must be between 0 and 1000 m, got {altitude_m}")
    if (hfov_deg is None) == (dfov_deg is None):
        raise ValueError("give either the horizontal or the diagonal field of view")
    if altitude_source not in ALTITUDE_SOURCES:
        raise ValueError(f"altitude source must be one of {', '.join(ALTITUDE_SOURCES)}")
    fov = hfov_deg if hfov_deg is not None else dfov_deg
    if not (math.isfinite(fov) and 0 < fov < 180):
        raise ValueError(f"field of view must be between 0 and 180 degrees, got {fov}")

    block = {"altitude_m": float(altitude_m), "altitude_source": altitude_source}
    if model and model.strip():
        block["model"] = model.strip()[:MODEL_LENGTH]
    if hfov_deg is not None:
        block["hfov_deg"] = float(hfov_deg)
    else:
        if sensor_aspect not in SENSOR_ASPECTS:
            raise ValueError(f"sensor aspect must be one of {', '.join(SENSOR_ASPECTS)}")
        w, h = SENSOR_ASPECTS[sensor_aspect]
        block.update(hfov_deg=round(hfov_from_dfov(dfov_deg, w, h), 2), dfov_deg=float(dfov_deg),
                     sensor_aspect=sensor_aspect)
    return block


def camera_from_form(data) -> dict | None:
    """The block from upload form fields, or None when altitude and FOV are both left empty.

    Fields: ``altitude_m``, ``fov_deg``, ``fov_kind`` (horizontal or diagonal, default
    horizontal), ``sensor_aspect``, ``camera_model``, ``altitude_source``.
    """
    def text(key: str) -> str:
        return str(data.get(key) or "").strip()

    altitude, fov = text("altitude_m"), text("fov_deg")
    if not altitude and not fov:
        return None
    if not altitude or not fov:
        raise ValueError("give both the altitude and the field of view, or neither")
    try:
        altitude_m, fov_deg = float(altitude), float(fov)
    except ValueError:
        raise ValueError("altitude and field of view must be numbers") from None
    kind = text("fov_kind") or "horizontal"
    if kind not in ("horizontal", "diagonal"):
        raise ValueError("fov_kind must be horizontal or diagonal")
    fovs = {"hfov_deg": fov_deg} if kind == "horizontal" else {"dfov_deg": fov_deg}
    return camera_block(altitude_m, **fovs, sensor_aspect=text("sensor_aspect") or "4:3",
                        model=text("camera_model") or None,
                        altitude_source=text("altitude_source") or "assumed")
