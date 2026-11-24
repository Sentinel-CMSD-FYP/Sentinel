"""GT targets for DM-Count (step 5): clipped head points and Gaussian density maps.

DM-Count trains on the points themselves (OT loss); its dataset reads one (N, 2) float32
(x, y) ``.npy`` per image and builds the per cell counts on the fly. So the points are
the training target, and the Gaussian maps here are only for visualisation and eval.

Pixel convention follows DM-Count's ``gen_discrete_map``: a head at (x, y) belongs to
pixel (round(x), round(y)), so cell j of size ``cell`` spans [cell*j - 0.5, cell*(j+1) - 0.5).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def clip_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip (N, 2) (x, y) into [0, width - 1] x [0, height - 1]. A few DroneCrowd heads sit
    on the border (x = 1921, y = 1080), which would fall outside DM-Count's maps."""
    out = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    out[:, 0] = np.clip(out[:, 0], 0, width - 1)
    out[:, 1] = np.clip(out[:, 1], 0, height - 1)
    return out


def _axis_mass(coords: np.ndarray, length: int, sigma: float, cell: int) -> np.ndarray:
    """(N, length // cell): share of each 1D Gaussian in every cell, renormalised to 1
    inside [-0.5, length - 0.5) so heads near the border keep their full mass."""
    edges = np.arange(0, length + 1, cell, dtype=np.float64) - 0.5
    mass = np.diff(ndtr((edges[None, :] - coords[:, None]) / sigma), axis=1)
    return mass / mass.sum(axis=1, keepdims=True)


def density_map(points: np.ndarray, width: int, height: int, sigma: float,
                cell: int = 8) -> np.ndarray:
    """Fixed sigma Gaussian density, integrated over ``cell`` x ``cell`` blocks.

    Returns (height // cell, width // cell) float32 that sums to len(points). ``cell=8``
    gives DM-Count's output grid (240 x 135 for 1920 x 1080); ``cell=1`` is full res.
    Points are clipped into the image first. The Gaussian is separable, so the map is
    one matrix product of per axis cell masses.
    """
    if width % cell or height % cell:
        raise ValueError(f"image {width}x{height} is not a multiple of cell {cell}")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    p = clip_points(points, width, height).astype(np.float64)
    if len(p) == 0:
        return np.zeros((height // cell, width // cell), dtype=np.float32)
    mx = _axis_mass(p[:, 0], width, sigma, cell)
    my = _axis_mass(p[:, 1], height, sigma, cell)
    return (my.T @ mx).astype(np.float32)
