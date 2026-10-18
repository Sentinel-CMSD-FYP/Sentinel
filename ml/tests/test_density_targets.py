"""Tests for sentinel_ml.density.targets (point clipping and Gaussian GT density maps)."""

import numpy as np
import pytest

from sentinel_ml.density.targets import clip_points, density_map


def test_clip_points_moves_border_heads_inside():
    pts = clip_points(np.array([[1921.0, 1080.0], [-2.0, 5.0], [10.0, 20.0]]), 1920, 1080)
    assert pts.dtype == np.float32
    np.testing.assert_array_equal(pts, [[1919, 1079], [0, 5], [10, 20]])


def test_clip_points_empty():
    assert clip_points(np.empty((0, 2)), 16, 16).shape == (0, 2)


@pytest.mark.parametrize("cell", [1, 8])
def test_map_sums_to_count_also_at_borders_and_corners(cell):
    pts = np.array([[0, 0], [1919, 1079], [960, 540], [1921, 1080], [5, 1079]], np.float32)
    m = density_map(pts, 1920, 1080, sigma=4.0, cell=cell)
    assert m.shape == (1080 // cell, 1920 // cell) and m.dtype == np.float32
    assert m.sum(dtype=np.float64) == pytest.approx(len(pts), abs=1e-4)


def test_map_is_empty_without_points():
    m = density_map(np.empty((0, 2)), 32, 16, sigma=2.0, cell=8)
    assert m.shape == (2, 4) and not m.any()


def test_cell_grid_matches_dmcount_discrete_cells():
    # A head at the centre of cell (row 1, col 2) with a tiny sigma lands in that cell only,
    # as in DM-Count's discrete map (round to pixel, then 8 x 8 blocks).
    m = density_map(np.array([[19.6, 11.0]]), 32, 24, sigma=0.5, cell=8)  # round(x)=20 -> col 2
    assert np.unravel_index(m.argmax(), m.shape) == (1, 2)
    assert m[1, 2] == pytest.approx(1.0, abs=1e-3)


def test_map_is_symmetric_gaussian_with_given_sigma():
    m = density_map(np.array([[50.0, 50.0]]), 101, 101, sigma=3.0, cell=1)
    assert np.unravel_index(m.argmax(), m.shape) == (50, 50)
    np.testing.assert_allclose(m, m.T, atol=1e-7)
    ys, xs = np.mgrid[:101, :101]
    var_x = (m * (xs - 50) ** 2).sum() / m.sum()
    assert var_x == pytest.approx(9.0 + 1 / 12, rel=1e-3)  # sigma^2 plus pixel binning


def test_bad_args():
    with pytest.raises(ValueError):
        density_map(np.zeros((1, 2)), 30, 24, sigma=4.0, cell=8)
    with pytest.raises(ValueError):
        density_map(np.zeros((1, 2)), 32, 24, sigma=0.0)
