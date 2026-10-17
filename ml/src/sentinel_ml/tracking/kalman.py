"""Constant-velocity Kalman filter for one head point.

State is (x, y, vx, vy); positions in original-frame pixels, velocities in pixels per
processed frame. Coordinates: x = column, y = row.
"""

from __future__ import annotations

import numpy as np


class ConstantVelocityKF:
    """2D constant-velocity Kalman filter with a position-only measurement.

    Args:
        xy: first measured (x, y), pixels.
        p_pos, p_vel: initial variance of position (px^2) and velocity (px^2 per frame^2).
            The velocity starts very uncertain because it is not observed yet.
        q_pos, q_vel: process noise variance added each frame to position and velocity.
        r_meas: measurement noise variance of a detected point, px^2.
    """

    def __init__(self, xy: tuple[float, float], p_pos: float, p_vel: float, q_pos: float,
                 q_vel: float, r_meas: float) -> None:
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([p_pos, p_pos, p_vel, p_vel])
        self.F = np.array([[1.0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.eye(2) * r_meas

    @property
    def xy(self) -> tuple[float, float]:
        """Current position estimate (x, y), pixels."""
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        """Current velocity estimate (vx, vy), pixels per processed frame."""
        return float(self.x[2]), float(self.x[3])

    def predict(self) -> tuple[float, float]:
        """Advance one frame. Returns the predicted (x, y)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.xy

    def update(self, xy: tuple[float, float]) -> None:
        """Correct the state with a measured (x, y)."""
        z = np.asarray(xy, dtype=float)
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ (z - self.H @ self.x)
        self.P = (np.eye(4) - k @ self.H) @ self.P
