"""Point-based SORT tracker (E-1 baseline, no camera-motion compensation yet).

Follows the no-DDCF baseline of the UP-COUNT tracker (Ptak and Kraft, repo
up-count/uav-dot-track, Apache-2.0, read at commit 505a728), re-implemented on points:
  * a detection is a point (x, y); matching uses a fixed gate of ``size_px`` pixels;
  * similarity = 1 - min(distance, gate) / gate, solved one-to-one with the Hungarian
    algorithm, and a match is kept only if it lies inside the gate;
  * a new track needs ``min_hits`` consecutive hits to be confirmed, a track dies after
    more than ``max_age`` consecutive misses or when its prediction leaves the frame;
  * only confirmed tracks matched in the current frame are reported.
Differences from the authors' code: a plain numpy Kalman filter over (x, y, vx, vy), and the
reported position is the matched detection, not the filter's prediction.

Units: pixels and processed frames. Coordinates: x = column, y = row, original frame pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel_ml.tracking.kalman import ConstantVelocityKF


class State(Enum):
    NEW = 1
    CONFIRMED = 2
    MISSING = 3
    DEAD = 4


@dataclass(frozen=True)
class TrackerConfig:
    """Tracker settings. The noise values are the authors' defaults, not tuned by us.

    size_px: matching gate in pixels; 20 is the authors' no-altitude value.
    score_thr: detections with a lower score are ignored.
    border_margin_px: a track whose predicted position comes closer than this to the frame
        border is killed. 10 is the authors' rule (half of size_px). It means heads within
        10 px of the border never get a confirmed track; 0 only kills tracks that leave.
    use_altitude: scale the gate with the clip's altitude (authors' rule
        ``size = int(ref_altitude_m / altitude_m * size_px)``). Off by default.
    ref_altitude_m: altitude at which the gate equals ``size_px`` (the authors use 100 m).
    max_age_s: if set, the miss limit in seconds; it replaces ``max_age`` (frames) once the
        clip's fps and frame_stride are known. ``resolve_config`` does the conversion.

    A config with ``use_altitude`` or ``max_age_s`` is "unresolved": pass it through
    :func:`resolve_config` (``run_tracker.track_clip`` does) before building a tracker.
    """

    max_age: int = 5
    min_hits: int = 3
    score_thr: float = 0.2
    size_px: int = 20
    border_margin_px: int = 10
    use_altitude: bool = False
    ref_altitude_m: float = 100.0
    max_age_s: float | None = None
    p_pos: float = 10.0
    p_vel: float = 10000.0
    q_pos: float = 1.0
    q_vel: float = 0.5
    r_meas: float = 1.0


def resolve_config(cfg: TrackerConfig, fps: float | None, frame_stride: int,
                   altitude_m: float | None) -> TrackerConfig:
    """Turn seconds and altitude into the frames and pixels the tracker works in.

    * ``max_age_s`` becomes ``max_age`` frames: ``ceil(max_age_s * fps / frame_stride)``, at
      least 1 (``frame_stride`` > 1 means each processed frame spans several video frames).
    * ``use_altitude`` becomes a new ``size_px``: ``int(ref_altitude_m / altitude_m * size_px)``.
    The result has ``max_age_s = None`` and ``use_altitude = False``, so it is applied once.
    Units: ``fps`` in frames per second, ``altitude_m`` in metres, results in frames and pixels.
    """
    out = cfg
    if cfg.max_age_s is not None:
        if not fps or fps <= 0:
            raise ValueError("max_age_s needs the clip's fps (the header has none)")
        frames = max(1, math.ceil(cfg.max_age_s * fps / max(1, frame_stride) - 1e-9))
        out = replace(out, max_age=frames, max_age_s=None)
    if cfg.use_altitude:
        if altitude_m is None or altitude_m <= 0:
            raise ValueError("use_altitude needs altitude_m > 0")
        out = replace(out, size_px=max(2, int(cfg.ref_altitude_m / altitude_m * cfg.size_px)),
                      use_altitude=False)
    return out


class Track:
    """One tracked head: a Kalman filter plus the hit/miss counters of the state machine."""

    def __init__(self, track_id: int, xy: tuple[float, float], score: float, state: State,
                 cfg: TrackerConfig) -> None:
        self.id = track_id
        self.cfg = cfg
        self.state = state
        self.kf = ConstantVelocityKF(xy, cfg.p_pos, cfg.p_vel, cfg.q_pos, cfg.q_vel, cfg.r_meas)
        self.score = score
        self.missing_counter = 0
        self.active_counter = cfg.min_hits if state is State.CONFIRMED else 1

    @property
    def xy(self) -> tuple[float, float]:
        return self.kf.xy

    def predict(self, frame_size: tuple[int, int]) -> None:
        """Move the filter one frame ahead; mark the track dead if it leaves the frame."""
        x, y = self.kf.predict()
        m = self.cfg.border_margin_px
        w, h = frame_size
        if x - m < 0 or y - m < 0 or x + m >= w or y + m >= h:
            self.state = State.DEAD

    def update(self, xy: tuple[float, float], score: float) -> None:
        self.kf.update(xy)
        self.score = score
        self.missing_counter = 0
        self.active_counter += 1
        if self.state is State.NEW and self.active_counter >= self.cfg.min_hits:
            self.state = State.CONFIRMED
        elif self.state is State.MISSING:
            self.state = State.CONFIRMED

    def mark_missed(self) -> None:
        if self.state is State.CONFIRMED:
            self.state = State.MISSING
        self.missing_counter += 1
        self.active_counter = 0
        if self.missing_counter > self.cfg.max_age:
            self.state = State.DEAD


class PointTracker:
    """Tracks head points over the frames of one clip. Use one instance per clip.

    Call :meth:`update` once per processed frame, in order. Track ids are integers,
    unique within the clip and never reused.
    """

    def __init__(self, frame_size: tuple[int, int], cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        if self.cfg.use_altitude or self.cfg.max_age_s is not None:
            raise ValueError("unresolved config: pass it through resolve_config first")
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.tracks: list[Track] = []
        self._next_id = 1
        self._initialized = False

    def _new_track(self, xy: tuple[float, float], score: float, state: State) -> None:
        self.tracks.append(Track(self._next_id, xy, score, state, self.cfg))
        self._next_id += 1

    def _gate(self) -> float:
        return float(2 * (self.cfg.size_px // 2))

    def _similarity(self, dets: np.ndarray, trks: np.ndarray) -> np.ndarray:
        """(N_det, N_trk) matrix of 1 - min(dist, gate) / gate; 0 means outside the gate."""
        gate = self._gate()
        dist = np.linalg.norm(dets[:, None, :] - trks[None, :, :], axis=-1)
        return 1.0 - np.minimum(dist, gate) / gate

    def update(self, points: np.ndarray) -> list[tuple[int, float, float, float]]:
        """Process one frame.

        Args:
            points: (N, 3) array of (x, y, score), pixels and score in [0, 1]. May be empty.

        Returns:
            Rows (track_id, x, y, score) of the confirmed tracks matched in this frame,
            with the matched detection's position.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        pts = pts[pts[:, 2] >= self.cfg.score_thr]

        if not self._initialized:
            if len(pts) == 0:
                return []
            for x, y, s in pts:
                self._new_track((float(x), float(y)), float(s), State.CONFIRMED)
            self._initialized = True
            return [(t.id, t.xy[0], t.xy[1], t.score) for t in self.tracks]

        for t in self.tracks:
            t.predict(self.frame_size)
        self.tracks = [t for t in self.tracks if t.state is not State.DEAD]

        matches: list[tuple[int, int]] = []
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        if len(pts) > 0 and len(self.tracks) > 0:
            sim = self._similarity(pts[:, :2], np.array([t.xy for t in self.tracks]))
            rows, cols = linear_sum_assignment(-sim)
            for d, k in zip(rows, cols, strict=True):
                if sim[d, k] > 0:
                    matches.append((int(d), int(k)))
                    matched_d.add(int(d))
                    matched_t.add(int(k))

        out = []
        for d, k in matches:
            t = self.tracks[k]
            x, y, s = pts[d]
            t.update((float(x), float(y)), float(s))
            if t.state is State.CONFIRMED:
                out.append((t.id, float(x), float(y), float(s)))
        for k, t in enumerate(self.tracks):
            if k not in matched_t:
                t.mark_missed()
        for d in range(len(pts)):
            if d not in matched_d:
                x, y, s = pts[d]
                self._new_track((float(x), float(y)), float(s), State.NEW)
        self.tracks = [t for t in self.tracks if t.state is not State.DEAD]
        return sorted(out)
