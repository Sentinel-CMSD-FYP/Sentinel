"""Tests for eval/benchmark_be.py: warm-up, sync, the common window, avg and p95 FPS."""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sentinel_ml.tracking.tracker import PointTracker, TrackerConfig

EVAL = Path(__file__).resolve().parents[2] / "eval"
SPEC = importlib.util.spec_from_file_location("benchmark_be", EVAL / "benchmark_be.py")
bm = importlib.util.module_from_spec(SPEC)
sys.modules["benchmark_be"] = bm
SPEC.loader.exec_module(bm)

CFG = TrackerConfig(max_age=5, min_hits=1, score_thr=0.2)
IMG = np.zeros((48, 64, 3), np.uint8)


class FakeClock:
    """Each call returns the next tick; ``step`` seconds apart."""

    def __init__(self, step: float = 0.1) -> None:
        self.t, self.step = 0.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def ft(start: float, dur: float) -> "bm.FrameTime":
    return bm.FrameTime(start, start + dur, 0.0, dur, 0.0)


def test_warmup_frames_are_run_but_not_timed_and_sync_is_called_each_stage():
    seen, syncs = [], []
    run = bm.run_stream(0, 11, lambda i: seen.append(i) or IMG, n_frames=5, warmup=3,
                        detector=lambda im: np.array([[10.0, 10.0, 0.9]]),
                        tracker=PointTracker((64, 48), CFG), sync=lambda: syncs.append(1),
                        clock=FakeClock())
    assert seen == list(range(8))        # 3 warm-up + 5 timed frames all processed
    assert len(run.frames) == 5          # only the timed ones are kept
    assert len(syncs) == 2 * 8           # after B and after E, on every frame
    assert run.error is None


def test_a_frames_stages_add_up_to_its_total():
    run = bm.run_stream(0, 1, lambda i: IMG, 2, 0, lambda im: np.zeros((0, 3)),
                        PointTracker((64, 48), CFG), lambda: None, FakeClock(0.5))
    f = run.frames[0]
    assert f.read == pytest.approx(0.5) and f.detect == pytest.approx(0.5)
    assert f.track == pytest.approx(0.5) and f.total == pytest.approx(1.5)


def test_avg_fps_is_frames_over_total_time_and_p95_is_the_slow_tail():
    # 19 frames of 0.1 s, then one of 0.5 s, back to back
    durs = [0.1] * 19 + [0.5]
    frames, t = [], 0.0
    for d in durs:
        frames.append(ft(t, d))
        t += d
    run = bm.StreamRun(0, 1, frames)
    s = bm.summarise_stream(run, (0.0, t))
    assert s["frames_used"] == 20
    assert s["avg_fps"] == pytest.approx(20 / sum(durs))                 # 20 / 2.4 = 8.33
    assert s["p95_fps"] == pytest.approx(1 / np.percentile(durs, 95))   # slow tail, below avg
    assert s["p95_fps"] < s["avg_fps"]


def test_only_frames_inside_the_common_window_are_counted():
    fast_early = bm.StreamRun(0, 1, [ft(i * 0.1, 0.1) for i in range(10)])    # runs 0 to 1 s
    slow_long = bm.StreamRun(1, 2, [ft(i * 0.5, 0.5) for i in range(10)])     # runs 0 to 5 s
    window = bm.common_window([fast_early, slow_long])
    assert window == (0.0, pytest.approx(1.0))
    assert bm.summarise_stream(slow_long, window)["frames_used"] == 2        # not all 10
    assert bm.summarise_stream(fast_early, window)["frames_used"] == 10


def test_summarise_pass_reports_the_lowest_stream_and_the_target_verdict():
    ok = bm.StreamRun(0, 1, [ft(i * 0.1, 0.1) for i in range(20)])           # 10 FPS
    slow = bm.StreamRun(1, 2, [ft(i * 0.1, 0.1) for i in range(10)] +
                        [ft(1.0 + i * 0.4, 0.4) for i in range(10)])        # avg 4 FPS
    p = bm.summarise_pass([ok, slow])
    assert p["n_streams"] == 2
    assert p["min_stream_avg_fps"] == pytest.approx(min(s["avg_fps"] for s in p["streams"]))
    assert p["avg_fps_meets_target"] is (p["min_stream_avg_fps"] >= 5.0)
    only_ok = bm.summarise_pass([ok])
    assert only_ok["avg_fps_meets_target"] is True and only_ok["total_avg_fps"] == pytest.approx(10)


def test_a_failed_stream_fails_the_pass_instead_of_reporting_numbers():
    bad = bm.StreamRun(0, 1, [], error="RuntimeError: boom")
    with pytest.raises(RuntimeError, match="boom"):
        bm.summarise_pass([bad])


def slow_detector(im):
    time.sleep(0.01)
    return np.array([[10.0, 10.0, 0.9]])


def test_run_pass_runs_streams_in_threads_with_one_shared_detector():
    clips = [11, 12, 13]
    frames = {c: [Path(f"{c}_{i}.jpg") for i in range(12)] for c in clips}
    calls = []

    def det(im):
        calls.append(1)
        return slow_detector(im)

    p = bm.run_pass(3, frames, clips, n_frames=8, warmup=2, detector=det, tracker_cfg=CFG,
                    frame_size_of=lambda c: (64, 48), fps=25.0, frame_stride=1, altitude_m=60.0,
                    sync=lambda: None, clock=time.perf_counter, read_image=lambda path: IMG)
    assert len(calls) == 3 * (2 + 8)                     # every stream ran warm-up + timed frames
    assert [s["clip"] for s in p["streams"]] == clips
    assert all(0 < s["frames_used"] <= 8 for s in p["streams"])
    assert all(s["avg_fps"] > 0 for s in p["streams"])


def test_run_pass_fails_when_one_stream_cannot_read_its_frames():
    clips = [11, 12]
    frames = {11: [Path("a.jpg")] * 6, 12: [Path("b.jpg")] * 6}

    def read(path):
        if path.name == "b.jpg":
            raise FileNotFoundError("cannot read image b.jpg")
        return IMG

    with pytest.raises(RuntimeError):
        bm.run_pass(2, frames, clips, 4, 1, lambda im: np.zeros((0, 3)), CFG, lambda c: (64, 48),
                    25.0, 1, 60.0, lambda: None, time.perf_counter, read)


def test_format_pass_shows_both_fps_and_the_target_verdict():
    run = bm.StreamRun(0, 11, [ft(i * 0.1, 0.1) for i in range(20)])
    text = bm.format_pass(bm.summarise_pass([run]))
    assert "avg FPS" in text and "p95 FPS" in text and "target 5 FPS" in text and "met" in text
