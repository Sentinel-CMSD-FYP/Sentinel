import threading
import time

import cv2
import numpy as np
import pytest

from sentinel_ml.flow.layer import FlowLayerConfig, FlowModule
from sentinel_ml.pipeline import run_clip
from sentinel_ml.pipeline.live import chains, open_source, run_live


class Sampler:
    """Writes one sample per wanted frame (its frame id), optionally taking ``cost`` seconds."""

    def __init__(self, name, every=1, cost=0.0, fail_at=None, needs=()):
        self.name, self.every, self.cost, self.fail_at = name, every, cost, fail_at
        self.needs = needs
        self.seen = []
        self._frames = {}

    def wants(self, frame_id):
        return frame_id % self.every == 0

    @property
    def samples(self):
        return self._frames

    def process(self, frame_id, frame):
        if not self.wants(frame_id):
            return
        if frame_id == self.fail_at:
            raise RuntimeError("boom")
        time.sleep(self.cost)
        self.seen.append(frame_id)
        self._frames[str(frame_id)] = frame_id

    def layer(self, clip_id):
        return {"clip_id": clip_id, "layer": self.name, "type": "points", "max_age_frames": 1,
                "frames": self._frames}


def frames(n, size=(6, 4)):
    return (np.full((size[1], size[0], 3), i % 256, np.uint8) for i in range(n))


def collect():
    got = []
    lock = threading.Lock()

    def on_sample(layer, frame_id, data, header, t_capture):
        with lock:
            got.append((layer, frame_id, data, header, t_capture))

    return got, on_sample


def test_fast_modules_get_every_wanted_frame_and_samples_come_in_order():
    a, b = Sampler("a"), Sampler("b", every=5)
    got, on_sample = collect()
    res = run_live(frames(20), [a, b], "live_x", fps=1000, on_sample=on_sample)
    assert res.n_frames == 20 and res.errors == {}
    assert a.seen == list(range(20)) and b.seen == [0, 5, 10, 15]
    assert [f for name, f, *_ in got if name == "a"] == list(range(20))
    assert [d for name, _, d, *_ in got if name == "b"] == [0, 5, 10, 15]
    assert res.layers["b"]["frames"] == {"0": 0, "5": 5, "10": 10, "15": 15}
    assert res.ran == {"a": 20, "b": 4} and res.skipped == {"a": 0, "b": 0}


def test_the_header_comes_once_without_frames_and_capture_times_are_set():
    got, on_sample = collect()
    run_live(frames(3), [Sampler("a")], "live_x", fps=1000, on_sample=on_sample)
    headers = [h for *_, h, _ in got]
    assert headers[0] == {"clip_id": "live_x", "layer": "a", "type": "points",
                          "max_age_frames": 1}
    assert headers[1:] == [None, None]
    assert all(isinstance(t, float) and t > 0 for *_, t in got)


def test_a_slow_module_skips_to_recent_frames_and_does_not_hold_up_a_fast_one():
    fast, slow = Sampler("fast"), Sampler("slow", cost=0.1)
    res = run_live(frames(60), [fast, slow], "live_x", fps=100, max_lag_s=0.05,
                   on_sample=lambda *a: None)
    assert fast.seen == list(range(60))  # its own thread, so the slow one costs it nothing
    assert 2 <= len(slow.seen) < 20 and res.skipped["slow"] > 0
    assert slow.seen == sorted(slow.seen) and slow.seen[-1] >= 50  # still reaches the end
    assert list(res.layers["slow"]["frames"]) == [str(f) for f in slow.seen]


def test_at_the_end_a_slow_module_skips_its_backlog_to_the_last_frame():
    slow = Sampler("slow", cost=0.1)
    run_live(frames(30), [slow], "live_x", fps=1000, max_lag_s=0.05)
    assert slow.seen == [0, 29]  # reading ended after ~30 ms; frames 1 to 28 waited too long


def test_a_file_is_read_at_its_own_pace():
    t = time.monotonic()
    res = run_live(frames(11), [Sampler("a")], "live_x", fps=50)
    assert time.monotonic() - t >= 0.19 and res.n_frames == 11
    t = time.monotonic()
    run_live(frames(11), [Sampler("a")], "live_x", fps=50, realtime=False)
    assert time.monotonic() - t < 0.19


def test_a_dependent_runs_in_the_chain_of_what_it_needs_and_fails_with_it():
    p, t, other = Sampler("p", fail_at=3), Sampler("t", needs=("p",)), Sampler("o")
    assert [[m.name for m in g] for g in chains([p, t, other])] == [["p", "t"], ["o"]]
    res = run_live(frames(8), [p, t, other], "live_x", fps=1000)
    assert res.errors["p"] == "process (frame 3) failed: RuntimeError: boom"
    assert res.errors["t"] == "needs ['p'], which failed"
    assert list(res.layers) == ["o"] and other.seen == list(range(8))
    assert p.seen == [0, 1, 2] and t.seen == [0, 1, 2]  # t stops on the frame p failed on


def test_needs_must_come_first():
    with pytest.raises(ValueError, match="needs"):
        chains([Sampler("t", needs=("p",)), Sampler("p")])


def test_stop_ends_the_reading_early_and_frames_go_to_on_frame():
    seen = []
    res = run_live(frames(100), [Sampler("a")], "live_x", fps=1000,
                   on_frame=lambda fid, frame, t: seen.append(fid), stop=lambda: len(seen) >= 7)
    assert res.n_frames == 7 and seen == list(range(7)) and list(res.layers["a"]["frames"]) == [
        str(i) for i in range(7)]


def test_a_failing_reader_is_raised_after_the_chains_finish():
    def bad():
        yield np.zeros((4, 6, 3), np.uint8)
        raise OSError("camera gone")

    a = Sampler("a")
    with pytest.raises(OSError, match="camera gone"):
        run_live(bad(), [a], "live_x", fps=1000)
    assert a.seen == [0]


def sliding(n, dx=2, h=240, w=320):
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 2.0)
    base = base.astype(np.uint8)
    for i in range(n):
        m = np.float32([[1, 0, dx * i], [0, 1, 0]])
        yield cv2.cvtColor(cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT),
                           cv2.COLOR_GRAY2BGR)


def test_the_flow_module_live_gives_the_same_layer_as_offline():
    cfg = FlowLayerConfig(every=10, gap=3, stabilise=False)
    offline = run_clip(enumerate(sliding(30)), [FlowModule(cfg)], "c")["flow"]
    live_module = FlowModule(cfg)
    assert [f for f in range(30) if live_module.wants(f)] == [0, 3, 10, 13, 20, 23]
    got, on_sample = collect()
    res = run_live(sliding(30), [live_module], "c", fps=1000, on_sample=on_sample)
    assert res.layers["flow"] == offline and [f for _, f, *_ in got] == [0, 10, 20]


def test_flow_forgets_a_first_frame_whose_pair_never_came():
    m = FlowModule(FlowLayerConfig(every=10, gap=3, stabilise=False))
    frames_ = list(sliding(14))
    m.process(0, frames_[0])
    m.process(10, frames_[10])  # frame 3 was skipped
    m.process(13, frames_[13])
    assert m._pending == {} and list(m.samples) == ["10"]


def test_open_source_reads_a_file_with_realtime_set(tmp_path):
    path = tmp_path / "v.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (32, 24))
    for i in range(5):
        out.write(np.full((24, 32, 3), i * 40, np.uint8))
    out.release()
    frames_, fps, size, realtime = open_source(str(path))
    assert (fps, size, realtime, len(list(frames_))) == (20.0, (32, 24), True, 5)
    with pytest.raises(ValueError, match="cannot open"):
        open_source(str(tmp_path / "missing.mp4"))
