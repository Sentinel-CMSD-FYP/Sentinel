"""Tests for eval/pipeline/fps_bench.py helpers (timing wrapper, stats, table)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "pipeline" / "fps_bench.py"
SPEC = importlib.util.spec_from_file_location("fps_bench", SCRIPT)
fb = importlib.util.module_from_spec(SPEC)
sys.modules["fps_bench"] = fb
SPEC.loader.exec_module(fb)


class Cfg:
    every = 5


class FakeDensity:
    name = "density"
    cfg = Cfg()

    def __init__(self):
        self.seen = []

    def process(self, frame_id, frame):
        self.seen.append(frame_id)

    def layer(self, clip_id):
        return {"frames": self.seen}


def test_stats_known_answer():
    s = fb.stats([10.0, 20.0, 30.0, 40.0])
    assert s["runs"] == 4 and s["mean_ms"] == 25.0 and s["p50_ms"] == 25.0
    assert s["fps"] == 40.0
    assert fb.stats([]) == {"runs": 0}


def test_timed_passes_every_frame_but_times_only_model_runs():
    pytest.importorskip("torch")
    inner = FakeDensity()
    t = fb.Timed(inner)
    for fid in range(12):
        t.process(fid, np.zeros((2, 2, 3), np.uint8))
    assert inner.seen == list(range(12))
    assert len(t.ms) == 3  # frames 0, 5, 10
    assert t.layer("c") == {"frames": list(range(12))}


def test_table_collects_model_and_stream_runs(tmp_path):
    model = {"kind": "model", "module": "density", "fp16": True, "device": "Tesla T4", "runs": 220,
             "mean_ms": 100.0, "p50_ms": 99.0, "p95_ms": 110.0, "fps": 10.0,
             "max_allocated_mb": 900}
    stream = {"stream": 0, "decode": {"mean_ms": 12.0}, "video_fps": 30.0}
    streams = {"kind": "streams", "n": 2, "density_fp16": True, "video_fps_min": 30.0,
               "video_fps_mean": 31.0, "keeps_up": True, "density_mean_ms": 150.0,
               "infrastructure_mean_ms": 40.0, "gpu_mem_loaded_mb": 2000.0,
               "gpu_mem_peak_mb": 2500.0, "gpu_util_mean_pct": 60.0,
               "streams": [stream, {**stream, "stream": 1}]}
    for name, r in (("m", model), ("s", streams)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps(r))
    fb.cmd_table(type("A", (), {"dir": tmp_path})())
    text = (tmp_path / "bench.md").read_text()
    assert "| density | True | Tesla T4 | 220 | 100.0 / 99.0 / 110.0 | 10.0 | 900 |" in text
    assert ("| 2 | True | 30.0 / 31.0 | True | 150.0 | 40.0 | 12.0 | 2000.0 / 2500.0 | 60.0 |"
            in text)
