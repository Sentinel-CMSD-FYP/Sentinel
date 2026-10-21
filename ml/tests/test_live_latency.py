"""Tests for eval/pipeline/live_latency.py (worker side live latency summary)."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "pipeline" / "live_latency.py"
SPEC = importlib.util.spec_from_file_location("live_latency", SCRIPT)
live_latency = importlib.util.module_from_spec(SPEC)
sys.modules["live_latency"] = live_latency
SPEC.loader.exec_module(live_latency)


def stream(folder: Path, sid: str, samples: list[tuple[str, float, float]]) -> Path:
    folder.mkdir()
    lines = [{"seq": i, "layer": layer, "frame": i, "t_capture": tc, "t_ready": tr, "data": None}
             for i, (layer, tc, tr) in enumerate(samples)]
    (folder / "samples.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (folder / "status.json").write_text(json.dumps({
        "stream_id": sid, "state": "done", "frames_read": 10, "ran": {"density": 2},
        "skipped": {"density": 8}, "late_s": 0.0, "errors": {}}))
    return folder


def test_only_samples_captured_while_every_stream_was_live_count(tmp_path):
    a = stream(tmp_path / "a", "a", [("density", 0, 1), ("density", 10, 12), ("flow", 20, 20.5)])
    b = stream(tmp_path / "b", "b", [("density", 5, 9), ("flow", 15, 15.5), ("density", 30, 40)])
    r = live_latency.summarize([a, b])
    assert r["overlap_s"] == 15  # a is live 0 to 20, b 5 to 30
    assert r["ready"]["density"] == {"n": 2, "median": 4, "p90": 4, "max": 4}  # 10->12, 5->9
    assert r["ready"]["flow"]["n"] == 2
    assert [w["id"] for w in r["workers"]] == ["a", "b"]
    md = live_latency.table([r])
    assert "| 2 | 15 | 4 / 4 (n 2) | 0.5 / 0.5 (n 2) |" in md
