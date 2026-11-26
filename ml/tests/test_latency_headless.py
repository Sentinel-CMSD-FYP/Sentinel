"""Tests for eval/pipeline/latency_headless.py (per-minute figures; no browser needed)."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "pipeline" / "latency_headless.py"
SPEC = importlib.util.spec_from_file_location("latency_headless", SCRIPT)
lh = importlib.util.module_from_spec(SPEC)
sys.modules["latency_headless"] = lh
SPEC.loader.exec_module(lh)


def rec(layer, capture, delay):
    return {"layer": layer, "tCapture": capture, "tDrawn": capture + delay}


def test_samples_are_grouped_by_layer_and_minute_from_the_first_picture():
    drawn = [rec("picture", 100.0, 0.03), rec("points", 101.0, 1.0), rec("points", 159.0, 3.0),
             rec("points", 130.0, 2.0), rec("points", 161.0, 5.0), rec("points", 99.0, 4.0)]
    got = lh.per_minute(drawn)["points"]
    assert [(g["minute"], g["n"]) for g in got] == [(-1, 1), (0, 3), (1, 1)]
    assert got[1]["median"] == 2.0 and got[1]["max"] == 3.0


def test_without_a_drawn_picture_there_is_nothing_to_report():
    assert lh.per_minute([rec("points", 1.0, 1.0)]) == {}
