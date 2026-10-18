"""Tests for eval/oracle_gap.py: the gap computation and the 0.15 decision rule."""

import importlib.util
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"
SPEC = importlib.util.spec_from_file_location("oracle_gap", EVAL / "oracle_gap.py")
og = importlib.util.module_from_spec(SPEC)
sys.modules["oracle_gap"] = og
SPEC.loader.exec_module(og)


def tracker_block(idf1_by_clip):
    per = {f"{c:03d}": {"IDF1": v, "HOTA": 0.3, "DetA": 0.25, "AssA": 0.38, "GT_IDs": 100}
           for c, v in idf1_by_clip.items()}
    mean = sum(idf1_by_clip.values()) / len(idf1_by_clip)
    return {"per_clip": per, "totals": {"IDSW": 10},
            "mean_over_clips": {"IDF1": mean, "HOTA": 0.3, "DetA": 0.25, "AssA": 0.38}}


def results(tracker, oracle):
    return {"iou_0.5": {"run2": tracker_block(tracker), "oracle": tracker_block(oracle)}}


def test_gain_is_oracle_minus_tracker_overall_and_per_clip():
    g = og.gap(results({11: 0.4, 15: 0.2}, {11: 0.7, 15: 0.5}), "run2", "oracle")
    assert g["idf1_tracker"] == pytest.approx(0.3) and g["idf1_oracle"] == pytest.approx(0.6)
    assert g["gain"] == pytest.approx(0.3)
    assert [r["clip"] for r in g["per_clip"]] == [11, 15]
    assert g["per_clip"][0]["gain"] == pytest.approx(0.3)


def test_the_rule_builds_only_at_a_gain_of_at_least_0_15():
    big = og.gap(results({11: 0.40}, {11: 0.56}), "run2", "oracle")      # +0.16
    exact = og.gap(results({11: 0.40}, {11: 0.55}), "run2", "oracle")    # +0.15
    small = og.gap(results({11: 0.40}, {11: 0.54}), "run2", "oracle")    # +0.14
    assert og.decide(big, 0.15) is True
    assert og.decide(exact, 0.15) is True   # "at least 0.15" includes exactly 0.15
    assert og.decide(small, 0.15) is False


def test_report_states_the_label_and_the_verdict():
    g = og.gap(results({11: 0.4}, {11: 0.5}), "run2", "oracle")
    text = og.format_report(g, 0.15)
    assert "oracle on test, measurement only" in text and "DO NOT BUILD" in text
    big = og.gap(results({11: 0.1}, {11: 0.5}), "run2", "oracle")
    build = og.format_report(big, 0.15)
    assert build.endswith("-> BUILD") and "DO NOT" not in build.splitlines()[-1]
