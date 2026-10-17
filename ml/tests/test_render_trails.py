"""Tests for eval/render_trails.py: trails must vanish together with their track."""

import importlib.util
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parents[2] / "eval"
SPEC = importlib.util.spec_from_file_location("render_trails", EVAL / "render_trails.py")
rt = importlib.util.module_from_spec(SPEC)
sys.modules["render_trails"] = rt
SPEC.loader.exec_module(rt)

BLANK = np.zeros((100, 200, 3), np.uint8)


def history():
    return defaultdict(lambda: deque(maxlen=40))


def test_a_track_is_drawn_with_a_trail_while_it_is_present():
    h = history()
    rt.draw_frame(BLANK, [[1, 20.0, 50.0, 0.9]], h)
    out = rt.draw_frame(BLANK, [[1, 60.0, 50.0, 0.9]], h)
    assert out[50, 40].any()  # the trail between the two positions has ink
    assert out[50, 60].any()  # and so does the current dot


def test_the_trail_of_a_track_that_is_gone_is_not_drawn():
    h = history()
    rt.draw_frame(BLANK, [[1, 20.0, 50.0, 0.9]], h)
    rt.draw_frame(BLANK, [[1, 60.0, 50.0, 0.9]], h)
    out = rt.draw_frame(BLANK, [], h)  # the track is not output in this frame
    assert not out.any()  # no ghost trail and no dot


def test_a_returning_track_keeps_its_color_and_other_tracks_are_independent():
    h = history()
    rt.draw_frame(BLANK, [[1, 20.0, 50.0, 0.9], [2, 20.0, 80.0, 0.9]], h)
    out = rt.draw_frame(BLANK, [[2, 60.0, 80.0, 0.9]], h)
    assert out[80, 40].any() and not out[50, 40].any()  # only track 2 is drawn
    assert rt.id_color(1) == rt.id_color(1) and rt.id_color(1) != rt.id_color(2)
