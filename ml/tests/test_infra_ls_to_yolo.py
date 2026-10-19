"""Tests for eval/infrastructure_yolo26/ls_to_yolo.py with small Label Studio exports."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "infrastructure_yolo26" / "ls_to_yolo.py"
SPEC = importlib.util.spec_from_file_location("infra_ls_to_yolo", SCRIPT)
conv = importlib.util.module_from_spec(SPEC)
sys.modules["infra_ls_to_yolo"] = conv
SPEC.loader.exec_module(conv)

CLASSES = ["gate_exit", "barrier", "wall", "obstruction"]
A = "/data/local-files/?d=images/a.jpg"
TRI = [[0, 0], [1, 0], [1, 1]]


def poly(label, points):
    return {"type": "polygonlabels", "id": "r",
            "value": {"points": points, "polygonlabels": [label], "closed": True}}


def task(image, *annotations):
    return {"data": {"image": image}, "annotations": list(annotations)}


def ann(result, updated="2026-10-19T10:00:00", cancelled=False):
    return {"result": result, "updated_at": updated, "was_cancelled": cancelled}


def test_percent_points_become_normalised_lines():
    t = task("/data/local-files/?d=images/img001150.jpg",
             ann([poly("wall", [[10, 20], [50, 20], [50, 100]]),
                  poly("barrier", [[0, 0], [100, 0], [100, 5]])]))
    assert conv.convert([t], CLASSES) == {"img001150": [
        "2 0.100000 0.200000 0.500000 0.200000 0.500000 1.000000",
        "1 0.000000 0.000000 1.000000 0.000000 1.000000 0.050000"]}


def test_points_outside_image_are_clipped():
    t = task(A, ann([poly("gate_exit", [[-1, 50], [101, 50], [50, 100.2]])]))
    expected = "0 0.000000 0.500000 1.000000 0.500000 0.500000 1.000000"
    assert conv.convert([t], CLASSES)["a"] == [expected]


def test_upload_prefix_is_removed_and_windows_paths_work():
    assert conv.image_name(task("/data/upload/3/1a2b3c4d-img070150.jpg")) == "img070150.jpg"
    assert conv.image_name(task("/data/local-files/?d=images%5Cimg070150.jpg")) == "img070150.jpg"


def test_empty_submission_is_background_and_skipped_task_has_no_file():
    tasks = [task("/data/local-files/?d=images/bg.jpg", ann([])),
             task("/data/local-files/?d=images/skip.jpg", ann([], cancelled=True)),
             task("/data/local-files/?d=images/todo.jpg")]
    assert conv.convert(tasks, CLASSES) == {"bg": []}


def test_newest_annotation_wins():
    t = task(A,
             ann([poly("wall", TRI)], updated="2026-10-19T09:00:00"),
             ann([poly("obstruction", TRI)], updated="2026-10-19T11:00:00"))
    assert conv.convert([t], CLASSES)["a"][0].startswith("3 ")


def test_bad_input_raises():
    with pytest.raises(ValueError, match="unknown label"):
        conv.convert([task(A, ann([poly("tree", TRI)]))], CLASSES)
    with pytest.raises(ValueError, match="2 points"):
        conv.convert([task(A, ann([poly("wall", [[0, 0], [1, 0]])]))], CLASSES)
    dup = task(A, ann([]))
    with pytest.raises(ValueError, match="more than one task"):
        conv.convert([dup, dup], CLASSES)
