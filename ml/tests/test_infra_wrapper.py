from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torchvision")

from sentinel_ml.infrastructure import (  # noqa: E402
    InfraConfig,
    InfraModel,
    class_nms,
    infrastructure_layer,
    largest_part,
    simplify_polygon,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "ml" / "configs" / "infrastructure_yolo26.yaml"
SAMPLE = REPO / "data" / "samples" / "dronecrowd_img100001.jpg"
CLASSES = ("gate_exit", "barrier", "wall", "obstruction")


def test_nms_drops_same_class_duplicates_only():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 100, 100], [5, 5, 100, 100], [300, 300, 400, 400]])
    scores = np.array([0.6, 0.9, 0.5, 0.7])
    cls = np.array([2, 2, 3, 2])
    keep = class_nms(boxes, scores, cls, 0.5)
    # Box 0 overlaps the better wall box 1; box 2 overlaps too but is another class.
    assert keep.tolist() == [1, 3, 2]


def test_simplify_polygon():
    square = np.array([[10, 10], [50, 10], [90, 10], [90, 90], [10, 90]], np.float32)
    poly = simplify_polygon(square, 1.0, (1920, 1080))
    assert len(poly) == 4  # the point on the straight edge is removed
    assert simplify_polygon(square[:2], 1.0, (1920, 1080)) is None
    outside = np.array([[-5, -5], [2000, 0], [2000, 1200]], np.float32)
    clipped = simplify_polygon(outside, 1.0, (1920, 1080))
    assert clipped.min() >= 0 and clipped[:, 0].max() <= 1919 and clipped[:, 1].max() <= 1079


class FakeModel:
    config = InfraConfig(weights="none", classes=CLASSES, every=125, max_age_frames=125)

    def predict(self, frame):
        return [{"cls": "wall", "score": 0.9, "box": [0, 0, 10, 10],
                 "polygon": [[0, 0], [10, 0], [10, 10]]}]


def test_layer_shape():
    frames = ((i, np.zeros((1080, 1920, 3), np.uint8)) for i in range(300))
    layer = infrastructure_layer(frames, FakeModel(), "seq011")
    assert list(layer["frames"]) == ["0", "125", "250"]
    assert (layer["type"], layer["layer"]) == ("polygons", "infrastructure")
    assert layer["schema_version"] == 1
    assert layer["frame_size"] == [1920, 1080] and layer["classes"] == list(CLASSES)
    assert layer["max_age_frames"] == 125
    assert layer["frames"]["0"][0]["cls"] == "wall"


def test_config_file_loads():
    cfg = InfraConfig.from_yaml(CONFIG)
    assert cfg.classes == CLASSES
    assert (cfg.imgsz, cfg.nms_iou, cfg.every) == (1280, 0.5, 125)


@pytest.mark.skipif(not SAMPLE.exists(), reason="needs data/samples/dronecrowd_img100001.jpg")
def test_real_model_on_a_dronecrowd_frame():
    pytest.importorskip("ultralytics")
    cfg = InfraConfig.from_yaml(CONFIG)
    weights = REPO / cfg.weights
    if not weights.exists():
        pytest.skip(f"needs {cfg.weights}")
    model = InfraModel(InfraConfig(**{**cfg.__dict__, "weights": str(weights), "device": "cpu"}))
    frame = cv2.imread(str(SAMPLE))
    objs = model.predict(frame)
    assert objs, "Model C finds stalls and buildings on this market frame"
    h, w = frame.shape[:2]
    scores = [o["score"] for o in objs]
    assert scores == sorted(scores, reverse=True)
    for o in objs:
        assert o["cls"] in CLASSES and cfg.conf <= o["score"] <= 1
        assert len(o["polygon"]) >= 3
        assert all(0 <= x < w and 0 <= y < h for x, y in o["polygon"])
    assert all(o["box"] == [round(v, cfg.decimals) for v in o["box"]] for o in objs)


def test_largest_part_keeps_the_biggest_blob():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:20, 10:20] = 1  # small part
    mask[40:90, 30:80] = 1  # big part
    part = largest_part(mask)
    assert part[:, 0].min() == 30 and part[:, 1].max() == 89
    assert largest_part(np.zeros((5, 5), np.uint8)).shape == (0, 2)
