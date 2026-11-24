"""The ``polygons`` layer type (Model C) in the layer tools of ``docs/layers`` (step 17)."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

LAYERS = Path(__file__).resolve().parents[2] / "docs" / "layers"
if not (LAYERS / "add_layer.py").is_file():
    pytest.skip("docs/layers is not available here", allow_module_level=True)


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"layers_{name}", LAYERS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


add_layer, check_layer = load("add_layer"), load("check_layer")

SIZE = [320, 240]
OBJ = {"cls": "wall", "score": 0.9, "box": [10.0, 20.0, 320.0, 60.0],
       "polygon": [[10.0, 20.0], [319.0, 20.0], [319.0, 59.5]]}


def layer(**frames):
    return {"schema_version": 1, "clip_id": "c1", "layer": "infrastructure", "type": "polygons",
            "frame_size": SIZE, "classes": ["gate_exit", "barrier", "wall", "obstruction"],
            "max_age_frames": 125, "frames": frames or {"0": [OBJ], "125": []}}


def broken(**changes):
    obj = copy.deepcopy(OBJ) | changes
    return layer(**{"0": [obj]})


def test_a_valid_polygons_layer_passes():
    assert check_layer.check_layer(layer(), n_frames=300) == []


@pytest.mark.parametrize("doc, problem", [
    (broken(cls="tree"), "not one of classes"),
    (broken(score=1.5), "score"),
    (broken(box=[10, 20, 321, 60]), "box"),
    (broken(box=[50, 20, 10, 60]), "box"),
    (broken(polygon=[[1, 1], [2, 2]]), "at least 3"),
    (broken(polygon=[[1, 1], [2, 2], [320, 5]]), "outside"),
    (layer(**{"0": [{"cls": "wall", "score": 0.5}]}), "exactly cls, score, box and polygon"),
    (layer(**{"0": {"cls": "wall"}}), "list of objects"),
    (layer() | {"classes": []}, "classes"),
    (layer() | {"classes": ["wall", "wall"]}, "classes"),
])
def test_bad_polygons_layers_are_explained(doc, problem):
    errs = check_layer.check_layer(doc)
    assert any(problem in e for e in errs), errs


def test_add_layer_stores_it_and_the_whole_clip_validates(tmp_path):
    pytest.importorskip("jsonschema")
    validate = load("validate")
    clip = tmp_path / "clip"
    (clip / "layers").mkdir(parents=True)
    (clip / "clip.json").write_text(json.dumps({
        "schema_version": 1, "clip_id": "c1", "fps": 25.0, "frame_size": SIZE,
        "n_frames": 300, "layers": []}))
    src = tmp_path / "infra.json"
    src.write_text(json.dumps(layer()))

    add_layer.add_layer(clip, src, label="Infrastructure")

    entry = json.loads((clip / "clip.json").read_text())["layers"][0]
    assert entry == {"layer": "infrastructure", "type": "polygons", "label": "Infrastructure",
                     "default_visible": False}
    results = validate.validate_clip_dir(clip)
    assert all(not errs for errs in results.values()), results


def test_the_validator_catches_what_the_schema_cannot():
    pytest.importorskip("jsonschema")
    validate = load("validate")
    assert validate.validate_layer(layer(), 300) == []
    errs = validate.validate_layer(broken(cls="tree"), 300)
    assert errs == ["frame 0: cls 'tree' is not one of classes"]
    errs = validate.validate_layer(broken(polygon=[[1, 1], [2, 2], [5, 240]]), 300)
    assert errs == ["frame 0: polygon point outside 320x240"]
