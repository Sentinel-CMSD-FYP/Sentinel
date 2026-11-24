import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docs" / "layers" / "add_layer.py"
if not SCRIPT.is_file():
    pytest.skip("docs/layers/add_layer.py is not available here", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("add_layer", SCRIPT)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

SIZE = [320, 240]


def make_clip(root: Path, layers=None) -> Path:
    clip = root / "clip"
    (clip / "layers").mkdir(parents=True)
    (clip / "clip.json").write_text(json.dumps({
        "schema_version": 1, "clip_id": "c1", "name": "Test clip", "fps": 25.0,
        "frame_size": SIZE, "n_frames": 100, "video_url": "/api/v1/clips/c1/video",
        "layers": layers or [], "produced_by": {"model": "farneback"},
    }))
    return clip


def base(name, kind, frames, **extra):
    return {"schema_version": 1, "clip_id": "c1", "layer": name, "type": kind,
            "frame_size": SIZE, "max_age_frames": 5, "frames": frames, **extra}


def points(name="points"):
    return base(name, "points", {"0": [[10.0, 20.0, 0.9]], "5": [[11.0, 21.0, 0.8]]})


def tracks():
    return base("tracks", "tracks", {"0": [[1, 10.0, 20.0, 0.9]]})


def density():
    return base("density", "grid", {"0": {"valid": [[1, 1, 1], [1, 1, 0]],
                                          "density": [[1, 2, 3], [4, 5, 0]]}},
                grid={"rows": 2, "cols": 3}, units="people_per_cell", fields=["density"])


def flow():
    z = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    frame = {"valid": [[1, 1, 1], [1, 1, 1]], "u": z, "v": z, "speed": z}
    return base("flow", "grid", {"0": frame}, grid={"rows": 2, "cols": 3},
                units="m_per_s", fields=["u", "v", "speed"])


def save(path: Path, doc) -> Path:
    path.write_text(json.dumps(doc))
    return path


def manifest(clip: Path) -> list[dict]:
    return json.loads((clip / "clip.json").read_text())["layers"]


def test_adds_a_layer_and_lists_it_in_the_clip(tmp_path):
    clip = make_clip(tmp_path)
    msg = tool.add_layer(clip, save(tmp_path / "p.json", points()), label="Detected heads")

    assert "added layer 'points'" in msg
    stored = json.loads((clip / "layers" / "points.json").read_text())
    assert stored == points()
    assert manifest(clip) == [
        {"layer": "points", "type": "points", "label": "Detected heads", "default_visible": False}
    ]


def test_other_clip_json_fields_are_kept(tmp_path):
    clip = make_clip(tmp_path)
    tool.add_layer(clip, save(tmp_path / "p.json", points()))
    data = json.loads((clip / "clip.json").read_text())
    assert data["produced_by"] == {"model": "farneback"}
    assert data["video_url"] == "/api/v1/clips/c1/video" and data["n_frames"] == 100


def test_a_density_grid_gets_units_a_heatmap_and_starts_visible(tmp_path):
    clip = make_clip(tmp_path)
    tool.add_layer(clip, save(tmp_path / "d.json", density()), label="Density")
    assert manifest(clip) == [{
        "layer": "density", "type": "grid", "label": "Density", "units": "people_per_cell",
        "style": {"render": "heatmap", "field": "density"}, "default_visible": True,
    }]


def test_a_grid_with_u_and_v_is_drawn_as_arrows(tmp_path):
    clip = make_clip(tmp_path)
    tool.add_layer(clip, save(tmp_path / "f.json", flow()))
    entry = manifest(clip)[0]
    assert entry["style"] == {"render": "arrows", "u": "u", "v": "v"}
    assert entry["label"] == "Flow" and entry["units"] == "m_per_s"


def test_visibility_flags_and_field_choice(tmp_path):
    clip = make_clip(tmp_path)
    tool.add_layer(clip, save(tmp_path / "t.json", tracks()), visible=True)
    tool.add_layer(clip, save(tmp_path / "d.json", density()), visible=False, field="density")
    by = {e["layer"]: e for e in manifest(clip)}
    assert by["tracks"]["default_visible"] is True
    assert by["density"]["default_visible"] is False
    with pytest.raises(tool.LayerError, match="not one of"):
        tool.add_layer(clip, save(tmp_path / "d2.json", density()), name="density2", field="nope")


def test_the_layer_name_can_be_overridden(tmp_path):
    clip = make_clip(tmp_path)
    tool.add_layer(clip, save(tmp_path / "p.json", points("raw_name")), name="heads")
    assert (clip / "layers" / "heads.json").is_file()
    assert json.loads((clip / "layers" / "heads.json").read_text())["layer"] == "heads"
    assert manifest(clip)[0]["layer"] == "heads"


def test_an_existing_layer_is_not_overwritten_without_replace(tmp_path):
    clip = make_clip(tmp_path)
    tool.add_layer(clip, save(tmp_path / "p.json", points()))
    with pytest.raises(tool.LayerError, match="already exists"):
        tool.add_layer(clip, save(tmp_path / "p.json", points()))

    newer = points()
    newer["frames"]["10"] = [[12.0, 22.0, 0.7]]
    msg = tool.add_layer(clip, save(tmp_path / "p2.json", newer), replace=True, label="Heads v2")
    assert msg.startswith("replaced layer 'points'")
    assert len(manifest(clip)) == 1 and manifest(clip)[0]["label"] == "Heads v2"
    assert "10" in json.loads((clip / "layers" / "points.json").read_text())["frames"]


def test_a_layer_made_on_a_different_resolution_is_refused_and_nothing_is_written(tmp_path):
    clip = make_clip(tmp_path)
    before = (clip / "clip.json").read_text()
    wrong = points()
    wrong["frame_size"] = [1920, 1080]
    with pytest.raises(tool.LayerError, match=r"different video or resolution.*video\.mp4"):
        tool.add_layer(clip, save(tmp_path / "p.json", wrong))
    assert (clip / "clip.json").read_text() == before
    assert not list((clip / "layers").iterdir())


def test_a_layer_for_another_clip_is_refused_unless_told_to_fix_the_id(tmp_path):
    clip = make_clip(tmp_path)
    other = points()
    other["clip_id"] = "something_else"
    with pytest.raises(tool.LayerError, match="clip_id 'something_else'"):
        tool.add_layer(clip, save(tmp_path / "p.json", other))
    tool.add_layer(clip, save(tmp_path / "p.json", other), fix_clip_id=True)
    assert json.loads((clip / "layers" / "points.json").read_text())["clip_id"] == "c1"


def test_an_invalid_layer_is_refused_with_the_reasons(tmp_path):
    clip = make_clip(tmp_path)
    bad = points()
    bad["frames"]["0"] = [[5000.0, 20.0, 0.9]]  # outside the 320x240 frame
    bad["frames"]["500"] = [[1.0, 1.0, 0.5]]  # beyond the clip's 100 frames
    with pytest.raises(tool.LayerError, match=r"not valid \(2 problems\)") as e:
        tool.add_layer(clip, save(tmp_path / "p.json", bad))
    assert "outside" in str(e.value) and "beyond the clip length 100" in str(e.value)
    assert manifest(clip) == []


@pytest.mark.parametrize("bad_name", ["Heads", "2heads", "my layer", "a-b"])
def test_a_bad_layer_name_is_refused(tmp_path, bad_name):
    clip = make_clip(tmp_path)
    with pytest.raises(tool.LayerError, match="lowercase"):
        tool.add_layer(clip, save(tmp_path / "p.json", points()), name=bad_name)


def test_a_bad_name_inside_the_file_is_refused_too(tmp_path):
    clip = make_clip(tmp_path)
    with pytest.raises(tool.LayerError, match="lowercase"):
        tool.add_layer(clip, save(tmp_path / "p.json", points("Bad Name")))
    # ...but --name can fix it without editing the file
    tool.add_layer(clip, save(tmp_path / "p.json", points("Bad Name")), name="heads")
    assert manifest(clip)[0]["layer"] == "heads"


def test_dry_run_checks_but_writes_nothing(tmp_path):
    clip = make_clip(tmp_path)
    msg = tool.add_layer(clip, save(tmp_path / "p.json", points()), dry_run=True)
    assert msg.startswith("dry run") and manifest(clip) == []
    assert not (clip / "layers" / "points.json").exists()
    bad = points()
    bad["frame_size"] = [1, 1]
    with pytest.raises(tool.LayerError):  # a dry run still refuses what a real run would
        tool.add_layer(clip, save(tmp_path / "b.json", bad), dry_run=True)


def test_missing_or_broken_inputs_are_explained(tmp_path):
    clip = make_clip(tmp_path)
    with pytest.raises(tool.LayerError, match="layer file not found"):
        tool.add_layer(clip, tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json")
    with pytest.raises(tool.LayerError, match="not valid JSON"):
        tool.add_layer(clip, broken)
    with pytest.raises(tool.LayerError, match="clip.json not found"):
        tool.add_layer(tmp_path / "empty", save(tmp_path / "p.json", points()))


def test_remove_deletes_the_file_and_the_listing(tmp_path):
    clip = make_clip(tmp_path)
    tool.add_layer(clip, save(tmp_path / "p.json", points()))
    tool.add_layer(clip, save(tmp_path / "d.json", density()))
    msg = tool.remove_layer(clip, "points")
    assert "layers now: density" in msg
    assert not (clip / "layers" / "points.json").exists()
    assert [e["layer"] for e in manifest(clip)] == ["density"]
    with pytest.raises(tool.LayerError, match="not in this clip"):
        tool.remove_layer(clip, "points")


def test_remove_dry_run_changes_nothing(tmp_path):
    clip = make_clip(tmp_path)
    tool.add_layer(clip, save(tmp_path / "p.json", points()))
    assert tool.remove_layer(clip, "points", dry_run=True).startswith("dry run")
    assert (clip / "layers" / "points.json").is_file() and len(manifest(clip)) == 1


def test_the_command_line_reports_success_and_failure(tmp_path, capsys):
    clip = make_clip(tmp_path)
    layer = save(tmp_path / "p.json", points())
    assert tool.main([str(clip), str(layer), "--label", "Heads", "--hidden"]) == 0
    assert "added layer 'points'" in capsys.readouterr().out

    assert tool.main([str(clip), str(layer)]) == 1  # already there
    assert "already exists" in capsys.readouterr().err

    assert tool.main([str(clip), "--remove", "points"]) == 0
    with pytest.raises(SystemExit):  # neither a file nor --remove
        tool.main([str(clip)])
    with pytest.raises(SystemExit):  # both at once
        tool.main([str(clip), str(layer), "--remove", "points"])


def test_a_saved_clip_still_passes_the_contract_validator(tmp_path):
    """What the backend serves is what the validator accepts."""
    validate = SCRIPT.with_name("validate.py")
    pytest.importorskip("jsonschema")
    spec = importlib.util.spec_from_file_location("validate_layers", validate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    clip = make_clip(tmp_path)
    tool.add_layer(clip, save(tmp_path / "p.json", points()))
    tool.add_layer(clip, save(tmp_path / "d.json", density()))
    tool.add_layer(clip, save(tmp_path / "f.json", flow()))
    tool.add_layer(clip, save(tmp_path / "t.json", tracks()))
    results = mod.validate_clip_dir(clip)
    assert all(not errs for errs in results.values()), results


# ---- files in the Models B and E format (docs/contracts.md v1.0) ----

SAMPLES = Path(__file__).resolve().parents[2] / "docs" / "samples"


def contract_doc(frames, **extra):
    return {"schema_version": "1.0", "clip_id": "c1", "frame_size": SIZE, "fps": 25,
            "frame_stride": 1, "altitude_m": 60.0, "altitude_source": "assumed_constant",
            "frames": frames, **extra}


def write(path: Path, doc) -> Path:
    path.write_text(json.dumps(doc))
    return path


def test_a_v1_points_file_is_added_as_it_is(tmp_path):
    clip = make_clip(tmp_path)
    f = write(tmp_path / "c1_points.json", contract_doc({"0": [[10.0, 20.0, 0.9]]}))
    tool.add_layer(clip, f)
    saved = json.loads((clip / "layers" / "points.json").read_text())
    assert (saved["type"], saved["layer"], saved["schema_version"]) == ("points", "points", 1)
    assert saved["max_age_frames"] == 1
    assert saved["altitude_m"] == 60.0 and saved["frame_stride"] == 1  # extra fields are kept
    assert json.loads((clip / "clip.json").read_text())["layers"][0]["type"] == "points"


def test_a_v1_tracks_file_is_added_as_it_is(tmp_path):
    clip = make_clip(tmp_path)
    f = write(tmp_path / "c1_tracks.json", contract_doc({"0": [[7, 10.0, 20.0, 0.9]]}))
    tool.add_layer(clip, f)
    assert json.loads((clip / "layers" / "tracks.json").read_text())["type"] == "tracks"


def test_frame_stride_sets_how_long_a_sample_stays_up(tmp_path):
    clip = make_clip(tmp_path)
    f = write(tmp_path / "c1_points.json", contract_doc({"0": [[1.0, 2.0, 1.0]]}, frame_stride=5))
    tool.add_layer(clip, f)
    assert json.loads((clip / "layers" / "points.json").read_text())["max_age_frames"] == 5


def test_max_age_overrides_the_default(tmp_path):
    clip = make_clip(tmp_path)
    f = write(tmp_path / "c1_points.json", contract_doc({"0": [[1.0, 2.0, 1.0]]}, frame_stride=5))
    tool.add_layer(clip, f, max_age=12)
    assert json.loads((clip / "layers" / "points.json").read_text())["max_age_frames"] == 12


def test_a_v1_file_needs_a_kind_when_the_name_does_not_say(tmp_path):
    clip = make_clip(tmp_path)
    f = write(tmp_path / "output.json", contract_doc({"0": [[1.0, 2.0, 1.0]]}))
    with pytest.raises(tool.LayerError, match="points or tracks"):
        tool.add_layer(clip, f)
    tool.add_layer(clip, f, kind="points")


def test_an_unknown_contract_version_is_refused(tmp_path):
    clip = make_clip(tmp_path)
    f = write(tmp_path / "c1_points.json", contract_doc({"0": []}, schema_version="2.0"))
    with pytest.raises(tool.LayerError, match="2.0"):
        tool.add_layer(clip, f)


def test_a_v1_file_still_gets_the_usual_checks(tmp_path):
    clip = make_clip(tmp_path)
    wrong_clip = contract_doc({"0": [[1.0, 2.0, 1.0]]}, clip_id="other")
    with pytest.raises(tool.LayerError, match="clip_id"):
        tool.add_layer(clip, write(tmp_path / "a_points.json", wrong_clip))
    beyond = contract_doc({"500": [[1.0, 2.0, 1.0]]})
    with pytest.raises(tool.LayerError, match="beyond"):
        tool.add_layer(clip, write(tmp_path / "b_points.json", beyond))
    assert not (clip / "layers" / "points.json").exists()


@pytest.mark.parametrize("sample, kind", [("clip_011_points.json", "points"),
                                          ("clip_011_tracks.json", "tracks")])
def test_the_real_contract_samples_load_into_a_clip(tmp_path, sample, kind):
    if not (SAMPLES / sample).is_file():
        pytest.skip("contract samples are not available here")
    clip = tmp_path / "dc_011"
    (clip / "layers").mkdir(parents=True)
    (clip / "clip.json").write_text(json.dumps({
        "schema_version": 1, "clip_id": "dc_011", "name": "DroneCrowd 011", "fps": 25.0,
        "frame_size": [1920, 1080], "n_frames": 300, "layers": []}))
    tool.add_layer(clip, SAMPLES / sample, fix_clip_id=True)
    saved = json.loads((clip / "layers" / f"{kind}.json").read_text())
    assert saved["type"] == kind and saved["clip_id"] == "dc_011"
    assert json.loads((clip / "clip.json").read_text())["layers"][0]["layer"] == kind
