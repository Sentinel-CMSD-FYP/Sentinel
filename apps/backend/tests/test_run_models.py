"""``manage.py run_models``: modules run on a clip's video and their layers land in the clip.

Needs ``sentinel_ml`` and OpenCV (the repo .venv); skipped where only the API is installed.
"""

import json

import pytest
from django.core.management import CommandError, call_command

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("sentinel_ml.pipeline")

from sentinel_ml.pipeline import registry  # noqa: E402

SIZE = (64, 48)  # width, height
N_FRAMES = 10


class FakeDensity:
    """Counts the frames it sees and writes one 1x1 grid sample every 5 frames."""

    def __init__(self, name="density"):
        self.name = name
        self.seen = []

    def process(self, frame_id, frame):
        self.seen.append(frame_id)

    def layer(self, clip_id):
        return {"schema_version": 1, "clip_id": clip_id, "layer": self.name, "type": "grid",
                "frame_size": list(SIZE), "grid": {"rows": 1, "cols": 1},
                "units": "people_per_cell", "fields": ["density"], "max_age_frames": 5,
                "frames": {str(f): {"valid": [[1]], "density": [[float(f)]]}
                           for f in self.seen if f % 5 == 0}}


def write_video(path, n, size=SIZE):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, size)
    for i in range(n):
        out.write(np.full((size[1], size[0], 3), i * 20, np.uint8))
    out.release()


@pytest.fixture
def clip(tmp_path, settings):
    root = tmp_path / "clips"
    clip_dir = root / "demo"
    (clip_dir / "layers").mkdir(parents=True)
    (clip_dir / "clip.json").write_text(json.dumps({
        "schema_version": 1, "clip_id": "demo", "name": "Demo", "fps": 25.0,
        "frame_size": list(SIZE), "n_frames": N_FRAMES, "layers": []}))
    write_video(clip_dir / "video.mp4", N_FRAMES + 3)  # longer than clip.json says
    settings.CLIPS_ROOT = root
    return clip_dir


@pytest.fixture
def fake(monkeypatch):
    made = []

    def build(device):
        made.append(FakeDensity())
        return made[-1]

    monkeypatch.setitem(registry.BUILDERS, "fake_density", build)
    return made


def manifest(clip_dir):
    return json.loads((clip_dir / "clip.json").read_text())["layers"]


def test_runs_a_module_and_adds_its_layer(clip, fake, capsys):
    call_command("run_models", "demo", "--modules", "fake_density")

    assert fake[0].seen == list(range(N_FRAMES))  # stops at n_frames, every frame once
    doc = json.loads((clip / "layers" / "density.json").read_text())
    assert doc["clip_id"] == "demo" and list(doc["frames"]) == ["0", "5"]
    assert [e["layer"] for e in manifest(clip)] == ["density"]
    assert "added layer 'density'" in capsys.readouterr().out


def test_the_api_serves_the_new_layer(clip, fake, client):
    call_command("run_models", "demo", "--modules", "fake_density")
    resp = client.get("/api/v1/clips/demo/layers/density")
    assert resp.status_code == 200 and resp.json()["frames"]["5"]["density"] == [[5.0]]


def test_an_existing_layer_needs_replace(clip, fake):
    call_command("run_models", "demo", "--modules", "fake_density")
    with pytest.raises(CommandError, match="already has layers \\['density'\\]; use --replace"):
        call_command("run_models", "demo", "--modules", "fake_density")
    assert len(fake) == 2 and fake[1].seen == []  # refused before any frame was decoded
    call_command("run_models", "demo", "--modules", "fake_density", "--replace")
    assert [e["layer"] for e in manifest(clip)] == ["density"]


def test_unknown_module_lists_the_available_ones(clip):
    with pytest.raises(CommandError, match="unknown modules \\['nope'\\]; available: density"):
        call_command("run_models", "demo", "--modules", "nope")


def test_the_real_flow_module_adds_an_arrows_layer(clip):
    """Flow needs no weights, so the registered module runs here as it does on a real clip."""
    call_command("run_models", "demo", "--modules", "flow")
    doc = json.loads((clip / "layers" / "flow.json").read_text())
    assert doc["type"] == "grid" and doc["fields"] == ["u", "v", "speed"]
    assert list(doc["frames"]) == ["0"]  # every 25 frames, the pair needs frame 5 (10 frames)
    entry = manifest(clip)[0]
    assert entry["layer"] == "flow" and entry["style"]["render"] == "arrows"
    assert entry["default_visible"] is False  # registry.SHOWN, not add_layer's grid default
    made = json.loads((clip / "clip.json").read_text())["produced_by"]["flow"]
    assert made["model"] == "farneback" and made["every"] == 25 and made["frames"] == N_FRAMES


def test_a_clip_camera_makes_flow_metres_per_second(clip):
    doc = json.loads((clip / "clip.json").read_text())
    doc["camera"] = {"altitude_m": 60.0, "hfov_deg": 71.53, "altitude_source": "assumed"}
    (clip / "clip.json").write_text(json.dumps(doc))
    call_command("run_models", "demo", "--modules", "flow")
    assert json.loads((clip / "layers" / "flow.json").read_text())["units"] == "m_per_s"
    assert manifest(clip)[0]["units"] == "m_per_s"
    made = json.loads((clip / "clip.json").read_text())["produced_by"]["flow"]
    assert made["units"] == "m_per_s"


def test_produced_by_keeps_other_layers_and_a_flat_export_entry(clip, monkeypatch):
    doc = json.loads((clip / "clip.json").read_text())
    doc["produced_by"] = {"model": "farneback", "gap_frames": 5}  # as flow/export.py writes it
    (clip / "clip.json").write_text(json.dumps(doc))
    fake_module = FakeDensity()
    fake_module.produced_by = {"model": "fake", "weights": "w.pth"}
    monkeypatch.setitem(registry.BUILDERS, "fake_density", lambda device: fake_module)
    call_command("run_models", "demo", "--modules", "fake_density")
    made = json.loads((clip / "clip.json").read_text())["produced_by"]
    assert made["flow"] == {"model": "farneback", "gap_frames": 5}
    entry = made["density"]  # keyed by layer name
    assert entry["model"] == "fake" and entry["weights"] == "w.pth"
    assert entry["frames"] == N_FRAMES and entry["seconds"] >= 0 and len(entry["date"]) == 10


def test_all_runs_every_registered_module(clip, fake, monkeypatch):
    monkeypatch.setattr(registry, "BUILDERS", {"fake_density": registry.BUILDERS["fake_density"]})
    call_command("run_models", "demo", "--modules", "all")
    assert [e["layer"] for e in manifest(clip)] == ["density"] and fake[0].seen


def test_tracks_brings_its_points_module_and_the_clip_fps(clip, monkeypatch):
    """Asking for tracks runs one points module first; the real tracker links its rows."""
    from sentinel_ml.localization.layer import PointsModule

    def fake_points(device):
        return PointsModule(lambda rgb: np.array([[20.0, 20.0, 0.9], [40.0, 30.0, 0.8]]))

    monkeypatch.setitem(registry.BUILDERS, "points", fake_points)
    call_command("run_models", "demo", "--modules", "tracks")
    assert [e["layer"] for e in manifest(clip)] == ["points", "tracks"]
    doc = json.loads((clip / "layers" / "tracks.json").read_text())
    assert doc["type"] == "tracks" and doc["camera_motion_compensated"] is False
    assert len(doc["frames"]) == N_FRAMES
    assert all([r[0] for r in rows] == [1, 2] for rows in doc["frames"].values())
    with pytest.raises(CommandError, match="already has layers \\['points', 'tracks'\\]"):
        call_command("run_models", "demo", "--modules", "tracks")


def test_unknown_clip_or_missing_video(clip, fake):
    with pytest.raises(CommandError, match="no clip 'nope'"):
        call_command("run_models", "nope", "--modules", "fake_density")
    (clip / "video.mp4").unlink()
    with pytest.raises(CommandError, match="has no video"):
        call_command("run_models", "demo", "--modules", "fake_density")


def test_a_video_of_another_size_is_refused(clip, fake):
    write_video(clip / "video.mp4", 3, size=(32, 24))
    with pytest.raises(CommandError, match="video is \\[32, 24\\]"):
        call_command("run_models", "demo", "--modules", "fake_density")
    assert fake[0].seen == []


def test_an_invalid_layer_is_refused_and_nothing_is_written(clip, monkeypatch):
    bad = FakeDensity()
    bad.layer = lambda clip_id: {"type": "grid", "layer": "density", "clip_id": clip_id,
                                 "frame_size": list(SIZE)}
    monkeypatch.setitem(registry.BUILDERS, "bad", lambda device: bad)
    with pytest.raises(CommandError, match="modules failed .*density: layer refused"):
        call_command("run_models", "demo", "--modules", "bad")
    assert manifest(clip) == [] and not (clip / "layers" / "density.json").exists()
    assert "produced_by" not in json.loads((clip / "clip.json").read_text())


def test_a_failing_module_keeps_the_other_layers(clip, fake, monkeypatch):
    """One module raises mid clip, another cannot be built: the good layer is still added."""
    broken = FakeDensity("broken")

    def explode(frame_id, frame):
        if frame_id == 4:
            raise RuntimeError("out of memory")

    broken.process = explode

    def no_weights(device):
        raise FileNotFoundError("weights missing")

    monkeypatch.setitem(registry.BUILDERS, "broken", lambda device: broken)
    monkeypatch.setitem(registry.BUILDERS, "unbuilt", no_weights)
    with pytest.raises(CommandError) as e:
        call_command("run_models", "demo", "--modules", "broken,unbuilt,fake_density")
    assert "unbuilt: build failed: FileNotFoundError: weights missing" in str(e.value)
    assert "broken: process (frame 4) failed: RuntimeError: out of memory" in str(e.value)
    assert [x["layer"] for x in manifest(clip)] == ["density"]
    assert sorted(json.loads((clip / "clip.json").read_text())["produced_by"]) == ["density"]
    assert fake[0].seen == list(range(N_FRAMES))


def test_progress_is_logged_while_frames_are_decoded(clip, fake, monkeypatch, caplog):
    from apps.pipeline import runner

    monkeypatch.setattr(runner, "PROGRESS_EVERY", 4)
    with caplog.at_level("INFO", logger="apps.pipeline.runner"):
        call_command("run_models", "demo", "--modules", "fake_density")
    lines = [r.getMessage() for r in caplog.records if "frames/s" in r.getMessage()]
    assert [line.split(",")[0] for line in lines] == ["demo: 4/10 frames", "demo: 8/10 frames"]
