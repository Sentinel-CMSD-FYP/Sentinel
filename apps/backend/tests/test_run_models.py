"""``manage.py run_models``: modules run on a clip's video and their layers land in the clip.

Needs ``sentinel_ml`` and OpenCV (the repo .venv); skipped where only the API is installed.
"""

import json

import numpy as np
import pytest
from django.core.management import CommandError, call_command

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
    with pytest.raises(CommandError, match="unknown modules \\['flow'\\]; available: density"):
        call_command("run_models", "demo", "--modules", "flow")


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
    with pytest.raises(CommandError, match="layer 'density' was refused"):
        call_command("run_models", "demo", "--modules", "bad")
    assert manifest(clip) == [] and not (clip / "layers" / "density.json").exists()
