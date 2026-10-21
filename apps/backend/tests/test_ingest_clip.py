"""``manage.py ingest_clip``: a video or a frame folder becomes a clip the API serves.

Needs ``sentinel_ml``, OpenCV and ffmpeg (imageio-ffmpeg in the repo .venv); skipped where only
the API is installed.
"""

import json

import pytest
from django.core.management import CommandError, call_command

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("sentinel_ml.flow.export")
from sentinel_ml.flow import export  # noqa: E402

if export.find_ffmpeg() is None:
    pytest.skip("no ffmpeg (pip install imageio-ffmpeg)", allow_module_level=True)

from sentinel_ml.pipeline import registry  # noqa: E402

from tests.test_run_models import N_FRAMES, SIZE, FakeDensity, write_video  # noqa: E402


def write_frames(folder, prefix, n, size=SIZE):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(folder / f"{prefix}{i + 1:03d}.jpg"),
                    np.full((size[1], size[0], 3), i * 20, np.uint8))


@pytest.fixture
def root(tmp_path, settings):
    settings.CLIPS_ROOT = tmp_path / "clips"
    return settings.CLIPS_ROOT


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return codec, n


def test_a_frame_folder_becomes_a_clip(tmp_path, root, capsys):
    frames = tmp_path / "images"
    write_frames(frames, "img011", N_FRAMES)
    write_frames(frames, "img012", 3)  # another sequence in the same folder is left out

    call_command("ingest_clip", str(frames), "seq011", "--seq", "011", "--fps", "25",
                 "--name", "Test 11")

    clip = json.loads((root / "seq011" / "clip.json").read_text())
    assert clip == {"schema_version": 1, "clip_id": "seq011", "name": "Test 11", "fps": 25.0,
                    "frame_size": list(SIZE), "n_frames": N_FRAMES, "layers": []}
    assert read_video(root / "seq011" / "video.mp4") == ("h264", N_FRAMES)
    assert (root / "seq011" / "layers").is_dir()
    assert [p.name for p in root.iterdir()] == ["seq011"]  # no temporary folder left
    assert "10 frames at 25 fps, 64x48" in capsys.readouterr().out


def test_a_video_keeps_its_frame_rate(tmp_path, root):
    write_video(tmp_path / "in.mp4", N_FRAMES)
    call_command("ingest_clip", str(tmp_path / "in.mp4"), "demo")
    clip = json.loads((root / "demo" / "clip.json").read_text())
    assert clip["fps"] == 25.0 and clip["n_frames"] == N_FRAMES and clip["name"] == "demo"


def test_camera_flags_go_into_clip_json(tmp_path, root, capsys):
    write_video(tmp_path / "in.mp4", N_FRAMES)
    call_command("ingest_clip", str(tmp_path / "in.mp4"), "demo", "--altitude", "60",
                 "--dfov", "82.1", "--camera-model", "DJI Mini 3", "--altitude-source", "stated")
    clip = json.loads((root / "demo" / "clip.json").read_text())
    assert clip["camera"] == {"altitude_m": 60.0, "altitude_source": "stated",
                              "model": "DJI Mini 3", "hfov_deg": 69.73, "dfov_deg": 82.1,
                              "sensor_aspect": "4:3"}
    assert "camera: 60 m (stated), horizontal FOV 69.73 degrees" in capsys.readouterr().out


def test_a_fov_without_altitude_is_refused_before_ingest(tmp_path, root):
    write_video(tmp_path / "in.mp4", N_FRAMES)
    with pytest.raises(CommandError, match="--altitude is needed"):
        call_command("ingest_clip", str(tmp_path / "in.mp4"), "demo", "--hfov", "70")
    assert not root.exists() or not any(root.iterdir())


def test_ingest_then_run_models_then_the_api(tmp_path, root, client, monkeypatch):
    write_video(tmp_path / "in.mp4", N_FRAMES)
    monkeypatch.setitem(registry.BUILDERS, "fake_density", lambda device: FakeDensity())

    call_command("ingest_clip", str(tmp_path / "in.mp4"), "demo")
    call_command("run_models", "demo", "--modules", "fake_density")

    clips = client.get("/api/v1/clips").json()
    assert [(c["clip_id"], c["layers"]) for c in clips] == [("demo", ["density"])]
    assert client.get("/api/v1/clips/demo").json()["video_url"] == "/api/v1/clips/demo/video"
    assert list(client.get("/api/v1/clips/demo/layers/density").json()["frames"]) == ["0", "5"]


def test_refusals_leave_nothing_behind(tmp_path, root, monkeypatch):
    write_video(tmp_path / "in.mp4", N_FRAMES)
    write_frames(tmp_path / "odd", "img", 2, size=(63, 48))
    write_frames(tmp_path / "mixed", "img", 2)
    write_frames(tmp_path / "mixed", "img9", 1, size=(32, 24))  # img9001.jpg sorts last
    cases = [
        (("nope.mp4", "demo"), "source not found"),
        (("in.mp4", "../demo"), "bad clip id"),
        (("odd", "demo"), "63x48; H.264 \\(yuv420p\\) needs an even"),
        (("empty", "demo"), "no image frames"),
        (("mixed", "demo"), "frame 2 is missing or not 64x48"),
    ]
    (tmp_path / "empty").mkdir()
    for (src, clip_id), match in cases:
        with pytest.raises(CommandError, match=match):
            call_command("ingest_clip", str(tmp_path / src), clip_id)

    monkeypatch.setattr(export, "find_ffmpeg", lambda: None)
    with pytest.raises(CommandError, match="no ffmpeg found"):
        call_command("ingest_clip", str(tmp_path / "in.mp4"), "demo")
    assert not root.exists() or list(root.iterdir()) == []


def test_an_existing_clip_is_refused(tmp_path, root):
    write_video(tmp_path / "in.mp4", N_FRAMES)
    call_command("ingest_clip", str(tmp_path / "in.mp4"), "demo")
    with pytest.raises(CommandError, match="already exists; delete it first"):
        call_command("ingest_clip", str(tmp_path / "in.mp4"), "demo")
