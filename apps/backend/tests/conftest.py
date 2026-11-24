import json

import pytest

VIDEO_BYTES = bytes(range(256)) * 8  # 2048 bytes, so a byte range can be checked exactly

CLIP = {
    "schema_version": 1,
    "clip_id": "demo",
    "name": "Demo clip",
    "fps": 25.0,
    "frame_size": [1920, 1080],
    "n_frames": 300,
    "video_url": "/wrong/and/should/be/replaced",
    "layers": [
        {"layer": "flow", "type": "grid", "label": "Optical flow"},
        {"layer": "points", "type": "points", "label": "Heads"},
        {"layer": "ghost", "type": "points", "label": "Declared but no file"},
    ],
}


def layer(name: str, kind: str, frames: dict) -> dict:
    return {"schema_version": 1, "clip_id": "demo", "layer": name, "type": kind,
            "frame_size": [1920, 1080], "max_age_frames": 25, "frames": frames}


@pytest.fixture
def clips_root(tmp_path, settings):
    """A clips folder with one complete clip, one with no video and some noise."""
    clip = tmp_path / "demo"
    (clip / "layers").mkdir(parents=True)
    (clip / "clip.json").write_text(json.dumps(CLIP))
    (clip / "video.mp4").write_bytes(VIDEO_BYTES)
    (clip / "layers" / "flow.json").write_text(json.dumps(
        layer("flow", "grid", {"0": {"valid": [[1]]}, "25": {"valid": [[1]]},
                               "50": {"valid": [[0]]}})))
    (clip / "layers" / "points.json").write_text(json.dumps(
        layer("points", "points", {"0": [[10.0, 20.0, 0.9]]})))

    silent = tmp_path / "silent"
    silent.mkdir()
    (silent / "clip.json").write_text(json.dumps(CLIP | {"clip_id": "silent", "layers": []}))

    (tmp_path / "not_a_clip").mkdir()  # no clip.json
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "clip.json").write_text("{ not json")

    settings.CLIPS_ROOT = tmp_path
    return tmp_path
