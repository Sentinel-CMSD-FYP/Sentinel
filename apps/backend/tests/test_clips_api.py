import json

import pytest

from .conftest import VIDEO_BYTES

pytestmark = pytest.mark.usefixtures("clips_root")


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_clip_list_summarises_readable_clips_only(client):
    r = client.get("/api/v1/clips")
    assert r.status_code == 200
    by_id = {c["clip_id"]: c for c in r.json()}
    assert set(by_id) == {"demo", "silent"}  # not_a_clip and broken are skipped
    assert by_id["demo"] == {
        "clip_id": "demo", "name": "Demo clip", "fps": 25.0, "frame_size": [1920, 1080],
        "n_frames": 300, "layers": ["flow", "points", "ghost"],
    }


def test_clip_list_is_empty_when_root_is_missing(client, settings, tmp_path):
    settings.CLIPS_ROOT = tmp_path / "does_not_exist"
    assert client.get("/api/v1/clips").json() == []


def test_clip_detail_points_video_url_at_the_video_endpoint(client):
    body = client.get("/api/v1/clips/demo").json()
    assert body["video_url"] == "/api/v1/clips/demo/video"
    assert [e["layer"] for e in body["layers"]] == ["flow", "points", "ghost"]


def test_clip_detail_drops_video_url_when_there_is_no_video(client):
    assert "video_url" not in client.get("/api/v1/clips/silent").json()


@pytest.mark.parametrize("clip_id", ["nope", "not_a_clip", "..", "%2e%2e", "demo%2f..", ".hidden"])
def test_unknown_or_unsafe_clip_ids_are_404(client, clip_id):
    assert client.get(f"/api/v1/clips/{clip_id}").status_code == 404


def test_layer_is_served_as_stored(client, clips_root):
    r = client.get("/api/v1/clips/demo/layers/points")
    assert r.status_code == 200
    assert r["Content-Type"] == "application/json"
    assert r.json() == json.loads((clips_root / "demo" / "layers" / "points.json").read_text())


def test_layer_frame_range_is_inclusive_and_keeps_the_rest(client):
    doc = client.get("/api/v1/clips/demo/layers/flow?from=25&to=50").json()
    assert sorted(doc["frames"], key=int) == ["25", "50"]
    assert doc["layer"] == "flow" and doc["max_age_frames"] == 25


def test_layer_open_ended_range(client):
    doc = client.get("/api/v1/clips/demo/layers/flow?from=26").json()
    assert list(doc["frames"]) == ["50"]


@pytest.mark.parametrize("query", ["from=abc", "to=1.5", "from=-1", "from=30&to=10"])
def test_bad_frame_range_is_400(client, query):
    assert client.get(f"/api/v1/clips/demo/layers/flow?{query}").status_code == 400


@pytest.mark.parametrize("layer", ["ghost", "density", "FLOW", "..", "flow.json"])
def test_undeclared_or_missing_layer_is_404(client, layer):
    assert client.get(f"/api/v1/clips/demo/layers/{layer}").status_code == 404


def test_layer_of_unknown_clip_is_404(client):
    assert client.get("/api/v1/clips/nope/layers/flow").status_code == 404


def test_video_whole_file(client):
    r = client.get("/api/v1/clips/demo/video")
    assert r.status_code == 200
    assert r["Accept-Ranges"] == "bytes"
    assert r["Content-Type"] == "video/mp4"
    assert b"".join(r.streaming_content) == VIDEO_BYTES


def test_video_byte_range_is_206_with_exact_bytes(client):
    r = client.get("/api/v1/clips/demo/video", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206
    assert r["Content-Range"] == f"bytes 100-199/{len(VIDEO_BYTES)}"
    assert r.content == VIDEO_BYTES[100:200]


def test_video_open_ended_and_suffix_ranges(client):
    r = client.get("/api/v1/clips/demo/video", headers={"Range": "bytes=2000-"})
    assert r.status_code == 206 and r.content == VIDEO_BYTES[2000:]
    r = client.get("/api/v1/clips/demo/video", headers={"Range": "bytes=-48"})
    assert r.status_code == 206 and r.content == VIDEO_BYTES[-48:]


def test_video_range_end_past_file_is_clamped(client):
    r = client.get("/api/v1/clips/demo/video", headers={"Range": "bytes=2040-9999"})
    assert r.status_code == 206
    assert r["Content-Range"] == f"bytes 2040-2047/{len(VIDEO_BYTES)}"


@pytest.mark.parametrize("header", ["bytes=5000-", "bytes=-0", "bytes=300-200"])
def test_video_unsatisfiable_range_is_416(client, header):
    r = client.get("/api/v1/clips/demo/video", headers={"Range": header})
    assert r.status_code == 416
    assert r["Content-Range"] == f"bytes */{len(VIDEO_BYTES)}"


@pytest.mark.parametrize("header", ["bytes=0-1,5-6", "items=0-5", "bytes=-", "garbage"])
def test_video_malformed_or_multi_range_falls_back_to_whole_file(client, header):
    r = client.get("/api/v1/clips/demo/video", headers={"Range": header})
    assert r.status_code == 200


def test_video_missing_is_404(client):
    assert client.get("/api/v1/clips/silent/video").status_code == 404
    assert client.get("/api/v1/clips/nope/video").status_code == 404


def test_cors_allows_the_frontend_dev_server(client):
    r = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert r["Access-Control-Allow-Origin"] == "http://localhost:5173"
