"""Live streams: ``POST /streams`` and friends, the stream worker and ``run_stream``.

The models are small stand ins that write one sample per wanted frame (the trained ones are
covered by the live runner's own tests and the manual runs in progress.md). Needs
``sentinel_ml``, OpenCV and ffmpeg; skipped where only the API is installed.
"""

import json
import time

import pytest
from django.core.management import call_command

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("sentinel_ml.pipeline.live")
from sentinel_ml.flow import export  # noqa: E402

if export.find_ffmpeg() is None:
    pytest.skip("no ffmpeg (pip install imageio-ffmpeg)", allow_module_level=True)

from sentinel_ml.pipeline import registry  # noqa: E402

from apps.pipeline import jobs, streams, views  # noqa: E402
from apps.pipeline.ingest import ingest_clip  # noqa: E402
from tests.test_run_models import SIZE, write_video  # noqa: E402

N_FRAMES = 12  # write_video brightens by 20 per frame, so at most 12


class Fake:
    """Writes one 1x1 grid sample (the frame id) on every ``every``-th frame."""

    def __init__(self, name, every, fail_at=None):
        self.name, self.every, self.fail_at = name, every, fail_at
        self._frames = {}

    def wants(self, frame_id):
        return frame_id % self.every == 0

    @property
    def samples(self):
        return self._frames

    def process(self, frame_id, frame):
        if not self.wants(frame_id):
            return
        if frame_id == self.fail_at:
            raise RuntimeError("boom")
        self._frames[str(frame_id)] = {"valid": [[1]], "density": [[float(frame_id)]]}

    def layer(self, clip_id):
        return {"schema_version": 1, "clip_id": clip_id, "layer": self.name, "type": "grid",
                "frame_size": list(SIZE), "grid": {"rows": 1, "cols": 1},
                "units": "people_per_cell", "fields": ["density"], "max_age_frames": 5,
                "frames": self._frames}


@pytest.fixture
def root(tmp_path, settings):
    settings.CLIPS_ROOT = tmp_path / "clips"
    video = tmp_path / "source.mp4"
    write_video(video, N_FRAMES)
    ingest_clip(settings.CLIPS_ROOT, video, "demo", name="Demo")
    return settings.CLIPS_ROOT


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(registry, "BUILDERS", {
        "dense": lambda device: Fake("dense", 5),
        "every": lambda device: Fake("every", 1),
        "broken": lambda device: Fake("broken", 1, fail_at=3),
    })


@pytest.fixture
def spawned(monkeypatch):
    started = []
    monkeypatch.setattr(streams, "spawn", started.append)
    return started


def start(client, **body):
    return client.post("/api/v1/streams", body, content_type="application/json")


def all_samples(client, stream_id):
    got, after = [], 0
    while True:
        res = client.get(f"/api/v1/streams/{stream_id}/samples?after={after}").json()
        if not res["samples"]:
            return got, res
        got += res["samples"]
        after = res["next"]


# ---- the API ----

def test_a_stream_is_queued_from_a_stored_clip(client, root, fakes, spawned):
    res = start(client, clip_id="demo", modules="dense,every")
    assert res.status_code == 202, res.json()
    s = res.json()
    assert spawned == [s["stream_id"]] and s["stream_id"].startswith("live_")
    assert s["state"] == "queued" and s["modules"] == ["dense", "every"]
    assert s["source"] == str(root / "demo" / "video.mp4") and s["source_clip"] == "demo"
    assert s["name"] == "Live: Demo"
    assert client.get(f"/api/v1/streams/{s['stream_id']}").json()["state"] == "queued"
    assert [x["stream_id"] for x in client.get("/api/v1/streams").json()["streams"]] == [
        s["stream_id"]]
    assert "live_" not in str([c["clip_id"] for c in client.get("/api/v1/clips").json()])


def test_all_modules_by_default_and_bad_requests(client, root, fakes, spawned):
    assert start(client, clip_id="demo").json()["modules"] == ["broken", "dense", "every"]
    assert start(client).status_code == 400
    assert start(client, clip_id="nope").status_code == 404
    res = start(client, clip_id="demo", modules=["dense", "nope"])
    assert res.status_code == 400 and "unknown modules ['nope']" in res.json()["detail"]
    assert start(client, clip_id="demo", modules=[1]).status_code == 400
    assert client.get("/api/v1/streams/nope").status_code == 404
    assert client.get("/api/v1/streams/..%2Fdemo").status_code == 404


def test_streams_are_capped(client, root, fakes, spawned, settings):
    settings.MAX_STREAMS = 2
    assert start(client, clip_id="demo").status_code == 202
    assert start(client, clip_id="demo").status_code == 202
    res = start(client, clip_id="demo")
    assert res.status_code == 409 and "limit is 2" in res.json()["detail"]


def test_a_dead_worker_frees_its_place(client, root, fakes, spawned, settings):
    settings.MAX_STREAMS = 1
    s = start(client, clip_id="demo").json()
    path = streams.streams_dir() / s["stream_id"] / "status.json"
    doc = json.loads(path.read_text())
    doc["updated"] = "2026-01-01T00:00:00+00:00"
    path.write_text(json.dumps(doc))
    assert client.get(f"/api/v1/streams/{s['stream_id']}").json()["state"] == "failed"
    assert start(client, clip_id="demo").status_code == 202


def test_stop_asks_a_running_stream_and_refuses_an_ended_one(client, root, fakes, spawned):
    s = start(client, clip_id="demo", modules="dense").json()
    assert client.post(f"/api/v1/streams/{s['stream_id']}/stop").status_code == 202
    assert (streams.streams_dir() / s["stream_id"] / "stop").exists()
    status = streams.execute(s["stream_id"])  # stops before the first frame
    assert status["state"] == "failed" and "no frames" in status["error"]
    assert client.post(f"/api/v1/streams/{s['stream_id']}/stop").status_code == 409


def test_samples_leave_a_half_written_line_for_the_next_call(client, root, fakes, spawned):
    s = start(client, clip_id="demo").json()
    path = streams.streams_dir() / s["stream_id"] / "samples.jsonl"
    path.write_bytes(b'{"seq": 0}\n{"seq": 1}\n{"se')
    res = client.get(f"/api/v1/streams/{s['stream_id']}/samples").json()
    assert res["samples"] == [{"seq": 0}, {"seq": 1}] and res["next"] == 22
    assert res["state"] == "queued"
    with path.open("ab") as f:
        f.write(b'q": 2}\n')
    res = client.get(f"/api/v1/streams/{s['stream_id']}/samples?after=22").json()
    assert res["samples"] == [{"seq": 2}] and res["next"] == path.stat().st_size
    assert client.get(f"/api/v1/streams/{s['stream_id']}/samples?after=-1").status_code == 400


# ---- the worker ----

@pytest.fixture
def finished(client, root, fakes, spawned):
    s = start(client, clip_id="demo", modules="dense,every,broken").json()
    t = time.monotonic()
    status = streams.execute(s["stream_id"])
    status["wall_s"] = time.monotonic() - t
    return status


def test_the_worker_streams_samples_in_order_and_saves_a_clip(client, finished):
    sid = finished["stream_id"]
    assert finished["state"] == "done" and finished["clip_id"] == sid
    assert finished["frames_read"] == N_FRAMES and finished["fps"] == 25.0
    assert finished["wall_s"] >= (N_FRAMES - 1) / 25  # read at the file's own pace
    assert finished["errors"] == {"broken": "process (frame 3) failed: RuntimeError: boom"}
    assert finished["ran"]["every"] == N_FRAMES and finished["ran"]["dense"] == 3

    got, last = all_samples(client, sid)
    assert last["state"] == "done" and [x["seq"] for x in got] == list(range(len(got)))
    assert [x["frame"] for x in got if x["layer"] == "dense"] == [0, 5, 10]
    assert [x["frame"] for x in got if x["layer"] == "every"] == list(range(N_FRAMES))
    first = next(x for x in got if x["layer"] == "dense")
    assert first["header"]["type"] == "grid" and "frames" not in first["header"]
    assert first["meta"] == {"layer": "dense", "type": "grid", "label": "Dense",
                             "units": "people_per_cell",
                             "style": {"render": "heatmap", "field": "density"},
                             "default_visible": False}  # not in the registry SHOWN set
    assert all("meta" not in x for x in got if x is not first and x["layer"] == "dense")
    assert first["data"] == {"valid": [[1]], "density": [[0.0]]}
    assert all(x["t_ready"] >= x["t_capture"] for x in got)
    assert set(finished["layers"]) == {"dense", "every", "broken"}  # broken sent frames 0 to 2

    clip = client.get(f"/api/v1/clips/{sid}").json()
    assert clip["n_frames"] == N_FRAMES and clip["name"] == "Live: Demo"
    assert [e["layer"] for e in clip["layers"]] == ["dense", "every"]
    layer = client.get(f"/api/v1/clips/{sid}/layers/dense").json()
    assert list(layer["frames"]) == ["0", "5", "10"]
    assert client.get(f"/api/v1/clips/{sid}/video").status_code == 200
    assert not (streams.streams_dir() / sid / "video.mp4").exists()


def test_the_live_picture_is_mjpeg_with_the_frame_id(client, finished):
    res = client.get(f"/api/v1/streams/{finished['stream_id']}/video")
    assert res["Content-Type"] == "multipart/x-mixed-replace; boundary=frame"
    body = b"".join(res.streaming_content)  # an ended stream sends its last frame and stops
    head, _, rest = body.partition(b"\r\n\r\n")
    assert head.startswith(b"--frame\r\nContent-Type: image/jpeg")
    assert f"X-Frame-Id: {N_FRAMES - 1}".encode() in head
    n = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
    image = cv2.imdecode(np.frombuffer(rest[:n], np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (SIZE[1], SIZE[0], 3)
    assert client.get("/api/v1/streams/nope/video").status_code == 404


def test_the_live_picture_goes_on_when_a_status_read_fails_once(client, finished, monkeypatch):
    sid = finished["stream_id"]
    running = {**finished, "state": "running"}
    reads = iter([running, None, finished])  # None: status.json was being replaced (Windows)
    monkeypatch.setattr(streams, "read_stream", lambda _id: next(reads))
    clock = iter(range(0, 100, 2))  # every check is more than 1 s after the last one
    monkeypatch.setattr(views.time, "monotonic", lambda: next(clock))
    res = client.get(f"/api/v1/streams/{sid}/video")
    body = b"".join(res.streaming_content)
    assert body.count(b"Content-Type: image/jpeg") == 1  # the last frame once, then the end
    with pytest.raises(StopIteration):  # it asked again after the failed read, until "done"
        next(reads)


def test_run_stream_takes_a_file_and_saves_a_clip(root, fakes, tmp_path, capsys):
    call_command("run_stream", str(root / "demo" / "video.mp4"), "--modules", "dense")
    out = capsys.readouterr().out
    assert f"{N_FRAMES} frames, 3 samples" in out and "failed none" in out
    (stream,) = streams.list_streams()
    assert stream["state"] == "done" and stream["name"] == "Live: video.mp4"
    assert (root / stream["clip_id"] / "layers" / "dense.json").is_file()


def test_run_stream_reads_a_frame_folder_at_the_given_fps(root, fakes, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(3):
        image = np.full((SIZE[1], SIZE[0], 3), 40 * i, np.uint8)
        cv2.imwrite(str(folder / f"img{i:03d}.jpg"), image)
    call_command("run_stream", str(folder), "--modules", "every", "--fps", "12.5")
    (stream,) = streams.list_streams()
    assert stream["state"] == "done" and stream["fps"] == 12.5 and stream["frames_read"] == 3


def test_the_worker_process_runs_a_stream(client, root, settings):
    """A real worker process (manage.py run_stream --worker) with the real flow module."""
    s = start(client, clip_id="demo", modules="flow").json()
    deadline = time.monotonic() + 120
    while (status := streams.read_stream(s["stream_id"]))["state"] in jobs.ACTIVE:
        assert time.monotonic() < deadline, status
        time.sleep(0.5)
    log = (streams.streams_dir() / s["stream_id"] / "log.txt").read_text()
    assert status["state"] == "done", log
    assert status["frames_read"] == N_FRAMES and (root / status["clip_id"] / "video.mp4").is_file()
