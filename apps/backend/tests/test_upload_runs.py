"""Upload a video and run the models from the API: ``POST /clips``, ``POST /clips/<id>/runs``,
``GET /clips/<id>/runs/<run_id>``, the run worker and the Kaggle runner.

The models here are the small stand ins of test_run_models (the trained ones run in
test_end_to_end.py), and the kaggle CLI is replaced by a fake that answers like Kaggle, so these
tests need no GPU, no weights and no Kaggle account. Needs ``sentinel_ml``, OpenCV and ffmpeg;
skipped where only the API is installed.
"""

import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from django.core.management import call_command

cv2 = pytest.importorskip("cv2")
pytest.importorskip("sentinel_ml.flow.export")
from sentinel_ml.flow import export  # noqa: E402

if export.find_ffmpeg() is None:
    pytest.skip("no ffmpeg (pip install imageio-ffmpeg)", allow_module_level=True)

from sentinel_ml.pipeline import registry  # noqa: E402

from apps.pipeline import jobs, kaggle_run  # noqa: E402
from apps.pipeline.views import clip_id_for  # noqa: E402
from tests.test_run_models import N_FRAMES, SIZE, FakeDensity, write_video  # noqa: E402


@pytest.fixture
def root(tmp_path, settings):
    settings.CLIPS_ROOT = tmp_path / "clips"
    settings.PIPELINE_RUNNER = "local"
    return settings.CLIPS_ROOT


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "my drone clip.mp4"
    write_video(path, N_FRAMES)
    return path


@pytest.fixture
def spawned(monkeypatch):
    """Record started workers instead of starting processes."""
    started = []
    monkeypatch.setattr(jobs, "spawn", started.append)
    return started


def upload(client, path, **fields):
    with path.open("rb") as f:
        return client.post("/api/v1/clips", {"file": f, **fields})


@pytest.fixture
def clip_id(client, root, video):
    assert upload(client, video).status_code == 201
    return "my_drone_clip"


# ---- POST /clips ----

def test_an_uploaded_video_becomes_a_clip(client, root, video):
    res = upload(client, video, name="Field test")
    assert res.status_code == 201, res.json()
    clip = res.json()
    assert clip["clip_id"] == "my_drone_clip" and clip["name"] == "Field test"
    assert clip["n_frames"] == N_FRAMES and clip["frame_size"] == list(SIZE)
    assert clip["layers"] == [] and clip["video_url"] == "/api/v1/clips/my_drone_clip/video"
    assert [c["clip_id"] for c in client.get("/api/v1/clips").json()] == ["my_drone_clip"]
    assert sorted(p.name for p in root.iterdir()) == ["my_drone_clip"]  # no upload left over


def test_the_same_name_twice_gets_a_new_id_and_the_file_name_as_label(client, root, video):
    upload(client, video)
    clip = upload(client, video).json()
    assert clip["clip_id"] == "my_drone_clip_2" and clip["name"] == "my drone clip.mp4"


def test_camera_fields_become_the_clip_camera(client, root, video):
    res = upload(client, video, altitude_m="45", fov_deg="82.1", fov_kind="diagonal",
                 camera_model="DJI Mini 3", altitude_source="telemetry")
    assert res.status_code == 201, res.json()
    assert res.json()["camera"] == {"altitude_m": 45.0, "altitude_source": "telemetry",
                                    "model": "DJI Mini 3", "hfov_deg": 69.73,
                                    "dfov_deg": 82.1, "sensor_aspect": "4:3"}


def test_empty_camera_fields_leave_no_camera(client, root, video):
    res = upload(client, video, altitude_m="", fov_deg="", fov_kind="horizontal")
    assert res.status_code == 201 and "camera" not in res.json()


@pytest.mark.parametrize("fields, why", [
    ({"altitude_m": "60"}, "both the altitude and the field of view"),
    ({"altitude_m": "sixty", "fov_deg": "70"}, "must be numbers"),
    ({"altitude_m": "60", "fov_deg": "200"}, "between 0 and 180"),
    ({"altitude_m": "60", "fov_deg": "70", "altitude_source": "guess"}, "altitude source"),
])
def test_bad_camera_fields_are_refused_and_nothing_is_kept(client, root, video, fields, why):
    res = upload(client, video, **fields)
    assert res.status_code == 400 and why in res.json()["detail"]
    assert not root.exists() or not any(p for p in root.iterdir() if not p.name.startswith("."))


def test_clip_ids_are_safe(tmp_path):
    assert clip_id_for(tmp_path, "..\\..\\evil name!.mp4") == "evil_name"
    assert clip_id_for(tmp_path, "../x/.mp4") == "clip"
    assert clip_id_for(tmp_path, "a" * 60 + ".mov") == "a" * 40
    (tmp_path / "clip").mkdir()
    assert clip_id_for(tmp_path, "%%%.mp4") == "clip_2"


def test_a_file_over_the_limit_is_refused(client, root, video, settings):
    settings.MAX_UPLOAD_MB = 0
    res = upload(client, video)
    assert res.status_code == 413 and "0 MB" in res.json()["detail"]


def test_missing_wrong_type_and_unreadable_files_are_refused(client, root, tmp_path):
    assert client.post("/api/v1/clips", {}).status_code == 400
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    res = upload(client, notes)
    assert res.status_code == 400 and ".mp4" in res.json()["detail"]
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video" * 100)
    res = upload(client, junk)
    assert res.status_code == 400 and "could not read" in res.json()["detail"]
    assert [p.name for p in root.iterdir()] == []  # no half made clip


# ---- POST /clips/<id>/runs and GET .../runs/<run_id> ----

def test_a_run_is_queued_and_its_worker_started(client, clip_id, spawned):
    res = client.post(f"/api/v1/clips/{clip_id}/runs")
    assert res.status_code == 202
    run = res.json()
    assert run["state"] == "queued" and run["clip_id"] == clip_id and run["runner"] == "local"
    assert spawned == [run["run_id"]]
    assert client.get(f"/api/v1/clips/{clip_id}/runs/{run['run_id']}").json() == run


def test_the_active_run_can_be_found_again(client, clip_id, spawned, fake_density):
    assert client.get("/api/v1/runs/active").json() == {"run": None}
    run = client.post(f"/api/v1/clips/{clip_id}/runs").json()
    assert client.get("/api/v1/runs/active").json() == {"run": run}
    jobs.execute(run["run_id"])
    assert client.get("/api/v1/runs/active").json() == {"run": None}


def test_one_run_at_a_time(client, clip_id, spawned):
    first = client.post(f"/api/v1/clips/{clip_id}/runs").json()["run_id"]
    res = client.post(f"/api/v1/clips/{clip_id}/runs")
    assert res.status_code == 409 and res.json()["run_id"] == first
    assert len(spawned) == 1


def test_unknown_clips_and_runs(client, clip_id, spawned):
    assert client.post("/api/v1/clips/nope/runs").status_code == 404
    assert client.get(f"/api/v1/clips/{clip_id}/runs/nope").status_code == 404
    run = client.post(f"/api/v1/clips/{clip_id}/runs").json()["run_id"]
    assert client.get(f"/api/v1/clips/other/runs/{run}").status_code == 404
    assert client.get(f"/api/v1/clips/{clip_id}/runs/..%2F..").status_code == 404


def test_a_clip_with_layers_needs_replace(client, clip_id, spawned, fake_density):
    jobs.execute(client.post(f"/api/v1/clips/{clip_id}/runs").json()["run_id"])
    res = client.post(f"/api/v1/clips/{clip_id}/runs")
    assert res.status_code == 409 and "replace" in res.json()["detail"]
    res = client.post(f"/api/v1/clips/{clip_id}/runs", {"replace": True},
                      content_type="application/json")
    assert res.status_code == 202 and res.json()["replace"] is True


def test_a_dead_worker_reads_as_failed_and_frees_the_server(client, clip_id, spawned):
    run = client.post(f"/api/v1/clips/{clip_id}/runs").json()
    path = jobs.runs_dir() / run["run_id"] / "status.json"
    old = (datetime.now(UTC) - timedelta(seconds=jobs.STALE_S + 1)).isoformat()
    path.write_text(json.dumps({**run, "updated": old}))
    seen = client.get(f"/api/v1/clips/{clip_id}/runs/{run['run_id']}").json()
    assert seen["state"] == "failed" and "stopped answering" in seen["error"]
    assert client.post(f"/api/v1/clips/{clip_id}/runs").status_code == 202


# ---- the worker ----

@pytest.fixture
def fake_density(monkeypatch):
    monkeypatch.setattr(registry, "BUILDERS", {"density": lambda device: FakeDensity()})


def test_the_local_runner_adds_every_layer(client, clip_id, spawned, fake_density, monkeypatch):
    from apps.pipeline import runner
    monkeypatch.setattr(runner, "PROGRESS_EVERY", 4)
    run = client.post(f"/api/v1/clips/{clip_id}/runs").json()

    jobs.execute(run["run_id"])

    done = client.get(f"/api/v1/clips/{clip_id}/runs/{run['run_id']}").json()
    assert done["state"] == "done" and done["layers"] == ["density"] and done["errors"] == {}
    assert done["progress"] == {"done": 8, "total": N_FRAMES} and done["finished"]
    clip = client.get(f"/api/v1/clips/{clip_id}").json()
    assert [e["layer"] for e in clip["layers"]] == ["density"]


def test_a_failing_run_says_why(client, clip_id, spawned, monkeypatch):
    monkeypatch.setattr(jobs, "build_modules", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    run = client.post(f"/api/v1/clips/{clip_id}/runs").json()
    status = jobs.execute(run["run_id"])
    assert status["state"] == "failed" and status["error"] == "OSError: disk"


def test_the_worker_process_starts_and_reports(client, clip_id, settings):
    """A real worker process (manage.py run_job) with a runner name it does not know."""
    settings.PIPELINE_RUNNER = "nope"
    run = client.post(f"/api/v1/clips/{clip_id}/runs").json()
    deadline = time.monotonic() + 60
    while (status := jobs.read_status(run["run_id"]))["state"] in jobs.ACTIVE:
        assert time.monotonic() < deadline, status
        time.sleep(0.5)
    assert status["state"] == "failed" and "unknown PIPELINE_RUNNER 'nope'" in status["error"]
    assert "failed" in (jobs.runs_dir() / run["run_id"] / "log.txt").read_text()


# ---- the Kaggle runner, against a fake kaggle CLI ----

class FakeKaggle:
    """Answers kaggle CLI calls as Kaggle does; ``states`` is what kernels status says."""

    def __init__(self, states, run_id_of=lambda run_id: run_id, dataset_exists=False):
        self.calls, self.states = [], list(states)
        self.run_id_of, self.exists = run_id_of, dataset_exists
        self.run_id = None

    def __call__(self, *args, timeout=None):
        self.calls.append(args[:2])
        ok = lambda out="": kaggle_run.subprocess.CompletedProcess(args, 0, out, "")  # noqa: E731
        if args[:2] == ("datasets", "status"):
            if not self.exists:
                return kaggle_run.subprocess.CompletedProcess(args, 1, "", "404 not found")
            return ok("ready")
        if args[:2] in (("datasets", "create"), ("datasets", "version")):
            folder = kaggle_run.Path(args[args.index("-p") + 1])
            self.run_id = json.loads((folder / "request.json").read_text())["run_id"]
            assert (folder / "video.mp4").stat().st_size > 0
            self.exists = True
            return ok()
        if args[:2] == ("kernels", "push"):
            meta = json.loads((kaggle_run.Path(args[3]) / "kernel-metadata.json").read_text())
            assert meta["id"] == "tester/sentinel-run-models"
            return ok("Kernel version 1 successfully pushed")
        if args[:2] == ("kernels", "status"):
            state = self.states.pop(0)
            if state is None:
                return kaggle_run.subprocess.CompletedProcess(args, 1, "", "connection reset")
            return ok(f'tester/sentinel-run-models has status "KernelWorkerStatus.{state}"')
        if args[:2] == ("kernels", "output"):
            self.write_output(kaggle_run.Path(args[args.index("-p") + 1]))
            return ok()
        raise AssertionError(args)

    def write_output(self, out):
        layers = out / "clip" / "layers"
        layers.mkdir(parents=True)
        density = FakeDensity()
        density.seen = list(range(N_FRAMES))
        (layers / "density.json").write_text(json.dumps(density.layer(self.clip_id)))
        (out / "clip" / "clip.json").write_text(json.dumps({
            "layers": [{"layer": "density", "default_visible": False}],
            "produced_by": {"density": {"model": "fake", "device": "cuda"}}}))
        (out / "eval.log").write_text("kernel log")
        (out / "result.json").write_text(json.dumps({
            "run_id": self.run_id_of(self.run_id),
            "report": {"layers": ["density"], "errors": {"points": "build failed: no ckpt"}},
            "error": None}))


@pytest.fixture
def on_kaggle(client, clip_id, spawned, settings, monkeypatch):
    settings.PIPELINE_RUNNER, settings.KAGGLE_USERNAME = "kaggle", "tester"
    monkeypatch.setattr(kaggle_run, "sleep", lambda s: None)

    def start(fake):
        fake.clip_id = clip_id
        monkeypatch.setattr(kaggle_run, "cli", fake)
        return client.post(f"/api/v1/clips/{clip_id}/runs").json()["run_id"]
    return start


def test_a_kaggle_run_brings_the_layers_back(client, clip_id, on_kaggle):
    fake = FakeKaggle(["QUEUED", None, "RUNNING", "COMPLETE"])
    status = jobs.execute(on_kaggle(fake))

    assert status["state"] == "done", status
    assert status["layers"] == ["density"]
    assert status["errors"] == {"points": "build failed: no ckpt"}
    assert ("datasets", "create") in fake.calls and ("kernels", "push") in fake.calls
    clip = client.get(f"/api/v1/clips/{clip_id}").json()
    assert clip["layers"][0]["layer"] == "density" and clip["layers"][0]["default_visible"] is False
    assert clip["produced_by"] == {"density": {"model": "fake", "device": "cuda"}}
    assert client.get(f"/api/v1/clips/{clip_id}/layers/density").status_code == 200
    folder = jobs.runs_dir() / status["run_id"]
    assert (folder / "kaggle.log").read_text() == "kernel log"
    assert not (folder / "dataset").exists()  # the video copy is gone after the upload


def test_a_kaggle_run_versions_an_existing_dataset(on_kaggle):
    fake = FakeKaggle(["COMPLETE"], dataset_exists=True)
    assert jobs.execute(on_kaggle(fake))["state"] == "done"
    assert ("datasets", "version") in fake.calls


def test_the_previous_runs_output_is_not_taken(on_kaggle):
    seen = []

    def first_stale(run_id):  # the first download is the previous run's output
        seen.append(run_id)
        return "older_run" if len(seen) == 1 else run_id

    fake = FakeKaggle(["COMPLETE", "RUNNING", "COMPLETE"], run_id_of=first_stale)
    assert jobs.execute(on_kaggle(fake))["state"] == "done"
    assert fake.calls.count(("kernels", "output")) == 2


def test_a_failed_kernel_fails_the_run(on_kaggle):
    fake = FakeKaggle(["RUNNING", "ERROR"], run_id_of=lambda run_id: None)
    status = jobs.execute(on_kaggle(fake))
    assert status["state"] == "failed" and "ended with ERROR" in status["error"]


def test_kaggle_needs_a_username(on_kaggle, settings):
    settings.KAGGLE_USERNAME = ""
    status = jobs.execute(on_kaggle(FakeKaggle([])))
    assert status["state"] == "failed" and "KAGGLE_USERNAME" in status["error"]


def test_run_job_command_reports_the_end(clip_id, spawned, fake_density, client, capsys):
    run = client.post(f"/api/v1/clips/{clip_id}/runs").json()["run_id"]
    call_command("run_job", run)
    assert "layers ['density']" in capsys.readouterr().out
