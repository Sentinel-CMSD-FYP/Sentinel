"""End to end: real DroneCrowd frames are ingested, every registered model runs on CPU with its
real weights, and the API serves all layers; once through the commands, once through the upload
and run endpoints (local runner, worker run in this process).

About 1 min on CPU. Needs the repo .venv (sentinel_ml with the density, infra and points extras,
ffmpeg), the weights named in ``ml/configs`` and the local DroneCrowd copy; skipped otherwise, so
CI (no weights, no dataset) skips it.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from django.conf import settings
from django.core.management import call_command

yaml = pytest.importorskip("yaml")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("sentinel_ml.pipeline")
for _dep in ("torch", "ultralytics", "segmentation_models_pytorch", "pytorch_lightning"):
    pytest.importorskip(_dep)

from sentinel_ml.flow.export import find_ffmpeg  # noqa: E402
from sentinel_ml.pipeline import available  # noqa: E402

REPO = Path(settings.REPO_ROOT)
CONFIGS = REPO / "ml" / "configs"
FRAMES = REPO / "data" / "dronecrowd" / "test_data-001" / "test_data" / "images"
N_FRAMES = 10  # flow pairs frame 0 with frame 5; tracks confirm a track after 3 frames
GT_FRAME_0 = 211  # people in test clip 11, frame 0


def _weights() -> list[Path]:
    def key(config, name):
        return REPO / yaml.safe_load((CONFIGS / config).read_text(encoding="utf-8"))[name]

    return [key("density_dmcount.yaml", "weights"), key("infrastructure_yolo26.yaml", "weights"),
            key("infer_b.yaml", "checkpoint")]


_missing = [str(p) for p in [*_weights(), FRAMES / "img011001.jpg"] if not p.is_file()]
if _missing:
    pytest.skip(f"missing {_missing}", allow_module_level=True)
if find_ffmpeg() is None:
    pytest.skip("no ffmpeg (pip install imageio-ffmpeg)", allow_module_level=True)


def validator():
    path = REPO / "docs" / "layers" / "validate.py"
    spec = importlib.util.spec_from_file_location("layer_validate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ingest_all_models_and_the_api(tmp_path, settings, client):
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(1, N_FRAMES + 1):
        shutil.copy(FRAMES / f"img011{i:03d}.jpg", src)
    settings.CLIPS_ROOT = tmp_path / "clips"

    call_command("ingest_clip", str(src), "seq011", "--seq", "011", "--fps", "25")
    call_command("run_models", "seq011", "--modules", "all", "--device", "cpu")

    clip_dir = settings.CLIPS_ROOT / "seq011"
    problems = validator().validate_clip_dir(clip_dir)
    assert len(problems) == 1 + len(available()) and not any(problems.values()), problems
    clip = client.get("/api/v1/clips/seq011").json()
    assert clip["n_frames"] == N_FRAMES and clip["frame_size"] == [1920, 1080]
    assert sorted(e["layer"] for e in clip["layers"]) == available()
    made = json.loads((clip_dir / "clip.json").read_text())["produced_by"]
    assert sorted(made) == available() and all(made[n]["frames"] == N_FRAMES for n in made)

    layers = {n: client.get(f"/api/v1/clips/seq011/layers/{n}").json() for n in available()}
    assert {n: d["type"] for n, d in layers.items()} == {
        "density": "grid", "flow": "grid", "infrastructure": "polygons", "points": "points",
        "tracks": "tracks"}

    # Not only valid JSON: the counts on frame 0 must be near the ground truth.
    count = sum(map(sum, layers["density"]["frames"]["0"]["density"]))
    assert 0.7 * GT_FRAME_0 < count < 1.3 * GT_FRAME_0
    points = layers["points"]["frames"]
    assert len(points) == N_FRAMES and 0.7 * GT_FRAME_0 < len(points["0"]) < 1.4 * GT_FRAME_0
    assert len(layers["tracks"]["frames"][str(N_FRAMES - 1)]) > 0.5 * GT_FRAME_0
    assert layers["infrastructure"]["frames"]["0"]  # walls and vehicles on frame 0
    assert "0" in layers["flow"]["frames"]


def test_upload_and_run_through_the_api(tmp_path, settings, client, monkeypatch):
    from apps.pipeline import jobs

    settings.CLIPS_ROOT = tmp_path / "clips"
    settings.PIPELINE_RUNNER, settings.PIPELINE_DEVICE = "local", "cpu"
    video = tmp_path / "test 11.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080))
    for i in range(1, N_FRAMES + 1):
        out.write(cv2.imread(str(FRAMES / f"img011{i:03d}.jpg")))
    out.release()
    started = []
    monkeypatch.setattr(jobs, "spawn", started.append)  # the worker runs below, in this process

    with video.open("rb") as f:
        res = client.post("/api/v1/clips", {"file": f})
    assert res.status_code == 201 and res.json()["clip_id"] == "test_11"
    res = client.post("/api/v1/clips/test_11/runs")
    assert res.status_code == 202 and started == [res.json()["run_id"]]
    jobs.execute(started[0])

    run = client.get(f"/api/v1/clips/test_11/runs/{started[0]}").json()
    assert run["state"] == "done" and run["errors"] == {}, run
    assert sorted(run["layers"]) == available()
    problems = validator().validate_clip_dir(settings.CLIPS_ROOT / "test_11")
    assert not any(problems.values()), problems
    density = client.get("/api/v1/clips/test_11/layers/density").json()
    count = sum(map(sum, density["frames"]["0"]["density"]))
    assert 0.7 * GT_FRAME_0 < count < 1.3 * GT_FRAME_0  # mp4v upload, then crf 18
