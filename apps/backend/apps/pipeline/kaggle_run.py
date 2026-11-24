"""The ``kaggle`` runner: a clip's models run on a Kaggle GPU and the layers come back here.

1. The clip's ``video.mp4`` and ``clip.json`` plus a ``request.json`` (run and clip id) become a
   new version of the private dataset ``<user>/sentinel-run-clip``.
2. Once Kaggle says the dataset is ready, the kernel ``<user>/sentinel-run-models``
   (``kaggle_kernel/``) is pushed. It runs ``manage.py run_models <clip> --modules all --device
   cuda`` with the trained weights, the same command as on this machine.
3. The kernel status is polled (Kaggle queues it until a GPU is free, then runs it).
4. Its output (layers, clip.json ``produced_by``, a report of failed modules, its log) is
   downloaded and each layer is added here with ``add_layer``, which checks it again.

The kernel takes the code from the dataset ``sentinel-ml-src``, so code changes reach Kaggle only
after ``python eval/density_dmcount/kaggle/push.py src``. The kaggle CLI needs the API token in
``~/.kaggle/access_token`` and ``KAGGLE_USERNAME``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from django.conf import settings

from apps.clips.store import ClipStore
from apps.pipeline.runner import RunError, layer_tool

logger = logging.getLogger(__name__)

DATASET = "sentinel-run-clip"
KERNEL = "sentinel-run-models"
KERNEL_DIR = Path(__file__).resolve().parent / "kaggle_kernel"
POLL_S = 30
READY_TIMEOUT_S = 1800  # a new dataset version usually takes under a minute
STATUS = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
MAX_CLI_FAILURES = 40  # consecutive failed status calls (a dropped connection) before giving up
STALE_GRACE_S = 600  # how long after the push a finished output may still be the previous run's


def cli(*args: str, timeout: float | None = 120) -> subprocess.CompletedProcess:
    """Run the kaggle CLI of this Python environment; never raises on a non zero exit."""
    cmd = [sys.executable, "-m", "kaggle", *args]
    logger.info("$ kaggle %s", " ".join(args))
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "PYTHONUTF8": "1"}, check=False)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 1, "", f"timed out after {timeout} s")


def checked(*args: str, timeout: float | None = 120) -> str:
    res = cli(*args, timeout=timeout)
    if res.returncode:
        raise RunError(f"kaggle {' '.join(args[:2])} failed: {(res.stderr or res.stdout).strip()}")
    return res.stdout


def user() -> str:
    if not settings.KAGGLE_USERNAME:
        raise RunError("set KAGGLE_USERNAME (the account that runs the kernel) before starting "
                       "the backend, or use PIPELINE_RUNNER=local on a machine with a GPU")
    return settings.KAGGLE_USERNAME


def sleep(seconds: float) -> None:  # tests replace it
    time.sleep(seconds)


def upload_clip(job, clip_dir: Path, owner: str) -> None:
    """Steps 1 and 2: the clip as a new dataset version, then wait until Kaggle has it ready."""
    stage = job.folder / "dataset"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    try:
        try:
            os.link(clip_dir / "video.mp4", stage / "video.mp4")  # no copy of a 1 GB file
        except OSError:
            shutil.copy2(clip_dir / "video.mp4", stage / "video.mp4")
        clip = json.loads((clip_dir / "clip.json").read_text(encoding="utf-8"))
        clip.update(layers=[])  # the kernel makes every layer afresh
        clip.pop("produced_by", None)
        (stage / "clip.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
        request = {"run_id": job.status["run_id"], "clip_id": job.status["clip_id"]}
        (stage / "request.json").write_text(json.dumps(request), encoding="utf-8")
        ref = f"{owner}/{DATASET}"
        (stage / "dataset-metadata.json").write_text(json.dumps(
            {"title": DATASET, "id": ref, "licenses": [{"name": "other"}]}), encoding="utf-8")

        job.update(stage="uploading the clip to Kaggle")
        if cli("datasets", "status", ref).returncode == 0:
            checked("datasets", "version", "-p", str(stage), "-m", request["run_id"],
                    timeout=None)
        else:  # new datasets are private unless --public is passed
            checked("datasets", "create", "-p", str(stage), timeout=None)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    # A kernel pushed before the new version is ready runs on the old one.
    job.update(stage="Kaggle is preparing the clip")
    deadline = time.monotonic() + READY_TIMEOUT_S
    while True:
        sleep(POLL_S)
        res = cli("datasets", "status", ref)
        if res.returncode == 0 and res.stdout.strip().lower().endswith("ready"):
            return
        if time.monotonic() > deadline:
            raise RunError(f"dataset {ref} not ready after {READY_TIMEOUT_S} s: "
                           f"{(res.stdout or res.stderr).strip()}")


def push_kernel(job, owner: str) -> str:
    """Step 2: push the kernel folder with the account filled in; returns the kernel ref."""
    stage = job.folder / "kernel"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(KERNEL_DIR, stage, ignore=shutil.ignore_patterns("__pycache__"))
    meta = stage / "kernel-metadata.json"
    meta.write_text(meta.read_text(encoding="utf-8").replace("{user}", owner), encoding="utf-8")
    job.update(stage="starting the Kaggle kernel")
    checked("kernels", "push", "-p", str(stage))
    return f"{owner}/{KERNEL}"


def wait_and_download(job, ref: str) -> dict:
    """Steps 3 and 4: poll the kernel until it ends, download its output, return result.json.

    Right after a push the status can still be the previous version's, so a finished output
    whose run id is not ours is not taken: polling goes on until our version ends.
    """
    out = job.folder / "output"
    pushed = time.monotonic()
    deadline = pushed + settings.KAGGLE_RUN_TIMEOUT_H * 3600
    failures = 0
    while True:
        if time.monotonic() > deadline:
            raise RunError(f"the Kaggle kernel did not finish within "
                           f"{settings.KAGGLE_RUN_TIMEOUT_H} h; see https://www.kaggle.com/code/"
                           f"{ref}")
        sleep(POLL_S)
        res = cli("kernels", "status", ref)
        found = STATUS.search(res.stdout or "")
        if not found:  # a dropped connection: the kernel goes on regardless
            failures += 1
            if failures > MAX_CLI_FAILURES:
                raise RunError(f"kaggle kernels status keeps failing: {res.stderr.strip()}")
            continue
        failures = 0
        state = found.group(1)
        if state in ("QUEUED", "RUNNING"):
            job.update(stage="Kaggle: waiting for a GPU" if state == "QUEUED"
                       else "Kaggle: running the models on the GPU")
            continue

        job.update(stage="downloading the layers from Kaggle")
        shutil.rmtree(out, ignore_errors=True)
        checked("kernels", "output", ref, "-p", str(out), timeout=1800)
        log = out / "eval.log"
        if log.is_file():
            shutil.copy2(log, job.folder / "kaggle.log")
        try:
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = {}
        if result.get("run_id") != job.status["run_id"]:
            if state == "COMPLETE" and time.monotonic() - pushed < STALE_GRACE_S:
                continue  # the previous version's output; ours is still on its way
            raise RunError(f"the Kaggle kernel ended with {state}; see {job.folder / 'kaggle.log'}"
                           f" or https://www.kaggle.com/code/{ref}")
        return result


def add_layers(clip_dir: Path, out: Path, result: dict, replace: bool) -> tuple[list, dict]:
    """Add each returned layer to the local clip, then its clip.json ``produced_by`` entry."""
    report = result["report"]
    errors = dict(report.get("errors", {}))
    made = json.loads((out / "clip" / "clip.json").read_text(encoding="utf-8"))
    visible = {e["layer"]: e.get("default_visible") for e in made["layers"]}
    tool = layer_tool()
    added = []
    for name in report.get("layers", []):
        try:
            tool.add_layer(clip_dir, out / "clip" / "layers" / f"{name}.json", replace=replace,
                           visible=visible.get(name))
            added.append(name)
        except (tool.LayerError, OSError) as e:
            errors[name] = f"layer refused: {e}"

    path = clip_dir / "clip.json"
    clip = json.loads(path.read_text(encoding="utf-8"))
    producers = clip.get("produced_by") if isinstance(clip.get("produced_by"), dict) else {}
    producers.update({n: made["produced_by"][n] for n in added if n in made.get("produced_by", {})})
    clip["produced_by"] = producers
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clip, indent=2), encoding="utf-8")
    tmp.replace(path)
    return added, errors


def run_kaggle(job) -> tuple[list[str], dict[str, str]]:
    """The runner: returns the added layers and the failed modules (as ``run_local`` does)."""
    owner = user()
    store = ClipStore(Path(settings.CLIPS_ROOT))
    clip_dir = store.video_path(job.status["clip_id"]).parent
    upload_clip(job, clip_dir, owner)
    ref = push_kernel(job, owner)
    result = wait_and_download(job, ref)
    if result.get("error"):
        raise RunError(f"on Kaggle: {result['error']}; see {job.folder / 'kaggle.log'}")
    job.update(stage="adding the layers")
    return add_layers(clip_dir, job.folder / "output", result, job.status["replace"])
