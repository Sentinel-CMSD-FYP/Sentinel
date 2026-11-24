"""Model runs started from the API: one background process per run, its state in a JSON file.

``start_run`` writes ``<CLIPS_ROOT>/.runs/<run_id>/status.json`` (state ``queued``) and starts
``manage.py run_job <run_id>`` as a separate process, so a run of many minutes outlives the
request and a reload of the dev server. The worker (``execute``) runs every registered module
with the runner named by ``settings.PIPELINE_RUNNER``:

* ``kaggle`` (default): the clip goes to a Kaggle GPU kernel and the layers it makes come back
  (``apps.pipeline.kaggle_run``). This PC has no CUDA GPU.
* ``local``: ``run_models`` on this machine, for a backend on a machine with a CUDA GPU.

The worker rewrites status.json at every stage and at least every ``HEARTBEAT_S`` seconds. A
queued or running status older than ``STALE_S`` belongs to a worker that died (killed, crashed),
so it is reported as failed and no longer blocks new runs. Only one run at a time on the whole
server, since the models take all of the GPU (or CPU).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from django.conf import settings

from apps.clips.store import CLIP_ID, ClipStore
from apps.pipeline.runner import RunError, build_modules, run_models

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
HEARTBEAT_S = 30
STALE_S = 300

_start_lock = threading.Lock()  # the dev server is threaded: two POSTs must not both start


class RunBusy(Exception):
    """Another run is still going; the message is its run id."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def runs_dir() -> Path:
    return Path(settings.CLIPS_ROOT) / ".runs"  # a dot folder, so the clip list skips it


def write_status(folder: Path, status: dict) -> None:
    tmp = folder / "status.json.tmp"  # never half written for a reader
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(folder / "status.json")
            return
        except PermissionError:  # Windows: the API is reading it this very moment
            if attempt == 19:
                raise
            time.sleep(0.01)


def load_status(folder: Path, stale_s: float) -> dict | None:
    """``folder/status.json``, or None if missing. A dead worker's job reads as failed."""
    path = folder / "status.json"
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    age = (datetime.now(UTC) - datetime.fromisoformat(status["updated"])).total_seconds()
    if status["state"] in ACTIVE and age > stale_s:
        status.update(state="failed", stage="failed",
                      error=f"the worker stopped answering {age:.0f} s ago (killed or crashed); "
                            f"see {folder / 'log.txt'}")
    return status


def read_status(run_id: str) -> dict | None:
    """The run's status, or None if there is no such run. A dead worker's run reads as failed."""
    if not CLIP_ID.match(run_id):
        return None
    return load_status(runs_dir() / run_id, STALE_S)


def active_run() -> dict | None:
    if not runs_dir().is_dir():
        return None
    for folder in sorted(runs_dir().iterdir()):
        status = read_status(folder.name)
        if status and status["state"] in ACTIVE:
            return status
    return None


def start_run(store: ClipStore, clip_id: str, replace: bool = False) -> dict:
    """Queue a run of every module on the clip and start its worker; returns the status.

    Raises NotFound for an unknown clip, RunError if the clip already has layers and ``replace``
    is not set, RunBusy if another run is still going.
    """
    clip = store.get_clip(clip_id)
    if clip["layers"] and not replace:
        raise RunError(f"clip {clip_id!r} already has layers "
                       f"{[e['layer'] for e in clip['layers']]}; send replace to redo them")
    with _start_lock:
        if busy := active_run():
            raise RunBusy(busy["run_id"])
        run_id = f"{clip_id}_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
        folder = runs_dir() / run_id
        folder.mkdir(parents=True)
        status = {"run_id": run_id, "clip_id": clip_id, "runner": settings.PIPELINE_RUNNER,
                  "replace": replace, "state": "queued", "stage": "waiting for the worker",
                  "created": now(), "updated": now(), "finished": None, "progress": None,
                  "layers": [], "errors": {}, "error": None}
        write_status(folder, status)
        spawn(run_id)
    logger.info("run %s queued (%s runner)", run_id, status["runner"])
    return status


def spawn(run_id: str) -> None:
    """Start ``manage.py run_job <run_id>`` detached, its output in the run's log.txt."""
    start_worker(runs_dir() / run_id, "run_job", run_id)


def start_worker(folder: Path, *command: str) -> None:
    """Start ``manage.py <command>`` detached, its output in ``folder/log.txt``."""
    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {
        "start_new_session": True}  # Ctrl+C on the dev server does not kill the run
    env = {**os.environ, "CLIPS_ROOT": str(settings.CLIPS_ROOT), "PYTHONUTF8": "1"}
    with (folder / "log.txt").open("ab") as log:
        subprocess.Popen([sys.executable, str(Path(settings.BASE_DIR) / "manage.py"), *command],
                         cwd=settings.BASE_DIR, env=env, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, **flags)


class Job:
    """A run's status as the worker updates it; every update is written at once."""

    def __init__(self, run_id: str, folder: Path | None = None):
        self.folder = folder or runs_dir() / run_id
        self.status = json.loads((self.folder / "status.json").read_text(encoding="utf-8"))
        self._lock = threading.Lock()

    def update(self, **fields) -> None:
        with self._lock:
            self.status.update(fields, updated=now())
            write_status(self.folder, self.status)


def run_local(job: Job) -> tuple[list[str], dict[str, str]]:
    """Every module on this machine; returns the added layers and the failed modules."""
    store = ClipStore(Path(settings.CLIPS_ROOT))
    clip_id = job.status["clip_id"]
    clip = store.get_clip(clip_id)
    errors: dict[str, str] = {}
    job.update(stage="loading the models")
    modules = build_modules(["all"], settings.PIPELINE_DEVICE, clip.get("fps"), errors,
                            clip.get("camera"))
    if modules:
        job.update(stage="running the models", progress={"done": 0, "total": clip["n_frames"]})
        run_models(store, clip_id, modules, job.status["replace"], errors,
                   progress=lambda done, total: job.update(progress={"done": done,
                                                                     "total": total}))
    return [m.name for m in modules if m.name not in errors], errors


def runners() -> dict:
    from apps.pipeline.kaggle_run import run_kaggle

    return {"local": run_local, "kaggle": run_kaggle}


def execute(run_id: str) -> dict:
    """The worker: run the job to its end, keeping status.json current; returns the status."""
    job = Job(run_id)
    job.update(state="running", stage="starting")
    stop = threading.Event()

    def beat():
        while not stop.wait(HEARTBEAT_S):
            job.update()

    threading.Thread(target=beat, daemon=True).start()
    try:
        runner = runners().get(job.status["runner"])
        if runner is None:
            raise RunError(f"unknown PIPELINE_RUNNER {job.status['runner']!r}; use kaggle or local")
        layers, errors = runner(job)
        job.update(state="done", stage="done", layers=layers, errors=errors, finished=now())
        logger.info("run %s done: layers %s, failed %s", run_id, layers, errors or "none")
    except Exception as e:  # any failure ends up in the status, with the traceback in log.txt
        logger.exception("run %s failed", run_id)
        why = str(e) if isinstance(e, RunError) else f"{type(e).__name__}: {e}"
        job.update(state="failed", stage="failed", error=why, finished=now())
    finally:
        stop.set()
    return job.status
