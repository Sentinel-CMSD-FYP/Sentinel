"""Live streams: a source read at real time, the models run on it as it arrives.

``start_stream`` (the API: a stored clip's video replayed as if it came from a drone) or the
``run_stream`` command (any file, folder of frames, URL or webcam) writes
``<CLIPS_ROOT>/.streams/<stream_id>/status.json``; the worker (``execute``) runs
``sentinel_ml.pipeline.live.run_live`` and keeps these files in that folder current:

* ``samples.jsonl``: one line per layer sample, written as soon as the model made it:
  ``{"seq", "layer", "frame", "t_capture", "t_ready", "data"}`` (times are Unix seconds;
  ``data`` is that frame's value in the stored layer format, and the first sample of a layer
  also carries ``header``, the layer without its frames, and ``meta``, the clip.json entry
  ``add_layer`` will write for it: label, style, default_visible). Readers go on from a byte
  offset.
* ``preview.bin``: the newest frame read, as one JSON line (``frame``, ``t_capture``)
  followed by the JPEG (at most ``PREVIEW_WIDTH`` wide), replaced on every frame.
* ``video.mp4``: every frame read, recorded while it runs.
* ``stop``: made by ``request_stop``; the reader stops at the next frame.

At the end the recording and every finished layer become a normal clip ``<CLIPS_ROOT>/<id>``
(through ``add_layer``), so the stream replays in the viewer like any clip. Several streams
may run at once (``MAX_STREAMS``), one process each.
"""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path

from django.conf import settings

from apps.clips.store import CLIP_ID, ClipStore
from apps.pipeline import jobs
from apps.pipeline.ingest import Recorder
from apps.pipeline.runner import (
    RunError,
    add_layers,
    build_modules,
    layer_tool,
    module_names,
    record_producers,
)

logger = logging.getLogger(__name__)

STATUS_EVERY_S = 1.0  # the worker rewrites status.json this often (frames read, samples)
STALE_S = 60  # a running stream whose status is older than this lost its worker
MAX_LAG_S = 1.0  # a model further behind the newest frame than this skips frames
PREVIEW_WIDTH = 960
PREVIEW_QUALITY = 80
RECORD_PRESET = "veryfast"  # x264 speed, so recording keeps up next to the models

_start_lock = threading.Lock()


class StreamBusy(Exception):
    """``MAX_STREAMS`` streams are running already."""


def streams_dir() -> Path:
    return Path(settings.CLIPS_ROOT) / ".streams"  # a dot folder, so the clip list skips it


def folder_of(stream_id: str) -> Path | None:
    return streams_dir() / stream_id if CLIP_ID.match(stream_id) else None


def read_stream(stream_id: str) -> dict | None:
    """The stream's status, or None if there is none. A dead worker's stream reads as failed."""
    folder = folder_of(stream_id)
    return jobs.load_status(folder, STALE_S) if folder else None


def list_streams() -> list[dict]:
    """Every stream, newest first."""
    if not streams_dir().is_dir():
        return []
    found = (read_stream(f.name) for f in sorted(streams_dir().iterdir(), reverse=True))
    return [s for s in found if s]


def create(source: str, modules: list[str], name: str, source_clip: str | None = None,
           fps: float | None = None, camera: dict | None = None) -> dict:
    """Write a queued stream's status (the caller starts its worker); returns the status.

    ``fps`` is the frame rate of a frame folder (which carries none; 30 without it), ``camera``
    the source clip's clip.json ``camera`` block (flow in m/s; kept in the saved clip).
    """
    names = module_names(modules)
    with _start_lock:
        running = [s for s in list_streams() if s["state"] in jobs.ACTIVE]
        if len(running) >= settings.MAX_STREAMS:
            raise StreamBusy(f"{len(running)} streams are running, the limit is "
                             f"{settings.MAX_STREAMS} (MAX_STREAMS)")
        stream_id = f"live_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
        folder = streams_dir() / stream_id
        folder.mkdir(parents=True)
        status = {"stream_id": stream_id, "name": name, "source": source,
                  "source_clip": source_clip, "source_fps": fps, "camera": camera,
                  "modules": names,
                  "state": "queued",
                  "stage": "waiting for the worker", "created": jobs.now(),
                  "updated": jobs.now(), "started": None, "finished": None, "fps": None,
                  "frame_size": None, "frames_read": 0, "samples": 0, "layers": {},
                  "ran": {}, "skipped": {}, "late_s": 0.0, "errors": {}, "error": None,
                  "clip_id": None}
        jobs.write_status(folder, status)
    return status


def start_stream(store: ClipStore, clip_id: str, modules: list[str] | None = None) -> dict:
    """Replay a stored clip as a live stream in a background worker; returns the status.

    Raises NotFound for an unknown clip or one without video, RunError for unknown modules,
    StreamBusy at ``MAX_STREAMS``. Only the clip's own video is read, never a path from the
    request.
    """
    clip = store.get_clip(clip_id)
    video = store.video_path(clip_id)
    status = create(str(video), modules or ["all"], f"Live: {clip['name']}", clip_id,
                    camera=clip.get("camera"))
    spawn(status["stream_id"])
    logger.info("stream %s queued from clip %s", status["stream_id"], clip_id)
    return status


def spawn(stream_id: str) -> None:
    jobs.start_worker(streams_dir() / stream_id, "run_stream", "--worker", stream_id)


def request_stop(stream_id: str) -> None:
    (streams_dir() / stream_id / "stop").touch()


def read_samples(stream_id: str, after: int = 0, limit: int = 1 << 20) -> tuple[list, int]:
    """The samples written after byte ``after`` (about ``limit`` bytes at most) and the offset
    to ask next. A line still being written is left for the next call."""
    path = streams_dir() / stream_id / "samples.jsonl"
    try:
        with path.open("rb") as f:
            f.seek(after)
            chunk = f.read(limit)
            if len(chunk) == limit and b"\n" not in chunk:  # one huge line: read all of it
                chunk += f.readline()
    except FileNotFoundError:
        return [], after
    end = chunk.rfind(b"\n") + 1
    return [json.loads(line) for line in chunk[:end].splitlines()], after + end


def read_preview(stream_id: str) -> tuple[dict, bytes] | None:
    """``(header, jpeg)`` of the newest frame, or None if there is none (yet)."""
    try:
        raw = (streams_dir() / stream_id / "preview.bin").read_bytes()
    except OSError:  # missing, or being replaced right now (Windows)
        return None
    head, _, jpeg = raw.partition(b"\n")
    return json.loads(head), jpeg


class Writer:
    """The worker's side of the stream folder: samples, preview, recording and status."""

    def __init__(self, job: jobs.Job, recorder: Recorder, modules: list) -> None:
        self.job = job
        self.visible = {m.name: getattr(m, "default_visible", None) for m in modules}
        self.folder = job.folder
        self.recorder = recorder
        self.samples = (self.folder / "samples.jsonl").open("ab")
        self.seq = 0
        self.layers: dict[str, dict] = {}
        self.frames_read = 0

    def frame(self, frame_id: int, frame, t_capture: float) -> None:
        import cv2

        self.recorder.write(frame)
        self.frames_read = frame_id + 1
        h, w = frame.shape[:2]
        if w > PREVIEW_WIDTH:
            frame = cv2.resize(frame, (PREVIEW_WIDTH, round(h * PREVIEW_WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
        head = json.dumps({"frame": frame_id, "t_capture": t_capture}).encode()
        tmp = self.folder / "preview.bin.tmp"
        tmp.write_bytes(head + b"\n" + jpeg.tobytes())
        try:
            tmp.replace(self.folder / "preview.bin")
        except PermissionError:  # Windows: being read right now; the next frame comes soon
            pass

    def sample(self, layer: str, frame_id: int, data, header: dict | None,
               t_capture: float | None) -> None:
        line = {"seq": self.seq, "layer": layer, "frame": frame_id, "t_capture": t_capture,
                "t_ready": time.time(), "data": data}
        if header is not None:
            line["header"] = header
            line["meta"] = layer_tool().manifest_entry(header, None, self.visible.get(layer), None)
            self.layers[layer] = header
        self.samples.write(json.dumps(line).encode() + b"\n")
        self.samples.flush()
        self.seq += 1

    def report(self, **fields) -> None:
        self.job.update(frames_read=self.frames_read, samples=self.seq,
                        layers=dict(self.layers), **fields)


def execute(stream_id: str) -> dict:
    """The worker: run the stream to its end, then save it as a clip; returns the status."""
    from sentinel_ml.flow.export import find_ffmpeg
    from sentinel_ml.pipeline.live import open_source, run_live

    folder = streams_dir() / stream_id
    job = jobs.Job(stream_id, folder)
    job.update(state="running", stage="opening the source", started=jobs.now())
    writer = None
    stop = threading.Event()

    def beat():
        while not stop.wait(STATUS_EVERY_S):
            if writer:
                writer.report()
            else:
                job.update()

    threading.Thread(target=beat, daemon=True).start()
    try:
        try:
            frames, fps, (w, h), realtime = open_source(job.status["source"],
                                                        job.status.get("source_fps"))
        except ValueError as e:
            raise RunError(str(e)) from e
        if w % 2 or h % 2:
            raise RunError(f"frames are {w}x{h}; H.264 (yuv420p) needs an even width and height")
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise RunError('no ffmpeg found; pip install -e "ml[export]" (imageio-ffmpeg)')
        job.update(stage="loading the models", fps=float(fps), frame_size=[w, h])
        errors: dict[str, str] = {}
        camera = job.status.get("camera")
        modules = build_modules(job.status["modules"], settings.PIPELINE_DEVICE, fps, errors,
                                camera)
        if not modules:
            raise RunError(f"no module could be built: {errors}")

        writer = Writer(job, Recorder(ffmpeg, folder / "video.mp4", fps, (w, h), RECORD_PRESET),
                        modules)
        job.update(stage="live", errors=errors)
        try:
            result = run_live(frames, modules, stream_id, fps=fps, realtime=realtime,
                              max_lag_s=MAX_LAG_S, on_frame=writer.frame, on_sample=writer.sample,
                              stop=(folder / "stop").exists)
        except BaseException:
            writer.recorder.abort()
            raise
        finally:
            writer.samples.close()
        errors.update(result.errors)
        n = writer.recorder.close()
        writer.report(stage="saving the clip", ran=result.ran, skipped=result.skipped,
                      late_s=round(result.late_s, 3), errors=errors)
        clip_id = save_clip(stream_id, job.status["name"], fps, (w, h), n, modules, result,
                            errors, camera)
        writer.report(state="done", stage="done", clip_id=clip_id, errors=errors,
                      finished=jobs.now())
        logger.info("stream %s done: %d frames, clip %s, failed %s", stream_id, n, clip_id,
                    errors or "none")
    except Exception as e:  # any failure ends up in the status, with the traceback in log.txt
        logger.exception("stream %s failed", stream_id)
        why = str(e) if isinstance(e, RunError) else f"{type(e).__name__}: {e}"
        (writer.report if writer else job.update)(state="failed", stage="failed", error=why,
                                                  finished=jobs.now())
    finally:
        stop.set()
    return job.status


def save_clip(stream_id: str, name: str, fps: float, size: tuple[int, int], n_frames: int,
              modules: list, result, errors: dict[str, str], camera: dict | None = None) -> str:
    """Move the recording into a new clip ``<CLIPS_ROOT>/<stream_id>`` and add its layers."""
    root = Path(settings.CLIPS_ROOT)
    final = root / stream_id
    with tempfile.TemporaryDirectory(dir=root, prefix=f".{stream_id}.") as tmp:
        work = Path(tmp)
        (work / "layers").mkdir()
        shutil.move(streams_dir() / stream_id / "video.mp4", work / "video.mp4")
        clip = {"schema_version": 1, "clip_id": stream_id, "name": name, "fps": float(fps),
                "frame_size": list(size), "n_frames": n_frames, "layers": []}
        if camera:
            clip["camera"] = camera
        (work / "clip.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
        add_layers(work, result.layers, modules, errors=errors)
        added = [m for m in modules if m.name in result.layers and m.name not in errors]
        if added:
            record_producers(work, added, result.times, n_frames)
        work.rename(final)
    return stream_id
