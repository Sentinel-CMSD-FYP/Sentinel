"""Upload a video and run the models on it from the browser.

* ``POST /clips`` (multipart: ``file``, optional ``name``): the video becomes a clip with
  ``ingest_clip`` before the reply (201 + clip.json). The clip id comes from the file name.
  Optional camera fields (``altitude_m``, ``fov_deg``, ``fov_kind``, ``sensor_aspect``,
  ``camera_model``, ``altitude_source``; ``apps.pipeline.camera``) become clip.json ``camera``.
* ``POST /clips/<id>/runs`` (optional ``replace``): starts every module in the background
  (``apps.pipeline.jobs``); 202 + the run status. One run at a time (409 otherwise).
* ``GET /clips/<id>/runs/<run_id>``: the run status (``state`` queued, running, done, failed).
* ``GET /runs/active``: ``{"run": status}`` of the run going now (any clip), null if none, so
  a reloaded page picks it up.

Live streams (``apps.pipeline.streams``):

* ``POST /streams`` (``clip_id``, optional ``modules``): replays a stored clip at real time as a
  live stream; 202 + the stream status. 409 at ``MAX_STREAMS``.
* ``GET /streams``: every stream, newest first. ``GET /streams/<id>``: one stream's status.
* ``POST /streams/<id>/stop``: the stream stops reading and becomes a clip; 202.
* ``GET /streams/<id>/samples?after=<offset>``: the layer samples made since ``offset``, in
  order, and ``next``, the offset to ask next.
* ``GET /streams/<id>/video``: the live picture as MJPEG; each part has ``X-Frame-Id`` and
  ``X-Capture-Time`` headers, so a client can match overlays to the frame shown.
"""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from rest_framework import status as http
from rest_framework.exceptions import NotFound as ApiNotFound
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.clips.store import NotFound
from apps.clips.views import ClipListView, get_store
from apps.pipeline import jobs, streams
from apps.pipeline.camera import camera_from_form
from apps.pipeline.ingest import ingest_clip
from apps.pipeline.runner import RunError

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
ID_LENGTH = 40
NAME_LENGTH = 100


def clip_id_for(root: Path, filename: str) -> str:
    """A safe, unused clip id from a file name: ``my drone clip.mp4`` -> ``my_drone_clip``.

    Only letters, digits, ``_`` and ``-`` are kept, so the client's name never reaches a path;
    ``_2``, ``_3``... is added when the id is taken.
    """
    stem = re.split(r"[\\/]", filename)[-1].rsplit(".", 1)[0]
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")[:ID_LENGTH].rstrip("_-") or "clip"
    clip_id, n = base, 2
    while (root / clip_id).exists():
        clip_id, n = f"{base}_{n}", n + 1
    return clip_id


def detail(message: str, code: int, **extra) -> Response:
    return Response({"detail": message, **extra}, status=code)


class ClipsView(ClipListView):
    """GET lists the clips (as before); POST uploads a video as a new clip."""

    def post(self, request: Request) -> Response:
        limit = settings.MAX_UPLOAD_MB * 1024 * 1024
        if int(request.META.get("CONTENT_LENGTH") or 0) > limit:  # refused before it is read
            return detail(f"the file is over the {settings.MAX_UPLOAD_MB} MB limit", 413)
        upload = request.FILES.get("file")
        if upload is None:
            return detail("send the video as the form field 'file'", 400)
        suffix = Path(upload.name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            return detail(f"not a video file ({suffix or 'no extension'}); use "
                          f"{', '.join(sorted(VIDEO_SUFFIXES))}", 400)
        name = (request.data.get("name") or "").strip()[:NAME_LENGTH] or upload.name[:NAME_LENGTH]
        try:
            camera = camera_from_form(request.data)
        except ValueError as e:
            return detail(f"camera: {e}", 400)

        store = get_store()
        store.root.mkdir(parents=True, exist_ok=True)
        clip_id = clip_id_for(store.root, upload.name)
        with tempfile.TemporaryDirectory(dir=store.root, prefix=".upload.") as tmp:
            source = Path(tmp) / f"video{suffix}"
            with source.open("wb") as f:
                for chunk in upload.chunks():
                    f.write(chunk)
            try:
                ingest_clip(store.root, source, clip_id, name=name, camera=camera)
            except RunError as e:
                return detail(f"could not read the video: {e}", 400)
        return Response(store.get_clip(clip_id), status=http.HTTP_201_CREATED)


class RunsView(APIView):
    def post(self, request: Request, clip_id: str) -> Response:
        replace = request.data.get("replace") in (True, "true", "1")
        try:
            run = jobs.start_run(get_store(), clip_id, replace)
        except NotFound as e:
            raise ApiNotFound(str(e)) from e
        except jobs.RunBusy as e:
            return detail(f"run {e} is still going; one run at a time", 409, run_id=str(e))
        except RunError as e:
            return detail(str(e), 409)
        return Response(run, status=http.HTTP_202_ACCEPTED)


class RunView(APIView):
    def get(self, request: Request, clip_id: str, run_id: str) -> Response:
        run = jobs.read_status(run_id)
        if run is None or run["clip_id"] != clip_id:
            raise ApiNotFound(f"clip {clip_id!r} has no run {run_id!r}")
        return Response(run)


class ActiveRunView(APIView):
    def get(self, request: Request) -> Response:
        return Response({"run": jobs.active_run()})  # a bare null would be an empty body


def stream_or_404(stream_id: str) -> dict:
    status = streams.read_stream(stream_id)
    if status is None:
        raise ApiNotFound(f"no stream {stream_id!r}")
    return status


class StreamsView(APIView):
    def get(self, request: Request) -> Response:
        return Response({"streams": streams.list_streams()})

    def post(self, request: Request) -> Response:
        clip_id = request.data.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            return detail("send the clip to replay as 'clip_id'", 400)
        modules = request.data.get("modules") or ["all"]
        if isinstance(modules, str):
            modules = [m.strip() for m in modules.split(",") if m.strip()]
        if not isinstance(modules, list) or not all(isinstance(m, str) for m in modules):
            return detail("'modules' is a list of module names or a comma separated string", 400)
        try:
            status = streams.start_stream(get_store(), clip_id, modules)
        except NotFound as e:
            raise ApiNotFound(str(e)) from e
        except streams.StreamBusy as e:
            return detail(str(e), 409)
        except RunError as e:
            return detail(str(e), 400)
        return Response(status, status=http.HTTP_202_ACCEPTED)


class StreamView(APIView):
    def get(self, request: Request, stream_id: str) -> Response:
        return Response(stream_or_404(stream_id))


class StreamStopView(APIView):
    def post(self, request: Request, stream_id: str) -> Response:
        status = stream_or_404(stream_id)
        if status["state"] not in jobs.ACTIVE:
            return detail(f"stream {stream_id!r} has ended ({status['state']})", 409)
        streams.request_stop(stream_id)
        return Response(status, status=http.HTTP_202_ACCEPTED)


class StreamSamplesView(APIView):
    def get(self, request: Request, stream_id: str) -> Response:
        status = stream_or_404(stream_id)
        after = request.query_params.get("after", "0")
        if not after.isdigit():
            return detail("'after' is the 'next' offset of the previous reply (0 at first)", 400)
        samples, nxt = streams.read_samples(stream_id, int(after))
        return Response({"samples": samples, "next": nxt, "state": status["state"]})


MJPEG_BOUNDARY = "frame"
MJPEG_POLL_S = 0.02


def stream_video(request: HttpRequest, stream_id: str) -> HttpResponse:
    """The newest frames as ``multipart/x-mixed-replace`` JPEGs until the stream ends."""
    if streams.read_stream(stream_id) is None:
        return JsonResponse({"detail": f"no stream {stream_id!r}"}, status=404)

    def parts():
        last, checked, active = None, 0.0, True
        while True:
            got = streams.read_preview(stream_id)
            if got and got[0]["frame"] != last:
                head, jpeg = got
                last = head["frame"]
                yield (f"--{MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                       f"Content-Length: {len(jpeg)}\r\nX-Frame-Id: {head['frame']}\r\n"
                       f"X-Capture-Time: {head['t_capture']:.3f}\r\n\r\n").encode() + jpeg + b"\r\n"
                continue
            if not active:
                return  # ended, and its last frame has gone out
            if time.monotonic() - checked > 1.0:
                checked = time.monotonic()
                status = streams.read_stream(stream_id)
                # None is a read that met the worker replacing status.json (Windows): ask again.
                if status is not None:
                    active = status["state"] in jobs.ACTIVE
                continue
            time.sleep(MJPEG_POLL_S)

    resp = StreamingHttpResponse(
        parts(), content_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}")
    resp["Cache-Control"] = "no-store"
    return resp
