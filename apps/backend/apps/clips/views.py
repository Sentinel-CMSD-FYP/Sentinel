import json

from django.conf import settings
from django.http import FileResponse, HttpRequest, HttpResponse
from rest_framework.exceptions import NotFound as ApiNotFound
from rest_framework.exceptions import ParseError
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .ranges import RangeNotSatisfiable, parse_range
from .store import ClipStore, NotFound


def get_store() -> ClipStore:
    # Read per request so tests (and a changed CLIPS_ROOT) take effect immediately.
    return ClipStore(settings.CLIPS_ROOT)


class HealthView(APIView):
    def get(self, request: Request) -> Response:
        return Response({"status": "ok"})


class ClipListView(APIView):
    def get(self, request: Request) -> Response:
        return Response(get_store().list_clips())


class ClipDetailView(APIView):
    def get(self, request: Request, clip_id: str) -> Response:
        try:
            return Response(get_store().get_clip(clip_id))
        except NotFound as e:
            raise ApiNotFound(str(e)) from e


class LayerView(APIView):
    """A layer file, optionally cut to the frames in ``?from=&to=`` (inclusive)."""

    def get(self, request: Request, clip_id: str, layer: str) -> HttpResponse:
        try:
            path = get_store().layer_path(clip_id, layer)
        except NotFound as e:
            raise ApiNotFound(str(e)) from e

        lo, hi = request.query_params.get("from"), request.query_params.get("to")
        if lo is None and hi is None:
            return HttpResponse(path.read_bytes(), content_type="application/json")

        try:
            lo_n = int(lo) if lo is not None else 0
            hi_n = int(hi) if hi is not None else None
        except ValueError as e:
            raise ParseError("'from' and 'to' must be whole frame numbers") from e
        if lo_n < 0 or (hi_n is not None and hi_n < lo_n):
            raise ParseError("need 0 <= from <= to")

        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["frames"] = {
            k: v for k, v in doc["frames"].items()
            if int(k) >= lo_n and (hi_n is None or int(k) <= hi_n)
        }
        return HttpResponse(json.dumps(doc), content_type="application/json")


def video(request: HttpRequest, clip_id: str) -> HttpResponse:
    """The clip's video with byte-range support, so the player can seek."""
    try:
        path = get_store().video_path(clip_id)
    except NotFound as e:
        return HttpResponse(
            json.dumps({"detail": str(e)}), status=404, content_type="application/json"
        )

    size = path.stat().st_size
    try:
        span = parse_range(request.headers.get("Range"), size)
    except RangeNotSatisfiable:
        resp = HttpResponse(status=416)
        resp["Content-Range"] = f"bytes */{size}"
        return resp

    if span is None:
        resp = FileResponse(path.open("rb"), content_type="video/mp4")
        resp["Content-Length"] = str(size)
    else:
        start, end = span
        with path.open("rb") as fh:
            fh.seek(start)
            data = fh.read(end - start + 1)
        resp = HttpResponse(data, status=206, content_type="video/mp4")
        resp["Content-Range"] = f"bytes {start}-{end}/{size}"
    resp["Accept-Ranges"] = "bytes"
    return resp
