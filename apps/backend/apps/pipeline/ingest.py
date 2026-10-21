"""Turn a video file or a folder of frames into a clip folder under ``CLIPS_ROOT``.

The clip gets ``video.mp4`` (H.264, yuv420p, so browsers can play it) and a ``clip.json`` with
no layers yet; ``run_models`` then adds the model layers. Frames are read with the same reader
the models use (``sentinel_ml.flow.sources.open_frames``), so a mp4 file and a DroneCrowd style
folder of numbered images both work.

Frames are piped straight into ffmpeg and encoded once at ``CRF`` 18. The models later decode
this video, and DM-Count is sensitive to compression (people are a few pixels wide): on test
clip 11 its counts differ from the original jpgs by 2.6% on average at crf 18, against 4.7% at
ffmpeg's default crf 23 and 3.0% when OpenCV's mp4v is re-encoded.

The clip is written into a temporary folder next to it and renamed at the end, so a failed
ingest leaves no half written clip behind.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from apps.clips.store import CLIP_ID
from apps.pipeline.runner import RunError

logger = logging.getLogger(__name__)

CRF = 18


def ingest_clip(root: Path, source: Path, clip_id: str, *, seq: str | None = None,
                fps: float | None = None, name: str | None = None,
                camera: dict | None = None) -> dict:
    """Write ``<root>/<clip_id>/{video.mp4, clip.json, layers/}`` and return the clip.json.

    ``camera`` (from ``apps.pipeline.camera.camera_block``) is stored as clip.json ``camera``;
    with it, flow reports metres per second.
    """
    from sentinel_ml.flow.export import find_ffmpeg
    from sentinel_ml.flow.sources import open_frames

    if not CLIP_ID.match(clip_id):
        raise RunError(f"bad clip id {clip_id!r}: use letters, digits, '_' and '-'")
    root, source = Path(root), Path(source)
    final = root / clip_id
    if final.exists():
        raise RunError(f"{final} already exists; delete it first to ingest again")
    if not source.exists():
        raise RunError(f"source not found: {source}")
    try:
        frames, src_fps, (w, h) = open_frames(source, seq, fps)
    except SystemExit as e:  # open_frames reports a bad source this way
        raise RunError(str(e)) from e
    if w % 2 or h % 2:
        raise RunError(f"frames are {w}x{h}; H.264 (yuv420p) needs an even width and height")
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RunError("no ffmpeg found, so no browser playable video can be made; "
                       'pip install -e "ml[export]" (imageio-ffmpeg) and try again')

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=f".{clip_id}.") as tmp:
        work = Path(tmp)
        (work / "layers").mkdir()
        video = work / "video.mp4"
        n = encode(ffmpeg, frames, video, src_fps, (w, h))

        clip = {"schema_version": 1, "clip_id": clip_id, "name": name or clip_id,
                "fps": float(src_fps), "frame_size": [w, h], "n_frames": n, "layers": []}
        if camera:
            clip["camera"] = camera
        (work / "clip.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
        work.rename(final)  # the temporary folder is gone now, so cleanup has nothing to do
    logger.info("ingested %s: %d frames at %g fps, %dx%d", clip_id, n, src_fps, w, h)
    return clip


def encode(ffmpeg: str, frames, video: Path, fps: float, size: tuple[int, int]) -> int:
    """Pipe BGR frames into ffmpeg as H.264; returns the number of frames written."""
    rec = Recorder(ffmpeg, video, fps, size)
    try:
        for frame in frames:
            rec.write(frame)
    except (RunError, OSError):
        rec.abort()
        raise
    return rec.close()


class Recorder:
    """Frames pushed one at a time into ffmpeg as browser playable H.264 (``video.mp4``).

    ``preset`` is ffmpeg's speed setting; a live stream uses a fast one, so recording keeps up
    with the source next to the models.
    """

    def __init__(self, ffmpeg: str, video: Path, fps: float, size: tuple[int, int],
                 preset: str = "medium") -> None:
        w, h = self.size = size
        self.n = 0
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", preset,
             "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        w, h = self.size
        if frame is None or frame.shape != (h, w, 3):
            raise RunError(f"frame {self.n} is missing or not {w}x{h} colour")
        self.proc.stdin.write(frame.tobytes())
        self.n += 1

    def abort(self) -> None:
        self.proc.kill()
        self.proc.wait()

    def close(self) -> int:
        """Finish the file; returns the number of frames written."""
        _, err = self.proc.communicate()
        if self.proc.returncode != 0:
            raise RunError(f"ffmpeg failed: {err.decode(errors='replace').strip()}")
        if self.n == 0:
            raise RunError("no frames read from the source")
        return self.n
