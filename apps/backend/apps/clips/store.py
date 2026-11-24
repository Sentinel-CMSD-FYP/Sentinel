"""Read clips and their layers from disk.

A clip is a folder under ``CLIPS_ROOT``::

    <clip_id>/clip.json
    <clip_id>/video.mp4
    <clip_id>/layers/<layer>.json

Names reach the filesystem only after matching a strict pattern, so a request cannot
walk out of the clips folder, and a layer is served only if ``clip.json`` declares it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CLIP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
LAYER_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class NotFound(Exception):
    """The clip, layer or video does not exist (or the name is not allowed)."""


class ClipIdMismatch(NotFound):
    """``clip.json`` names a different clip than its folder, e.g. a folder copied and renamed.

    Everything is looked up by folder name but listed by the ``clip_id`` inside, so a mismatch
    would show the wrong clip's data under another name. It is refused instead.
    """


@dataclass(frozen=True)
class ClipStore:
    root: Path

    def _clip_dir(self, clip_id: str) -> Path:
        if not CLIP_ID.match(clip_id):
            raise NotFound(f"no clip {clip_id!r}")
        path = self.root / clip_id
        if not (path / "clip.json").is_file():
            raise NotFound(f"no clip {clip_id!r}")
        return path

    def get_clip(self, clip_id: str) -> dict:
        """The clip's ``clip.json``, with ``video_url`` set only if a video file exists."""
        path = self._clip_dir(clip_id)
        clip = json.loads((path / "clip.json").read_text(encoding="utf-8"))
        if clip.get("clip_id") != clip_id:
            raise ClipIdMismatch(
                f"clip folder {clip_id!r} says clip_id {clip.get('clip_id')!r} in its clip.json; "
                f"they must match (edit clip.json, or rename the folder)"
            )
        if (path / "video.mp4").is_file():
            clip["video_url"] = f"/api/v1/clips/{clip_id}/video"
        else:
            clip.pop("video_url", None)
        return clip

    def list_clips(self) -> list[dict]:
        """A short summary of every readable clip, sorted by id. Broken clips are skipped."""
        if not self.root.is_dir():
            return []
        clips = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or not CLIP_ID.match(child.name):
                continue
            try:
                clip = self.get_clip(child.name)
            except ClipIdMismatch as e:
                logger.warning("skipping clip: %s", e)
                continue
            except NotFound:
                continue
            except (OSError, ValueError):
                logger.warning("skipping unreadable clip %s", child.name, exc_info=True)
                continue
            clips.append({
                "clip_id": clip["clip_id"],
                "name": clip.get("name", clip["clip_id"]),
                "fps": clip["fps"],
                "frame_size": clip["frame_size"],
                "n_frames": clip["n_frames"],
                "layers": [entry["layer"] for entry in clip["layers"]],
            })
        return clips

    def layer_path(self, clip_id: str, layer: str) -> Path:
        clip = self.get_clip(clip_id)
        declared = {entry["layer"] for entry in clip["layers"]}
        if not LAYER_NAME.match(layer) or layer not in declared:
            raise NotFound(f"clip {clip_id!r} has no layer {layer!r}")
        path = self._clip_dir(clip_id) / "layers" / f"{layer}.json"
        if not path.is_file():
            raise NotFound(f"layer {layer!r} of clip {clip_id!r} has no data yet")
        return path

    def video_path(self, clip_id: str) -> Path:
        self.get_clip(clip_id)  # refuses a folder whose clip.json names another clip
        path = self._clip_dir(clip_id) / "video.mp4"
        if not path.is_file():
            raise NotFound(f"clip {clip_id!r} has no video")
        return path
