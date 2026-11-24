"""Frames from a video file or from a folder of images, behind one interface.

Datasets come both ways (DroneCrowd is numbered jpgs, YouTube clips are mp4), and the flow
tools should not care which.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def open_frames(
    path: Path, seq: str | None = None, fps: float | None = None
) -> tuple[Iterator[np.ndarray], float, tuple[int, int]]:
    """``(frames, fps, (width, height))`` for an mp4, or a folder of numbered images.

    In a folder, frames are taken in name order; ``seq`` keeps only the DroneCrowd-style
    names ``img<seq>*``. A folder carries no frame rate, so it is ``fps`` or 30.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(
            f for f in path.iterdir()
            if f.suffix.lower() in IMAGE_SUFFIXES and (not seq or f.name.startswith(f"img{seq}"))
        )
        if not files:
            raise SystemExit(f"no image frames in {path} (seq={seq})")
        first = cv2.imread(str(files[0]))
        if first is None:
            raise SystemExit(f"cannot read {files[0]}")
        h, w = first.shape[:2]
        return (cv2.imread(str(f)) for f in files), fps or 30.0, (w, h)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read() -> Iterator[np.ndarray]:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                return
            yield frame

    return read(), fps or cap.get(cv2.CAP_PROP_FPS) or 30.0, size
