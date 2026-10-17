"""Render a short video with coloured trails per track id, from a ``*_tracks.json`` file.

Each confirmed track is drawn as a dot at its current position plus a trail of its last
``--trail`` positions, in one colour per id. A steady colour on a person means a stable id;
a colour that keeps changing on the same person means id switches.

Units: pixels. Coordinates: (x, y) = (column, row), original frame pixels.

    python eval/render_trails.py --config ml/configs/dronecrowd_data.yaml --split test \
        --tracks data/outputs/tracks/test/clip_011_tracks.json --start 0 --seconds 5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from sentinel_ml.localization import data as D


def id_color(track_id: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour for a track id (golden-ratio hue steps)."""
    hue = int((track_id * 0.61803398875 % 1.0) * 179)
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_frame(image_bgr: np.ndarray, rows: list[list[float]],
               history: dict[int, deque]) -> np.ndarray:
    """Draw trails and current dots for the track rows of one frame; updates ``history``."""
    out = image_bgr.copy()
    for tid, x, y, _ in rows:
        history[int(tid)].append((round(x), round(y)))
    for tid, trail in history.items():
        color = id_color(tid)
        pts = list(trail)
        for a, b in zip(pts[:-1], pts[1:], strict=False):
            cv2.line(out, a, b, color, 2, cv2.LINE_AA)
    for tid, x, y, _ in rows:
        cv2.circle(out, (round(x), round(y)), 6, id_color(int(tid)), -1, cv2.LINE_AA)
        cv2.circle(out, (round(x), round(y)), 6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the dronecrowd_data.yaml")
    ap.add_argument("--split", choices=D.SPLITS, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0, help="first frame_id")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--trail", type=int, default=40, help="trail length in frames")
    ap.add_argument("--width", type=int, default=1280, help="output width; height keeps 16:9")
    ap.add_argument("--out", type=Path, help="default: data/outputs/trails/<clip>_f<start>.mp4")
    args = ap.parse_args()

    doc = json.loads(args.tracks.read_text())
    clip = int(doc["clip_id"].split("_")[1])
    fps = float(doc["fps"] or 25)
    n_frames = round(args.seconds * fps / int(doc.get("frame_stride", 1)))
    ds = D.dataset_from_config(D.load_config(args.config), args.split)
    refs = {r.frame_id: r for r in ds.refs if r.clip == clip}
    ids = [i for i in range(args.start, args.start + n_frames * int(doc["frame_stride"]),
                            int(doc["frame_stride"])) if i in refs]
    size = (args.width, round(args.width * doc["frame_size"][1] / doc["frame_size"][0]))
    out_path = args.out or Path("data/outputs/trails") / f"{doc['clip_id']}_f{args.start}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / int(doc["frame_stride"]), size)
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.trail))
    for fid in ids:
        img = cv2.imread(str(refs[fid].image_path))
        rows = doc["frames"].get(str(fid), [])
        frame = draw_frame(img, rows, history)
        cv2.putText(frame, f"{doc['clip_id']} frame {fid}  tracks {len(rows)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    writer.release()
    print(f"wrote {out_path}: {len(ids)} frames at {fps / int(doc['frame_stride']):g} fps")


if __name__ == "__main__":
    main()
