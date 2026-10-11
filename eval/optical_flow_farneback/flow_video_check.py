"""Run Farnebäck flow over a video and write an annotated copy plus timing stats.

Qualitative check on real footage; the synthetic-frame unit tests live in
``ml/tests``. The clip should come from a hovering drone or a static camera,
because frame registration is not built yet and camera drift would show up as
crowd motion.

    python eval/optical_flow_farneback/flow_video_check.py clip.mp4 --out out/clip_flow.mp4
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from sentinel_ml.flow.farneback import FarnebackConfig, FarnebackFlow, FlowField, pool_regions


def colour_overlay(frame: np.ndarray, flow: FlowField, max_speed: float) -> np.ndarray:
    """Blend an HSV flow map (hue = direction, brightness = speed) onto the frame."""
    h, w = frame.shape[:2]
    u = cv2.resize(flow.u, (w, h), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(flow.v, (w, h), interpolation=cv2.INTER_LINEAR)
    mag, ang = cv2.cartToPolar(u, v)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)  # OpenCV hue is 0-179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_speed * 255, 0, 255).astype(np.uint8)
    return cv2.addWeighted(frame, 0.5, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0.5, 0)


def draw_arrows(img: np.ndarray, flow: FlowField, rows: int, cols: int, gain: float) -> None:
    """Draw one mean-flow arrow per grid region, ``gain`` times the per-frame motion."""
    h, w = img.shape[:2]
    s = pool_regions(flow, rows, cols)
    for r in range(rows):
        for c in range(cols):
            x, y = int((c + 0.5) * w / cols), int((r + 0.5) * h / rows)
            tip = (int(x + s.mean_u[r, c] * gain), int(y + s.mean_v[r, c] * gain))
            cv2.arrowedLine(img, (x, y), tip, (255, 255, 255), 2, tipLength=0.3)


def open_frames(
    path: Path, seq: str | None, fps: float | None
) -> tuple[Iterator[np.ndarray], float, tuple[int, int]]:
    """Frames from an mp4, or from a folder of ``img<seq><frame>.jpg`` (DroneCrowd layout)."""
    if path.is_dir():
        files = sorted(path.glob(f"img{seq}*.jpg" if seq else "*.jpg"))
        if not files:
            raise SystemExit(f"no jpg frames in {path} (seq={seq})")
        h, w = cv2.imread(str(files[0])).shape[:2]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path, help="mp4 file, or a folder of jpg frames")
    ap.add_argument("--seq", default=None, help="sequence id in a frame folder, e.g. 011")
    ap.add_argument("--fps", type=float, default=None, help="override the frame rate")
    ap.add_argument("--out", type=Path, default=None, help="annotated mp4 (omit for stats only)")
    ap.add_argument("--config", type=Path, default=Path("ml/configs/flow_farneback.yaml"))
    ap.add_argument("--start", type=int, default=0, help="first frame to read")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (lower fps)")
    ap.add_argument("--grid", type=int, nargs=2, default=(6, 10), metavar=("ROWS", "COLS"))
    ap.add_argument("--max-speed", type=float, default=20.0,
                    help="px/frame that maps to full brightness in the colour overlay")
    ap.add_argument("--no-arrows", action="store_true")
    args = ap.parse_args()

    frames, src_fps, size = open_frames(args.video, args.seq, args.fps)

    cfg = FarnebackConfig.from_yaml(args.config)
    flow = FarnebackFlow(cfg)
    writer = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps / args.stride, size
        )

    print(f"{args.video}  {size[0]}x{size[1]} @ {src_fps:.1f} fps  downscale={cfg.downscale}")
    print(f"{'frame':>6} {'ms':>7} {'mean px/f':>10} {'p95 px/f':>9}")

    ms_all: list[float] = []
    mean_all: list[float] = []
    used = 0
    for idx, frame in enumerate(islice(frames, args.start, None), args.start):
        if (idx - args.start) % args.stride:
            continue
        t0 = time.perf_counter()
        field = flow.update(frame)
        ms = (time.perf_counter() - t0) * 1000
        if field is None:
            continue
        used += 1

        mag = field.magnitude
        ms_all.append(ms)
        mean_all.append(float(mag.mean()))
        if used % 25 == 1:
            print(f"{idx:>6} {ms:>7.1f} {mag.mean():>10.2f} {np.percentile(mag, 95):>9.2f}")

        if writer:
            out = colour_overlay(frame, field, args.max_speed)
            if not args.no_arrows:
                draw_arrows(out, field, *args.grid, gain=4.0)
            writer.write(out)
        if args.max_frames and used >= args.max_frames:
            break

    if writer:
        writer.release()
    if not ms_all:
        raise SystemExit("no flow computed; need at least two frames")

    ms_arr = np.array(ms_all)
    print(f"\nflow fields: {len(ms_arr)}")
    print(f"flow time: median {np.median(ms_arr):.1f} ms, p95 {np.percentile(ms_arr, 95):.1f} ms "
          f"(~{1000 / np.median(ms_arr):.0f} fps)")
    print(f"mean speed: {np.mean(mean_all):.2f} px/frame (original-frame pixels)")
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
