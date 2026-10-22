"""E-5: speed of the full B -> E loop per stream, with 1 to 4 streams sharing one model.

One thread per stream. Model B is loaded once and shared; every stream has its own tracker.
Per frame the loop is: read the image (cv2.imread + RGB), B (preprocess, network, postprocess),
E (tracker update). The device is synchronised before each timer stops
(``torch.cuda.synchronize()`` on cuda), so no GPU work is left uncounted.

Warm-up frames (default 10) run first and are not timed. All streams then start together and
only frames that lie fully inside the window where *every* stream is running are counted, so a
stream that finishes early does not make the others look faster.

Per stream we report
  * avg FPS = frames / total time of those frames (a throughput, not a mean of per-frame rates);
  * p95 FPS = 1 / (95th-percentile frame time): the rate of the slowest 5% of frames.
The Phase 0 target is >= 5 FPS per stream on the RTX 4070. This script measures; it decides
nothing. It reads images from disk inside the timed loop (a real stream also decodes frames).
Units: seconds and frames per second. Nothing here uses ground truth.

    python eval/benchmark_be.py --infer-config ml/configs/infer_b.yaml \
        --tracker-config ml/configs/tracker_e2_candidate.yaml --split test \
        --streams 1 3 4 --label alone --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from sentinel_ml.localization import data as D
from sentinel_ml.localization.infer import Detector, DotDetector
from sentinel_ml.tracking.tracker import PointTracker, TrackerConfig, resolve_config

TARGET_FPS = 5.0  # Phase 0 sheet / ROADMAP E-5: per stream, on the RTX 4070


@dataclass
class FrameTime:
    """Wall-clock stamps (seconds, perf_counter) and the stage durations of one frame."""

    start: float
    end: float
    read: float
    detect: float
    track: float

    @property
    def total(self) -> float:
        return self.end - self.start


@dataclass
class StreamRun:
    stream_id: int
    clip: int
    frames: list[FrameTime] = field(default_factory=list)  # measured frames only
    error: str | None = None


def percentile(values: list[float], q: float) -> float:
    """q-th percentile (0-100) with linear interpolation."""
    return float(np.percentile(np.asarray(values, dtype=float), q))


def run_stream(stream_id: int, clip: int, read_frame: Callable[[int], np.ndarray],
               n_frames: int, warmup: int, detector: Detector, tracker: PointTracker,
               sync: Callable[[], None], clock: Callable[[], float],
               ready: threading.Barrier | None = None) -> StreamRun:
    """Run ``warmup + n_frames`` frames through B then E; time the last ``n_frames``.

    ``ready`` is waited on after the warm-up so every stream starts measuring together.
    """
    run = StreamRun(stream_id, clip)
    try:
        for i in range(warmup + n_frames):
            if i == warmup and ready is not None:
                ready.wait()
            t0 = clock()
            image = read_frame(i)
            t1 = clock()
            points = detector(image)
            sync()
            t2 = clock()
            tracker.update(np.asarray(points, dtype=float).reshape(-1, 3))
            sync()
            t3 = clock()
            if i >= warmup:
                run.frames.append(FrameTime(t0, t3, t1 - t0, t2 - t1, t3 - t2))
    except Exception as exc:  # report it, and release the other threads waiting on the barrier
        run.error = f"{type(exc).__name__}: {exc}"
        if ready is not None:
            ready.abort()
    return run


def common_window(runs: list[StreamRun]) -> tuple[float, float]:
    """Time window in which every stream was running (latest first start, earliest last end)."""
    return (max(r.frames[0].start for r in runs), min(r.frames[-1].end for r in runs))


def summarise_stream(run: StreamRun, window: tuple[float, float]) -> dict:
    """Stats over the frames of ``run`` that lie fully inside ``window``."""
    used = [f for f in run.frames if f.start >= window[0] and f.end <= window[1]]
    if not used:
        raise ValueError(f"stream {run.stream_id}: no frame inside the common window")
    totals = [f.total for f in used]
    return {
        "stream": run.stream_id, "clip": run.clip, "frames_used": len(used),
        "avg_fps": len(used) / sum(totals),
        "p95_fps": 1.0 / percentile(totals, 95),
        "frame_time_ms": {"mean": 1000 * float(np.mean(totals)),
                          "p50": 1000 * percentile(totals, 50),
                          "p95": 1000 * percentile(totals, 95), "max": 1000 * max(totals)},
        "stage_mean_ms": {"read": 1000 * float(np.mean([f.read for f in used])),
                          "detect_B": 1000 * float(np.mean([f.detect for f in used])),
                          "track_E": 1000 * float(np.mean([f.track for f in used]))},
    }


def summarise_pass(runs: list[StreamRun]) -> dict:
    """Summary of one pass (all streams of one concurrency level)."""
    failed = [r for r in runs if r.error]
    if failed:
        raise RuntimeError("; ".join(f"stream {r.stream_id}: {r.error}" for r in failed))
    window = common_window(runs)
    if window[1] <= window[0]:
        raise RuntimeError("the streams never ran at the same time; nothing to report")
    streams = [summarise_stream(r, window) for r in runs]
    return {
        "n_streams": len(runs), "window_s": window[1] - window[0], "streams": streams,
        "min_stream_avg_fps": min(s["avg_fps"] for s in streams),
        "min_stream_p95_fps": min(s["p95_fps"] for s in streams),
        "total_avg_fps": sum(s["avg_fps"] for s in streams),
        "avg_fps_meets_target": all(s["avg_fps"] >= TARGET_FPS for s in streams),
    }


def run_pass(n_streams: int, clip_frames: dict[int, list[Path]], clips: list[int],
             n_frames: int, warmup: int, detector: Detector, tracker_cfg: TrackerConfig,
             frame_size_of: Callable[[int], tuple[int, int]], fps: float, frame_stride: int,
             altitude_m: float, sync: Callable[[], None], clock: Callable[[], float],
             read_image: Callable[[Path], np.ndarray]) -> dict:
    """Run ``n_streams`` streams at once (clip ``clips[i % len(clips)]`` for stream i)."""
    cfg = resolve_config(tracker_cfg, fps, frame_stride, altitude_m)
    ready = threading.Barrier(n_streams)
    results: list[StreamRun | None] = [None] * n_streams

    def work(i: int) -> None:
        clip = clips[i % len(clips)]
        paths = clip_frames[clip]
        tracker = PointTracker(frame_size_of(clip), cfg)
        results[i] = run_stream(i, clip, lambda k: read_image(paths[k]), n_frames, warmup,
                                detector, tracker, sync, clock, ready)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n_streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return summarise_pass([r for r in results if r is not None])


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def format_pass(p: dict) -> str:
    lines = [f"{p['n_streams']} stream(s), common window {p['window_s']:.1f} s"]
    lines.append("stream clip frames  avg FPS  p95 FPS  frame ms (mean/p95/max)   "
                 "read / B / E ms")
    for s in p["streams"]:
        ft, st = s["frame_time_ms"], s["stage_mean_ms"]
        lines.append(f"{s['stream']:6d} {s['clip']:4d} {s['frames_used']:6d} {s['avg_fps']:8.2f} "
                     f"{s['p95_fps']:8.2f}  {ft['mean']:7.0f}/{ft['p95']:.0f}/{ft['max']:.0f}"
                     f"   {st['read']:.0f} / {st['detect_B']:.0f} / {st['track_E']:.0f}")
    lines.append(f"lowest stream: avg {p['min_stream_avg_fps']:.2f} FPS, p95 "
                 f"{p['min_stream_p95_fps']:.2f} FPS; all streams together "
                 f"{p['total_avg_fps']:.2f} FPS; target {TARGET_FPS:g} FPS per stream (avg): "
                 f"{'met' if p['avg_fps_meets_target'] else 'NOT met'}")
    return "\n".join(lines)


def gpu_snapshot() -> str:
    """nvidia-smi line(s) taken before the run, as evidence of what else was on the GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=False)
        return out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi unavailable: {exc}"


def environment(device: str) -> dict:
    env = {"device": device, "gpu_before_run": gpu_snapshot()}
    try:
        import torch
        env["torch"] = torch.__version__
        if device.startswith("cuda"):
            env["cuda_device"] = torch.cuda.get_device_name(torch.device(device))
    except ImportError:
        env["torch"] = None
    return env


def make_sync(device: str) -> Callable[[], None]:
    if device.startswith("cuda"):
        import torch
        return torch.cuda.synchronize
    return lambda: None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--infer-config", type=Path, required=True)
    ap.add_argument("--tracker-config", type=Path, required=True)
    ap.add_argument("--split", choices=D.SPLITS, default="test")
    ap.add_argument("--clips", type=int, nargs="+", help="clips to use (default: first four)")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 3, 4],
                    help="concurrency levels to run, one pass each")
    ap.add_argument("--frames", type=int, default=250, help="timed frames per stream (>= 200)")
    ap.add_argument("--warmup", type=int, default=10, help="untimed frames per stream")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", required=True, help="e.g. alone, with_teammates")
    ap.add_argument("--out-dir", type=Path, default=Path("data/outputs/benchmark"))
    ap.add_argument("--root", type=Path, help="override the dataset root")
    ap.add_argument("--checkpoint", type=Path, help="override the checkpoint path")
    ap.add_argument("--repo-dir", type=Path, help="override the DOT repo folder")
    ap.add_argument("--allow-short", action="store_true",
                    help="allow fewer than 200 timed frames (smoke tests only)")
    args = ap.parse_args()
    if args.frames < 200 and not args.allow_short:
        raise SystemExit("--frames must be >= 200 (ROADMAP E-5); --allow-short is for smoke tests")
    out_path = args.out_dir / f"{args.label}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; pick another --label (results are not overwritten)")

    cfg = yaml.safe_load(args.infer_config.read_text())
    for key, val in (("checkpoint", args.checkpoint), ("repo_dir", args.repo_dir)):
        if val is not None:
            cfg[key] = str(val)
    data_cfg = D.load_config(Path(cfg["data_config"]))
    if args.root is not None:
        data_cfg["root"] = str(args.root)
    ds = D.dataset_from_config(data_cfg, args.split, frame_stride=int(cfg["frame_stride"]))
    available = sorted({r.clip for r in ds.refs})
    clips = args.clips or available[:4]
    need = args.warmup + args.frames
    clip_frames: dict[int, list[Path]] = {}
    for c in clips:
        paths = [r.image_path for r in sorted(ds.refs, key=lambda r: r.frame_id) if r.clip == c]
        if len(paths) < need:
            raise SystemExit(f"clip {c} has {len(paths)} frames, need {need}")
        clip_frames[c] = paths

    def frame_size_of(c: int) -> tuple[int, int]:
        h, w = read_rgb(clip_frames[c][0]).shape[:2]
        return (w, h)

    tcfg = TrackerConfig(**yaml.safe_load(args.tracker_config.read_text())["tracker"])
    print(f"loading model B on {args.device} (once, shared by all streams)")
    detector = DotDetector(cfg, args.device)
    sync = make_sync(args.device)
    env = environment(args.device)
    print(f"environment: {env}")

    passes = []
    for n in args.streams:
        print(f"\n=== {n} stream(s): {args.warmup} warm-up + {args.frames} timed frames each ===")
        p = run_pass(n, clip_frames, clips, args.frames, args.warmup, detector, tcfg,
                     frame_size_of, float(cfg["fps"]), int(cfg["frame_stride"]),
                     float(cfg["altitude_m"]), sync, time.perf_counter, read_rgb)
        print(format_pass(p))
        passes.append(p)

    doc = {"label": args.label, "split": args.split, "clips": clips,
           "warmup_frames": args.warmup, "timed_frames_per_stream": args.frames,
           "tracker_config": str(args.tracker_config), "infer_config": str(args.infer_config),
           "environment": env, "target_fps_per_stream": TARGET_FPS,
           "note": ("B+E per stream, images read from disk inside the timed loop, one shared "
                    "model B, threads in one process; avg FPS = frames / total frame time, "
                    "p95 FPS = 1 / p95 frame time, over the window where all streams run"),
           "passes": passes}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
