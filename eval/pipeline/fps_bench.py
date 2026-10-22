"""Step 19: FPS of Models A and C on one GPU, alone and as 1 to 4 concurrent streams.

    python eval/pipeline/fps_bench.py model density --out runs/fps/model_density_fp32
    python eval/pipeline/fps_bench.py model density --half --out runs/fps/model_density_fp16
    python eval/pipeline/fps_bench.py model infrastructure --out runs/fps/model_infrastructure
    python eval/pipeline/fps_bench.py streams --n 4 --half --out runs/fps/streams4_fp16
    python eval/pipeline/fps_bench.py table runs/fps

``model`` times one module alone on 1920x1080 DroneCrowd test frames held in memory (decode not
timed): ``--warmup`` calls are skipped, then ``--frames`` calls are timed, each between two cuda
synchronize calls. It goes through the pipeline code (``DensityModule``: DM-Count + zone grid;
``InfraModule``: YOLO26 + class NMS + polygons), so the number is what a stream pays per run.

``streams`` starts ``--n`` processes, one per stream, each a full copy of the pipeline with its
own models (as ``run_models`` would be for one drone). Each reads its own test clips with
``open_frames`` and runs ``run_clip`` as fast as it can (density every 25th frame,
infrastructure every 125th, as in the configs). All streams load and warm up, then start
together. Per stream: video FPS (``--live-fps`` or more keeps up with live video), ms per model
run and per decoded frame. GPU memory and load come from nvidia-smi, sampled every 0.5 s.

``--half`` turns on fp16 for DM-Count; Model C follows its config (fp16 on CUDA). Each run writes
``<out>/result.json``; ``table`` collects all of them under a folder into ``<dir>/bench.md``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "manifests" / "dronecrowd_split.json"
MODULES = ("density", "infrastructure")
BARRIER_S = 900  # time allowed for every stream to load its models and warm up


def sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def every_of(module) -> int:
    """How often a module runs its model (frame ids that are a multiple of this)."""
    return module.cfg.every if module.name == "density" else module.model.config.every


class Timed:
    """A Module wrapper that records the ms of every call that runs the model."""

    def __init__(self, module) -> None:
        self.module, self.name = module, module.name
        self.every = every_of(module)
        self.ms: list[float] = []

    def process(self, frame_id: int, frame: np.ndarray) -> None:
        if frame_id % self.every:
            self.module.process(frame_id, frame)
            return
        sync()
        t = time.perf_counter()
        self.module.process(frame_id, frame)
        sync()
        self.ms.append((time.perf_counter() - t) * 1000)

    def layer(self, clip_id: str) -> dict:
        return self.module.layer(clip_id)


def stats(ms: list[float]) -> dict:
    if not ms:
        return {"runs": 0}
    a = np.asarray(ms)
    mean = float(a.mean())
    return {"runs": len(a), "mean_ms": round(mean, 1),
            "p50_ms": round(float(np.percentile(a, 50)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1), "fps": round(1000 / mean, 2)}


def smi() -> tuple[float, float] | None:
    """(memory used MB, utilisation %) of GPU 0 from nvidia-smi, or None without it."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    mem, util = out.strip().split(",")
    return float(mem), float(util)


class GpuSampler(threading.Thread):
    """Samples ``smi()`` every ``period`` s until ``stop()``."""

    def __init__(self, period: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.period, self.samples = period, []
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            if (s := smi()) is not None:
                self.samples.append(s)
            self._done.wait(self.period)

    def stop(self) -> dict:
        self._done.set()
        self.join()
        if not self.samples:
            return {}
        mem, util = np.asarray(self.samples).T
        return {"gpu_mem_peak_mb": float(mem.max()), "gpu_util_mean_pct": round(float(util.mean()), 1)}


def device_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {"device": torch.cuda.get_device_name(0),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)}


def build(name: str, a: argparse.Namespace):
    from sentinel_ml.pipeline import build_module

    overrides = {}
    weights = a.density_weights if name == "density" else a.infra_weights
    if weights:
        overrides["weights"] = str(weights)
    if name == "density":
        overrides["half"] = a.half
    return build_module(name, a.device, **overrides)


def default_test_dir() -> Path:
    import yaml
    from sentinel_ml.density.data import find_split_dirs

    root = REPO / yaml.safe_load((REPO / "ml/configs/density_data.yaml").read_text())["root"]
    return root / find_split_dirs(root)["test"] / "images"


def load_frames(test_dir: Path, clips: list[int], step: int = 25) -> list[np.ndarray]:
    """Every ``step``-th frame of each clip, decoded once."""
    import cv2

    frames = []
    for c in clips:
        for f in range(1, 301, step):
            path = test_dir / f"img{c:03d}{f:03d}.jpg"
            if (img := cv2.imread(str(path))) is None:
                raise FileNotFoundError(path)
            frames.append(img)
    return frames


def write(out: Path, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def cmd_model(a: argparse.Namespace) -> None:
    frames = load_frames(a.test_dir, a.clips[:5])
    m = Timed(build(a.module, a))
    for i, frame in enumerate(itertools.islice(itertools.cycle(frames), a.warmup + a.frames)):
        m.process(i * m.every, frame)
    used = smi()
    write(a.out, {"kind": "model", "module": a.module, "fp16": m.module.model.half,
                  "frame_size": list(frames[0].shape[1::-1]), "warmup": a.warmup,
                  **stats(m.ms[a.warmup:]), **device_info(),
                  "gpu_mem_used_mb": used[0] if used else None})


def timed_frames(test_dir: Path, clips: list[int], limit: int | None, decode_ms: list[float]):
    """``(frame_id, frame)`` over the clips in a row, recording the ms each decode takes."""
    from sentinel_ml.flow.sources import open_frames

    clips_frames = (open_frames(test_dir, seq=f"{c:03d}")[0] for c in clips)
    frames = itertools.islice(itertools.chain.from_iterable(clips_frames), limit)
    for fid in itertools.count():
        t = time.perf_counter()
        frame = next(frames, None)
        if frame is None:
            return
        decode_ms.append((time.perf_counter() - t) * 1000)
        yield fid, frame


def stream_worker(idx: int, clips: list[int], a: argparse.Namespace, barrier, queue) -> None:
    try:
        from sentinel_ml.flow.sources import open_frames
        from sentinel_ml.pipeline import run_clip

        mods = [Timed(build(name, a)) for name in MODULES]
        first = next(open_frames(a.test_dir, seq=f"{clips[0]:03d}")[0])
        for m in mods:  # warm up outside the timing
            for _ in range(3):
                m.module.process(0, first)
        barrier.wait(timeout=BARRIER_S)
        decode_ms: list[float] = []
        t0 = time.perf_counter()
        run_clip(timed_frames(a.test_dir, clips, a.limit, decode_ms), mods, f"stream{idx}")
        wall = time.perf_counter() - t0
        queue.put({"stream": idx, "clips": clips, "frames": len(decode_ms),
                   "wall_s": round(wall, 2), "video_fps": round(len(decode_ms) / wall, 2),
                   "decode": stats(decode_ms), **{m.name: stats(m.ms) for m in mods},
                   **device_info()})
    except Exception as e:  # noqa: BLE001 (report instead of leaving the others waiting)
        barrier.abort()
        queue.put({"stream": idx, "error": repr(e)})


def cmd_streams(a: argparse.Namespace) -> None:
    ctx = mp.get_context("spawn")  # CUDA cannot be used in forked children
    barrier, queue = ctx.Barrier(a.n + 1), ctx.Queue()
    k = a.clips_per_stream
    procs = []
    for i in range(a.n):
        clips = [a.clips[(i * k + j) % len(a.clips)] for j in range(k)]
        procs.append(ctx.Process(target=stream_worker, args=(i, clips, a, barrier, queue)))
        procs[-1].start()
    try:
        barrier.wait(timeout=BARRIER_S)
    except threading.BrokenBarrierError:
        pass  # a stream failed while loading; its error is in the queue
    loaded = smi()
    sampler = GpuSampler()
    sampler.start()
    t0 = time.perf_counter()
    streams = sorted((queue.get(timeout=a.timeout) for _ in procs), key=lambda r: r["stream"])
    wall = time.perf_counter() - t0
    gpu = sampler.stop()
    for p in procs:
        p.join()
    errors = [s for s in streams if "error" in s]
    ok = [s for s in streams if "error" not in s]
    fps = [s["video_fps"] for s in ok]
    write(a.out, {
        "kind": "streams", "n": a.n, "density_fp16": a.half, "live_fps": a.live_fps,
        "wall_s": round(wall, 2), "video_fps_min": min(fps, default=None),
        "video_fps_mean": round(float(np.mean(fps)), 2) if fps else None,
        "keeps_up": bool(fps) and not errors and min(fps) >= a.live_fps,
        **{f"{name}_mean_ms": round(float(np.mean([s[name]["mean_ms"] for s in ok])), 1)
           for name in MODULES if ok},
        "gpu_mem_loaded_mb": loaded[0] if loaded else None, **gpu, "streams": streams})
    if errors:
        raise SystemExit(f"{len(errors)} stream(s) failed: {[e['error'] for e in errors]}")


def cmd_table(a: argparse.Namespace) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(a.dir.rglob("result.json"))]
    lines = ["| module | fp16 | device | runs | mean / p50 / p95 ms | FPS | max alloc MB |",
             "|---|---|---|---|---|---|---|"]
    for r in (r for r in runs if r["kind"] == "model"):
        lines.append(f"| {r['module']} | {r['fp16']} | {r['device']} | {r['runs']} | "
                     f"{r['mean_ms']} / {r['p50_ms']} / {r['p95_ms']} | {r['fps']} | "
                     f"{r.get('max_allocated_mb', '')} |")
    lines += ["", ("| streams | A fp16 | video FPS min / mean | keeps up | A ms mean | C ms mean "
                   "| decode ms | GPU MB loaded / peak | GPU util % |"),
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in runs if r["kind"] == "streams"),
                    key=lambda r: (r["density_fp16"], r["n"])):
        dec = np.mean([s["decode"]["mean_ms"] for s in r["streams"] if "decode" in s])
        lines.append(f"| {r['n']} | {r['density_fp16']} | {r['video_fps_min']} / "
                     f"{r['video_fps_mean']} | {r['keeps_up']} | {r.get('density_mean_ms')} | "
                     f"{r.get('infrastructure_mean_ms')} | {dec:.1f} | "
                     f"{r['gpu_mem_loaded_mb']} / {r.get('gpu_mem_peak_mb')} | "
                     f"{r.get('gpu_util_mean_pct')} |")
    text = "\n".join(lines) + "\n"
    (a.dir / "bench.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, required=True)
    common.add_argument("--test-dir", type=Path, help="DroneCrowd test images folder")
    common.add_argument("--clips", type=lambda s: [int(c) for c in s.split(",")],
                        help="test clip ids, comma separated (default: manifest test order)")
    common.add_argument("--device", help="overrides the configs (cpu, cuda)")
    common.add_argument("--half", action="store_true", help="fp16 for DM-Count (CUDA only)")
    common.add_argument("--density-weights", type=Path, help="overrides density_dmcount.yaml")
    common.add_argument("--infra-weights", type=Path, help="overrides infrastructure_yolo26.yaml")
    m = sub.add_parser("model", parents=[common], help="one model alone, frames in memory")
    m.add_argument("module", choices=MODULES)
    m.add_argument("--warmup", type=int, default=20)
    m.add_argument("--frames", type=int, default=220)
    s = sub.add_parser("streams", parents=[common], help="N concurrent pipeline processes")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--clips-per-stream", type=int, default=2)
    s.add_argument("--limit", type=int, help="frames per stream (default: all of its clips)")
    s.add_argument("--live-fps", type=float, default=25.0)
    s.add_argument("--timeout", type=float, default=1800, help="s to wait for each stream")
    t = sub.add_parser("table", help="collect result.json files into bench.md")
    t.add_argument("dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "table":
        cmd_table(a)
        return
    a.test_dir = a.test_dir or default_test_dir()
    a.clips = a.clips or json.loads(MANIFEST.read_text())["test"]
    if a.cmd == "model":
        cmd_model(a)
    else:
        cmd_streams(a)


if __name__ == "__main__":
    main()
