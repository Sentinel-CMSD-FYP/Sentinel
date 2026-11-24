"""Kaggle kernel (GPU): step 29 live latency, 1, 3 and 4 streams at once on one T4.

Each stream is ``manage.py run_stream <frames> --fps 25 --modules all`` (the step 28 worker: frames
read at real time, all five models, samples written as they are made), on a 1200 frame source
(DroneCrowd test 11, 23, 70 and val 71 back to back, 48 s; each stream starts at a different clip).
Frames are copied to local disk first (the dataset mount reads cold JPEGs at ~350 ms each, step
19). Per stream count, ``eval/pipeline/live_latency.py`` gives capture to sample ready per layer
over the time all streams were live; GPU memory is sampled every second with nvidia-smi.

Setup (code to /tmp/repo, weights placed and md5 checked, uav-dot, pip) as in demo_clips. Output:
/kaggle/working/{eval.log, result.json, latency.md, streams_<n>/<id>/{samples.jsonl,status.json}}.
"""

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")
FRAMES = Path("/tmp/frames")
CLIPS = Path("/tmp/clips")
PIP = ["ultralytics==8.4.182", "segmentation-models-pytorch==0.3.4", "timm==0.9.7",
       "pytorch-lightning==2.6.6", "hydra-core==1.3.7", "imageio-ffmpeg==0.6.0"]
SOURCE = [("test_data", "011"), ("test_data", "023"), ("test_data", "070"), ("train_data", "071")]
COUNTS = [1, 3, 4]
WATCHDOG_H = 1.0  # ~25 min expected
LOG = WORK / "eval.log"
DEADLINE = time.monotonic() + WATCHDOG_H * 3600


def log(msg: str) -> None:
    with LOG.open("a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def find(name: str, max_depth: int = 6) -> Path:
    """First ``name`` under /kaggle/input, searched level by level (a full rglob takes minutes)."""
    for depth in range(1, max_depth + 1):
        if hit := next(INPUT.glob("/".join(["*"] * depth) + "/" + name), None):
            return hit
    raise FileNotFoundError(name)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sh(*cmd: str, env: dict, cwd: Path | None = None) -> bool:
    log("$ " + " ".join(cmd))
    with LOG.open("a") as f:
        try:
            subprocess.run(cmd, check=True, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                           timeout=max(DEADLINE - time.monotonic(), 1))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log(f"failed: {e}")
            return False
    return True


def place(src: Path, dst: Path, want_md5: str) -> None:
    got = md5(src)
    log(f"{src} md5 {got}")
    if not got.startswith(want_md5):
        raise RuntimeError(f"{src} has md5 {got}, expected {want_md5}...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_sources(images: dict[str, Path]) -> list[Path]:
    """One 1200 frame folder per stream (4), each starting at a different clip; files are local."""
    base = FRAMES / "base"
    base.mkdir(parents=True)
    names = []
    for split, seq in SOURCE:
        for f in sorted(images[split].glob(f"img{seq}*.jpg")):
            names.append(f"{len(names):04d}.jpg")
            shutil.copy2(f, base / names[-1])
    folders = []
    for i in range(max(COUNTS)):
        folder = FRAMES / f"s{i}"
        folder.mkdir()
        start = i * len(names) // len(SOURCE)
        for j, name in enumerate(names[start:] + names[:start]):
            (folder / f"{j:04d}.jpg").symlink_to(base / name)
        folders.append(folder)
    log(f"sources: {len(folders)} x {len(names)} frames")
    return folders


def gpu_memory_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False).stdout
    return int(out.split()[0]) if out.strip() else -1


def run_streams(n: int, sources: list[Path], env: dict, backend: Path) -> dict:
    """``n`` run_stream workers at once; returns live_latency's summary plus GPU memory."""
    from live_latency import summarize

    streams = CLIPS / ".streams"
    before = set(streams.iterdir()) if streams.is_dir() else set()
    peak = 0
    with contextlib.ExitStack() as stack:
        logs = [stack.enter_context(open(WORK / f"stream_{n}_{i}.log", "w")) for i in range(n)]
        procs = [subprocess.Popen([sys.executable, "manage.py", "run_stream", str(sources[i]),
                                   "--fps", "25", "--modules", "all", "--name", f"s{i}"],
                                  env=env, cwd=backend, stdout=logs[i], stderr=subprocess.STDOUT)
                 for i in range(n)]
        while any(p.poll() is None for p in procs):
            if time.monotonic() > DEADLINE:
                for p in procs:
                    p.kill()
                raise RuntimeError("watchdog")
            peak = max(peak, gpu_memory_mb())
            time.sleep(1)
    log(f"{n} streams: exit codes {[p.returncode for p in procs]}, GPU memory peak {peak} MB")
    folders = sorted(set(streams.iterdir()) - before)
    out = WORK / f"streams_{n}"
    for f in folders:
        (out / f.name).mkdir(parents=True)
        for name in ("samples.jsonl", "status.json", "log.txt"):
            if (f / name).exists():
                shutil.copy2(f / name, out / f.name / name)
    result = summarize(folders)
    result["gpu_memory_peak_mb"] = peak
    log(json.dumps(result))
    return result


def main() -> None:
    log("kernel started")
    try:
        src = find("predict_split.py").parents[2]
        shutil.copytree(src, REPO)
        weights = REPO / "data" / "weights"
        place(find("best.pth"), weights / "dmcount" / "finetune_e4.pth", "725694ba")
        place(find("weights/best.pt", 8), weights / "yolo26" / "model_c_best.pt", "12213bdea884")
        uavdot = find("checkpoints/dot_pd_dronecrowd_51.00.ckpt", 7).parents[1]
        (REPO / "ml" / "third_party").mkdir(parents=True, exist_ok=True)
        (REPO / "ml" / "third_party" / "uav-dot").symlink_to(uavdot, target_is_directory=True)
        images = {split: find(f"{split}/images", 7) for split in ("test_data", "train_data")}
        sys.path.insert(0, str(REPO / "eval" / "pipeline"))
        from live_latency import table

        env = {**os.environ, "PYTHONPATH": str(REPO / "ml" / "src"), "CLIPS_ROOT": str(CLIPS),
               "CUDA_VISIBLE_DEVICES": "0", "PIPELINE_DEVICE": "cuda", "MAX_STREAMS": "4",
               "YOLO_CONFIG_DIR": str(WORK / "ultralytics")}
        backend = REPO / "apps" / "backend"
        CLIPS.mkdir(parents=True)
        ok = sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", env=env)
        ok = ok and sh(sys.executable, "-m", "pip", "install", "-q", *PIP, "-r",
                       str(backend / "requirements.txt"), env=env)
        if not ok:
            raise RuntimeError("setup failed")
        sources = make_sources(images)
        results = []
        for n in COUNTS:
            results.append(run_streams(n, sources, env, backend))
            (WORK / "result.json").write_text(json.dumps(results, indent=2))
        (WORK / "latency.md").write_text(table(results) + "\n")
        log("done\n" + table(results))
    except Exception as e:  # keep the log even when a step fails
        log(f"failed: {e!r}")
    finally:
        shutil.rmtree(WORK / "ultralytics", ignore_errors=True)
        print(LOG.read_text(), flush=True)


if __name__ == "__main__":
    main()
