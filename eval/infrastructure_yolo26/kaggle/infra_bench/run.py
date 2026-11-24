"""Kaggle kernel (GPU): benchmark Model C, val mAP + FPS in fp32 and fp16 (step 15).

Takes ``best.pt`` from the attached output of the step 14 kernel ``sentinel-infra-train``
(md5 logged, it should match ``data/weights/yolo26/model_c_best.pt``), copies the attached
``sentinel-infra-labels`` dataset to /tmp/infra and runs ``benchmark.py`` from ``sentinel-ml-src``
once per precision. Output in /kaggle/working/bench/{fp32,fp16}/ (bench.json, bench.md).
All steps log to eval.log under one watchdog.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
DATA = Path("/tmp/infra")
ULTRALYTICS = "ultralytics==8.4.182"
IMGSZ = 1280
FRAMES = 300
WATCHDOG_H = 0.5  # ~10 min expected
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


def sh(*cmd: str) -> bool:
    """Run ``cmd`` with its output in the log; False if the watchdog killed it."""
    log("$ " + " ".join(cmd))
    with LOG.open("a") as f:
        try:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT,
                           timeout=max(DEADLINE - time.monotonic(), 1))
        except subprocess.TimeoutExpired:
            log(f"watchdog: killed after {WATCHDOG_H} h")
            return False
    return True


def main() -> None:
    log("kernel started")
    try:
        scripts = find("infrastructure_yolo26")
        split = scripts.parents[1] / "data" / "manifests" / "infra_split.json"
        weights = find("weights/best.pt", 8)
        log(f"weights {weights} md5 {hashlib.md5(weights.read_bytes()).hexdigest()}")
        labels = find("sentinel-infra-labels", 3)
        for sub, pattern in (("images", "*.jpg"), ("labels", "*.txt")):  # mount may be flat
            (DATA / sub).mkdir(parents=True, exist_ok=True)
            for p in labels.rglob(pattern):
                shutil.copy2(p, DATA / sub / p.name)
        os.environ["YOLO_CONFIG_DIR"] = str(WORK / "ultralytics")
        ok = sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")
        ok = ok and sh(sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS)
        for tag, extra in (("fp32", []), ("fp16", ["--half"])):
            ok = ok and sh(sys.executable, "-u", str(scripts / "benchmark.py"),
                           "--out", str(WORK / "bench" / tag), "--data", str(DATA),
                           "--split", str(split), "--weights", str(weights),
                           "--imgsz", str(IMGSZ), "--device", "0", "--frames", str(FRAMES), *extra)
        log("done" if ok else "stopped by the watchdog")
    except Exception as e:  # keep the log even when a step fails
        log(f"failed: {e!r}")
    finally:
        print(LOG.read_text(), flush=True)


if __name__ == "__main__":
    main()
