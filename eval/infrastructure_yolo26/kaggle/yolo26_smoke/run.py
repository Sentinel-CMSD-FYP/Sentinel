"""Kaggle kernel (GPU): YOLO26 smoke test (step 11).

Installs Ultralytics (same version as the local venv; Kaggle's torch is kept), then runs
``smoke_test.py`` from the attached sentinel-ml-src dataset on every image in ``data/samples/``.
The pretrained weights download to /kaggle/working/weights (the input is read only).
All steps log to /kaggle/working/eval.log under one watchdog (WATCHDOG_H), as in finetune_eval.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
ULTRALYTICS = "ultralytics==8.4.182"
WATCHDOG_H = 0.5  # a few minutes expected
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
        script = find("infrastructure_yolo26") / "smoke_test.py"
        repo = script.parents[2]
        os.environ["YOLO_CONFIG_DIR"] = str(WORK / "ultralytics")
        ok = sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")
        ok = ok and sh(sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS)
        for img in sorted((repo / "data" / "samples").glob("*")):
            ok = ok and sh(sys.executable, "-u", str(script), str(img), "--device", "cuda",
                           "--weights", str(WORK / "weights" / "yolo26s-seg.pt"),
                           "--out", str(WORK / "annotated"))
        log("done" if ok else "stopped by the watchdog")
    except Exception as e:  # keep the log even when a step fails
        log(f"failed: {e!r}")
    finally:
        print(LOG.read_text(), flush=True)


if __name__ == "__main__":
    main()
