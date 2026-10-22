"""Kaggle kernel (GPU): fine tune YOLO26 seg for Model C on the scene split (step 14).

Copies the attached ``sentinel-infra-labels`` dataset (``push.py labels``: images/ + YOLO
labels/) to /tmp/infra and runs ``train.py`` with ``data/manifests/infra_split.json`` from the
``sentinel-ml-src`` dataset. Output in /kaggle/working: ``run/`` (Ultralytics run:
weights/best.pt, results.csv, curve plots). All steps log to eval.log under one watchdog.
Set EPOCHS 3 for the pipeline check, 150 for the full run.
"""

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
EPOCHS = 150
PATIENCE = 50
IMGSZ = 1280
BATCH = 4         # 8 ran out of memory on a T4 at imgsz 1280 (step 13)
WATCHDOG_H = 1.5
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
    log(f"kernel started, EPOCHS {EPOCHS}")
    try:
        scripts = find("infrastructure_yolo26")
        split = scripts.parents[1] / "data" / "manifests" / "infra_split.json"
        labels = find("sentinel-infra-labels", 3)
        for sub, pattern in (("images", "*.jpg"), ("labels", "*.txt")):  # mount may be flat
            (DATA / sub).mkdir(parents=True, exist_ok=True)
            for p in labels.rglob(pattern):
                shutil.copy2(p, DATA / sub / p.name)
        log(f"{len(list((DATA / 'images').glob('*.jpg')))} images, "
            f"{len(list((DATA / 'labels').glob('*.txt')))} labels")
        os.environ["YOLO_CONFIG_DIR"] = str(WORK / "ultralytics")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        ok = sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")
        ok = ok and sh(sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS)
        ok = ok and sh(sys.executable, "-u", str(scripts / "train.py"), "--out", str(WORK / "run"),
                       "--data", str(DATA), "--split", str(split),
                       "--weights", "/tmp/weights/yolo26s-seg.pt", "--epochs", str(EPOCHS),
                       "--patience", str(PATIENCE), "--imgsz", str(IMGSZ), "--batch", str(BATCH),
                       "--device", "0")
        log("done" if ok else "stopped by the watchdog")
    except Exception as e:  # keep the log even when a step fails
        log(f"failed: {e!r}")
    finally:
        print(LOG.read_text(), flush=True)


if __name__ == "__main__":
    main()
