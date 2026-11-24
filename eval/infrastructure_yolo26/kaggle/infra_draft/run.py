"""Kaggle kernel (GPU): train the Model C draft model and draft the unlabelled images (step 13).

Copies the attached ``sentinel-infra-labels`` dataset (``push.py labels``: images/ + YOLO
labels/) to /tmp/infra, runs ``draft_model.py train`` on the labelled images, then
``draft_model.py predict`` on the images without labels. Output in /kaggle/working:
``run/`` (Ultralytics run, weights/best.pt), ``_drafts/<NAME>.json`` (Label Studio
predictions) and ``drafts_overlay/``. All steps log to eval.log under one watchdog.
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
NAME = "batch4"     # drafts file name, change per round
EPOCHS = 100
IMGSZ = 1280
BATCH = 4         # 8 ran out of memory on a T4 at imgsz 1280
WATCHDOG_H = 1.0
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
        script = find("infrastructure_yolo26") / "draft_model.py"
        labels = find("sentinel-infra-labels", 3)
        for sub, pattern in (("images", "*.jpg"), ("labels", "*.txt")):  # mount may be flat
            (DATA / sub).mkdir(parents=True, exist_ok=True)
            for p in labels.rglob(pattern):
                shutil.copy2(p, DATA / sub / p.name)
        log(f"{len(list((DATA / 'images').glob('*.jpg')))} images, "
            f"{len(list((DATA / 'labels').glob('*.txt')))} labels")
        os.environ["YOLO_CONFIG_DIR"] = str(WORK / "ultralytics")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        common = ["--data", str(DATA), "--imgsz", str(IMGSZ), "--device", "0"]
        ok = sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")
        ok = ok and sh(sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS)
        ok = ok and sh(sys.executable, "-u", str(script), "train", "--out", str(WORK / "run"),
                       "--weights", str(WORK / "weights" / "yolo26s-seg.pt"),
                       "--epochs", str(EPOCHS), "--batch", str(BATCH), *common)
        ok = ok and sh(sys.executable, "-u", str(script), "predict",
                       str(WORK / "run" / "weights" / "best.pt"), "--name", NAME, *common)
        if ok:
            shutil.copytree(DATA / "images" / "_drafts", WORK / "_drafts")
            shutil.copytree(DATA / "drafts_overlay", WORK / "drafts_overlay")
        log("done" if ok else "stopped by the watchdog")
    except Exception as e:  # keep the log even when a step fails
        log(f"failed: {e!r}")
    finally:
        print(LOG.read_text(), flush=True)


if __name__ == "__main__":
    main()
