"""Kaggle kernel (GPU): steps 8 and 9, fine tune DM-Count on DroneCrowd from the QNRF weights.

Resume across 12 h sessions: after a session, download its output and upload run/ as the
private dataset sentinel-dmcount-ckpt (``push.py ckpt``); the next push attaches it, the run
folder (last.pt, best.pth, history.csv) is copied to /kaggle/working/run and
``sentinel_ml.density.train`` resumes from last.pt. Without that dataset the run starts from
the QNRF weights (fetched from the DM-Count GitHub repo); to start fresh, delete the dataset. Set EPOCHS to the total
epoch to train up to; the train config stops before an epoch that would pass 11 h.

Kaggle keeps no log or output of a cancelled run, so training runs under a watchdog
(WATCHDOG_H): on timeout it is killed and the kernel still ends normally. Training output goes
to run/train.log and is printed into the kernel log at the end. LIMIT > 0 trains and
validates on the first LIMIT frames only (diagnostic runs).
"""

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RUN = WORK / "run"
EPOCHS = 2
LIMIT = 0
WATCHDOG_H = 11.5  # hard stop, so a stalled session still ends with its log
WEIGHTS_URL = ("https://github.com/cvlab-stonybrook/DM-Count/raw/"
               "cc5f2132e0d1328909f31b6d665b8e0b15c30467/pretrained_models/model_qnrf.pth")


def main() -> None:
    print("kernel started", flush=True)
    src = next(INPUT.rglob("density_train.yaml")).parents[2]  # the sentinel-ml-src copy
    env = {**os.environ, "PYTHONPATH": str(src / "ml" / "src")}
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=True)

    prev = next((p.parent for p in INPUT.rglob("last.pt")), None)  # the ckpt dataset
    if prev is not None:
        shutil.copytree(prev, RUN, dirs_exist_ok=True)
        print(f"resuming from {prev}", flush=True)
    else:
        print("no previous run found, starting from the QNRF weights", flush=True)
    weights = WORK / "weights" / "model_qnrf.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(WEIGHTS_URL, weights)

    cmd = [sys.executable, "-m", "sentinel_ml.density.train", "--out", str(RUN),
           "--epochs", str(EPOCHS), "--root", str(INPUT), "--init-weights", str(weights)]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    print("$", " ".join(cmd), flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    log = RUN / "train.log"
    with log.open("a") as f:
        try:
            code = subprocess.run(cmd, check=False, env=env, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=WATCHDOG_H * 3600).returncode
        except subprocess.TimeoutExpired:
            code = None
    print(log.read_text(), flush=True)
    print(f"watchdog: training killed after {WATCHDOG_H} h" if code is None
          else f"training exit code {code}", flush=True)
    shutil.rmtree(weights.parent)  # keep the output small
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
