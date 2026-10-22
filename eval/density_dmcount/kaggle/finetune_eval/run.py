"""Kaggle kernel (GPU): step 10 eval of the fine tuned DM-Count weights (step 9 best epoch).

Same as zeroshot_eval, but the weights are best.pth from the attached private dataset
sentinel-dmcount-ckpt. Val (300 frames) runs first as a sanity check (it must give the
training val MAE), then test (all 9000, with overlays). Outputs land in
/kaggle/working/<NAME> (preds, overlays) and /kaggle/working/count_error (tables).

Kaggle shows no log of a running or cancelled kernel, and v1 sat in RUNNING for 3.6 h, so all
steps write to /kaggle/working/eval.log with timestamps and share one watchdog (WATCHDOG_H):
on timeout the step is killed, the kernel still ends normally and prints the log.
"""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
NAME = "finetune_e4"
WATCHDOG_H = 2.0  # ~80 min expected
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


def sh(*cmd: str, env: dict) -> bool:
    """Run ``cmd`` with its output in the log; False if the watchdog killed it."""
    log("$ " + " ".join(cmd))
    with LOG.open("a") as f:
        try:
            subprocess.run(cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=max(DEADLINE - time.monotonic(), 1))
        except subprocess.TimeoutExpired:
            log(f"watchdog: killed after {WATCHDOG_H} h")
            return False
    return True


def main() -> None:
    log("kernel started")
    try:
        script_dir = find("predict_split.py").parent
        env = {**os.environ, "PYTHONPATH": str(script_dir.parents[1] / "ml" / "src")}
        weights = find("best.pth")
        log(f"weights {weights} md5 {hashlib.md5(weights.read_bytes()).hexdigest()}")
        ok = sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", env=env)
        for split in ("val", "test"):
            extra = ["--overlays"] if split == "test" else []
            ok = ok and sh(sys.executable, "-u", str(script_dir / "predict_split.py"),
                           "--split", split, "--root", str(INPUT), "--weights", str(weights),
                           "--device", "cuda", "--out", str(WORK / NAME), *extra, env=env)
            ok = ok and sh(sys.executable, "-u", str(script_dir / "count_error.py"),
                           "--split", split, "--preds", str(WORK / NAME / split / "preds.csv"),
                           "--name", NAME, "--root", str(INPUT), "--out", str(WORK / "count_error"),
                           env=env)
        log("done" if ok else "stopped by the watchdog")
    finally:
        print(LOG.read_text(), flush=True)


if __name__ == "__main__":
    main()
