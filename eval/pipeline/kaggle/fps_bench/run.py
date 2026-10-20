"""Kaggle kernel (GPU): step 19 FPS of Models A and C, alone and as 1 to 4 concurrent streams.

Weights: DM-Count ``best.pth`` from the attached dataset sentinel-dmcount-ckpt (finetune_e4, md5
725694ba...) and Model C ``best.pt`` from the attached output of sentinel-infra-train (md5
12213bdea884...). Frames: DroneCrowd test clips of onvithong/dronecrowddata. Everything runs on
one T4 (CUDA_VISIBLE_DEVICES=0), the stand in for the single RTX 4070.

Steps (``eval/pipeline/fps_bench.py`` from sentinel-ml-src): density fp32, density fp16 and
infrastructure alone; the fp16 val recheck (300 val frames, never test); 1 to 4 streams with
density fp32, then fp16; the table. Output in /kaggle/working/fps/ (result.json per run,
bench.md) and /kaggle/working/count_error/. All steps log to eval.log under one watchdog.
"""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
ULTRALYTICS = "ultralytics==8.4.182"
STREAMS = (1, 2, 3, 4)
WATCHDOG_H = 1.0  # ~30 min expected
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


def sh(*cmd: str, env: dict) -> bool:
    """Run ``cmd`` with its output in the log; False if the watchdog killed it or it failed."""
    log("$ " + " ".join(cmd))
    with LOG.open("a") as f:
        try:
            subprocess.run(cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=max(DEADLINE - time.monotonic(), 1))
        except subprocess.TimeoutExpired:
            log(f"watchdog: killed after {WATCHDOG_H} h")
            return False
        except subprocess.CalledProcessError as e:
            log(f"failed with exit code {e.returncode}")
            return False
    return True


def main() -> None:
    log("kernel started")
    try:
        scripts = find("predict_split.py").parent
        src = scripts.parents[1]
        bench = src / "eval" / "pipeline" / "fps_bench.py"
        dweights, iweights = find("best.pth"), find("weights/best.pt", 8)
        test_dir = find("test_data/images", 7)
        root = test_dir.parents[2]  # dronecrowddata/{train_data,test_data}/...
        log(f"density weights {dweights} md5 {md5(dweights)}")
        log(f"infra weights {iweights} md5 {md5(iweights)}")
        log(f"test frames {test_dir}")
        env = {**os.environ, "PYTHONPATH": str(src / "ml" / "src"), "CUDA_VISIBLE_DEVICES": "0",
               "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "YOLO_CONFIG_DIR": str(WORK / "ultralytics")}
        common = ["--test-dir", str(test_dir), "--density-weights", str(dweights),
                  "--infra-weights", str(iweights), "--device", "cuda"]
        fps = WORK / "fps"
        ok = sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", env=env)
        ok = ok and sh(sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS, env=env)
        for module, half, tag in (("density", [], "density_fp32"),
                                  ("density", ["--half"], "density_fp16"),
                                  ("infrastructure", [], "infrastructure")):
            ok = ok and sh(sys.executable, "-u", str(bench), "model", module, *half, *common,
                           "--out", str(fps / f"model_{tag}"), env=env)
        ok = ok and sh(sys.executable, "-u", str(scripts / "predict_split.py"), "--split", "val",
                       "--root", str(root), "--weights", str(dweights), "--device", "cuda",
                       "--half", "--out", str(WORK / "finetune_e4_fp16"), env=env)
        ok = ok and sh(sys.executable, "-u", str(scripts / "count_error.py"), "--split", "val",
                       "--preds", str(WORK / "finetune_e4_fp16" / "val" / "preds.csv"),
                       "--name", "finetune_e4_fp16", "--root", str(root),
                       "--out", str(WORK / "count_error"), env=env)
        for half, tag in (([], "fp32"), (["--half"], "fp16")):
            for n in STREAMS:
                ok = ok and sh(sys.executable, "-u", str(bench), "streams", "--n", str(n), *half,
                               *common, "--out", str(fps / f"streams{n}_{tag}"), env=env)
        sh(sys.executable, str(bench), "table", str(fps), env=env)
        log("done" if ok else "stopped early, see above")
    except Exception as e:  # keep the log even when a step fails
        log(f"failed: {e!r}")
    finally:
        print(LOG.read_text(), flush=True)


if __name__ == "__main__":
    main()
