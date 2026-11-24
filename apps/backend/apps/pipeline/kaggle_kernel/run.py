"""Kaggle kernel (GPU): every model on one uploaded clip, for a run started from the API.

Pushed by ``apps/pipeline/kaggle_run.py`` after it uploaded the clip as the dataset
``sentinel-run-clip`` (video.mp4, clip.json with no layers, request.json with the run id). Runs
``manage.py run_models <clip> --modules all --device cuda --report`` with the trained weights,
set up as in the step 25 demo_clips kernel: code from sentinel-ml-src copied to /tmp/repo,
DM-Count ``best.pth`` from sentinel-dmcount-ckpt (md5 725694ba...), Model C ``best.pt`` from the
output of sentinel-infra-train (md5 12213bdea884...), Model B from sentinel-b-ckpt.

Output in /kaggle/working: ``clip/{clip.json, layers/*.json}``, ``eval.log`` and ``result.json``
(run id, the report of added layers and failed modules, or the error that stopped the run). The
video is not copied back; the backend has it. Everything runs under one watchdog.
"""

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
CLIPS = Path("/tmp/clips")
PIP = ["ultralytics==8.4.182", "segmentation-models-pytorch==0.3.4", "timm==0.9.7",
       "pytorch-lightning==2.6.6", "hydra-core==1.3.7", "imageio-ffmpeg==0.6.0"]
WATCHDOG_H = 2.0  # 300 frames take ~7 min; minutes of video take longer
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


def sh(*cmd: str, env: dict, cwd: Path | None = None) -> int:
    """Run ``cmd`` with its output in the log; its exit code (-1 if the watchdog killed it)."""
    log("$ " + " ".join(cmd))
    with LOG.open("a") as f:
        try:
            return subprocess.run(cmd, check=False, env=env, cwd=cwd, stdout=f,
                                  stderr=subprocess.STDOUT,
                                  timeout=max(DEADLINE - time.monotonic(), 1)).returncode
        except subprocess.TimeoutExpired:
            log(f"watchdog: killed after {WATCHDOG_H} h")
            return -1


def place(src: Path, dst: Path, want_md5: str) -> None:
    """Copy weights to the path the configs name, refusing a different checkpoint."""
    got = md5(src)
    log(f"{src} md5 {got}")
    if not got.startswith(want_md5):
        raise RuntimeError(f"{src} has md5 {got}, expected {want_md5}...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    log("kernel started")
    result = {"run_id": None, "report": None, "error": None}
    try:
        request_path = find("request.json", 4)
        request = json.loads(request_path.read_text())
        result["run_id"], clip_id = request["run_id"], request["clip_id"]
        log(f"run {result['run_id']} on clip {clip_id}")

        src = find("predict_split.py").parents[2]  # <src>/eval/density_dmcount/predict_split.py
        shutil.copytree(src, REPO)
        weights = REPO / "data" / "weights"
        place(find("best.pth"), weights / "dmcount" / "finetune_e4.pth", "725694ba")
        place(find("weights/best.pt", 8), weights / "yolo26" / "model_c_best.pt", "12213bdea884")
        uavdot = find("checkpoints/dot_pd_dronecrowd_51.00.ckpt", 7).parents[1]
        (REPO / "ml" / "third_party").mkdir(parents=True, exist_ok=True)
        (REPO / "ml" / "third_party" / "uav-dot").symlink_to(uavdot, target_is_directory=True)

        clip_dir = CLIPS / clip_id
        (clip_dir / "layers").mkdir(parents=True)
        for name in ("clip.json", "video.mp4"):
            shutil.copy2(request_path.parent / name, clip_dir / name)

        env = {**os.environ, "PYTHONPATH": str(REPO / "ml" / "src"), "CLIPS_ROOT": str(CLIPS),
               "CUDA_VISIBLE_DEVICES": "0", "YOLO_CONFIG_DIR": str(WORK / "ultralytics")}
        backend = REPO / "apps" / "backend"
        sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", env=env)
        if sh(sys.executable, "-m", "pip", "install", "-q", *PIP, "-r",
              str(backend / "requirements.txt"), env=env):
            raise RuntimeError("pip install failed")
        report = WORK / "report.json"
        # Exit code 1 with a report = some modules failed and the others' layers were added.
        code = sh(sys.executable, "manage.py", "run_models", clip_id, "--modules", "all",
                  "--device", "cuda", "--report", str(report), env=env, cwd=backend)
        if not report.is_file():
            raise RuntimeError(f"run_models stopped with exit code {code} and no report")
        result["report"] = json.loads(report.read_text())
        shutil.copytree(clip_dir, WORK / "clip", ignore=shutil.ignore_patterns("video.mp4"))
        log(f"done: {result['report']}")
    except Exception as e:  # keep the log and say why in result.json
        result["error"] = repr(e)
        log(f"failed: {e!r}")
    finally:
        shutil.rmtree(WORK / "ultralytics", ignore_errors=True)
        (WORK / "result.json").write_text(json.dumps(result, indent=2))
        print(LOG.read_text(), flush=True)


if __name__ == "__main__":
    main()
