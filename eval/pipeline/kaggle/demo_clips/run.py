"""Kaggle kernel (GPU): step 25 demo clips, every model on each clip through the real backend.

For each clip, ``manage.py ingest_clip`` (DroneCrowd frames to video.mp4 + clip.json) and then
``manage.py run_models <clip> --modules all --device cuda`` (density, flow, infrastructure,
points, tracks in one decode; layers added with docs/layers/add_layer.py, clip.json
``produced_by`` written), then ``docs/layers/validate.py`` on the clip folder.

The code comes from sentinel-ml-src (ml, apps/backend, docs/layers), copied to /tmp/repo so the
registry finds its configs and weights at the usual repo paths (/kaggle/working holds only the
output). Weights: DM-Count ``best.pth`` from sentinel-dmcount-ckpt (finetune_e4, md5 725694ba...),
Model C ``best.pt`` from the output of sentinel-infra-train (md5 12213bdea884...), Model B code +
checkpoint from sentinel-b-ckpt (uav-dot a6d3664). Output: /kaggle/working/clips/<clip_id>/ and
eval.log, all under one watchdog.
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
REPO = Path("/tmp/repo")
CLIPS = WORK / "clips"
PIP = ["ultralytics==8.4.182", "segmentation-models-pytorch==0.3.4", "timm==0.9.7",
       "pytorch-lightning==2.6.6", "hydra-core==1.3.7", "imageio-ffmpeg==0.6.0"]
# (clip id, DroneCrowd split folder, clip number, name shown in the viewer)
DEMO = [("seq011", "test_data", "011", "DroneCrowd test 11 (day)"),
        ("seq023", "test_data", "023", "DroneCrowd test 23 (dense)"),
        ("seq070", "test_data", "070", "DroneCrowd test 70 (night)"),
        ("seq071", "train_data", "071", "DroneCrowd val 71 (dusk)")]
WATCHDOG_H = 1.0  # ~20 min expected
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
    """Run ``cmd`` with its output in the log; False if the watchdog killed it or it failed."""
    log("$ " + " ".join(cmd))
    with LOG.open("a") as f:
        try:
            subprocess.run(cmd, check=True, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                           timeout=max(DEADLINE - time.monotonic(), 1))
        except subprocess.TimeoutExpired:
            log(f"watchdog: killed after {WATCHDOG_H} h")
            return False
        except subprocess.CalledProcessError as e:
            log(f"failed with exit code {e.returncode}")
            return False
    return True


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
    try:
        src = find("predict_split.py").parents[2]  # <src>/eval/density_dmcount/predict_split.py
        shutil.copytree(src, REPO)
        weights = REPO / "data" / "weights"
        place(find("best.pth"), weights / "dmcount" / "finetune_e4.pth", "725694ba")
        place(find("weights/best.pt", 8), weights / "yolo26" / "model_c_best.pt", "12213bdea884")
        uavdot = find("checkpoints/dot_pd_dronecrowd_51.00.ckpt", 7).parents[1]
        (REPO / "ml" / "third_party").mkdir(parents=True, exist_ok=True)
        (REPO / "ml" / "third_party" / "uav-dot").symlink_to(uavdot, target_is_directory=True)
        log(f"uav-dot {uavdot}")
        frames = {split: find(f"{split}/images", 7) for split in ("test_data", "train_data")}
        log(f"frames {frames}")

        env = {**os.environ, "PYTHONPATH": str(REPO / "ml" / "src"), "CLIPS_ROOT": str(CLIPS),
               "CUDA_VISIBLE_DEVICES": "0", "YOLO_CONFIG_DIR": str(WORK / "ultralytics")}
        backend = REPO / "apps" / "backend"
        CLIPS.mkdir(parents=True, exist_ok=True)
        ok = sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", env=env)
        ok = ok and sh(sys.executable, "-m", "pip", "install", "-q", *PIP, "-r",
                       str(backend / "requirements.txt"), env=env)
        ok = ok and sh(sys.executable, "-c", "import torch; print('torch', torch.__version__, "
                       "torch.cuda.is_available())", env=env)
        for clip_id, split, seq, name in DEMO:
            if not ok:
                break
            log(f"--- {clip_id}")
            ok = sh(sys.executable, "manage.py", "ingest_clip", str(frames[split]), clip_id,
                    "--seq", seq, "--fps", "25", "--name", name, env=env, cwd=backend)
            ok = ok and sh(sys.executable, "manage.py", "run_models", clip_id, "--modules", "all",
                           "--device", "cuda", env=env, cwd=backend)
            ok = ok and sh(sys.executable, str(REPO / "docs" / "layers" / "validate.py"),
                           str(CLIPS / clip_id), env=env)
        shutil.rmtree(WORK / "ultralytics", ignore_errors=True)
        log("done" if ok else "stopped early, see above")
    except Exception as e:  # keep the log even when a step fails
        log(f"failed: {e!r}")
    finally:
        print(LOG.read_text(), flush=True)


if __name__ == "__main__":
    main()
