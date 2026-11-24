"""Kaggle kernel (GPU): step 7 zero shot baseline of the released DM-Count QNRF weights.

Fetches the weights from the DM-Count GitHub repo, then for val (300 frames) and test
(all 9000) runs ``predict_split.py`` and scores the counts with ``count_error.py``, both
from the attached sentinel-ml-src (with ``ml/src`` on PYTHONPATH) over the public
DroneCrowd copy in /kaggle/input. Test also saves a few overlays. Outputs land in
/kaggle/working/zeroshot_qnrf (preds, overlays) and /kaggle/working/count_error (tables).
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
NAME = "zeroshot_qnrf"
WEIGHTS_URL = ("https://github.com/cvlab-stonybrook/DM-Count/raw/"
               "cc5f2132e0d1328909f31b6d665b8e0b15c30467/pretrained_models/model_qnrf.pth")


def sh(*cmd: str, env: dict) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    script_dir = next(INPUT.rglob("predict_split.py")).parent
    env = {**os.environ, "PYTHONPATH": str(script_dir.parents[1] / "ml" / "src")}
    weights = WORK / "weights" / "model_qnrf.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(WEIGHTS_URL, weights)
    sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", env=env)

    for split in ("val", "test"):
        extra = ["--overlays"] if split == "test" else []
        sh(sys.executable, str(script_dir / "predict_split.py"), "--split", split,
           "--root", str(INPUT), "--weights", str(weights), "--device", "cuda",
           "--out", str(WORK / NAME), *extra, env=env)
        sh(sys.executable, str(script_dir / "count_error.py"), "--split", split,
           "--preds", str(WORK / NAME / split / "preds.csv"), "--name", NAME,
           "--root", str(INPUT), "--out", str(WORK / "count_error"), env=env)
    weights.unlink()  # keep the output small


if __name__ == "__main__":
    main()
