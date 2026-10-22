"""Kaggle kernel: DM-Count smoke test on the GPU.

Puts ``ml/src`` from the attached sentinel-ml-src dataset on PYTHONPATH (no pip:
numpy, OpenCV, PyYAML and torch ship with Kaggle), fetches the released
QNRF weights from the DM-Count GitHub repo and runs ``smoke_test.py`` on every
image in ``data/samples/``. Outputs land in /kaggle/working.
"""

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = WORK / "repo"
WEIGHTS_URL = ("https://github.com/cvlab-stonybrook/DM-Count/raw/"
               "cc5f2132e0d1328909f31b6d665b8e0b15c30467/pretrained_models/model_qnrf.pth")


def find_src() -> Path:
    """Dataset root; Kaggle may or may not have unpacked the folder zips."""
    hits = sorted(INPUT.rglob("dataset-metadata.json")) or sorted(INPUT.rglob("pyproject.toml"))
    if hits:
        root = hits[0].parent
        return root.parent if root.name == "ml" else root
    zips = sorted(INPUT.rglob("*.zip"))
    if not zips:
        raise SystemExit(f"sentinel-ml-src not found under {INPUT}")
    return zips[0].parent


def stage(src: Path) -> None:
    """Copy (read only input) into a writable repo layout so pip can build."""
    REPO.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.suffix == ".zip":
            with zipfile.ZipFile(item) as z:
                z.extractall(REPO / item.stem)
        elif item.is_dir():
            shutil.copytree(item, REPO / item.name, dirs_exist_ok=True)
    print("staged:", sorted(str(p.relative_to(REPO)) for p in REPO.glob("*/*")))


def sh(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    stage(find_src())
    os.environ["PYTHONPATH"] = str(REPO / "ml" / "src")

    weights = WORK / "weights" / "model_qnrf.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, weights)
    print(f"weights: {weights.stat().st_size / 1e6:.1f} MB")
    sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")

    script = REPO / "eval" / "density_dmcount" / "smoke_test.py"
    for img in sorted((REPO / "data" / "samples").glob("*")):
        sh(sys.executable, str(script), str(img), "--weights", str(weights),
           "--device", "cuda", "--out", str(WORK / "overlays"))


if __name__ == "__main__":
    main()
