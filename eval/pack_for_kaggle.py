"""Pack what B-3 inference needs into one folder (and optional zip) to upload to Kaggle.

Copies, keeping the repo layout so the same config files work unchanged:
  * ml/src, ml/configs, ml/pyproject.toml          (our code and configs)
  * ml/third_party/uav-dot                         (DOT repo code; no .git, no checkpoints)
  * data/manifests/dronecrowd_split.json           (frozen split)
  * the images of one split under data/dronecrowd/<train_dir or test_dir>/images/

Ground truth is not needed to run inference, so it is not copied.

    python eval/pack_for_kaggle.py --config ml/configs/dronecrowd_data.yaml --split val --zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import yaml

IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "checkpoints", "*.egg-info",
                                ".pytest_cache", ".ruff_cache")


def split_images(cfg: dict, manifest: dict, split: str) -> list[Path]:
    """Image files of the clips in ``split`` (val clips live in the official train folder)."""
    folder = Path(cfg["root"]) / (cfg["test_dir"] if split == "test" else cfg["train_dir"])
    clips = {int(c) for c in manifest[split]}
    return sorted(p for p in (folder / "images").glob("img*.jpg") if int(p.name[3:6]) in clips)


def pack(cfg: dict, split: str, out: Path) -> tuple[int, int]:
    """Copy everything into ``out``. Returns (n_files, total_bytes)."""
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree("ml/src", out / "ml" / "src", ignore=IGNORE)
    shutil.copytree("ml/configs", out / "ml" / "configs", ignore=IGNORE)
    shutil.copy("ml/pyproject.toml", out / "ml" / "pyproject.toml")
    shutil.copytree("ml/third_party/uav-dot", out / "ml" / "third_party" / "uav-dot",
                    ignore=IGNORE)
    (out / "data" / "manifests").mkdir(parents=True)
    shutil.copy(cfg["manifest"], out / "data" / "manifests" / Path(cfg["manifest"]).name)
    images = split_images(cfg, manifest, split)
    if not images:
        raise FileNotFoundError(f"no images found for split '{split}'")
    folder = Path(cfg["root"]) / (cfg["test_dir"] if split == "test" else cfg["train_dir"])
    dest = out / folder / "images"
    dest.mkdir(parents=True)
    for p in images:
        shutil.copy(p, dest / p.name)
    files = [p for p in out.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def make_zip(folder: Path) -> Path:
    """Zip ``folder`` without compression (JPEGs do not compress), next to it."""
    path = folder.with_suffix(".zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the dronecrowd_data.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--out", type=Path, default=Path("data/outputs/kaggle_pack"))
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    n, size = pack(cfg, args.split, args.out)
    print(f"packed {n} files, {size / 1e6:.0f} MB into {args.out}")
    if args.zip:
        z = make_zip(args.out)
        print(f"wrote {z} ({z.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
