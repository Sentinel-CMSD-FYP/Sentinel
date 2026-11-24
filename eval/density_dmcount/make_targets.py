"""Step 5: DM-Count point files and Gaussian GT density maps, written once and checked.

    python eval/density_dmcount/make_targets.py make
    python eval/density_dmcount/make_targets.py check
    python eval/density_dmcount/make_targets.py make --root /kaggle/input --out /kaggle/working/dmcount

``make`` writes, for every frame the configured strides keep (``ml/configs/density_data.yaml``):
  * ``points/{split}/img*.npy``: (N, 2) float32 (x, y), clipped into the image (DM-Count format);
  * ``density/{split}/img*.npz``: key ``density``, (135, 240) float32 fixed sigma Gaussian map.
``check`` reloads every file and asserts that it matches the raw GT: same point count, points
inside the image, and sum(map) equal to that count. It then saves GT map overlays for the
``gt_check.py`` frames under ``<out>/check``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from gt_check import DRAW
from sentinel_ml.density import DensityMap, overlay
from sentinel_ml.density.data import load_split_from_config
from sentinel_ml.density.targets import clip_points, density_map
from sentinel_ml.localization.data import SPLITS, load_gt

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "ml" / "configs" / "density_data.yaml"
SUM_TOL = 1e-3  # |sum(map) - N| allowed per frame (float32 rounding)


def make(cfg: dict, ds, split: str, out: Path) -> None:
    t = cfg["targets"]
    (w, h), sigma, cell = t["image_size"], float(t["sigma"]), int(t["cell"])
    pts_dir, map_dir = out / "points" / split, out / "density" / split
    pts_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    for ref in ds.refs:
        points = clip_points(load_gt(ref.gt_path)[0], w, h)
        np.save(pts_dir / f"{ref.image_path.stem}.npy", points)
        np.savez_compressed(map_dir / f"{ref.image_path.stem}.npz",
                            density=density_map(points, w, h, sigma, cell))
    print(f"{split}: wrote {len(ds.refs)} point files and maps under {out}")


def check(cfg: dict, ds, split: str, out: Path) -> None:
    (w, h), cell = cfg["targets"]["image_size"], int(cfg["targets"]["cell"])
    worst, clipped = 0.0, 0
    for ref in ds.refs:
        raw = load_gt(ref.gt_path)[0]
        points = np.load(out / "points" / split / f"{ref.image_path.stem}.npy")
        density = np.load(out / "density" / split / f"{ref.image_path.stem}.npz")["density"]
        name = ref.image_path.name
        assert points.dtype == np.float32 and points.shape == (len(raw), 2), name
        assert (points >= 0).all() and (points[:, 0] <= w - 1).all() \
            and (points[:, 1] <= h - 1).all(), name
        assert density.shape == (h // cell, w // cell), name
        err = abs(float(density.sum(dtype=np.float64)) - len(raw))
        assert err <= SUM_TOL, f"{name}: sum {density.sum()} vs {len(raw)} points"
        worst = max(worst, err)
        clipped += int((np.abs(points - raw) > 0).any(axis=1).sum())
    print(f"{split}: {len(ds.refs)} frames OK, max |sum(map) - N| {worst:.2e}, "
          f"{clipped} points clipped")


def draw(cfg: dict, ds, split: str, out: Path) -> None:
    index = {(r.clip, r.frame_id): i for i, r in enumerate(ds.refs)}
    (out / "check").mkdir(parents=True, exist_ok=True)
    for s, clip, frame_id in DRAW:
        if s != split:
            continue
        sample = ds[index[(clip, frame_id)]]
        stem = sample.ref.image_path.stem
        density = np.load(out / "density" / split / f"{stem}.npz")["density"]
        bgr = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
        path = out / "check" / f"{split}_{sample.ref.clip_id}_f{frame_id:03d}_gtmap.jpg"
        cv2.imwrite(str(path), overlay(bgr, DensityMap(density, bgr.shape[:2])))
        print(f"  saved {path} ({len(sample.points)} points, sum {density.sum():.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["make", "check"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--root", type=Path, help="dataset root, overrides the config")
    ap.add_argument("--out", type=Path, help="overrides targets.out_dir in the config")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = args.out or REPO / cfg["targets"]["out_dir"]

    for split in SPLITS:
        ds = load_split_from_config(cfg, split, root=args.root, repo=REPO)
        if args.mode == "make":
            make(cfg, ds, split, out)
        else:
            check(cfg, ds, split, out)
            draw(cfg, ds, split, out)


if __name__ == "__main__":
    main()
