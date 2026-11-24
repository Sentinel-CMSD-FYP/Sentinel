"""Step 4 check: split sizes with the configured strides, and GT points drawn on 5 frames.

    python eval/density_dmcount/gt_check.py
    python eval/density_dmcount/gt_check.py --root /kaggle/input --out /kaggle/working/gt_check

Uses ``ml/configs/density_data.yaml``. Prints frames, clips and points per frame for each
split, then saves the frames in ``DRAW`` (dense train, day and dusk val, day and night
test) with GT dots so they can be checked by eye against the people.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from sentinel_ml.density.data import load_split_from_config
from sentinel_ml.localization.data import SPLITS, draw_points, load_gt

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "ml" / "configs" / "density_data.yaml"
# (split, clip, 0 based frame id); each must survive the split's frame stride.
DRAW = [("train", 100, 0), ("val", 6, 100), ("val", 71, 150), ("test", 11, 0), ("test", 70, 150)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--root", type=Path, help="dataset root, overrides the config")
    ap.add_argument("--out", type=Path, help="overrides check_dir in the config")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = args.out or REPO / cfg["check_dir"]
    out.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        ds = load_split_from_config(cfg, split, root=args.root, repo=REPO)
        n = np.array([len(load_gt(r.gt_path)[0]) for r in ds.refs])
        clips = sorted({r.clip for r in ds.refs})
        print(f"{split}: {len(ds)} frames, {len(clips)} clips, points per frame "
              f"median {np.median(n):.0f} min {n.min()} max {n.max()}")
        index = {(r.clip, r.frame_id): i for i, r in enumerate(ds.refs)}
        for s, clip, frame_id in DRAW:
            if s != split:
                continue
            sample = ds[index[(clip, frame_id)]]
            path = out / f"{split}_{sample.ref.clip_id}_f{frame_id:03d}.jpg"
            cv2.imwrite(str(path), draw_points(sample, radius=4))
            print(f"  saved {path} ({len(sample.points)} points)")


if __name__ == "__main__":
    main()
