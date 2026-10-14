"""Draw ground truth and predictions on chosen frames, to see why a clip scores badly.

GT points are drawn as red filled dots, predictions (score >= --thr) as cyan rings.
Diagnosis only: do not tune thresholds or models on what you see in test clips.

Units: pixels. Coordinates: (x, y) = (column, row), original frame pixels.

    python eval/overlay_check.py --config ml/configs/dronecrowd_data.yaml --split test \
        --pred-dir data/outputs/points/test --clips 74 17 --frame 150 --thr 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import point_metrics as pm  # eval/ is on sys.path when this file is run as a script

from sentinel_ml.localization import data as D


def draw(image_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray, label: str) -> np.ndarray:
    """Return a copy of the image with GT (red dots) and predictions (cyan rings)."""
    out = image_bgr.copy()
    for x, y, _ in pred:
        cv2.circle(out, (round(x), round(y)), 7, (255, 255, 0), 2)
    for x, y in gt:
        cv2.circle(out, (round(x), round(y)), 3, (0, 0, 255), -1)
    cv2.putText(out, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the dronecrowd_data.yaml")
    ap.add_argument("--split", choices=D.SPLITS, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--clips", type=int, nargs="+", required=True)
    ap.add_argument("--frame", type=int, default=150, help="0-based frame_id in each clip")
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--out-dir", type=Path, default=Path("data/outputs/diagnose"))
    args = ap.parse_args()

    ds = D.dataset_from_config(D.load_config(args.config), args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        idx = next(i for i, r in enumerate(ds.refs)
                   if r.clip == clip and r.frame_id == args.frame)
        sample = ds[idx]
        pred = pm.load_points_json(args.pred_dir / f"clip_{clip:03d}_points.json")
        p = pm.filter_by_score(pred[str(args.frame)], args.thr)
        c = pm.frame_counts(p, sample.points, args.radius)
        gray = float(sample.image.mean())
        label = (f"clip {clip} f{args.frame} GT {len(sample.points)} pred {len(p)} "
                 f"TP {c.tp} FP {c.fp} FN {c.fn} bright {gray:.0f}")
        print(label)
        bgr = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
        path = args.out_dir / f"{args.split}_clip{clip:03d}_f{args.frame}.jpg"
        cv2.imwrite(str(path), draw(bgr, sample.points, p, label))


if __name__ == "__main__":
    main()
