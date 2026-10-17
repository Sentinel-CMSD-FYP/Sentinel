"""Point-localisation metrics: one-to-one matching within a pixel radius, then P/R/F1.

Matching rule (the Phase 0 sheet: "within 10 px of the true head"):
  * a prediction matches a GT point if their Euclidean distance is <= ``radius_px``;
  * one-to-one, nearest-first: all candidate pairs are sorted by distance and taken
    greedily, skipping any pair whose prediction or GT point is already used;
  * ties are broken by prediction index, then GT index (stable, so results repeat).
This greedy rule is not the optimal (Hungarian) assignment; it is the rule agreed in
ROADMAP B-2. Counts are summed over all frames before P/R/F1 (micro average).

Units: pixels. Coordinates: (x, y) = (column, row) in original frame pixels.

Edge cases: if TP+FP+FN == 0 (no GT, no predictions) precision = recall = F1 = 1.0;
otherwise a zero denominator gives 0.0 (e.g. no predictions -> precision 0.0).

    python eval/point_metrics.py --config ml/configs/point_metrics.yaml \
        --pred data/outputs/clip_011_points.json --gt data/outputs/clip_011_gt.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Counts:
    """True positives, false positives, false negatives."""

    tp: int
    fp: int
    fn: int

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def _xy(points: np.ndarray) -> np.ndarray:
    """Return the (N, 2) x, y part of an (N, >=2) array; empty input gives (0, 2)."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2))
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"points must have shape (N, >=2), got {arr.shape}")
    return arr[:, :2]


def match_points(
    pred_xy: np.ndarray, gt_xy: np.ndarray, radius_px: float
) -> list[tuple[int, int]]:
    """One-to-one nearest-first matching. Returns (pred_index, gt_index) pairs.

    Args:
        pred_xy: (N, 2) predicted (x, y), pixels.
        gt_xy: (M, 2) ground-truth (x, y), pixels.
        radius_px: maximum match distance in pixels (inclusive), must be > 0.
    """
    if radius_px <= 0:
        raise ValueError("radius_px must be > 0")
    pred, gt = _xy(pred_xy), _xy(gt_xy)
    if len(pred) == 0 or len(gt) == 0:
        return []
    dist = np.hypot(pred[:, None, 0] - gt[None, :, 0], pred[:, None, 1] - gt[None, :, 1])
    pairs = np.argwhere(dist <= radius_px)  # row-major: sorted by pred then gt index
    order = np.argsort(dist[pairs[:, 0], pairs[:, 1]], kind="stable")
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int]] = []
    for p, g in pairs[order]:
        if p in used_pred or g in used_gt:
            continue
        used_pred.add(int(p))
        used_gt.add(int(g))
        matches.append((int(p), int(g)))
    return matches


def frame_counts(pred_xy: np.ndarray, gt_xy: np.ndarray, radius_px: float) -> Counts:
    """TP/FP/FN for one frame."""
    n_pred, n_gt = len(_xy(pred_xy)), len(_xy(gt_xy))
    tp = len(match_points(pred_xy, gt_xy, radius_px))
    return Counts(tp=tp, fp=n_pred - tp, fn=n_gt - tp)


def precision_recall_f1(c: Counts) -> dict[str, float]:
    """Precision, recall, F1 from counts. See module docstring for edge cases."""
    if c.tp + c.fp + c.fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = c.tp / (c.tp + c.fp) if c.tp + c.fp else 0.0
    recall = c.tp / (c.tp + c.fn) if c.tp + c.fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def filter_by_score(points: np.ndarray, score_thr: float) -> np.ndarray:
    """Keep rows with score >= score_thr. Rows without a score column count as 1.0."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3))
    if arr.shape[1] < 3:
        return arr
    return arr[arr[:, 2] >= score_thr]


def evaluate(
    pred_frames: dict[str, np.ndarray],
    gt_frames: dict[str, np.ndarray],
    radius_px: float = 10.0,
    score_thr: float = 0.0,
) -> dict:
    """Score predictions against GT over the GT frames.

    Only frames present in ``gt_frames`` are scored. A GT frame with no predictions
    counts all its GT points as FN. Prediction frames absent from GT are ignored and
    counted in ``pred_frames_not_in_gt``.
    """
    total = Counts(0, 0, 0)
    for fid, gt in gt_frames.items():
        pred = filter_by_score(pred_frames.get(fid, np.zeros((0, 3))), score_thr)
        total = total + frame_counts(pred, gt, radius_px)
    return {
        "radius_px": radius_px,
        "score_thr": score_thr,
        "n_frames": len(gt_frames),
        "pred_frames_not_in_gt": len(set(pred_frames) - set(gt_frames)),
        "tp": total.tp,
        "fp": total.fp,
        "fn": total.fn,
        **precision_recall_f1(total),
    }


def load_points_json(path: Path) -> dict[str, np.ndarray]:
    """Load ``frames`` from a points file (docs/contracts.md): {frame_id: (N, 3) array}."""
    doc = json.loads(Path(path).read_text())
    return {fid: np.asarray(rows, dtype=np.float64).reshape(-1, 3) for fid, rows in
            doc["frames"].items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True, help="predicted *_points.json")
    ap.add_argument("--gt", type=Path, required=True, help="GT in the same points format")
    ap.add_argument("--radius", type=float, help="override radius_px from config")
    ap.add_argument("--score-thr", type=float, help="override score_thr from config")
    ap.add_argument("--label", default="", help="source label to print, e.g. 'val, thr=0.4'")
    ap.add_argument("--out", type=Path, help="write the result as JSON")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    radius = args.radius if args.radius is not None else float(cfg["radius_px"])
    thr = args.score_thr if args.score_thr is not None else float(cfg["score_thr"])
    result = evaluate(load_points_json(args.pred), load_points_json(args.gt), radius, thr)
    result["label"] = args.label
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
