"""Score-threshold sweep for Model B on one split (B-4).

Compares predicted ``*_points.json`` files (docs/contracts.md) with ground truth read
through the frozen loader, using the 10 px one-to-one matching in ``point_metrics.py``.
Writes a table (CSV + JSON) and an F1-vs-threshold plot.

Rule: thresholds are chosen on the ``val`` split only. The ``test`` split may be scored
at exactly ONE threshold (the one already chosen on val); this script refuses a sweep
over several thresholds on ``test``.

Units: pixels. Coordinates: (x, y) = (column, row), original frame pixels.

    python eval/threshold_sweep.py --config ml/configs/threshold_sweep.yaml
    python eval/threshold_sweep.py --config ml/configs/threshold_sweep.yaml \
        --split test --pred-dir data/outputs/points/test --thresholds 0.4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import point_metrics as pm  # eval/ is on sys.path when this file is run as a script
import yaml

from sentinel_ml.localization import data as D


def check_split_rule(split: str, thresholds: list[float]) -> None:
    """Refuse to sweep several thresholds on the test split."""
    if split == "test" and len(thresholds) != 1:
        raise SystemExit("test split: give exactly one threshold (chosen on val), "
                         f"got {len(thresholds)}. Never tune on test.")


def load_pred_clip(path: Path) -> dict[str, np.ndarray]:
    """{frame_id: (N, 3) x, y, score} from one points file."""
    return pm.load_points_json(path)


def gt_by_clip(ds: D.DroneCrowdDataset) -> dict[int, dict[str, np.ndarray]]:
    """Ground-truth points per clip and frame_id, read with the frozen loader."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for ref in ds.refs:
        pts, _ = D.load_gt(ref.gt_path)
        out.setdefault(ref.clip, {})[str(ref.frame_id)] = pts
    return out


def sweep(pred: dict[int, dict[str, np.ndarray]], gt: dict[int, dict[str, np.ndarray]],
          radius_px: float, thresholds: list[float]) -> list[dict]:
    """One row per threshold: micro-averaged TP/FP/FN/P/R/F1 and the same per clip."""
    missing = sorted(set(gt) - set(pred))
    if missing:
        raise FileNotFoundError(f"no predictions for clips {missing}")
    rows = []
    for thr in thresholds:
        per_clip, total = {}, pm.Counts(0, 0, 0)
        for clip in sorted(gt):
            r = pm.evaluate(pred[clip], gt[clip], radius_px, thr)
            counts = pm.Counts(r["tp"], r["fp"], r["fn"])
            per_clip[clip] = {**{k: r[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}}
            total = total + counts
        rows.append({"threshold": thr, "tp": total.tp, "fp": total.fp, "fn": total.fn,
                     **pm.precision_recall_f1(total), "per_clip": per_clip})
    return rows


def format_table(rows: list[dict]) -> str:
    lines = ["thr     TP      FP      FN     precision  recall   F1"]
    for r in rows:
        lines.append(f"{r['threshold']:.2f}  {r['tp']:7d} {r['fp']:7d} {r['fn']:7d}"
                     f"   {r['precision']:.4f}    {r['recall']:.4f}   {r['f1']:.4f}")
    return "\n".join(lines)


def write_outputs(rows: list[dict], meta: dict, out_dir: Path) -> list[Path]:
    """Write sweep.json, sweep.csv and (if matplotlib is installed) the F1 plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps({**meta, "rows": rows}, indent=1) + "\n")
    with (out_dir / "sweep.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
        for r in rows:
            w.writerow([r["threshold"], r["tp"], r["fp"], r["fn"],
                        round(r["precision"], 4), round(r["recall"], 4), round(r["f1"], 4)])
    written = [out_dir / "sweep.json", out_dir / "sweep.csv"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipped the plot (pip install matplotlib)")
        return written
    thr = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("f1", "F1"), ("precision", "precision"), ("recall", "recall")):
        ax.plot(thr, [r[key] for r in rows], marker="o", label=label)
    ax.set_xlabel("score threshold")
    ax.set_ylabel(f"@ {meta['radius_px']:g} px")
    ax.set_title(meta["label"], fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "f1_vs_threshold.png", dpi=150)
    plt.close(fig)
    return written + [out_dir / "f1_vs_threshold.png"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=D.SPLITS, help="override the config's split")
    ap.add_argument("--pred-dir", type=Path, help="override the folder of *_points.json")
    ap.add_argument("--thresholds", type=float, nargs="+", help="override the threshold list")
    ap.add_argument("--out-dir", type=Path, help="override the output folder")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    split = args.split or cfg["split"]
    thresholds = args.thresholds or [float(t) for t in cfg["thresholds"]]
    check_split_rule(split, thresholds)
    pred_dir = args.pred_dir or Path(cfg["pred_dir"])
    out_dir = args.out_dir or Path(cfg["out_dir"]) / split
    radius = float(cfg["radius_px"])

    ds = D.dataset_from_config(D.load_config(Path(cfg["data_config"])), split)
    gt = gt_by_clip(ds)
    pred = {int(p.name.split("_")[1]): load_pred_clip(p)
            for p in sorted(pred_dir.glob("clip_*_points.json"))}
    pred = {c: v for c, v in pred.items() if c in gt}
    n_frames = sum(len(v) for v in gt.values())
    label = f"DroneCrowd {split} split ({len(gt)} clips, {n_frames} frames), {radius:g} px"
    rows = sweep(pred, gt, radius, thresholds)
    print(label)
    print(format_table(rows))
    meta = {"label": label, "split": split, "radius_px": radius, "n_clips": len(gt),
            "n_frames": n_frames, "pred_dir": str(pred_dir)}
    for path in write_outputs(rows, meta, out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
