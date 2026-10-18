"""Per-frame point-count error of Model B, by crowd-density bin and by lighting.

For every frame: count predictions with score >= threshold, compare with the GT count.
  relative error = |predicted - GT| / GT        (GT = 0 frames are skipped and counted)
  signed bias    = (predicted - GT) / GT        (negative = under-counting)
Density bin comes from the frame's GT count: sparse < c0, moderate c0..c1, dense > c1
(cutoffs fixed in the config before any results). Reported two ways: all clips (primary)
and day-only clips (secondary), when lighting tags exist. Clips are never dropped after
seeing results.

This is B's point count, not the tracking count error of Model E.

    python eval/count_error.py --config ml/configs/count_error.yaml --split val
    python eval/count_error.py --config ml/configs/count_error.yaml --split test   # once only
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

BINS = ("sparse", "moderate", "dense")
LIGHTING = ("day", "dusk", "night")


def bin_of(n_gt: int, cutoffs: list[int]) -> str:
    """Density bin of a frame from its GT count: sparse < c0, moderate c0..c1, dense > c1."""
    lo, hi = cutoffs
    if n_gt < lo:
        return "sparse"
    return "moderate" if n_gt <= hi else "dense"


def frame_rows(pred: dict[int, dict[str, np.ndarray]], gt: dict[int, dict[str, np.ndarray]],
               threshold: float) -> list[dict]:
    """One row per GT frame: clip, frame_id, GT count, predicted count at ``threshold``."""
    rows = []
    for clip in sorted(gt):
        if clip not in pred:
            raise FileNotFoundError(f"no predictions for clip {clip}")
        for fid, pts in gt[clip].items():
            p = pm.filter_by_score(pred[clip].get(fid, np.zeros((0, 3))), threshold)
            rows.append({"clip": clip, "frame_id": int(fid), "gt": len(pts), "pred": len(p)})
    return rows


def load_tags(path: Path, split: str) -> dict[int, str] | None:
    """{clip: day|dusk|night} for ``split``, or None if the file is missing or not filled in."""
    if not Path(path).exists():
        return None
    tags = yaml.safe_load(Path(path).read_text()).get(split) or {}
    if not tags or any(v is None for v in tags.values()):
        return None
    bad = {c: v for c, v in tags.items() if v not in LIGHTING}
    if bad:
        raise ValueError(f"unknown lighting tags {bad}; use one of {LIGHTING}")
    return {int(c): v for c, v in tags.items()}


def stats(rows: list[dict]) -> dict:
    """Frames, clips, mean/median relative error (%) and mean signed bias (%) of ``rows``."""
    usable = [r for r in rows if r["gt"] > 0]
    if not usable:
        return {"frames": 0, "clips": 0, "skipped_gt0": len(rows)}
    rel = np.array([(r["pred"] - r["gt"]) / r["gt"] for r in usable])
    return {"frames": len(usable), "clips": len({r["clip"] for r in usable}),
            "skipped_gt0": len(rows) - len(usable),
            "mean_abs_error_pct": float(np.abs(rel).mean() * 100),
            "median_abs_error_pct": float(np.median(np.abs(rel)) * 100),
            "mean_bias_pct": float(rel.mean() * 100)}


def report(rows: list[dict], cutoffs: list[int], tags: dict[int, str] | None) -> dict:
    """Stats overall, per density bin and (if tagged) per lighting tag, for ``rows``."""
    out = {"overall": stats(rows),
           "by_bin": {b: stats([r for r in rows if bin_of(r["gt"], cutoffs) == b])
                      for b in BINS}}
    if tags is not None:
        out["by_lighting"] = {t: stats([r for r in rows if tags[r["clip"]] == t])
                              for t in LIGHTING}
        out["clips_per_lighting"] = {t: sorted(c for c, v in tags.items() if v == t)
                                     for t in LIGHTING}
    return out


def full_report(rows: list[dict], cutoffs: list[int], tags: dict[int, str] | None) -> dict:
    """Primary (all clips) and, with tags, secondary (day-only) reports."""
    result = {"all_clips": report(rows, cutoffs, tags)}
    if tags is not None:
        day = [r for r in rows if tags[r["clip"]] == "day"]
        result["day_only"] = report(day, cutoffs, {c: v for c, v in tags.items() if v == "day"})
    return result


def format_block(title: str, rep: dict, targets: dict[str, float]) -> str:
    def line(name, s, target=""):
        if not s.get("frames"):
            return f"{name:<10} no frames"
        return (f"{name:<10} {s['clips']:>3} clips {s['frames']:>6} frames   "
                f"mean err {s['mean_abs_error_pct']:6.1f}%  median {s['median_abs_error_pct']:6.1f}%"
                f"  bias {s['mean_bias_pct']:+6.1f}%  {target}")
    lines = [title, line("overall", rep["overall"])]
    for b in BINS:
        lines.append(line(b, rep["by_bin"][b],
                          f"[provisional draft target <= {targets[b]:g}%]" if b in targets else ""))
    for t, s in rep.get("by_lighting", {}).items():
        lines.append(line(t, s))
    return "\n".join(lines)


def write_outputs(result: dict, meta: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "count_error.json").write_text(json.dumps({**meta, **result}, indent=1) + "\n")
    with (out_dir / "count_error.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "group", "clips", "frames", "mean_abs_error_pct",
                    "median_abs_error_pct", "mean_bias_pct"])
        for subset, rep in result.items():
            groups = {"overall": rep["overall"], **rep["by_bin"], **rep.get("by_lighting", {})}
            for name, s in groups.items():
                if s.get("frames"):
                    w.writerow([subset, name, s["clips"], s["frames"],
                                round(s["mean_abs_error_pct"], 2),
                                round(s["median_abs_error_pct"], 2), round(s["mean_bias_pct"], 2)])
    return [out_dir / "count_error.json", out_dir / "count_error.csv"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--pred-dir", type=Path, help="default: data/outputs/points/<split>")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["out_dir"]) / args.split
    if args.split == "test" and (out_dir / "count_error.json").exists():
        raise SystemExit(f"{out_dir / 'count_error.json'} exists: the test split is scored "
                         "once. Delete the file only if the run was invalid.")
    ds = D.dataset_from_config(D.load_config(Path(cfg["data_config"])), args.split)
    gt: dict[int, dict[str, np.ndarray]] = {}
    for ref in ds.refs:
        gt.setdefault(ref.clip, {})[str(ref.frame_id)] = D.load_gt(ref.gt_path)[0]
    pred_dir = args.pred_dir or Path("data/outputs/points") / args.split
    pred = {int(p.name.split("_")[1]): pm.load_points_json(p)
            for p in sorted(pred_dir.glob("clip_*_points.json"))}
    cutoffs = [int(c) for c in cfg["bin_cutoffs"]]
    tags = load_tags(Path(cfg["lighting_tags"]), args.split)
    if tags is None:
        print("lighting tags missing or incomplete: lighting and day-only views are skipped")
    result = full_report(frame_rows(pred, gt, float(cfg["threshold"])), cutoffs, tags)
    targets = {k: float(v) for k, v in cfg["provisional_targets_pct"].items()}
    label = (f"DroneCrowd {args.split} split, B point count at score >= {cfg['threshold']}, "
             f"bins {cutoffs}")
    print(label)
    print(format_block("ALL CLIPS (primary)", result["all_clips"], targets))
    if "day_only" in result:
        print(format_block("DAY-ONLY CLIPS (secondary)", result["day_only"], targets))
    meta = {"label": label, "split": args.split, "threshold": cfg["threshold"],
            "bin_cutoffs": cutoffs, "tags_used": tags is not None,
            "targets_note": "provisional, draft Phase 0 sheet s4.2, not pass/fail"}
    for path in write_outputs(result, meta, out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
