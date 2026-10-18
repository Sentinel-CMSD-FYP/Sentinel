"""Step 6: per frame count error of DM-Count, by crowd density bin and by lighting.

    python eval/density_dmcount/count_error.py --split val  --preds preds.csv --name zeroshot_qnrf
    python eval/density_dmcount/count_error.py --split test --preds preds.csv --name zeroshot_qnrf

Follows the team protocol of decision 0005 (Ahmed's ``eval/count_error.py`` for Model B),
for float density counts instead of point counts:
  * per frame: absolute error |pred - GT|, relative error |pred - GT| / GT, signed bias
    (pred - GT) / GT (negative = undercounting). Frames with GT < ``min_gt_for_pct`` stay in
    MAE and RMSE but are left out of the % numbers, and are counted;
  * density bin from the frame's GT count: sparse < c0, moderate c0 to c1, dense > c1;
  * reported for all clips (primary) and day only clips (secondary), plus per lighting tag;
  * provisional targets are shown for reference, nothing is called pass or fail;
  * a test result is never overwritten (the test split is scored once per run name).

``--preds`` is a CSV with columns ``clip,frame_id,pred``: clip number, 0 based frame index in
the clip, predicted count (sum of the density map). It must hold exactly the frames of the
split as ``ml/configs/density_data.yaml`` strides them. GT counts come from the raw GT files.
Outputs in ``<out_dir>/<name>/<split>/``: ``count_error.{json,csv,md}`` and ``frames.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml
from sentinel_ml.density.data import load_split_from_config
from sentinel_ml.localization.data import load_gt

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "ml" / "configs" / "density_data.yaml"
BINS = ("sparse", "moderate", "dense")
LIGHTING = ("day", "dusk", "night")
COLUMNS = ("frames", "clips", "mean_gt", "mae", "rmse", "pct_frames", "mean_abs_error_pct",
           "median_abs_error_pct", "mean_bias_pct")


def bin_of(n_gt: int, cutoffs: list[int]) -> str:
    """Density bin of a frame from its GT count: sparse < c0, moderate c0 to c1, dense > c1."""
    lo, hi = cutoffs
    if n_gt < lo:
        return "sparse"
    return "moderate" if n_gt <= hi else "dense"


def load_preds(path: Path) -> dict[tuple[int, int], float]:
    """{(clip, frame_id): predicted count} from a ``clip,frame_id,pred`` CSV."""
    preds: dict[tuple[int, int], float] = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"clip", "frame_id", "pred"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: needs columns clip, frame_id, pred")
        for r in reader:
            key, pred = (int(r["clip"]), int(r["frame_id"])), float(r["pred"])
            if key in preds:
                raise ValueError(f"{path}: duplicate prediction for clip {key[0]} frame {key[1]}")
            if not math.isfinite(pred):
                raise ValueError(f"{path}: prediction {pred} for clip {key[0]} frame {key[1]}")
            preds[key] = pred
    return preds


def join(gt: list[dict], preds: dict[tuple[int, int], float]) -> list[dict]:
    """Add ``pred`` to every GT row. Raises if a GT frame has no prediction or a prediction
    has no GT frame, so a wrong split or stride cannot pass silently."""
    keys = {(r["clip"], r["frame_id"]) for r in gt}
    missing, extra = keys - preds.keys(), preds.keys() - keys
    if missing or extra:
        raise ValueError(f"{len(missing)} GT frames without prediction (e.g. "
                         f"{sorted(missing)[:3]}), {len(extra)} predictions without GT frame "
                         f"(e.g. {sorted(extra)[:3]})")
    return [{**r, "pred": preds[(r["clip"], r["frame_id"])]} for r in gt]


def load_tags(path: Path, split: str) -> dict[int, str] | None:
    """{clip: day|dusk|night} for ``split``, or None if the file is missing or not filled in."""
    if not Path(path).exists():
        return None
    tags = (yaml.safe_load(Path(path).read_text()) or {}).get(split) or {}
    if not tags or any(v is None for v in tags.values()):
        return None
    bad = {c: v for c, v in tags.items() if v not in LIGHTING}
    if bad:
        raise ValueError(f"unknown lighting tags {bad}; use one of {LIGHTING}")
    return {int(c): v for c, v in tags.items()}


def stats(rows: list[dict], min_gt: int) -> dict:
    """Frames, clips, MAE and RMSE over all ``rows``; mean/median relative error (%) and mean
    signed bias (%) over the rows with GT >= ``min_gt`` (``pct_frames`` of them)."""
    if not rows:
        return {"frames": 0}
    gt = np.array([r["gt"] for r in rows], dtype=np.float64)
    diff = np.array([r["pred"] for r in rows], dtype=np.float64) - gt
    out = {"frames": len(rows), "clips": len({r["clip"] for r in rows}),
           "mean_gt": float(gt.mean()), "mae": float(np.abs(diff).mean()),
           "rmse": float(np.sqrt((diff ** 2).mean()))}
    keep = gt >= max(min_gt, 1)  # GT = 0 has no relative error whatever min_gt is
    out["pct_frames"] = int(keep.sum())
    if keep.any():
        rel = diff[keep] / gt[keep]
        out |= {"mean_abs_error_pct": float(np.abs(rel).mean() * 100),
                "median_abs_error_pct": float(np.median(np.abs(rel)) * 100),
                "mean_bias_pct": float(rel.mean() * 100)}
    return out


def report(rows: list[dict], cutoffs: list[int], min_gt: int,
           tags: dict[int, str] | None) -> dict:
    """Stats overall, per density bin and (if tagged) per lighting tag."""
    out = {"overall": stats(rows, min_gt),
           "by_bin": {b: stats([r for r in rows if bin_of(r["gt"], cutoffs) == b], min_gt)
                      for b in BINS}}
    if tags is not None:
        out["by_lighting"] = {t: stats([r for r in rows if tags[r["clip"]] == t], min_gt)
                              for t in LIGHTING}
        clips = {r["clip"] for r in rows}
        out["clips_per_lighting"] = {t: sorted(c for c in clips if tags[c] == t)
                                     for t in LIGHTING}
    return out


def full_report(rows: list[dict], cutoffs: list[int], min_gt: int,
                tags: dict[int, str] | None) -> dict:
    """Primary (all clips) and, with tags, secondary (day only) reports."""
    if tags is not None and {r["clip"] for r in rows} - tags.keys():
        raise ValueError(f"no lighting tag for clips {sorted({r['clip'] for r in rows} - tags.keys())}")
    result = {"all_clips": report(rows, cutoffs, min_gt, tags)}
    if tags is not None:
        day = [r for r in rows if tags[r["clip"]] == "day"]
        result["day_only"] = report(day, cutoffs, min_gt, tags)
    return result


def groups(rep: dict) -> dict[str, dict]:
    return {"overall": rep["overall"], **rep["by_bin"], **rep.get("by_lighting", {})}


def markdown(result: dict, cutoffs: list[int], targets: dict[str, float]) -> str:
    """One markdown table per subset, one row per group with frames."""
    lo, hi = cutoffs
    names = {"sparse": f"sparse (GT < {lo})", "moderate": f"moderate ({lo} to {hi})",
             "dense": f"dense (GT > {hi})"}
    out = []
    for subset, rep in result.items():
        out += [f"**{subset.replace('_', ' ')}**", "",
                "| group | clips | frames | mean GT | MAE | RMSE | mean err | median err "
                "| bias | provisional target |", "|" + "---|" * 10]
        for name, s in groups(rep).items():
            if not s["frames"]:
                continue
            pct = (f"{s['mean_abs_error_pct']:.1f}% | {s['median_abs_error_pct']:.1f}% | "
                   f"{s['mean_bias_pct']:+.1f}%") if "mean_abs_error_pct" in s else "n/a | n/a | n/a"
            target = f"<= {targets[name]:g}%" if name in targets else ""
            out.append(f"| {names.get(name, name)} | {s['clips']} | {s['frames']} | "
                       f"{s['mean_gt']:.1f} | {s['mae']:.2f} | {s['rmse']:.2f} | {pct} | {target} |")
        out.append("")
    return "\n".join(out)


def write_outputs(result: dict, rows: list[dict], meta: dict, md: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / n for n in ("count_error.json", "count_error.csv", "count_error.md",
                                   "frames.csv")]
    paths[0].write_text(json.dumps({**meta, **result}, indent=1) + "\n")
    with paths[1].open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "group", *COLUMNS])
        for subset, rep in result.items():
            for name, s in groups(rep).items():
                if s["frames"]:
                    w.writerow([subset, name, *(round(s[c], 3) if c in s else "" for c in COLUMNS)])
    paths[2].write_text(f"{meta['label']}\n\n{md}")
    with paths[3].open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "frame_id", "gt", "pred", "bin"])
        w.writeheader()
        w.writerows({**r, "bin": bin_of(r["gt"], meta["bin_cutoffs"])} for r in rows)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--preds", type=Path, required=True, help="CSV with clip,frame_id,pred")
    ap.add_argument("--name", required=True, help="run name, e.g. zeroshot_qnrf")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--root", type=Path, help="dataset root, overrides the config")
    ap.add_argument("--out", type=Path, help="overrides count_error.out_dir in the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ce = cfg["count_error"]
    out_dir = (args.out or REPO / ce["out_dir"]) / args.name / args.split
    if args.split == "test" and (out_dir / "count_error.json").exists():
        raise SystemExit(f"{out_dir / 'count_error.json'} exists: the test split is scored "
                         "once per run. Delete it only if the run was invalid.")
    ds = load_split_from_config(cfg, args.split, root=args.root, repo=REPO)
    gt = [{"clip": r.clip, "frame_id": r.frame_id, "gt": len(load_gt(r.gt_path)[0])}
          for r in ds.refs]
    rows = join(gt, load_preds(args.preds))
    cutoffs, min_gt = [int(c) for c in ce["bin_cutoffs"]], int(ce["min_gt_for_pct"])
    tags = load_tags(REPO / ce["lighting_tags"], args.split)
    if tags is None:
        print("lighting tags missing or incomplete: lighting and day only views are skipped")
    result = full_report(rows, cutoffs, min_gt, tags)
    targets = {k: float(v) for k, v in ce["provisional_targets_pct"].items()}
    label = (f"DroneCrowd {args.split} split, DM-Count run '{args.name}', bins {cutoffs}, "
             f"% error over frames with GT >= {min_gt}; targets are provisional (draft Phase 0 "
             "sheet s4.2), not pass/fail")
    md = markdown(result, cutoffs, targets)
    print(label + "\n\n" + md)
    meta = {"label": label, "split": args.split, "name": args.name, "preds": str(args.preds),
            "bin_cutoffs": cutoffs, "min_gt_for_pct": min_gt, "tags_used": tags is not None,
            "provisional_targets_pct": targets}
    for path in write_outputs(result, rows, meta, md, out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
