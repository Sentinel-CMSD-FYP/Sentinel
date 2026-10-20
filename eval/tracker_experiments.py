"""Val-only tracker experiments: score_thr sweep (step 1), then min_hits x max_age (step 2).

The grids, the metrics and the selection rule are fixed in ``ml/configs/tracker_experiments.yaml``
and docs/project-log.md before any experiment runs. For every configuration the tracker is run on all
val clips and scored with the E-3 count error (without the 45-frame filter = primary, with it =
secondary), ids per person, GT-people coverage, spurious ids and extra ids.

Selection rule: among configurations whose mean GT-people coverage >= ``min_coverage``, take the
lowest primary count error; configurations within ``tie_points`` of it are tied and the tie-break
of the step decides (step 1: score_thr closest to 0.20, then lower; step 2: smaller min_hits,
then smaller max_age). The full table is always written, not only the winner.

This script refuses the test split. B's val points are optimistic (B trained on those clips).

    python eval/tracker_experiments.py --config ml/configs/tracker_experiments.yaml --step 1
    python eval/tracker_experiments.py --config ml/configs/tracker_experiments.yaml --step 2
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import diagnose_tracking as dg  # eval/ is on sys.path when this file is run as a script
import numpy as np
import track_count_error as tc
import yaml

from sentinel_ml.localization import data as D
from sentinel_ml.tracking import TrackerConfig
from sentinel_ml.tracking.run_tracker import track_clip

B_THRESHOLD = 0.20  # the score threshold chosen for B on val (docs/project-log.md)


def evaluate_config(points_docs: dict[int, dict], gt_by_clip: dict[int, dict], cfg: TrackerConfig,
                    min_len: int) -> dict:
    """Run the tracker with ``cfg`` on every clip and score it. Means are over clips."""
    rows, coverage, spurious, extra = [], [], 0, 0
    for clip, doc in sorted(points_docs.items()):
        tracks_doc = track_clip(doc, cfg)
        gt = gt_by_clip[clip]
        row = tc.clip_row(clip, tracks_doc, gt, min_len)
        outputs = {int(k): v for k, v in tracks_doc["frames"].items()}
        ident = dg.gt_identity(outputs, gt, row["gt_people"])
        rows.append(row)
        coverage.append(ident["gt_people_tracked"] / row["gt_people"])
        spurious += ident["spurious_ids"]
        extra += ident["extra_ids_for_tracked"]
    agg = tc.aggregate(rows)
    return {"mean_err_pct": agg["all"]["mean_err_pct"],
            "mean_err_pct_min_len": agg["min_len"]["mean_err_pct"],
            "ids_per_person": agg["all"]["mean_ids_per_person"],
            "mean_coverage": float(np.mean(coverage)),
            "spurious_ids": spurious, "extra_ids_for_tracked": extra,
            "clips": len(rows)}


def select(results: list[dict], min_coverage: float, tie_points: float, step: int) -> dict | None:
    """Apply the pre-declared selection rule. Returns the chosen result, or None."""
    eligible = [r for r in results if r["mean_coverage"] >= min_coverage]
    if not eligible:
        return None
    best = min(r["mean_err_pct"] for r in eligible)
    tied = [r for r in eligible if r["mean_err_pct"] <= best + tie_points]
    if step == 1:
        return min(tied, key=lambda r: (round(abs(r["score_thr"] - B_THRESHOLD), 9), r["score_thr"]))
    return min(tied, key=lambda r: (r["min_hits"], r["max_age"]))


def check_val_only(points_dir: Path) -> None:
    """Refuse anything that looks like the test split."""
    if "test" in {p.lower() for p in points_dir.parts}:
        raise SystemExit("tracker experiments are val-only; never tune on test")


def write_results(results: list[dict], chosen: dict | None, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({**meta, "results": results, "selected": chosen}, indent=1) + "\n")
    fields = ["score_thr", "min_hits", "max_age", "mean_err_pct", "mean_err_pct_min_len",
              "ids_per_person", "mean_coverage", "spurious_ids", "extra_ids_for_tracked"]
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            w.writerow([round(r[k], 4) if isinstance(r[k], float) else r[k] for k in fields])


def format_table(results: list[dict], chosen: dict | None, min_coverage: float) -> str:
    lines = ["score_thr min_hits max_age  err%(all)  err%(>=45)  ids/person  coverage  spurious  extra"]
    for r in results:
        mark = ""
        if chosen is r:
            mark = "  <= selected"
        elif r["mean_coverage"] < min_coverage:
            mark = "  (coverage below floor)"
        lines.append(f"{r['score_thr']:9.2f} {r['min_hits']:8d} {r['max_age']:7d} {r['mean_err_pct']:10.1f}"
                     f" {r['mean_err_pct_min_len']:11.1f} {r['ids_per_person']:11.2f} "
                     f"{r['mean_coverage']:9.3f} {r['spurious_ids']:9d} {r['extra_ids_for_tracked']:6d}{mark}")
    return "\n".join(lines)


def write_selected_config(base_path: Path, chosen: dict, out_path: Path) -> None:
    """Copy the baseline tracker config with the selected values (baseline file untouched)."""
    cfg = yaml.safe_load(base_path.read_text())
    cfg["tracker"].update(score_thr=chosen["score_thr"], min_hits=chosen["min_hits"],
                          max_age=chosen["max_age"])
    header = ("# Selected on val by eval/tracker_experiments.py (docs/project-log.md). Frozen: do not\n"
              "# change after the single test scoring. B's val points are optimistic.\n")
    out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--step", type=int, choices=[1, 2], required=True)
    ap.add_argument("--score-thr", type=float, help="step 2: override the step-1 selection")
    ap.add_argument("--write-selected", type=Path, help="write the chosen config to this yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    points_dir = Path(cfg["points_dir"])
    check_val_only(points_dir)
    base = TrackerConfig(**yaml.safe_load(Path(cfg["tracker_config"]).read_text())["tracker"])
    out_dir = Path(cfg["out_dir"]) / f"step{args.step}"
    if args.step == 1:
        grid = [replace(base, score_thr=t, min_hits=3, max_age=5) for t in cfg["step1"]["score_thr"]]
    else:
        thr = args.score_thr
        if thr is None:
            thr = json.loads((Path(cfg["out_dir"]) / "step1" / "results.json").read_text())[
                "selected"]["score_thr"]
        grid = [replace(base, score_thr=thr, min_hits=h, max_age=a)
                for h in cfg["step2"]["min_hits"] for a in cfg["step2"]["max_age"]]

    data_cfg = D.load_config(Path(cfg["data_config"]))
    docs = {int(f.name.split("_")[1]): json.loads(f.read_text())
            for f in sorted(points_dir.glob("clip_*_points.json"))}
    gts = {clip: D.clip_gt_frames(data_cfg, "val", clip, sorted(int(k) for k in doc["frames"]))
           for clip, doc in docs.items()}
    results = []
    for k, c in enumerate(grid, 1):
        r = evaluate_config(docs, gts, c, int(cfg["min_len_frames"]))
        r.update(score_thr=c.score_thr, min_hits=c.min_hits, max_age=c.max_age)
        results.append(r)
        print(f"[{k}/{len(grid)}] score_thr {c.score_thr} min_hits {c.min_hits} max_age {c.max_age}: "
              f"err {r['mean_err_pct']:.1f}% coverage {r['mean_coverage']:.3f}", flush=True)
    sel = cfg["selection"]
    chosen = select(results, float(sel["min_coverage"]), float(sel["tie_points"]), args.step)
    print(format_table(results, chosen, float(sel["min_coverage"])))
    meta = {"split": "val", "clips": len(docs), "step": args.step,
            "note": "val only; B trained on these clips so results are optimistic",
            "selection_rule": sel}
    write_results(results, chosen, meta, out_dir)
    print(f"wrote {out_dir / 'results.json'} and results.csv; selected: {chosen}")
    if args.write_selected and chosen is not None:
        write_selected_config(Path(cfg["tracker_config"]), chosen, args.write_selected)
        print(f"wrote {args.write_selected}")


if __name__ == "__main__":
    main()
