"""Tracking count error of Model E per sequence (ROADMAP E-3).

For each clip:  error = |unique track ids - unique GT person ids| / unique GT person ids.
Reported two ways, never dropping a clip:
  * primary:   every output track id counts (no filter);
  * secondary: only ids that were output in at least ``min_len_frames`` frames (the authors'
               evaluation drops shorter tracks; this needs the whole clip, so it is offline).
Then the mean over clips (primary view: all clips; secondary view: day-only clips if the
lighting tags are filled in). The signed ratio shows over-counting (ids > people).

GT person count = number of distinct person ids in the clip's GT .mat files (the authors'
MOT-format GT has the same ids). The test split is scored once: this script refuses to
overwrite an existing test result.

Units: counts of ids. The draft target is shown for reference only; nothing is called pass or
fail.

    python eval/track_count_error.py --config ml/configs/track_count_error.yaml --split val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from sentinel_ml.localization import data as D

LIGHTING = ("day", "dusk", "night")


def clip_row(clip: int, tracks_doc: dict, gt: dict, min_len: int) -> dict:
    """Counts and errors for one clip. ``gt`` is ``{frame_id: (points, person_ids)}``."""
    lengths: dict[int, int] = {}
    for rows in tracks_doc["frames"].values():
        for tid, *_ in rows:
            lengths[tid] = lengths.get(tid, 0) + 1
    n_gt = len({int(i) for _, ids in gt.values() for i in ids})
    n_all = len(lengths)
    n_long = sum(1 for n in lengths.values() if n >= min_len)
    mean_gt = float(np.mean([len(ids) for _, ids in gt.values()]))
    return {"clip": clip, "gt_people": n_gt, "mean_gt_per_frame": mean_gt,
            "ids_all": n_all, "ids_min_len": n_long,
            "err_pct_all": 100.0 * abs(n_all - n_gt) / n_gt,
            "err_pct_min_len": 100.0 * abs(n_long - n_gt) / n_gt,
            "ratio_all": n_all / n_gt, "ratio_min_len": n_long / n_gt}


def aggregate(rows: list[dict]) -> dict:
    """Mean, median and max of the per-clip errors, and the mean ids-per-person ratio."""
    if not rows:
        return {"clips": 0}
    out = {"clips": len(rows)}
    for key in ("all", "min_len"):
        e = np.array([r[f"err_pct_{key}"] for r in rows])
        out[key] = {"mean_err_pct": float(e.mean()), "median_err_pct": float(np.median(e)),
                    "max_err_pct": float(e.max()),
                    "mean_ids_per_person": float(np.mean([r[f"ratio_{key}"] for r in rows]))}
    return out


def summarise(rows: list[dict], tags: dict[int, str] | None) -> dict:
    """Primary view (all clips) and, with tags, the day-only view and per-lighting views."""
    result = {"all_clips": aggregate(rows)}
    if tags is not None:
        result["day_only"] = aggregate([r for r in rows if tags[r["clip"]] == "day"])
        result["by_lighting"] = {t: aggregate([r for r in rows if tags[r["clip"]] == t])
                                 for t in LIGHTING}
    return result


def load_tags(path: Path, split: str) -> dict[int, str] | None:
    """{clip: tag} for ``split``, or None if the file is missing or not fully filled in."""
    if not Path(path).exists():
        return None
    tags = yaml.safe_load(Path(path).read_text()).get(split) or {}
    if not tags or any(v is None for v in tags.values()):
        return None
    bad = {c: v for c, v in tags.items() if v not in LIGHTING}
    if bad:
        raise ValueError(f"unknown lighting tags {bad}; use one of {LIGHTING}")
    return {int(c): v for c, v in tags.items()}


def format_report(rows: list[dict], summary: dict, target_pct: float) -> str:
    lines = ["clip  GT   ids   ids>=min  err%(all)  err%(>=min)  ids/person",
             *[f"{r['clip']:4d} {r['gt_people']:4d} {r['ids_all']:5d} {r['ids_min_len']:8d}"
               f"  {r['err_pct_all']:9.1f}  {r['err_pct_min_len']:11.1f}  {r['ratio_all']:9.2f}"
               for r in sorted(rows, key=lambda r: r["clip"])]]

    def line(name, a):
        if not a.get("clips"):
            return f"{name:<22} no clips"
        return (f"{name:<22} {a['clips']:3d} clips  mean err: no filter {a['all']['mean_err_pct']:6.1f}%"
                f" (median {a['all']['median_err_pct']:6.1f}%) | >=min {a['min_len']['mean_err_pct']:6.1f}%"
                f" (median {a['min_len']['median_err_pct']:6.1f}%) | ids/person {a['all']['mean_ids_per_person']:.2f}")

    lines.append("")
    lines.append(line("ALL CLIPS (primary)", summary["all_clips"]))
    if "day_only" in summary:
        lines.append(line("DAY-ONLY (secondary)", summary["day_only"]))
        for t, a in summary["by_lighting"].items():
            lines.append(line(f"  {t}", a))
    lines.append(f"provisional figure, draft Phase 0 sheet (unsigned): tracking count <= "
                 f"{target_pct:g}%; shown for reference, not pass or fail")
    return "\n".join(lines)


def write_outputs(rows: list[dict], summary: dict, meta: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "track_count_error.json").write_text(
        json.dumps({**meta, **summary, "per_clip": rows}, indent=1) + "\n")
    fields = ["clip", "gt_people", "mean_gt_per_frame", "ids_all", "ids_min_len", "err_pct_all",
              "err_pct_min_len", "ratio_all", "ratio_min_len"]
    with (out_dir / "track_count_error.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in sorted(rows, key=lambda r: r["clip"]):
            w.writerow([round(r[k], 2) if isinstance(r[k], float) else r[k] for k in fields])
    return [out_dir / "track_count_error.json", out_dir / "track_count_error.csv"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--tracks-dir", type=Path, help="default: <tracks_dir>/<split>")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["out_dir"]) / args.split
    if args.split == "test" and (out_dir / "track_count_error.json").exists():
        raise SystemExit(f"{out_dir / 'track_count_error.json'} exists: the test split is "
                         "scored once. Delete the file only if the run was invalid.")
    data_cfg = D.load_config(Path(cfg["data_config"]))
    manifest = json.loads(Path(data_cfg["manifest"]).read_text())
    tracks_dir = args.tracks_dir or Path(cfg["tracks_dir"]) / args.split
    rows = []
    for clip in manifest[args.split]:
        path = tracks_dir / f"clip_{clip:03d}_tracks.json"
        if not path.exists():
            raise SystemExit(f"missing {path}: run sentinel_ml.tracking.run_tracker first")
        doc = json.loads(path.read_text())
        gt = D.clip_gt_frames(data_cfg, args.split, clip, sorted(int(k) for k in doc["frames"]))
        rows.append(clip_row(clip, doc, gt, int(cfg["min_len_frames"])))
    tags = load_tags(Path(cfg["lighting_tags"]), args.split)
    summary = summarise(rows, tags)
    label = (f"DroneCrowd {args.split} split, {len(rows)} clips, tracker on B points at "
             f"score >= {cfg.get('score_thr_note', '0.2')}")
    print(label)
    print(format_report(rows, summary, float(cfg["provisional_target_pct"])))
    meta = {"label": label, "split": args.split, "min_len_frames": cfg["min_len_frames"],
            "tags_used": tags is not None,
            "target_note": "provisional, unsigned draft Phase 0 sheet, not pass/fail"}
    for p in write_outputs(rows, summary, meta, out_dir):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
