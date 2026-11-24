"""Step 10: side by side table of several DM-Count runs scored by ``count_error.py``.

    python eval/density_dmcount/compare_runs.py --split test zeroshot_qnrf finetune_e4

Reads ``<out_dir>/<run>/<split>/count_error.json`` and ``frames.csv`` of each run and writes
``<out_dir>/compare_<split>.md``: per group MAE, mean err and bias for each run (all clips
and day only), Model B's test numbers as a reference, and the mean error per clip.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml
from count_error import DEFAULT_CONFIG, REPO, groups

# Model B (Ahmed) point count on the same test frames: decision 0005 on
# origin/feat/ml-localization-tracking, as (mean err %, bias %).
MODEL_B_TEST = {
    "all_clips": {"overall": (21.6, -9.6), "sparse": (24.2, -14.9), "moderate": (26.3, -8.6),
                  "dense": (7.4, -1.7), "day": (18.9, -5.6), "night": (45.4, -45.4)},
    "day_only": {"overall": (18.9, -5.6), "sparse": (20.1, -8.6), "moderate": (24.3, -5.3),
                 "dense": (7.4, -1.7)},
}


def group_table(results: dict[str, dict], subset: str, split: str, targets: dict) -> list[str]:
    runs = list(results)
    with_b = split == "test"
    head = ["group", "frames", *(f"{r} MAE / err / bias" for r in runs)]
    head += ["Model B err / bias"] * with_b + ["provisional target"]
    lines = [f"**{subset.replace('_', ' ')}**", "", "| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    first = groups(results[runs[0]][subset])
    for name, s in first.items():
        if not s["frames"]:
            continue
        cells = [name, str(s["frames"])]
        for r in runs:
            g = groups(results[r][subset])[name]
            cells.append(f"{g['mae']:.1f} / {g['mean_abs_error_pct']:.1f}% / "
                         f"{g['mean_bias_pct']:+.1f}%")
        if with_b:
            b = MODEL_B_TEST.get(subset, {}).get(name)
            cells.append(f"{b[0]:.1f}% / {b[1]:+.1f}%" if b else "")
        cells.append(f"<= {targets[name]:g}%" if name in targets else "")
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def clip_errors(frames_csv: Path, min_gt: int) -> dict[int, tuple[float, float]]:
    """{clip: (mean GT, mean abs rel error %)} over frames with GT >= ``min_gt``."""
    per = defaultdict(list)
    with frames_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            gt, pred = int(row["gt"]), float(row["pred"])
            if gt >= min_gt:
                per[int(row["clip"])].append((gt, abs(pred - gt) / gt * 100))
    return {c: (sum(g for g, _ in v) / len(v), sum(e for _, e in v) / len(v))
            for c, v in sorted(per.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run names, e.g. zeroshot_qnrf finetune_e4")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    ce = yaml.safe_load(args.config.read_text())["count_error"]
    out_dir = REPO / ce["out_dir"]
    targets = {k: float(v) for k, v in ce["provisional_targets_pct"].items()}
    results = {r: json.loads((out_dir / r / args.split / "count_error.json").read_text())
               for r in args.runs}
    clips = {r: clip_errors(out_dir / r / args.split / "frames.csv", int(ce["min_gt_for_pct"]))
             for r in args.runs}

    lines = [(f"DroneCrowd {args.split} split: {', '.join(args.runs)}. Cells: MAE / mean abs rel "
              "error / bias (negative = undercount). Targets are provisional (draft Phase 0 "
              "sheet s4.2), not pass/fail."), ""]
    for subset in ("all_clips", "day_only"):
        if subset in results[args.runs[0]]:
            lines += group_table(results, subset, args.split, targets)
    lines += ["**mean err per clip**", "",
              "| clip | mean GT | " + " | ".join(args.runs) + " |", "|" + "---|" * (2 + len(args.runs))]
    for c, (gt, _) in clips[args.runs[0]].items():
        lines.append(f"| {c} | {gt:.0f} | " + " | ".join(f"{clips[r][c][1]:.1f}%" for r in args.runs)
                     + " |")
    path = out_dir / f"compare_{args.split}.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
