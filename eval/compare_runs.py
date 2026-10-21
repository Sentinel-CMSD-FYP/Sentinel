"""Per-clip comparison of two tracker runs: count error (E-3) and TrackEval IDF1, HOTA, MOTA.

Reads the ``track_count_error.json`` of each run and one TrackEval ``results.json`` that holds
both runs as trackers, and writes one row per clip plus the overall means. Nothing is selected
here: both runs are shown side by side and neither is called better.

Units: counts of ids and metric values. Metric threshold: IoU 0.5 unless ``--iou`` says otherwise.

    python eval/compare_runs.py --count-a data/outputs/track_count_error/test/track_count_error.json \
        --count-b data/outputs/track_count_error_run2/test/track_count_error.json \
        --trackeval data/outputs/trackeval/test/results.json \
        --tracker-a run1_step2_selected --tracker-b run2_frozen_e2 --out data/outputs/run_comparison/test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = ["clip", "gt_people",
          "ids_a", "err_pct_a", "err_pct_min45_a", "ids_b", "err_pct_b", "err_pct_min45_b",
          "idf1_a", "idf1_b", "hota_a", "hota_b", "mota_a", "mota_b", "idsw_a", "idsw_b"]


def compare(count_a: dict, count_b: dict, trackeval_res: dict, tracker_a: str, tracker_b: str,
            iou_key: str = "iou_0.5") -> list[dict]:
    """One row per clip, joining the count-error rows with the TrackEval per-clip metrics."""
    rows_a = {r["clip"]: r for r in count_a["per_clip"]}
    rows_b = {r["clip"]: r for r in count_b["per_clip"]}
    if set(rows_a) != set(rows_b):
        raise ValueError("the two runs cover different clips")
    te = trackeval_res[iou_key]
    out = []
    for clip in sorted(rows_a):
        seq = f"{clip:03d}"
        a, b = rows_a[clip], rows_b[clip]
        ta, tb = te[tracker_a]["per_clip"][seq], te[tracker_b]["per_clip"][seq]
        out.append({"clip": clip, "gt_people": a["gt_people"],
                    "ids_a": a["ids_all"], "err_pct_a": a["err_pct_all"],
                    "err_pct_min45_a": a["err_pct_min_len"],
                    "ids_b": b["ids_all"], "err_pct_b": b["err_pct_all"],
                    "err_pct_min45_b": b["err_pct_min_len"],
                    "idf1_a": ta["IDF1"], "idf1_b": tb["IDF1"],
                    "hota_a": ta["HOTA"], "hota_b": tb["HOTA"],
                    "mota_a": ta["MOTA"], "mota_b": tb["MOTA"],
                    "idsw_a": ta["IDSW"], "idsw_b": tb["IDSW"]})
    return out


def means(rows: list[dict]) -> dict:
    """Mean of every numeric column over the clips."""
    return {k: sum(r[k] for r in rows) / len(rows) for k in FIELDS if k != "clip"}


def format_table(rows: list[dict], name_a: str, name_b: str) -> str:
    head = (f"clip   GT | {name_a}: ids  err%  err%>=45  IDF1  HOTA | {name_b}: ids  err%  "
            f"err%>=45  IDF1  HOTA")
    lines = [head]
    for r in rows:
        lines.append(f"{r['clip']:4d} {r['gt_people']:4d} | {r['ids_a']:12d} {r['err_pct_a']:5.1f} "
                     f"{r['err_pct_min45_a']:8.1f} {r['idf1_a']:5.3f} {r['hota_a']:5.3f} | "
                     f"{r['ids_b']:12d} {r['err_pct_b']:5.1f} {r['err_pct_min45_b']:8.1f} "
                     f"{r['idf1_b']:5.3f} {r['hota_b']:5.3f}")
    m = means(rows)
    lines.append(f"mean {m['gt_people']:4.0f} | {m['ids_a']:12.0f} {m['err_pct_a']:5.1f} "
                 f"{m['err_pct_min45_a']:8.1f} {m['idf1_a']:5.3f} {m['hota_a']:5.3f} | "
                 f"{m['ids_b']:12.0f} {m['err_pct_b']:5.1f} {m['err_pct_min45_b']:8.1f} "
                 f"{m['idf1_b']:5.3f} {m['hota_b']:5.3f}")
    return chr(10).join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--count-a", type=Path, required=True)
    ap.add_argument("--count-b", type=Path, required=True)
    ap.add_argument("--trackeval", type=Path, required=True)
    ap.add_argument("--tracker-a", required=True)
    ap.add_argument("--tracker-b", required=True)
    ap.add_argument("--iou", default="iou_0.5", help="key of the TrackEval threshold block")
    ap.add_argument("--out", type=Path, help="write the per-clip rows as CSV")
    args = ap.parse_args()
    rows = compare(json.loads(args.count_a.read_text()), json.loads(args.count_b.read_text()),
                   json.loads(args.trackeval.read_text()), args.tracker_a, args.tracker_b, args.iou)
    print(format_table(rows, "A", "B"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
