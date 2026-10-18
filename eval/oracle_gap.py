"""Gap between a tracker and the perfect-association oracle, per clip, and the DDCF decision rule.

Reads a TrackEval ``results_oracle.json`` (from ``trackeval_eval.py --oracle``) and compares the
mean-over-clips and per-clip IDF1 of a tracker with the oracle on the same detections. The gain is
an upper bound on what any association method can add on those detections. The rule (decision
0009): the association step (DDCF) is built only if the gain is at least ``--min-gain`` IDF1
(0.15) at IoU 0.5.

The oracle uses ground-truth identities: it is not a tracker, and its numbers are labelled
"oracle, measurement only" and never reported as a result of ours.

    python eval/oracle_gap.py --trackeval data/outputs/trackeval/test/results_oracle.json \
        --tracker run2_frozen_e2 --oracle oracle_score_0.35
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def gap(results: dict, tracker: str, oracle: str, iou_key: str = "iou_0.5") -> dict:
    """Mean and per-clip IDF1 (and HOTA) of the tracker and the oracle, and the gain."""
    block = results[iou_key]
    t, o = block[tracker], block[oracle]
    per_clip = []
    for seq in sorted(t["per_clip"]):
        a, b = t["per_clip"][seq], o["per_clip"][seq]
        per_clip.append({"clip": int(seq), "gt_ids": a["GT_IDs"], "idf1_tracker": a["IDF1"],
                         "idf1_oracle": b["IDF1"], "gain": b["IDF1"] - a["IDF1"],
                         "hota_tracker": a["HOTA"], "hota_oracle": b["HOTA"],
                         "deta": b["DetA"]})
    mt, mo = t["mean_over_clips"], o["mean_over_clips"]
    return {"iou": iou_key, "tracker": tracker, "oracle": oracle,
            "idf1_tracker": mt["IDF1"], "idf1_oracle": mo["IDF1"], "gain": mo["IDF1"] - mt["IDF1"],
            "hota_tracker": mt["HOTA"], "hota_oracle": mo["HOTA"],
            "deta_oracle": mo["DetA"], "assa_tracker": mt["AssA"], "assa_oracle": mo["AssA"],
            "idsw_tracker": t["totals"]["IDSW"], "idsw_oracle": o["totals"]["IDSW"],
            "per_clip": per_clip}


def decide(g: dict, min_gain: float) -> bool:
    """True if the association step should be built under the rule (gain >= ``min_gain``)."""
    return bool(g["gain"] >= min_gain - 1e-12)


def format_report(g: dict, min_gain: float) -> str:
    lines = [f"oracle on test, measurement only ({g['iou']}): tracker {g['tracker']} vs {g['oracle']}",
             "clip  GT  IDF1 tracker  IDF1 oracle   gain   DetA(oracle)"]
    for r in g["per_clip"]:
        lines.append(f"{r['clip']:4d} {r['gt_ids']:4d} {r['idf1_tracker']:12.3f} {r['idf1_oracle']:12.3f}"
                     f" {r['gain']:+7.3f} {r['deta']:11.3f}")
    lines.append(f"mean IDF1: tracker {g['idf1_tracker']:.3f}, oracle {g['idf1_oracle']:.3f}, "
                 f"gain {g['gain']:+.3f}")
    lines.append(f"HOTA {g['hota_tracker']:.3f} -> {g['hota_oracle']:.3f}; AssA {g['assa_tracker']:.3f} "
                 f"-> {g['assa_oracle']:.3f}; DetA of the oracle {g['deta_oracle']:.3f}; ID switches "
                 f"{g['idsw_tracker']} -> {g['idsw_oracle']}")
    verdict = "BUILD" if decide(g, min_gain) else "DO NOT BUILD"
    lines.append(f"rule: DDCF is built only if the gain is >= {min_gain:g} IDF1 -> {verdict}")
    return chr(10).join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--trackeval", type=Path, required=True)
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--min-gain", type=float, default=0.15)
    ap.add_argument("--iou", default="iou_0.5")
    ap.add_argument("--out", type=Path, help="write the per-clip rows as CSV")
    args = ap.parse_args()
    g = gap(json.loads(args.trackeval.read_text()), args.tracker, args.oracle, args.iou)
    print(format_report(g, args.min_gain))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(g["per_clip"][0]))
            w.writeheader()
            for r in g["per_clip"]:
                w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
