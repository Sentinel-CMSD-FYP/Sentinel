"""E-4: render 10 s trail videos for a rule-picked set of test clips, for a human to judge.

Which clips (fixed by this rule, so the choice is not cherry-picked): rank the clips by the
frozen run-2 IDF1 in ``test_per_clip.csv`` (``eval/compare_runs.py``), take the ``--best``
highest, the ``--worst`` lowest, and ``--random`` more drawn with a fixed seed from the clips in
between. Video file names carry only the clip number, so the viewer is not told which group a clip
is in; ``index.csv`` holds the group and the numbers, to be opened after watching.

The qualitative judgement (are the ids visibly stable over 10 s?) is the viewer's, not a metric.
Tracks drawn are the confirmed tracks that the tracker outputs, so a person B never found has no
dot at all; that is a detection gap, not an id problem. Units: pixels, seconds.

    python eval/render_demo.py --config ml/configs/dronecrowd_data.yaml \
        --tracks-dir data/outputs/tracks_final/test \
        --per-clip data/outputs/run_comparison/test_per_clip.csv --out-dir data/outputs/demo
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from render_trails import render_video

from sentinel_ml.localization import data as D


def pick_clips(idf1_by_clip: dict[int, float], best: int, worst: int, n_random: int,
               seed: int) -> dict[int, str]:
    """Group per chosen clip: 'best', 'worst' or 'random'. Ties are broken by clip number."""
    if best + worst + n_random > len(idf1_by_clip):
        raise ValueError("asked for more clips than there are")
    ranked = sorted(idf1_by_clip, key=lambda c: (-idf1_by_clip[c], c))
    top, bottom = ranked[:best], ranked[len(ranked) - worst:] if worst else []
    rest = [c for c in ranked if c not in top and c not in bottom]
    rand = random.Random(seed).sample(rest, n_random)
    groups = {c: "best" for c in top}
    groups.update({c: "worst" for c in bottom})
    groups.update({c: "random" for c in rand})
    return dict(sorted(groups.items()))


def read_per_clip(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the dronecrowd_data.yaml")
    ap.add_argument("--split", choices=D.SPLITS, default="test")
    ap.add_argument("--tracks-dir", type=Path, required=True)
    ap.add_argument("--per-clip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/outputs/demo"))
    ap.add_argument("--best", type=int, default=4)
    ap.add_argument("--worst", type=int, default=4)
    ap.add_argument("--random", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--trail", type=int, default=40, help="trail length in frames")
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()

    rows = {int(r["clip"]): r for r in read_per_clip(args.per_clip)}
    groups = pick_clips({c: float(r["idf1_b"]) for c, r in rows.items()}, args.best, args.worst,
                        args.random, args.seed)
    ds = D.dataset_from_config(D.load_config(args.config), args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for clip, group in groups.items():
        doc = json.loads((args.tracks_dir / f"clip_{clip:03d}_tracks.json").read_text())
        refs = {r.frame_id: r for r in ds.refs if r.clip == clip}
        out = args.out_dir / f"clip_{clip:03d}.mp4"
        n = render_video(doc, refs, 0, args.seconds, args.trail, args.width, out)
        r = rows[clip]
        index.append({"clip": clip, "group": group, "frames": n, "gt_people": r["gt_people"],
                      "tracker_ids": r["ids_b"], "idf1": r["idf1_b"], "hota": r["hota_b"],
                      "id_switches": r["idsw_b"], "count_error_pct": r["err_pct_b"]})
        print(f"wrote {out} ({n} frames)")
    with (args.out_dir / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"wrote {args.out_dir / 'index.csv'} (open it AFTER watching)")


if __name__ == "__main__":
    main()
