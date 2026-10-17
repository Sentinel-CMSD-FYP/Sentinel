"""Rank DroneCrowd sequences by crowd density and head motion from the ground truth.

Reads the per-frame ``GT_img<seq><frame>.mat`` files (rows of ``x, y, id``) and
prints one line per sequence, densest first. Used to pick test clips for flow:
dense and moving beats sparse and static.

    python eval/optical_flow_farneback/dronecrowd_density.py data/DroneCrowd/test_data/ground_truth
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree

FRAME_W, FRAME_H = 1920, 1080
CELL = 120  # px; peak density is heads in the busiest CELL x CELL square


def load_heads(path: Path) -> np.ndarray:
    """``(n, 3)`` array of x, y, id for one frame."""
    info = sio.loadmat(path)["image_info"][0, 0]
    loc = np.asarray(info["location"][0, 0], dtype=np.float64)
    return loc.reshape(-1, 3)


def sequence_stats(files: list[Path]) -> dict[str, float]:
    counts, peaks, nn_dists, speeds = [], [], [], []
    prev: dict[int, tuple[float, float]] = {}

    for f in files:
        heads = load_heads(f)
        counts.append(len(heads))
        if len(heads) == 0:
            prev = {}
            continue

        # Busiest CELL x CELL square: how tightly the crowd packs.
        gx = np.clip((heads[:, 0] // CELL).astype(int), 0, FRAME_W // CELL)
        gy = np.clip((heads[:, 1] // CELL).astype(int), 0, FRAME_H // CELL)
        peaks.append(np.bincount(gx * 1000 + gy).max())

        if len(heads) > 1:
            d, _ = cKDTree(heads[:, :2]).query(heads[:, :2], k=2)
            nn_dists.append(np.median(d[:, 1]))

        # True head displacement between consecutive frames, matched by track id.
        cur = {int(i): (x, y) for x, y, i in heads}
        for i, (x, y) in cur.items():
            if i in prev:
                speeds.append(np.hypot(x - prev[i][0], y - prev[i][1]))
        prev = cur

    sp = np.array(speeds) if speeds else np.zeros(1)
    return {
        "frames": len(files),
        "mean_heads": float(np.mean(counts)),
        "max_heads": float(np.max(counts)),
        "peak_cell": float(np.mean(peaks)) if peaks else 0.0,
        "nn_px": float(np.median(nn_dists)) if nn_dists else float("nan"),
        "speed_mean": float(sp.mean()),
        "speed_p95": float(np.percentile(sp, 95)),
        "moving_frac": float((sp > 0.5).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gt_dir", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    by_seq: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(args.gt_dir.glob("GT_img*.mat")):
        by_seq[f.stem[6:9]].append(f)  # GT_img<seq:3><frame:3>
    if not by_seq:
        raise SystemExit(f"no GT_img*.mat files in {args.gt_dir}")

    rows = []
    for seq, files in by_seq.items():
        rows.append({"seq": seq, **sequence_stats(files)})
    rows.sort(key=lambda r: r["mean_heads"], reverse=True)

    cols = ["seq", "frames", "mean_heads", "max_heads", "peak_cell", "nn_px",
            "speed_mean", "speed_p95", "moving_frac"]
    print("peak_cell = heads in busiest 120x120 px square; nn_px = median nearest-neighbour")
    print("distance; speed in px/frame (1080p); moving_frac = share of head steps > 0.5 px\n")
    print(f"{'seq':>4} {'frames':>6} {'mean':>6} {'max':>5} {'peak':>5} {'nn_px':>6} "
          f"{'spd':>5} {'p95':>5} {'moving':>6}")
    for r in rows:
        print(f"{r['seq']:>4} {r['frames']:>6} {r['mean_heads']:>6.0f} {r['max_heads']:>5.0f} "
              f"{r['peak_cell']:>5.1f} {r['nn_px']:>6.1f} {r['speed_mean']:>5.2f} "
              f"{r['speed_p95']:>5.2f} {r['moving_frac']:>6.2f}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows({k: r[k] for k in cols} for r in rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
