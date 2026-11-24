"""Step 9: draw the training curve (train loss, train MAE, val MAE per epoch) from history.csv.

    python eval/density_dmcount/plot_history.py runs/kaggle/dmcount_train/v7/run/history.csv

Saves ``curve.png`` next to the CSV (or at ``--out``). Drawn with OpenCV, so no plotting
package is needed. The best val MAE epoch is marked.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

W, H, PAD = 560, 320, 50
PANELS = [("train loss", ["loss"]), ("MAE (count)", ["train_mae", "val_mae"])]
COLORS = {"loss": (180, 90, 30), "train_mae": (180, 90, 30), "val_mae": (40, 40, 220)}


def panel(title: str, epochs: list[int], series: dict[str, list[float]], best: int) -> np.ndarray:
    img = np.full((H, W, 3), 255, np.uint8)
    lo, hi = 0.0, max(max(v) for v in series.values()) * 1.05
    e0, e1 = epochs[0], max(epochs[-1], epochs[0] + 1)

    def xy(e: float, v: float) -> tuple[int, int]:
        return (int(PAD + (e - e0) / (e1 - e0) * (W - 2 * PAD)),
                int(H - PAD - (v - lo) / (hi - lo) * (H - 2 * PAD)))

    cv2.rectangle(img, xy(e0, hi), xy(e1, lo), (0, 0, 0), 1)
    for frac in (0, 0.5, 1):
        v = lo + frac * (hi - lo)
        cv2.putText(img, f"{v:.1f}", (2, xy(e0, v)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1)
    for e in (e0, e1):
        cv2.putText(img, str(e), (xy(e, lo)[0] - 6, H - PAD + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1)
    cv2.putText(img, f"{title} vs epoch", (PAD, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for i, (name, vals) in enumerate(series.items()):
        pts = np.array([xy(e, v) for e, v in zip(epochs, vals)], np.int32)
        cv2.polylines(img, [pts], False, COLORS[name], 2)
        for p in pts:
            cv2.circle(img, tuple(int(c) for c in p), 3, COLORS[name], -1)
        cv2.putText(img, name, (W - PAD - 90, 30 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLORS[name], 1)
    if "val_mae" in series:
        bv = series["val_mae"][epochs.index(best)]
        x, y = xy(best, bv)
        cv2.circle(img, (x, y), 7, (0, 160, 0), 2)
        cv2.putText(img, f"best {bv:.2f} (epoch {best})", (min(max(x - 80, PAD), W - PAD - 170),
                    y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("history", type=Path)
    ap.add_argument("--out", type=Path, help="default: curve.png next to the CSV")
    args = ap.parse_args()
    rows = list(csv.DictReader(args.history.open()))
    epochs = [int(r["epoch"]) for r in rows]
    best = min(rows, key=lambda r: float(r["val_mae"]))
    img = np.vstack([panel(t, epochs, {k: [float(r[k]) for r in rows] for k in keys},
                           int(best["epoch"])) for t, keys in PANELS])
    out = args.out or args.history.with_name("curve.png")
    cv2.imwrite(str(out), img)
    print(f"{len(rows)} epochs, best val MAE {float(best['val_mae']):.2f} at epoch {best['epoch']}, "
          f"saved {out}")


if __name__ == "__main__":
    main()
