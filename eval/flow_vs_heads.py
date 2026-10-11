"""Compare Farnebäck flow with DroneCrowd head motion at the annotated head positions.

For each frame pair ``(t, t + gap)`` the true displacement of a head is the change in
its annotated position (matched by track id); the predicted displacement is the mean
flow in a small window around the head. All displacements are original-frame pixels
over the whole gap. Zero flow is the baseline to beat: DroneCrowd heads barely move
(about 0.5 px/frame at 1080p), so a model can look fine while predicting nothing.

    python eval/flow_vs_heads.py data/DroneCrowd/test_data --seq 023 \\
        --gap 1 5 10 --downscale 0.25 0.5 1.0
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from dronecrowd_density import load_heads
from scipy.spatial import cKDTree

from sentinel_ml.flow.farneback import FarnebackConfig, FarnebackFlow, FlowField

BG_MIN_DIST = 40  # px from any annotated head to count as background
BG_POINTS = 300


def window_mean(field: np.ndarray, x: float, y: float, scale: float, radius: float) -> float:
    """Mean of ``field`` over a square of ``radius`` original px around (x, y)."""
    r = max(0, round(radius * scale))
    c, row = round(x * scale), round(y * scale)
    h, w = field.shape
    c, row = min(max(c, 0), w - 1), min(max(row, 0), h - 1)
    return float(field[max(0, row - r) : row + r + 1, max(0, c - r) : c + r + 1].mean())


def sample(flow: FlowField, pts: np.ndarray, scale: float, radius: float) -> np.ndarray:
    return np.array(
        [[window_mean(f, x, y, scale, radius) for f in (flow.u, flow.v)] for x, y in pts]
    ).reshape(-1, 2)


def evaluate(
    root: Path, seq: str, cfg: FarnebackConfig, gap: int, starts: list[int], radius: float,
    move_thresh: float, rng: np.random.Generator,
) -> dict[str, float]:
    model = FarnebackFlow(cfg)

    @lru_cache(maxsize=8)
    def heads(fr: int) -> np.ndarray:
        return load_heads(root / "ground_truth" / f"GT_img{seq}{fr:03d}.mat")

    true_d, pred_d, bg, ms = [], [], [], []
    for t in starts:
        a = cv2.imread(str(root / "images" / f"img{seq}{t:03d}.jpg"))
        b = cv2.imread(str(root / "images" / f"img{seq}{t + gap:03d}.jpg"))
        t0 = time.perf_counter()
        flow = model.compute(a, b)
        ms.append((time.perf_counter() - t0) * 1000)

        h1, h2 = heads(t), heads(t + gap)
        pos2 = {int(i): (x, y) for x, y, i in h2}
        keep = [(x, y, *pos2[int(i)]) for x, y, i in h1 if int(i) in pos2]
        if keep:
            k = np.array(keep)
            true_d.append(k[:, 2:4] - k[:, 0:2])
            pred_d.append(sample(flow, k[:, 0:2], cfg.downscale, radius))

        # Noise floor: flow far from every annotated head, where nothing should move.
        h, w = a.shape[:2]
        cand = rng.uniform((0, 0), (w, h), size=(BG_POINTS * 3, 2))
        if len(h1):
            far = cKDTree(h1[:, :2]).query(cand)[0] > BG_MIN_DIST
            cand = cand[far]
        bg.append(np.hypot(*sample(flow, cand[:BG_POINTS], cfg.downscale, radius).T))

    td, pd = np.vstack(true_d), np.vstack(pred_d)
    tmag, err = np.hypot(*td.T), np.hypot(*(pd - td).T)
    mov = tmag >= move_thresh
    out = {
        "n": len(td),
        "true_mean": tmag.mean(),
        "epe": err.mean(),
        "n_mov": int(mov.sum()),
        "epe_mov": err[mov].mean() if mov.any() else np.nan,
        "zero_epe_mov": tmag[mov].mean() if mov.any() else np.nan,
        "bg_med": float(np.median(np.concatenate(bg))),
        "ms": float(np.median(ms)),
    }
    if mov.sum() > 2:
        t, p = td[mov], pd[mov]
        out["slope"] = float((t * p).sum() / (t * t).sum())  # 1 = right size, <1 = too small
        ang = np.degrees(np.abs(np.arctan2(t[:, 0] * p[:, 1] - t[:, 1] * p[:, 0],
                                           (t * p).sum(1))))
        out["ang_med"] = float(np.median(ang))
        out["corr"] = float(np.corrcoef(t.ravel(), p.ravel())[0, 1])
    else:
        out.update(slope=np.nan, ang_med=np.nan, corr=np.nan)
    return {k: float(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, help="folder containing images/ and ground_truth/")
    ap.add_argument("--seq", required=True, help="sequence id, e.g. 023")
    ap.add_argument("--config", type=Path, default=Path("ml/configs/flow_farneback.yaml"))
    ap.add_argument("--gap", type=int, nargs="+", default=[1, 5, 10], help="frames between pair")
    ap.add_argument("--downscale", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--winsize", type=int, nargs="+", default=None,
                    help="Farnebäck window sizes to compare (default: the config's)")
    ap.add_argument("--pairs", type=int, default=20, help="frame pairs sampled per setting")
    ap.add_argument("--radius", type=float, default=6.0, help="window radius in original px")
    ap.add_argument("--move-thresh", type=float, default=2.0,
                    help="true displacement (px over the gap) that counts as moving")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    base = FarnebackConfig.from_yaml(args.config)
    n_frames = len(list((args.root / "images").glob(f"img{args.seq}*.jpg")))
    if n_frames == 0:
        raise SystemExit(f"no frames for sequence {args.seq} in {args.root / 'images'}")

    print(f"seq {args.seq}: {n_frames} frames, {args.pairs} pairs per setting, "
          f"window radius {args.radius:g} px, moving = true disp >= {args.move_thresh:g} px")
    print("displacements are original px over the gap; zero = error of predicting no motion\n")
    cols = ["downscale", "winsize", "gap", "n", "n_mov", "true_mean", "epe", "epe_mov",
            "zero_epe_mov", "slope", "corr", "ang_med", "bg_med", "ms"]
    print(f"{'scale':>5} {'win':>3} {'gap':>3} {'heads':>6} {'moving':>6} {'true':>5} "
          f"{'epe':>5} {'epe_m':>5} {'zero_m':>6} {'slope':>5} {'corr':>5} {'ang°':>5} "
          f"{'bg':>5} {'ms':>6}")

    rows = []
    wins = args.winsize or [base.params.winsize]
    for gap in args.gap:
        starts = [int(s) for s in np.linspace(1, n_frames - gap, args.pairs)]
        for s, win in ((s, w) for s in args.downscale for w in wins):
            cfg = replace(base, downscale=s, params=replace(base.params, winsize=win))
            r = evaluate(args.root, args.seq, cfg, gap, starts, args.radius,
                         args.move_thresh, np.random.default_rng(0))
            r = {"downscale": s, "winsize": win, "gap": gap, **r}
            rows.append(r)
            print(f"{s:>5g} {win:>3} {gap:>3} {r['n']:>6.0f} {r['n_mov']:>6.0f} "
                  f"{r['true_mean']:>5.2f} "
                  f"{r['epe']:>5.2f} {r['epe_mov']:>5.2f} {r['zero_epe_mov']:>6.2f} "
                  f"{r['slope']:>5.2f} {r['corr']:>5.2f} {r['ang_med']:>5.0f} "
                  f"{r['bg_med']:>5.2f} {r['ms']:>6.0f}")

    print("\nepe_m < zero_m means flow beats predicting no motion; slope < 1 means flow is too")
    print("small; bg is median |flow| away from heads (noise floor, includes cars and shadows).")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows({k: r[k] for k in cols} for r in rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
