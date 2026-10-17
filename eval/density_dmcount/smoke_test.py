"""DM-Count smoke test: one image in, count printed, heat map overlay saved.

    python eval/density_dmcount/smoke_test.py data/samples/dmcount_example_1.png

Runs on CPU locally and on the Kaggle GPU (see ``kaggle/``). Weights default to
``ml/configs/density_dmcount.yaml``; pass ``--weights`` to override.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from sentinel_ml.density import DMCount, DMCountConfig, overlay

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "ml" / "configs" / "density_dmcount.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--weights", type=Path, help="overrides the config")
    ap.add_argument("--device", help="overrides the config (cpu, cuda, auto)")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "dmcount_smoke")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs after one warmup")
    args = ap.parse_args()

    cfg = DMCountConfig.from_yaml(args.config).to_dict()
    if args.weights:
        cfg["weights"] = str(args.weights)
    elif not Path(cfg["weights"]).is_absolute():
        cfg["weights"] = str(REPO / cfg["weights"])
    if args.device:
        cfg["device"] = args.device
    model = DMCount(DMCountConfig(**cfg))

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"cannot read {args.image}")

    dm = model.predict(frame)  # warmup
    times = []
    for _ in range(args.repeats):
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dm = model.predict(frame)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.image.stem}_overlay.png"
    cv2.imwrite(str(out_path), overlay(frame, dm))

    dev = torch.cuda.get_device_name(0) if model.device.type == "cuda" else "cpu"
    print(f"image    {args.image.name} {frame.shape[1]}x{frame.shape[0]}")
    print(f"weights  {Path(cfg['weights']).name}")
    print(f"device   {dev} (torch {torch.__version__})")
    print(f"density  {dm.density.shape[1]}x{dm.density.shape[0]}")
    print(f"count    {dm.count:.1f}")
    print(f"time     {min(times) * 1000:.0f} ms best of {len(times)}")
    print(f"overlay  {out_path}")


if __name__ == "__main__":
    main()
