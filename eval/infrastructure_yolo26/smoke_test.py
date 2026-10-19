"""YOLO26 smoke test: one image in, detections printed, annotated image saved.

    python eval/infrastructure_yolo26/smoke_test.py data/samples/dronecrowd_img100001.jpg

Runs the pretrained COCO segmentation weights from ``ml/configs/infrastructure_yolo26.yaml``
on CPU locally and on the Kaggle GPU (see ``kaggle/``). It only checks the install;
the COCO classes are not our infrastructure classes.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "ml" / "configs" / "infrastructure_yolo26.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--weights", type=Path, help="overrides the config")
    ap.add_argument("--device", help="overrides the config (cpu, cuda, auto)")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "yolo26_smoke")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs after one warmup")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    weights = args.weights or Path(cfg["weights"])
    if not weights.is_absolute():
        weights = REPO / weights
    weights.parent.mkdir(parents=True, exist_ok=True)
    device = args.device or cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"cannot read {args.image}")

    model = YOLO(str(weights))  # Ultralytics downloads known asset names to this path
    kw = {"imgsz": cfg["imgsz"], "conf": cfg["conf"], "device": device, "verbose": False}
    res = model.predict(frame, **kw)[0]  # warmup
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        res = model.predict(frame, **kw)[0]  # returns CPU results, so the GPU is synced
        times.append(time.perf_counter() - t0)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.image.stem}_yolo26.jpg"
    cv2.imwrite(str(out_path), res.plot())

    names = Counter(res.names[int(c)] for c in res.boxes.cls)
    h, w = frame.shape[:2]
    print(f"image:     {args.image} ({w}x{h})")
    masks = 0 if res.masks is None else len(res.masks)
    print(f"weights:   {weights.name} (task {model.task}, {masks} masks)")
    print(f"device:    {device}")
    print(f"found:     {len(res.boxes)} objects {dict(names.most_common())}")
    print(f"time:      {min(times) * 1000:.0f} ms (best of {args.repeats}, imgsz {cfg['imgsz']})")
    print(f"annotated: {out_path}")


if __name__ == "__main__":
    main()
