"""Step 7: run DM-Count over one DroneCrowd split and write the per frame counts.

    python eval/density_dmcount/predict_split.py --split val  --out runs/zeroshot_qnrf
    python eval/density_dmcount/predict_split.py --split test --out runs/zeroshot_qnrf --overlays

Frames come from ``ml/configs/density_data.yaml`` (manifest + strides), so the CSV holds
exactly the frames ``count_error.py`` expects. Writes ``<out>/<split>/preds.csv`` with
columns ``clip,frame_id,pred,gt`` (pred = sum of the density map, at full 1920x1080 in fp32).
``--overlays`` also saves heat map overlays of a few frames to ``<out>/<split>/overlays/``:
the densest frame, the frame with the median GT count of each density bin, and the middle
frame of each night clip. Weights default to ``ml/configs/density_dmcount.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import yaml
from count_error import bin_of, load_tags
from sentinel_ml.density import DMCount, DMCountConfig, overlay
from sentinel_ml.density.data import load_split_from_config
from sentinel_ml.localization.data import load_gt

REPO = Path(__file__).resolve().parents[2]
DATA_CONFIG = REPO / "ml" / "configs" / "density_data.yaml"
MODEL_CONFIG = REPO / "ml" / "configs" / "density_dmcount.yaml"


def pick_overlay_frames(rows: list[dict], cutoffs: list[int],
                        tags: dict[int, str] | None) -> dict[str, tuple[int, int]]:
    """{label: (clip, frame_id)} of the frames worth a look (see module docstring)."""
    picks = {"densest": max(rows, key=lambda r: r["gt"])}
    for name in ("sparse", "moderate", "dense"):
        in_bin = sorted((r for r in rows if bin_of(r["gt"], cutoffs) == name),
                        key=lambda r: r["gt"])
        if in_bin:
            picks[f"{name}_median"] = in_bin[len(in_bin) // 2]
    for clip in sorted(c for c, t in (tags or {}).items() if t == "night"):
        frames = [r for r in rows if r["clip"] == clip]
        if frames:
            picks[f"night_clip{clip:03d}"] = frames[len(frames) // 2]
    return {k: (r["clip"], r["frame_id"]) for k, r in picks.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--out", type=Path, required=True, help="run folder; writes <out>/<split>/")
    ap.add_argument("--config", type=Path, default=DATA_CONFIG)
    ap.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    ap.add_argument("--weights", type=Path, help="overrides the model config")
    ap.add_argument("--device", help="overrides the model config (cpu, cuda, auto)")
    ap.add_argument("--half", action="store_true", help="fp16 autocast (CUDA only), step 19")
    ap.add_argument("--root", type=Path, help="dataset root, overrides the data config")
    ap.add_argument("--overlays", action="store_true", help="save a few heat map overlays")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    mcfg = DMCountConfig.from_yaml(args.model_config).to_dict()
    if args.weights:
        mcfg["weights"] = str(args.weights)
    elif not Path(mcfg["weights"]).is_absolute():
        mcfg["weights"] = str(REPO / mcfg["weights"])
    if args.device:
        mcfg["device"] = args.device
    if args.half:
        mcfg["half"] = True
    model = DMCount(DMCountConfig(**mcfg))

    ds = load_split_from_config(cfg, args.split, root=args.root, repo=REPO)
    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.split}: {len(ds.refs)} frames, weights {Path(mcfg['weights']).name}, "
          f"device {model.device}, fp16 {model.half}", flush=True)

    rows, paths = [], {}
    t0 = time.perf_counter()
    for i, ref in enumerate(ds.refs, 1):
        frame = cv2.imread(str(ref.image_path))
        if frame is None:
            raise FileNotFoundError(f"cannot read image {ref.image_path}")
        rows.append({"clip": ref.clip, "frame_id": ref.frame_id,
                     "pred": model.predict(frame).count, "gt": len(load_gt(ref.gt_path)[0])})
        paths[(ref.clip, ref.frame_id)] = ref.image_path
        if i % 500 == 0 or i == len(ds.refs):
            dt = time.perf_counter() - t0
            print(f"  {i}/{len(ds.refs)} frames, {dt / i * 1000:.0f} ms per frame", flush=True)

    with (out / "preds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "frame_id", "pred", "gt"])
        w.writeheader()
        w.writerows({**r, "pred": f"{r['pred']:.3f}"} for r in rows)
    print(f"wrote {out / 'preds.csv'}")

    if args.overlays:
        ce = cfg["count_error"]
        tags = load_tags(REPO / ce["lighting_tags"], args.split)
        by_key = {(r["clip"], r["frame_id"]): r for r in rows}
        (out / "overlays").mkdir(exist_ok=True)
        for label, key in pick_overlay_frames(rows, ce["bin_cutoffs"], tags).items():
            frame = cv2.imread(str(paths[key]))
            r = by_key[key]
            path = out / "overlays" / f"{label}_img{key[0]:03d}{key[1] + 1:03d}_gt{r['gt']}.jpg"
            cv2.imwrite(str(path), overlay(frame, model.predict(frame)))
            print(f"overlay {path.name}: pred {r['pred']:.1f}, gt {r['gt']}")


if __name__ == "__main__":
    main()
