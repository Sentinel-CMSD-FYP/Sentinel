"""Compute per-frame camera shifts for the clips of a split with a flow model.

For each clip it runs the flow provider (default: Mahid's Farneback with his default config,
``ml/configs/flow_farneback.yaml``, downscale 0.25) between consecutive frames and keeps the
median flow as the camera shift (original-frame pixels, previous frame to current frame). The
result is one ``clip_XXX_shifts.json`` per clip, read by ``tracker_experiments.py --step 5``.

The val clips come from the train folder, whose images live on the USB drive: plug it in first.

Units: pixels. Coordinates: x = column, y = row, original frame pixels.

    python eval/compute_camera_shifts.py --data-config ml/configs/dronecrowd_data.yaml --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel_ml.localization import data as D
from sentinel_ml.tracking.cmc import FarnebackProvider, camera_shifts_for_clip, shifts_to_doc


def image_paths_for_clip(data_cfg: dict, split: str, clip: int,
                         frame_ids: list[int]) -> dict[int, Path]:
    """``{frame_id: image path}`` for a clip. Raises if the first image is not reachable."""
    folder = Path(data_cfg["root"]) / (data_cfg["test_dir"] if split == "test"
                                      else data_cfg["train_dir"]) / "images"
    paths = {f: folder / f"img{clip:03d}{f + 1:03d}.jpg" for f in frame_ids}
    first = paths[min(frame_ids)]
    if not first.exists():
        raise SystemExit(f"cannot find {first}. Val and train images are on the USB drive: is it "
                         "plugged in (and does the junction point at it)?")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-config", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--points-dir", type=Path, help="default data/outputs/points/<split>")
    ap.add_argument("--flow-config", type=Path, default=Path("ml/configs/flow_farneback.yaml"))
    ap.add_argument("--clips", type=int, nargs="+", help="only these clips")
    ap.add_argument("--out-dir", type=Path, help="default data/outputs/camera_shifts/<split>")
    args = ap.parse_args()

    from sentinel_ml.flow.farneback import FarnebackConfig

    data_cfg = D.load_config(args.data_config)
    points_dir = args.points_dir or Path("data/outputs/points") / args.split
    out_dir = args.out_dir or Path("data/outputs/camera_shifts") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = FarnebackProvider(FarnebackConfig.from_yaml(args.flow_config))
    files = sorted(points_dir.glob("clip_*_points.json"))
    if args.clips:
        files = [f for f in files if int(f.name.split("_")[1]) in set(args.clips)]
    for f in files:
        clip = int(f.name.split("_")[1])
        frame_ids = sorted(int(k) for k in json.loads(f.read_text())["frames"])
        shifts = camera_shifts_for_clip(image_paths_for_clip(data_cfg, args.split, clip,
                                                             frame_ids), provider)
        doc = shifts_to_doc(f"clip_{clip:03d}", shifts, provider.downscale)
        (out_dir / f"clip_{clip:03d}_shifts.json").write_text(json.dumps(doc) + chr(10))
        mags = sorted((s.dx ** 2 + s.dy ** 2) ** 0.5 for s in shifts.values())
        print(f"clip {clip}: {len(shifts)} shifts, median {mags[len(mags) // 2]:.3f} px, "
              f"max {mags[-1]:.3f} px", flush=True)


if __name__ == "__main__":
    main()
