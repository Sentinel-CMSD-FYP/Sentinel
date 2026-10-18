"""Camera-motion check over all clips of a split (measurement only, nothing is tuned).

Per clip: optical-flow estimate of the camera translation between consecutive frames (the
authors' method: sparse flow on half-resolution grey frames, similarity transform with RANSAC,
translation in full-resolution pixels) and the ground-truth common-mode motion (median step of
all GT people per frame pair, a camera-motion proxy). Compared with the tracker's 20 px gate.

Units: pixels per frame. Coordinates: x = column, y = row, original frame pixels.

    python eval/camera_motion_check.py --config ml/configs/dronecrowd_data.yaml --split test \
        --frames 300 --out data/outputs/camera_motion/test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import diagnose_tracking as dg  # eval/ is on sys.path when this file is run as a script

from sentinel_ml.localization import data as D

GATE_PX = 20.0  # the tracker's default matching gate (ml/configs/tracker.yaml size_px)


def clip_motion(data_cfg: dict, split: str, clip: int, n_frames: int) -> dict:
    """Flow-based camera translation and GT common-mode motion for one clip."""
    folder = Path(data_cfg["root"]) / (data_cfg["test_dir"] if split == "test"
                                      else data_cfg["train_dir"])
    images = [folder / "images" / f"img{clip:03d}{f + 1:03d}.jpg" for f in range(n_frames)]
    gt = D.clip_gt_frames(data_cfg, split, clip, list(range(n_frames)))
    flow = dg.camera_translation(images)
    motion = dg.gt_motion(gt)
    return {"clip": clip, "flow_median_px": flow["median_px"], "flow_p95_px": flow["p95_px"],
            "flow_max_px": flow["max_px"], "gt_common_mode_median_px": motion["common_mode_median_px"],
            "gt_common_mode_max_px": motion["common_mode_max_px"],
            "gt_step_p95_px": motion["step_p95_px"],
            "share_gt_steps_above_gate": motion["share_steps_above_20px"]}


def summarise(rows: list[dict]) -> dict:
    """Worst-case and spread over clips, and how it compares with the gate."""
    worst = max(rows, key=lambda r: r["flow_max_px"])
    return {"clips": len(rows),
            "flow_max_over_clips_px": worst["flow_max_px"], "flow_max_clip": worst["clip"],
            "flow_median_of_medians_px": float(sorted(r["flow_median_px"] for r in rows)[len(rows) // 2]),
            "gt_common_mode_max_over_clips_px": max(r["gt_common_mode_max_px"] for r in rows),
            "share_gt_steps_above_gate_max_over_clips": max(r["share_gt_steps_above_gate"]
                                                            for r in rows),
            "worst_flow_max_as_share_of_gate": worst["flow_max_px"] / GATE_PX}


def format_table(rows: list[dict], summary: dict) -> str:
    lines = ["clip  flow median  flow p95  flow max | GT common-mode median/max | GT step p95 | steps>20px"]
    for r in sorted(rows, key=lambda r: -r["flow_max_px"]):
        lines.append(f"{r['clip']:4d} {r['flow_median_px']:11.2f} {r['flow_p95_px']:9.2f} "
                     f"{r['flow_max_px']:9.2f} | {r['gt_common_mode_median_px']:11.2f} "
                     f"{r['gt_common_mode_max_px']:8.2f} | {r['gt_step_p95_px']:11.2f} | "
                     f"{100 * r['share_gt_steps_above_gate']:9.3f}%")
    s = summary
    lines.append(f"over {s['clips']} clips: largest flow translation {s['flow_max_over_clips_px']:.2f} px "
                 f"per frame (clip {s['flow_max_clip']}) = {100 * s['worst_flow_max_as_share_of_gate']:.1f}% "
                 f"of the {GATE_PX:g} px gate; largest GT common-mode motion "
                 f"{s['gt_common_mode_max_over_clips_px']:.2f} px")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the dronecrowd_data.yaml")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--frames", type=int, default=300, help="frames per clip to use")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    data_cfg = D.load_config(args.config)
    manifest = json.loads(Path(data_cfg["manifest"]).read_text())
    rows = []
    for clip in manifest[args.split]:
        rows.append(clip_motion(data_cfg, args.split, clip, args.frames))
        print(f"clip {clip}: flow max {rows[-1]['flow_max_px']:.2f} px", flush=True)
    summary = summarise(rows)
    print(format_table(rows, summary))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"split": args.split, "frames": args.frames,
                                        "summary": summary, "per_clip": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
