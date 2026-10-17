"""Read ``<clip_id>_points.json`` (Model B), write ``<clip_id>_tracks.json`` (contracts.md v1.0).

The header (clip id, frame size, fps, frame_stride, altitude) is copied from the points file.
Frames are processed in numeric order of their frame_id; frames with no points still appear in
the output with an empty list.

Coordinates: (x, y) = (column, row), original frame pixels.

    python -m sentinel_ml.tracking.run_tracker --config ml/configs/tracker.yaml --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from sentinel_ml.tracking.altitude import altitude_from_size_tags, load_size_tags
from sentinel_ml.tracking.tracker import PointTracker, TrackerConfig, resolve_config

HEADER_KEYS = ("schema_version", "clip_id", "frame_size", "fps", "frame_stride", "altitude_m",
               "altitude_source")


def track_clip(points_doc: dict, cfg: TrackerConfig, altitude_m: float | None = None) -> dict:
    """Track one clip. Returns a tracks document in the contract format.

    ``cfg`` may use seconds and altitude; they are resolved with the clip's own header
    (fps, frame_stride, altitude_m). ``altitude_m`` overrides the header altitude, for example
    with the DroneCrowd per-clip proxy when benchmarking.
    """
    w, h = points_doc["frame_size"]
    alt_used = altitude_m if altitude_m is not None else points_doc.get("altitude_m")
    scaled_gate = cfg.use_altitude
    cfg = resolve_config(cfg, points_doc.get("fps"), int(points_doc.get("frame_stride", 1)),
                         alt_used)
    tracker = PointTracker((w, h), cfg)
    frames: dict[str, list[list[float]]] = {}
    for fid in sorted(points_doc["frames"], key=int):
        pts = np.asarray(points_doc["frames"][fid], dtype=float).reshape(-1, 3)
        frames[fid] = [[tid, round(x, 1), round(y, 1), round(s, 4)]
                       for tid, x, y, s in tracker.update(pts)]
    doc = {k: points_doc[k] for k in HEADER_KEYS}
    if scaled_gate:  # extra key (readers ignore unknown keys): the altitude the gate really used
        doc["altitude_used_m"] = alt_used
    doc["frames"] = frames
    return doc


def check_tracks_doc(doc: dict) -> None:
    """Raise ValueError if a tracks document breaks docs/contracts.md."""
    w, h = doc["frame_size"]
    for key in HEADER_KEYS:
        if key not in doc:
            raise ValueError(f"missing header field '{key}'")
    for fid, rows in doc["frames"].items():
        if not fid.isdigit():
            raise ValueError(f"frame key '{fid}' is not a digit string")
        ids = [r[0] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"frame {fid}: duplicate track id")
        for tid, x, y, s in rows:
            if not isinstance(tid, int) or not (0 <= x <= w and 0 <= y <= h and 0 <= s <= 1):
                raise ValueError(f"frame {fid}: bad row {[tid, x, y, s]}")


def write_tracks(doc: dict, path: Path) -> None:
    """Write JSON with one track row per line (keeps files readable and small)."""
    head = ",\n".join(f'"{k}": {json.dumps(v)}' for k, v in doc.items() if k != "frames")
    frame_lines = []
    for fid, rows in doc["frames"].items():
        body = ",\n".join(json.dumps(r) for r in rows)
        frame_lines.append(f'"{fid}": [\n{body}\n]' if rows else f'"{fid}": []')
    text = "{\n" + head + ',\n"frames": {\n' + ",\n".join(frame_lines) + "\n}\n}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--clips", type=int, nargs="+", help="only these clip numbers")
    ap.add_argument("--points-dir", type=Path, help="override <points_dir>/<split>")
    ap.add_argument("--out-dir", type=Path, help="override <out_dir>/<split>")
    ap.add_argument("--altitude-tags", type=Path,
                    help="object-size tags yaml: use the DroneCrowd per-clip altitude proxy "
                         "(50 m large, 100 m small) instead of the header altitude")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tracker_cfg = TrackerConfig(**cfg["tracker"])
    points_dir = args.points_dir or Path(cfg["points_dir"]) / args.split
    out_dir = args.out_dir or Path(cfg["out_dir"]) / args.split
    files = sorted(points_dir.glob("clip_*_points.json"))
    if args.clips:
        files = [f for f in files if int(f.name.split("_")[1]) in set(args.clips)]
    if not files:
        raise SystemExit(f"no points files in {points_dir}")
    altitude = None
    if args.altitude_tags:
        tags = load_size_tags(args.altitude_tags, args.split)
        if tags is None:
            raise SystemExit(f"no complete object-size tags for '{args.split}' in "
                             f"{args.altitude_tags}")
        altitude = altitude_from_size_tags(tags)
    for f in files:
        clip = int(f.name.split("_")[1])
        doc = track_clip(json.loads(f.read_text()), tracker_cfg,
                         altitude.get(clip) if altitude else None)
        check_tracks_doc(doc)
        out = out_dir / f.name.replace("_points.json", "_tracks.json")
        write_tracks(doc, out)
        ids = {r[0] for rows in doc["frames"].values() for r in rows}
        print(f"wrote {out.name}: {len(doc['frames'])} frames, {len(ids)} track ids")


if __name__ == "__main__":
    main()
