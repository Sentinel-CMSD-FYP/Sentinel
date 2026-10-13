"""Build sample ``*_points.json`` / ``*_tracks.json`` from DroneCrowd ground truth.

The samples show the format in ``docs/contracts.md`` (v1.0) for teammates. They are
NOT model output: points come from GT (score fixed at 1.0) and track IDs are the
dataset's person IDs. Each sample file carries a ``_note`` saying so.

Coordinates: x = column, y = row, original 1920x1080 frame pixels.
``frame_id`` in the output is the 0-based index in the clip (dataset frame - 1).

    python eval/make_contract_samples.py --config ml/configs/contract_samples.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import scipy.io as sio
import yaml

NOTE = (
    "SAMPLE built from DroneCrowd ground truth for format illustration only. "
    "Not model output: score is fixed at 1.0, track_id is the dataset person ID, "
    "altitude is assumed (60 m), not measured; fps 25 is the published DroneCrowd value."
)


def load_gt(path: Path) -> dict[int, tuple[float, float]]:
    """Return {person_id: (x, y)} in pixels from one ``GT_img*.mat``."""
    loc = sio.loadmat(str(path))["image_info"][0, 0][0, 0]["location"]
    return {int(r[2]): (float(r[0]), float(r[1])) for r in loc}


def dump(doc: dict) -> str:
    """JSON with indent, but each innermost list (a point or a track row) on one line."""
    text = json.dumps(doc, indent=1)
    def one_line(m: re.Match) -> str:
        return "[" + " ".join(m.group(1).split()) + "]"

    return re.sub(r"\[\s+([^\[\]]+?)\s+\]", one_line, text)


def header(cfg: dict) -> dict:
    return {
        "schema_version": "1.0",
        "clip_id": cfg["clip_id"],
        "frame_size": cfg["frame_size"],
        "fps": cfg["fps"],
        "frame_stride": cfg["frame_stride"],
        "altitude_m": cfg["altitude_m"],
        "altitude_source": "assumed_constant",
        "_note": NOTE,
    }


def pick_ids(frames: list[dict[int, tuple[float, float]]], n_persistent: int) -> list[int]:
    """Lowest ``n_persistent`` IDs present in every frame, plus IDs that appear mid-way."""
    in_all = sorted(set.intersection(*[set(f) for f in frames]))
    late = sorted(set(frames[-1]) - set(frames[0]))
    return in_all[:n_persistent] + late


def validate(points: dict, tracks: dict) -> None:
    """Check both documents against docs/contracts.md. Raises AssertionError."""
    for doc in (points, tracks):
        assert doc["schema_version"] == "1.0"
        assert isinstance(doc["clip_id"], str)
        w, h = doc["frame_size"]
        assert isinstance(doc["fps"], (int, float)) or doc["fps"] is None
        assert isinstance(doc["frame_stride"], int) and doc["frame_stride"] >= 1
        assert isinstance(doc["altitude_m"], (int, float))
        assert doc["altitude_source"] in ("assumed_constant", "telemetry")
        assert all(k.isdigit() for k in doc["frames"])
    for rows in points["frames"].values():
        for x, y, s in rows:
            assert 0 <= x <= points["frame_size"][0] and 0 <= y <= points["frame_size"][1]
            assert 0.0 <= s <= 1.0
    seen: dict[int, str] = {}
    for fid, rows in tracks["frames"].items():
        ids = [int(r[0]) for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate track id in frame {fid}"
        for tid in ids:
            seen.setdefault(tid, fid)
    assert points["frames"].keys() == tracks["frames"].keys()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text())

    gt_dir = Path(cfg["gt_dir"])
    first, n = int(cfg["start_frame"]), int(cfg["n_frames"])
    seq = int(cfg["clip_id"].split("_")[1])
    # contract frame_id is 0-based; dataset file names are 1-based
    frames = [load_gt(gt_dir / f"GT_img{seq:03d}{first + k + 1:03d}.mat") for k in range(n)]
    ids = pick_ids(frames, int(cfg["n_persistent_ids"]))

    points, tracks = header(cfg), header(cfg)
    points["frames"], tracks["frames"] = {}, {}
    for k, gt in enumerate(frames):
        fid = str(first + k)
        present = [i for i in ids if i in gt]
        points["frames"][fid] = [[gt[i][0], gt[i][1], 1.0] for i in present]
        tracks["frames"][fid] = [[i, gt[i][0], gt[i][1], 1.0] for i in present]

    validate(points, tracks)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, doc in (("points", points), ("tracks", tracks)):
        path = out / f"{cfg['clip_id']}_{name}.json"
        path.write_text(dump(doc) + "\n")
        print(f"wrote {path} ({sum(len(v) for v in doc['frames'].values())} rows)")
    print(f"track ids used: {ids}")
    print("ids that appear mid-window:", sorted(set(frames[-1]) - set(frames[0])))


if __name__ == "__main__":
    main()
