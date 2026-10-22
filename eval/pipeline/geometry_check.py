"""Check a clip's geometry in metres: known lengths and walking speeds (step 30).

Uses the clip.json ``camera`` block (altitude, horizontal FOV) and ``sentinel_ml.geometry``
(straight down camera, flat ground, no lens distortion). Two checks:

1. Known lengths: segments marked on one frame (parked cars are about 4.5 m, a pitch line, a
   road marking) are converted from px to m. Each segment also gives the altitude that would
   make it exactly its known length; with a right camera block these sit near the clip's own.
2. Speeds: per track speed of the tracks layer (E) over 1 s windows; people walking should come
   out near 1.2 m/s. The flow layer (D) cell speeds are summarised too, in m/s.

Every speed is also given at the median implied altitude, since metres scale with altitude.

    python eval/pipeline/geometry_check.py data/samples/seq011_60m \\
        --lines eval/pipeline/geometry_lines/seq011.yaml --out data/outputs/geometry_check/seq011_60m

Writes ``result.json``, ``result.md`` and ``lines.jpg`` (the segments drawn with their metres).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from sentinel_ml.geometry import NadirCamera

MOVING_M_S = 0.5  # a track faster than this (median over its 1 s windows) counts as moving
WALKING_M_S = 1.2


def segment_table(segments: list[dict], gsd: float, altitude_m: float, known_m: float) -> list:
    """Per segment: px length, metres at ``gsd``, its known length and the implied altitude."""
    rows = []
    for s in segments:
        px = math.dist(s["p1"], s["p2"])
        m = px * gsd
        known = float(s.get("known_m", known_m))
        rows.append({"label": s["label"], "px": round(px, 1), "m": round(m, 2),
                     "known_m": known, "ratio": round(m / known, 3),
                     "implied_altitude_m": round(altitude_m * known / m, 1)})
    return rows


def track_speeds(frames: dict, fps: float, gsd: float, window_s: float = 1.0) -> list[float]:
    """Median speed (m/s) of each track over ``window_s`` windows; tracks shorter are left out.

    ``frames`` is a tracks layer's ``frames``: frame id (str) to rows ``[id, x, y, score]``.
    """
    w = max(1, round(window_s * fps))
    pos: dict[int, dict[int, tuple[float, float]]] = {}
    for f, rows in frames.items():
        for tid, x, y, *_ in rows:
            pos.setdefault(int(tid), {})[int(f)] = (x, y)
    out = []
    for track in pos.values():
        v = [math.dist(p, track[f + w]) * gsd * fps / w
             for f, p in track.items() if f + w in track]
        if v:
            out.append(float(np.median(v)))
    return out


def flow_speeds(doc: dict, to_m_s: float) -> np.ndarray:
    """Every valid cell speed of a flow layer in m/s (``to_m_s`` = 1 if already in m/s)."""
    vals = [np.asarray(s["speed"], float)[np.asarray(s["valid"], bool)]
            for s in doc["frames"].values()]
    return np.concatenate(vals) * to_m_s if vals else np.zeros(0)


def stats(v) -> dict:
    v = np.asarray(v, float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": round(float(np.median(v)), 3),
            "p90": round(float(np.percentile(v, 90)), 3), "max": round(float(v.max()), 3)}


def read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    try:
        for _ in range(index + 1):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"{video} has no frame {index}")
    finally:
        cap.release()
    return frame


def draw(frame: np.ndarray, rows: list, segments: list) -> np.ndarray:
    out = frame.copy()
    for s, r in zip(segments, rows, strict=True):
        p1, p2 = (tuple(round(c) for c in s[k]) for k in ("p1", "p2"))
        cv2.line(out, p1, p2, (0, 255, 255), 2)
        for p in (p1, p2):
            cv2.circle(out, p, 3, (0, 0, 255), -1)
        cv2.putText(out, f"{r['label']}: {r['m']:.2f} m", (p1[0], p1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def check(clip_dir: Path, lines: dict) -> tuple[dict, list]:
    clip = json.loads((clip_dir / "clip.json").read_text(encoding="utf-8"))
    cam = clip.get("camera")
    if not cam:
        raise SystemExit(f"{clip_dir} has no clip.json camera block (ingest it with --altitude)")
    w = clip["frame_size"][0]
    gsd = NadirCamera(cam["altitude_m"], cam["hfov_deg"], w).gsd
    rows = segment_table(lines["segments"], gsd, cam["altitude_m"], lines.get("known_m", 4.5))
    implied = float(np.median([r["implied_altitude_m"] for r in rows])) if rows else None
    rescale = implied / cam["altitude_m"] if implied else 1.0

    result = {"clip_id": clip["clip_id"], "camera": cam, "frame_width_px": w,
              "gsd_m_per_px": round(gsd, 5), "segments": rows,
              "median_implied_altitude_m": implied}
    layers = clip_dir / "layers"
    if (layers / "tracks.json").is_file():
        doc = json.loads((layers / "tracks.json").read_text(encoding="utf-8"))
        v = np.array(track_speeds(doc["frames"], clip["fps"], gsd))
        moving = v[v >= MOVING_M_S]
        result["tracks"] = {"all": stats(v), "moving": stats(moving),
                            "moving_share": round(moving.size / v.size, 3) if v.size else None,
                            "moving_at_implied_altitude": stats(moving * rescale)}
    if (layers / "flow.json").is_file():
        doc = json.loads((layers / "flow.json").read_text(encoding="utf-8"))
        to = 1.0 if doc.get("units") == "m_per_s" else gsd * clip["fps"]
        f = flow_speeds(doc, to)
        result["flow"] = {"units_in_layer": doc.get("units"), "cells": stats(f),
                          "cells_at_implied_altitude": stats(f * rescale)}
    return result, rows


def markdown(r: dict) -> str:
    cam = r["camera"]
    camera = (f"Camera: {cam['altitude_m']:g} m ({cam['altitude_source']}), horizontal FOV "
              f"{cam['hfov_deg']:g} degrees, {cam.get('model', 'model unknown')}. "
              f"GSD {r['gsd_m_per_px'] * 100:.2f} cm per px at {r['frame_width_px']} px wide.")
    out = [f"# Geometry check: {r['clip_id']}", "", camera, "",
           "| segment | px | m | known m | ratio | implied altitude m |",
           "|---|---|---|---|---|---|"]
    out += [f"| {s['label']} | {s['px']} | {s['m']} | {s['known_m']:g} | {s['ratio']} | "
            f"{s['implied_altitude_m']} |" for s in r["segments"]]
    out += ["", f"Median implied altitude: {r['median_implied_altitude_m']} m.", ""]

    def row(name, s):
        if not s.get("n"):
            return f"| {name} | 0 | | | |"
        return f"| {name} | {s['n']} | {s['median']} | {s['p90']} | {s['max']} |"

    out += [f"Speeds in m/s (walking is about {WALKING_M_S} m/s):", "",
            "| what | n | median | p90 | max |", "|---|---|---|---|---|"]
    if "tracks" in r:
        t = r["tracks"]
        out += [row("tracks, all (per track median over 1 s)", t["all"]),
                row(f"tracks, moving (>= {MOVING_M_S} m/s)", t["moving"]),
                row("tracks, moving, at implied altitude", t["moving_at_implied_altitude"])]
    if "flow" in r:
        out += [row("flow cells", r["flow"]["cells"]),
                row("flow cells, at implied altitude", r["flow"]["cells_at_implied_altitude"])]
    if "tracks" in r and r["tracks"]["moving_share"] is not None:
        out += ["", f"Moving share of tracks: {r['tracks']['moving_share']:.0%}."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--lines", type=Path, required=True,
                    help="YAML: frame, known_m, segments [{label, p1: [x, y], p2: [x, y]}]")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    lines = yaml.safe_load(a.lines.read_text(encoding="utf-8"))
    result, rows = check(a.clip_dir, lines)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (a.out / "result.md").write_text(markdown(result), encoding="utf-8")
    frame = read_frame(a.clip_dir / "video.mp4", int(lines.get("frame", 0)))
    cv2.imwrite(str(a.out / "lines.jpg"), draw(frame, rows, lines["segments"]))
    print(markdown(result))


if __name__ == "__main__":
    main()
