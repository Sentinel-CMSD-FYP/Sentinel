"""Generate a fake demo clip (video plus all four layers) for building the frontend.

Nothing here comes from a model. A small simulated crowd is drawn into a video, and the
layers are derived from the same simulation, so the overlays line up with what is on
screen: two counter-flowing groups plus a blob converging on one spot.

The flow layer deliberately has a gap (no samples for frames 100-149) to exercise the
"gaps stay gaps" rule in the frontend.

    python docs/layers/make_sample_clip.py data/samples/demo_fake
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate import validate_clip_dir  # noqa: E402

CLIP_ID = "demo_fake"
W, H, FPS, N_FRAMES = 1920, 1080, 25.0, 300
FLOW_GRID = {"rows": 6, "cols": 10}
DENSITY_GRID = {"rows": 9, "cols": 16}
SAMPLE_EVERY = 25  # grid layers at 1 Hz
POINT_EVERY = 5
FLOW_GAP = range(100, 150)


def simulate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Positions ``(N_FRAMES, n, 2)`` and per-frame velocities ``(N_FRAMES, n, 2)``."""
    right = np.column_stack([rng.uniform(0, W, 70), rng.uniform(150, 400, 70)])
    left = np.column_stack([rng.uniform(0, W, 70), rng.uniform(600, 900, 70)])
    blob = rng.normal([1400, 550], 90, (60, 2))
    pos0 = np.vstack([right, left, blob])
    vel = np.vstack([
        rng.normal([1.8, 0.0], 0.3, (70, 2)),
        rng.normal([-1.4, 0.0], 0.3, (70, 2)),
        (np.array([960.0, 540.0]) - blob) / 400.0,  # drift towards the centre
    ])
    pos = np.empty((N_FRAMES, len(pos0), 2))
    vels = np.empty_like(pos)
    p, v = pos0.copy(), vel.copy()
    for t in range(N_FRAMES):
        pos[t], vels[t] = p, v
        p = p + v
        # Bounce off the edges so the crowd stays in view and tracks never jump across it.
        for axis, size in ((0, W), (1, H)):
            out = (p[:, axis] < 0) | (p[:, axis] > size - 1)
            v[out, axis] *= -1
            p[:, axis] = np.clip(p[:, axis], 0, size - 1)
    return pos, vels


def cell_index(p: np.ndarray, grid: dict) -> tuple[np.ndarray, np.ndarray]:
    r = np.clip((p[:, 1] / H * grid["rows"]).astype(int), 0, grid["rows"] - 1)
    c = np.clip((p[:, 0] / W * grid["cols"]).astype(int), 0, grid["cols"] - 1)
    return r, c


def envelope(layer: str, kind: str, max_age: int) -> dict:
    return {"schema_version": 1, "clip_id": CLIP_ID, "layer": layer, "type": kind,
            "frame_size": [W, H], "max_age_frames": max_age}


def flow_layer(pos: np.ndarray, vel: np.ndarray) -> dict:
    g = FLOW_GRID
    doc = envelope("flow", "grid", SAMPLE_EVERY) | {
        "grid": g, "units": "px_per_frame", "fields": ["u", "v", "speed"], "frames": {}}
    for t in range(0, N_FRAMES, SAMPLE_EVERY):
        if t in FLOW_GAP:
            continue
        r, c = cell_index(pos[t], g)
        shape = (g["rows"], g["cols"])
        n, u, v, sp = (np.zeros(shape) for _ in range(4))
        np.add.at(n, (r, c), 1)
        np.add.at(u, (r, c), vel[t][:, 0])
        np.add.at(v, (r, c), vel[t][:, 1])
        np.add.at(sp, (r, c), np.hypot(*vel[t].T))
        ok = n >= 2  # a real model would also be silent in sparsely covered cells
        div = np.where(ok, n, 1)
        doc["frames"][str(t)] = {
            "valid": ok.astype(int).tolist(),
            "u": np.round(u / div, 3).tolist(),
            "v": np.round(v / div, 3).tolist(),
            "speed": np.round(sp / div, 3).tolist(),
        }
    return doc


def density_layer(pos: np.ndarray) -> dict:
    g = DENSITY_GRID
    doc = envelope("density", "grid", SAMPLE_EVERY) | {
        "grid": g, "units": "people_per_cell", "fields": ["density"], "frames": {}}
    for t in range(0, N_FRAMES, SAMPLE_EVERY):
        r, c = cell_index(pos[t], g)
        d = np.zeros((g["rows"], g["cols"]))
        np.add.at(d, (r, c), 1)
        doc["frames"][str(t)] = {"valid": np.ones_like(d, dtype=int).tolist(),
                                 "density": d.tolist()}
    return doc


def points_layer(pos: np.ndarray, rng: np.random.Generator) -> dict:
    doc = envelope("points", "points", POINT_EVERY) | {"frames": {}}
    for t in range(0, N_FRAMES, POINT_EVERY):
        keep = rng.random(len(pos[t])) > 0.05  # the detector misses a few heads
        seen = pos[t][keep]
        scores = rng.uniform(0.5, 1.0, len(seen))
        doc["frames"][str(t)] = [[round(x, 1), round(y, 1), round(float(s), 2)]
                                 for (x, y), s in zip(seen, scores, strict=True)]
    return doc


def tracks_layer(pos: np.ndarray, rng: np.random.Generator) -> dict:
    doc = envelope("tracks", "tracks", POINT_EVERY) | {"altitude_m": 60.0, "frames": {}}
    for t in range(0, N_FRAMES, POINT_EVERY):
        keep = rng.random(len(pos[t])) > 0.02
        doc["frames"][str(t)] = [[int(i), round(x, 1), round(y, 1), round(float(s), 2)]
                                 for i, ((x, y), s) in enumerate(
                                     zip(pos[t], rng.uniform(0.6, 1.0, len(pos[t])), strict=True))
                                 if keep[i]]
    return doc


def render_video(path: Path, pos: np.ndarray) -> None:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    colours = [(60, 60, 200)] * 70 + [(200, 90, 60)] * 70 + [(40, 40, 40)] * 60
    for t in range(N_FRAMES):
        img = np.full((H, W, 3), 150, np.uint8)
        cv2.putText(img, f"FAKE DATA  frame {t}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3)
        for (x, y), col in zip(pos[t], colours, strict=True):
            cv2.circle(img, (int(x), int(y)), 7, col, -1)
        out.write(img)
    out.release()


def find_ffmpeg() -> str | None:
    """ffmpeg on PATH, else the binary bundled with the imageio-ffmpeg package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_browser_playable(path: Path) -> bool:
    """Re-encode to H.264 in place. OpenCV's mp4v is MPEG-4 Part 2, which browsers cannot play."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    tmp = path.with_name("video_h264.mp4")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        check=True,
    )
    tmp.replace(path)
    return True


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/samples/demo_fake")
    (root / "layers").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pos, vel = simulate(rng)

    layers = {
        "flow": flow_layer(pos, vel),
        "density": density_layer(pos),
        "points": points_layer(pos, rng),
        "tracks": tracks_layer(pos, rng),
    }
    for name, doc in layers.items():
        (root / "layers" / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")

    clip = {
        "schema_version": 1, "clip_id": CLIP_ID, "name": "Fake crowd (generated)",
        "fps": FPS, "frame_size": [W, H], "n_frames": N_FRAMES,
        "video_url": f"/api/v1/clips/{CLIP_ID}/video",
        "layers": [
            {"layer": "flow", "type": "grid", "label": "Optical flow", "units": "px_per_frame",
             "default_visible": True, "style": {"render": "arrows", "u": "u", "v": "v"}},
            {"layer": "density", "type": "grid", "label": "Density", "units": "people_per_cell",
             "default_visible": True, "style": {"render": "heatmap", "field": "density"}},
            {"layer": "points", "type": "points", "label": "Detected heads",
             "default_visible": False},
            {"layer": "tracks", "type": "tracks", "label": "Tracks", "default_visible": False},
        ],
    }
    (root / "clip.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
    render_video(root / "video.mp4", pos)
    playable = make_browser_playable(root / "video.mp4")

    print(f"wrote {root}")
    if not playable:
        print("  WARNING: no ffmpeg found, so video.mp4 is MPEG-4 Part 2 and will NOT play in a "
              "browser. Run `pip install imageio-ffmpeg` and generate again.")
    bad = False
    for name, errs in validate_clip_dir(root).items():
        print(f"  {'ok  ' if not errs else 'FAIL'} {name}")
        for e in errs[:5]:
            print(f"       {e}")
        bad = bad or bool(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
