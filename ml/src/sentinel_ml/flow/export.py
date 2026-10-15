"""Turn a video (or a folder of frames) into a clip folder with a real flow layer.

The output follows ``docs/layers/README.md``::

    <out>/clip.json
    <out>/video.mp4          H.264, the exact frames the layer's frame numbers refer to
    <out>/layers/flow.json   grid layer with fields u, v, speed (px per frame)

Every ``every`` frames, flow is computed between that frame and the one ``gap`` frames later,
averaged over a ``rows x cols`` grid, and divided by ``gap`` so the units are pixels per frame.

    python -m sentinel_ml.flow.export clip.mp4 --out data/samples/my_clip
    python -m sentinel_ml.flow.export data/DroneCrowd/test_data/images --seq 011 --fps 25 \\
        --out data/samples/dc_011
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from .camera import remove_camera_motion
from .farneback import FarnebackConfig, FarnebackFlow, pool_regions
from .sources import open_frames


@dataclass(frozen=True)
class ExportConfig:
    rows: int = 6
    cols: int = 10
    every: int | None = None  # frames between samples; None means about one per second
    gap: int = 5  # frames between the two images of each flow pair
    # A cell with less image texture than this has no reliable flow, so it is marked invalid
    # (a gap) and not drawn. Texture is the mean image gradient in grey levels per pixel.
    min_texture: float = 1.5
    max_width: int = 1920  # larger videos are shrunk to this; smaller ones are kept
    start: int = 0
    max_frames: int | None = None
    # Subtract the camera's own shift, zoom and rotation, fitted to the flow (see camera.py).
    # A sample where too little of the frame agreed on one camera motion is left out entirely:
    # a gap, not a guess.
    stabilise: bool = True
    min_inliers: float = 0.5
    # Keep only cells whose centre is in the middle `centre_fraction` of the frame (1.0 = all).
    # Other cells are marked invalid. Useful where the fit is weakest, at the edges of a zoom.
    centre_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be positive")
        if self.gap < 1:
            raise ValueError("gap must be at least 1 frame")
        if self.every is not None and self.every < 1:
            raise ValueError("every must be at least 1 frame")
        if not 0.0 < self.centre_fraction <= 1.0:
            raise ValueError("centre_fraction must be in (0, 1]")
        if not 0.0 <= self.min_inliers <= 1.0:
            raise ValueError("min_inliers must be in [0, 1]")


def centre_mask(rows: int, cols: int, fraction: float) -> np.ndarray:
    """True for cells whose centre lies inside the middle ``fraction`` of the frame."""
    cy = (np.arange(rows) + 0.5) / rows - 0.5
    cx = (np.arange(cols) + 0.5) / cols - 0.5
    half = fraction / 2 + 1e-9
    return (np.abs(cy)[:, None] <= half) & (np.abs(cx)[None, :] <= half)


def cell_texture(gray: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean gradient magnitude per grid cell, in grey levels per pixel."""
    g = gray.astype(np.float32)
    # The Sobel kernel has a gain of 8 on a unit gradient; divide it out.
    mag = (np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))) / 8.0
    h, w = mag.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    out = np.empty((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = mag[ys[r] : ys[r + 1], xs[c] : xs[c + 1]].mean()
    return out


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


def to_h264(path: Path) -> bool:
    """Re-encode in place. Browsers cannot play OpenCV's mp4v (MPEG-4 Part 2)."""
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


def export_flow(
    source: Path,
    out: Path,
    *,
    seq: str | None = None,
    fps: float | None = None,
    clip_id: str | None = None,
    name: str | None = None,
    config: ExportConfig | None = None,
    farneback: FarnebackConfig | None = None,
) -> dict:
    """Write the clip folder and return a summary of what was produced."""
    cfg = config or ExportConfig()
    fcfg = farneback or FarnebackConfig()
    out = Path(out)
    clip_id = clip_id or out.name

    frames, src_fps, (src_w, src_h) = open_frames(source, seq, fps)
    stop = None if cfg.max_frames is None else cfg.start + cfg.max_frames
    frames = islice(frames, cfg.start, stop)

    # H.264 needs even dimensions. Shrink wide videos, never enlarge small ones.
    scale = min(1.0, cfg.max_width / src_w)
    width, height = (max(2, int(round(d * scale)) // 2 * 2) for d in (src_w, src_h))
    resize = (width, height) != (src_w, src_h)

    (out / "layers").mkdir(parents=True, exist_ok=True)
    video = out / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {video}")

    model = FarnebackFlow(fcfg)
    every = cfg.every or max(1, round(src_fps))
    pending: dict[int, np.ndarray] = {}
    samples: dict[str, dict] = {}
    camera: dict[str, dict] = {}
    flow_ms: list[float] = []
    speeds: list[float] = []
    speeds_raw: list[float] = []
    inside = centre_mask(cfg.rows, cfg.cols, cfg.centre_fraction)
    n = 0

    for idx, frame in enumerate(frames):
        if resize:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        n = idx + 1

        if idx % every == 0:
            pending[idx] = frame
        prev = pending.pop(idx - cfg.gap, None)
        if prev is None:
            continue

        t0 = time.perf_counter()
        flow = model.compute(prev, frame)
        flow_ms.append((time.perf_counter() - t0) * 1000)
        k = 1.0 / cfg.gap  # pixels over the gap -> pixels per frame
        at = str(idx - cfg.gap)

        if cfg.stabilise:
            speeds_raw.append(float(flow.magnitude.mean()) * k)
            flow, cam = remove_camera_motion(flow, (width, height))
            reliable = cam.reliable(cfg.min_inliers)
            camera[at] = {**cam.per_second(cfg.gap, src_fps), "used": reliable}
            if not reliable:
                continue  # camera motion is ambiguous here; say nothing for this sample

        pooled = pool_regions(flow, cfg.rows, cfg.cols)
        valid = cell_texture(model.prepare(prev), cfg.rows, cfg.cols) >= cfg.min_texture
        valid &= inside
        samples[at] = {
            "valid": valid.astype(int).tolist(),
            "u": np.round(pooled.mean_u * k, 3).tolist(),
            "v": np.round(pooled.mean_v * k, 3).tolist(),
            "speed": np.round(pooled.mean_speed * k, 3).tolist(),
        }
        speeds.append(float(pooled.mean_speed[valid].mean() * k) if valid.any() else 0.0)

    writer.release()
    if n == 0:
        raise RuntimeError(f"no frames read from {source}")
    playable = to_h264(video)

    layer = {
        "schema_version": 1, "clip_id": clip_id, "layer": "flow", "type": "grid",
        "frame_size": [width, height], "grid": {"rows": cfg.rows, "cols": cfg.cols},
        "units": "px_per_frame", "fields": ["u", "v", "speed"], "max_age_frames": every,
        "frames": samples,
    }
    clip = {
        "schema_version": 1, "clip_id": clip_id, "name": name or clip_id, "fps": src_fps,
        "frame_size": [width, height], "n_frames": n,
        "video_url": f"/api/v1/clips/{clip_id}/video",
        "layers": [{
            "layer": "flow", "type": "grid", "label": "Optical flow", "units": "px_per_frame",
            "default_visible": True, "style": {"render": "arrows", "u": "u", "v": "v"},
        }],
        "produced_by": {
            "model": "farneback", "downscale": fcfg.downscale, "winsize": fcfg.params.winsize,
            "gap_frames": cfg.gap, "every_frames": every, "min_texture": cfg.min_texture,
            "stabilise": cfg.stabilise, "min_inliers": cfg.min_inliers,
            "centre_fraction": cfg.centre_fraction,
        },
    }
    (out / "layers" / "flow.json").write_text(json.dumps(layer), encoding="utf-8")
    (out / "clip.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
    if cfg.stabilise:
        # Not part of the layer contract: what was subtracted at each sample, for inspection.
        (out / "camera.json").write_text(
            json.dumps({"clip_id": clip_id, "gap_frames": cfg.gap, "frames": camera}),
            encoding="utf-8",
        )

    valid_share = [np.array(s["valid"]).mean() for s in samples.values()]
    return {
        "out": str(out), "frames": n, "fps": src_fps, "size": (width, height),
        "samples": len(samples), "every": every, "playable_h264": playable,
        "valid_cell_fraction": float(np.mean(valid_share)) if valid_share else 0.0,
        "mean_speed_px_per_frame": float(np.mean(speeds)) if speeds else 0.0,
        "mean_speed_before_px_per_frame": float(np.mean(speeds_raw)) if speeds_raw else None,
        "camera_samples_dropped": sum(1 for c in camera.values() if not c["used"]),
        "mean_zoom_percent_per_s": (
            float(np.mean([c["zoom_percent_per_s"] for c in camera.values()])) if camera else None
        ),
        "median_flow_ms": float(np.median(flow_ms)) if flow_ms else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path, help="mp4 file, or a folder of numbered images")
    ap.add_argument("--out", type=Path, required=True, help="clip folder to write")
    ap.add_argument("--seq", help="keep only img<seq>* in a frame folder (DroneCrowd)")
    ap.add_argument("--fps", type=float, help="frame rate (needed for frame folders)")
    ap.add_argument("--clip-id", help="default: the output folder's name")
    ap.add_argument("--name", help="label shown in the viewer")
    ap.add_argument("--config", type=Path, default=Path("ml/configs/flow_farneback.yaml"))
    ap.add_argument("--downscale", type=float, default=0.5, help="default 0.5; see the report")
    ap.add_argument("--winsize", type=int, default=9, help="default 9; see the report")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--every", type=int, help="frames between samples (default: about 1 s)")
    ap.add_argument("--gap", type=int, default=5, help="frames between a flow pair")
    ap.add_argument("--min-texture", type=float, default=1.5)
    ap.add_argument("--no-stabilise", dest="stabilise", action="store_false",
                    help="keep camera shift, zoom and rotation in the flow (default: remove them)")
    ap.add_argument("--min-inliers", type=float, default=0.5,
                    help="drop a sample when less of the frame than this agrees on camera motion")
    ap.add_argument("--centre-fraction", type=float, default=1.0,
                    help="only keep cells in the middle fraction of the frame, e.g. 0.6")
    ap.add_argument("--max-width", type=int, default=1920)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args(argv)

    base = FarnebackConfig.from_yaml(a.config) if a.config.is_file() else FarnebackConfig()
    farneback = replace(base, downscale=a.downscale, params=replace(base.params, winsize=a.winsize))
    cfg = ExportConfig(rows=a.rows, cols=a.cols, every=a.every, gap=a.gap,
                       min_texture=a.min_texture, max_width=a.max_width, start=a.start,
                       max_frames=a.max_frames, stabilise=a.stabilise,
                       min_inliers=a.min_inliers, centre_fraction=a.centre_fraction)
    if a.source.is_dir() and a.fps is None:
        print("note: a frame folder has no frame rate; assuming 30 fps (pass --fps to set it)")

    s = export_flow(a.source, a.out, seq=a.seq, fps=a.fps, clip_id=a.clip_id, name=a.name,
                    config=cfg, farneback=farneback)
    print(f"wrote {s['out']}")
    print(f"  {s['frames']} frames at {s['fps']:g} fps, {s['size'][0]}x{s['size'][1]}")
    print(f"  {s['samples']} flow samples (one every {s['every']} frames), "
          f"{s['median_flow_ms']:.0f} ms each")
    print(f"  cells with reliable texture: {s['valid_cell_fraction']:.0%}")
    print(f"  mean speed over valid cells: {s['mean_speed_px_per_frame']:.2f} px/frame")
    before = s["mean_speed_before_px_per_frame"]
    if before is not None:
        print(f"  camera motion removed: whole-frame speed {before:.2f}"
              f" -> {s['mean_speed_px_per_frame']:.2f} px/frame; mean zoom "
              f"{s['mean_zoom_percent_per_s']:+.2f} %/s; "
              f"{s['camera_samples_dropped']} samples dropped as ambiguous")
    if not s["playable_h264"]:
        print("  WARNING: no ffmpeg found, so video.mp4 will NOT play in a browser. "
              "Run `pip install imageio-ffmpeg` and export again.")


if __name__ == "__main__":
    main()
