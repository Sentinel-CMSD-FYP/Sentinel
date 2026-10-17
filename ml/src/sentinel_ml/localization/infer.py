"""Model B inference: DOT + Pixel Distill -> ``<clip_id>_points.json`` (docs/contracts.md v1.0).

For every clip of a split it runs the detector on each frame and writes one file with
all points above ``min_score`` and their scores. No other thresholding is applied; the
operating threshold is chosen later on validation (B-4).

Coordinates: (x, y) = (column, row), original 1920x1080 frame pixels. The network
input is the frame STRETCHED to 1920x1088 (not padded); y is scaled back by 1080/1088,
exactly as the repo's ``pt_pred.py`` does.

    python -m sentinel_ml.localization.infer --config ml/configs/infer_b.yaml --split val
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import sys
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import yaml

from sentinel_ml.localization import data as D

# A detector maps one original RGB frame (H, W, 3 uint8) to (N, 3) float (x, y, score),
# x/y in original pixels and score in [0, 1].
Detector = Callable[[np.ndarray], np.ndarray]

# Types the released DOT checkpoint stores (found by listing its pickle globals).
# Anything else is refused when the checkpoint is loaded.
ALLOWED_PICKLE_GLOBALS = {
    ("collections", "defaultdict"), ("collections", "OrderedDict"),
    ("omegaconf.base", "ContainerMetadata"), ("omegaconf.base", "Metadata"),
    ("omegaconf.dictconfig", "DictConfig"), ("omegaconf.listconfig", "ListConfig"),
    ("omegaconf.nodes", "AnyNode"), ("typing", "Any"),
    ("builtins", "list"), ("builtins", "dict"), ("builtins", "int"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "FloatStorage"), ("torch", "LongStorage"),
}


# ---------- pure helpers (numpy only, tested on CPU in CI) ----------

def preprocess(image_rgb: np.ndarray, net_size: tuple[int, int], mean: list[float],
               std: list[float]) -> np.ndarray:
    """Resize (linear, stretching) to ``net_size`` = (w, h), scale to 0-1, normalise.

    Returns a float32 array of shape (3, h, w), the network input for one frame.
    """
    resized = cv2.resize(image_rgb, tuple(net_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def net_to_original(xy_net: np.ndarray, orig_size: tuple[int, int],
                    net_size: tuple[int, int]) -> np.ndarray:
    """Map (N, 2) (x, y) from network-input pixels to original-frame pixels.

    ``orig_size`` and ``net_size`` are (width, height).
    """
    scale = np.array([orig_size[0] / net_size[0], orig_size[1] / net_size[1]])
    return np.asarray(xy_net, dtype=np.float64).reshape(-1, 2) * scale


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def build_doc(clip_id: str, frame_size: tuple[int, int], frames: dict[int, np.ndarray],
              cfg: dict) -> dict:
    """Points document in the contract format. Rounds x, y to 0.1 px and score to 4 places."""
    return {
        "schema_version": "1.0",
        "clip_id": clip_id,
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "fps": cfg["fps"],
        "frame_stride": int(cfg["frame_stride"]),
        "altitude_m": cfg["altitude_m"],
        "altitude_source": cfg["altitude_source"],
        "frames": {
            str(fid): [[round(float(x), 1), round(float(y), 1), round(float(s), 4)]
                       for x, y, s in np.asarray(pts).reshape(-1, 3)]
            for fid, pts in sorted(frames.items())
        },
    }


def write_doc(doc: dict, path: Path) -> None:
    """Write JSON with one point per line, to keep files readable and small."""
    lines = ["{"]
    head = [f'"{k}": {json.dumps(v)}' for k, v in doc.items() if k != "frames"]
    lines.append(",\n".join(head) + ",")
    lines.append('"frames": {')
    frame_lines = []
    for fid, rows in doc["frames"].items():
        body = ",\n".join(json.dumps(r) for r in rows)
        frame_lines.append(f'"{fid}": [\n{body}\n]' if rows else f'"{fid}": []')
    lines.append(",\n".join(frame_lines))
    lines.append("}\n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def check_doc(doc: dict) -> None:
    """Raise if a points document breaks the contract (scores in [0, 1], points in frame)."""
    w, h = doc["frame_size"]
    for fid, rows in doc["frames"].items():
        if not fid.isdigit():
            raise ValueError(f"frame key '{fid}' is not a digit string")
        for x, y, s in rows:
            if not (0 <= x <= w and 0 <= y <= h and 0.0 <= s <= 1.0):
                raise ValueError(f"frame {fid}: bad row {[x, y, s]} for frame size {w}x{h}")


# ---------- running a split ----------

def run_clip(detector: Detector, refs: list[D.FrameRef], max_frames: int | None = None
             ) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    """Detect on every frame of one clip. Returns ({frame_id: (N, 3)}, (w, h))."""
    out: dict[int, np.ndarray] = {}
    size = (0, 0)
    for ref in refs[:max_frames]:
        bgr = cv2.imread(str(ref.image_path))
        if bgr is None:
            raise FileNotFoundError(f"cannot read image {ref.image_path}")
        size = (bgr.shape[1], bgr.shape[0])
        out[ref.frame_id] = detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return out, size


def run_split(detector: Detector, ds: D.DroneCrowdDataset, out_dir: Path, cfg: dict,
              max_frames: int | None = None, log=print) -> list[Path]:
    """Write one points file per clip. Clips whose file already exists are skipped."""
    by_clip: dict[int, list[D.FrameRef]] = {}
    for ref in ds.refs:
        by_clip.setdefault(ref.clip, []).append(ref)
    written = []
    t_start = time.perf_counter()
    n_frames = 0
    for clip, refs in sorted(by_clip.items()):
        path = out_dir / f"clip_{clip:03d}_points.json"
        if path.exists():
            log(f"skip {path.name} (exists)")
            continue
        t0 = time.perf_counter()
        frames, size = run_clip(detector, refs, max_frames)
        doc = build_doc(f"clip_{clip:03d}", size, frames, cfg)
        check_doc(doc)
        write_doc(doc, path)
        written.append(path)
        n_frames += len(frames)
        log(f"wrote {path.name}: {len(frames)} frames, "
            f"{sum(len(v) for v in frames.values())} points, {time.perf_counter() - t0:.1f}s")
    log(f"done: {len(written)} files, {n_frames} frames, "
        f"{time.perf_counter() - t_start:.1f}s total")
    return written


# ---------- the real model (needs torch + the DOT repo's requirements) ----------

@contextlib.contextmanager
def restricted_torch_load() -> Iterator[None]:
    """Make ``torch.load`` refuse every pickled type outside ``ALLOWED_PICKLE_GLOBALS``.

    Newer PyTorch (>= 2.6) rejects this checkpoint in safe mode, and safe mode cannot be
    widened enough (it cannot restore a ``defaultdict``). A restricted unpickler keeps the
    protection: unknown types raise ``UnpicklingError`` instead of running.
    """
    import torch

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            module = "builtins" if module == "__builtin__" else module
            name = "int" if name == "long" else name
            if (module, name) in ALLOWED_PICKLE_GLOBALS:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"blocked {module}.{name}")

    mod = types.ModuleType("restricted_pickle")
    mod.__dict__.update(pickle.__dict__)
    mod.Unpickler = Restricted
    original = torch.load

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        kwargs["pickle_module"] = mod
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


class DotDetector:
    """The released DOT + Pixel Distill model as a :data:`Detector`.

    Needs ``torch`` and the DOT repo's requirements (pytorch-lightning, hydra-core,
    segmentation-models-pytorch==0.3.4, timm==0.9.7, ...). Runs one frame at a time.
    """

    def __init__(self, cfg: dict, device: str) -> None:
        import torch

        repo = str(Path(cfg["repo_dir"]).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.model.dot_regressor import DotRegressor  # the DOT repo's own package

        self._torch = torch
        self.cfg = cfg
        self.device = torch.device(device)
        with restricted_torch_load():
            model = DotRegressor.load_from_checkpoint(cfg["checkpoint"], map_location="cpu")
        self.model = model.eval().to(self.device)
        self.net_size = tuple(cfg["net_size"])

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = preprocess(image_rgb, self.net_size, self.cfg["data_mean"], self.cfg["data_std"])
        with torch.no_grad():
            y_pred, _, _ = self.model(torch.from_numpy(x).unsqueeze(0).to(self.device))
            # columns: class, x, y (network-input pixels), raw logit
            pts = self.model.postprocessing(y_pred, thresh=float(self.cfg["min_score"]))[0]
        pts = pts.detach().cpu().numpy()
        h, w = image_rgb.shape[:2]
        xy = net_to_original(pts[:, 1:3], (w, h), self.net_size)
        return np.hstack([xy, sigmoid(pts[:, 3])[:, None]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=D.SPLITS, required=True)
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--root", type=Path, help="override the dataset root in the data config")
    ap.add_argument("--manifest", type=Path, help="override the split manifest path")
    ap.add_argument("--checkpoint", type=Path, help="override the checkpoint path")
    ap.add_argument("--repo-dir", type=Path, help="override the DOT repo folder")
    ap.add_argument("--out-dir", type=Path, help="override the output folder")
    ap.add_argument("--max-frames", type=int, help="only the first N frames of each clip")
    ap.add_argument("--max-clips", type=int, help="only the first N clips (for quick tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = D.load_config(Path(cfg["data_config"]))
    for key, val in (("root", args.root), ("manifest", args.manifest)):
        if val is not None:
            data_cfg[key] = str(val)
    for key, val in (("checkpoint", args.checkpoint), ("repo_dir", args.repo_dir)):
        if val is not None:
            cfg[key] = str(val)

    ds = D.dataset_from_config(data_cfg, args.split, frame_stride=int(cfg["frame_stride"]))
    if args.max_clips is not None:
        keep = sorted({r.clip for r in ds.refs})[: args.max_clips]
        ds.refs = [r for r in ds.refs if r.clip in keep]
    out_dir = (args.out_dir or Path(cfg["out_dir"])) / args.split
    print(f"split={args.split} clips={len({r.clip for r in ds.refs})} frames={len(ds)} "
          f"device={args.device} out={out_dir}")
    run_split(DotDetector(cfg, args.device), ds, out_dir, cfg, args.max_frames)


if __name__ == "__main__":
    main()
