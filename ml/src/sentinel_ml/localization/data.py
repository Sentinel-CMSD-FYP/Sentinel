"""DroneCrowd data loader for Model B (and later E).

Splits used by this project (docs/project-log.md):
  * ``test``  = the official test clips (30). Used only for final numbers.
  * ``val``   = ``n_val`` clips held out from the official train clips, chosen with a
                fixed seed and frozen in a manifest. The dataset's own ``val_data``
                folder is NOT used: it is a copy of 12 frames per test clip.
  * ``train`` = the remaining official train clips.

Coordinates: points are (x, y) = (column, row) in ORIGINAL frame pixels, always, even
when the image is resized. ``frame_id`` is the 0-based index in the clip (dataset file
number minus 1), matching docs/contracts.md. Images are RGB uint8, shape (H, W, 3).

    python -m sentinel_ml.localization.data --config ml/configs/dronecrowd_data.yaml make-split
    python -m sentinel_ml.localization.data --config ml/configs/dronecrowd_data.yaml draw
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import yaml

FRAME_RE = re.compile(r"^img(\d{3})(\d{3})\.jpg$")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class FrameRef:
    """Where one frame lives. ``frame_id`` is 0-based; ``clip_id`` like ``clip_011``."""

    clip: int
    frame_id: int
    image_path: Path
    gt_path: Path

    @property
    def clip_id(self) -> str:
        return f"clip_{self.clip:03d}"


@dataclass(frozen=True)
class Sample:
    """One frame. ``points`` is (N, 2) float32 (x, y) in original pixels.

    ``scale`` is (sx, sy) = resized size / original size, (1.0, 1.0) if not resized.
    Divide model output by ``scale`` to get back to original pixels.
    """

    image: np.ndarray
    points: np.ndarray
    person_ids: np.ndarray
    scale: tuple[float, float]
    ref: FrameRef


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one ``GT_img*.mat``. Returns ((N, 2) float32 x, y in pixels, (N,) int32 ids)."""
    info = sio.loadmat(str(path))["image_info"][0, 0][0, 0]
    loc = np.asarray(info["location"], dtype=np.float64).reshape(-1, 3)
    return loc[:, :2].astype(np.float32), loc[:, 2].astype(np.int32)


def to_original_pixels(points: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    """Map (N, 2) (x, y) from resized-image pixels back to original-frame pixels."""
    return np.asarray(points, dtype=np.float64) / np.array([scale[0], scale[1]])


def read_clip_list(path: Path) -> list[int]:
    """Clip numbers from ``trainlist.txt`` / ``testlist.txt`` (one per line)."""
    return sorted(int(x) for x in Path(path).read_text().split())


def choose_val_clips(train_clips: list[int], n_val: int, seed: int) -> list[int]:
    """Pick ``n_val`` clips from the official train clips, reproducibly."""
    if not 0 < n_val < len(train_clips):
        raise ValueError(f"n_val must be in 1..{len(train_clips) - 1}, got {n_val}")
    return sorted(random.Random(seed).sample(sorted(train_clips), n_val))


def build_manifest(official_train: list[int], official_test: list[int], n_val: int,
                   seed: int) -> dict:
    """The frozen split. ``train`` + ``val`` = official train clips; ``test`` = official test."""
    val = choose_val_clips(official_train, n_val, seed)
    return {
        "dataset": "DroneCrowd",
        "note": "val is held out from the official train clips; the official val_data "
                "folder is a copy of test frames and is not used (docs/project-log.md)",
        "seed": seed,
        "n_val": n_val,
        "train": sorted(set(official_train) - set(val)),
        "val": val,
        "test": sorted(official_test),
    }


def check_manifest(manifest: dict) -> None:
    """Raise if splits overlap or are empty."""
    sets = {k: set(manifest[k]) for k in SPLITS}
    for a in SPLITS:
        if not sets[a]:
            raise ValueError(f"split '{a}' is empty")
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f"splits '{a}' and '{b}' overlap: {sorted(sets[a] & sets[b])}")


class DroneCrowdDataset:
    """Frames of one split. Index with ``ds[i]`` -> :class:`Sample`; ``len(ds)`` frames.

    Args:
        root: dataset root folder (contains the train/test folders).
        manifest: dict from :func:`build_manifest` (or loaded from its JSON).
        split: ``"train"``, ``"val"`` or ``"test"``.
        train_dir / test_dir: image+GT folders relative to ``root``; each holds
            ``images/`` and ``ground_truth/``. Val clips come from ``train_dir``.
        image_size: optional (width, height) to resize images to. Points stay in
            original pixels; use ``Sample.scale`` for the model side.
        frame_stride: keep every Nth frame (1 = all).
    """

    def __init__(self, root: Path, manifest: dict, split: str, train_dir: str, test_dir: str,
                 image_size: tuple[int, int] | None = None, frame_stride: int = 1) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        check_manifest(manifest)
        self.image_size = image_size
        folder = Path(root) / (test_dir if split == "test" else train_dir)
        self.refs = self._list_refs(folder, set(manifest[split]), frame_stride)
        if not self.refs:
            raise FileNotFoundError(f"no frames found for split '{split}' under {folder}")

    @staticmethod
    def _list_refs(folder: Path, clips: set[int], stride: int) -> list[FrameRef]:
        found: dict[int, list[int]] = {}
        for p in (folder / "images").iterdir():
            m = FRAME_RE.match(p.name)
            if m and int(m.group(1)) in clips:
                found.setdefault(int(m.group(1)), []).append(int(m.group(2)))
        refs = []
        for clip in sorted(found):
            for n in sorted(found[clip])[::stride]:
                name = f"img{clip:03d}{n:03d}"
                refs.append(FrameRef(clip, n - 1, folder / "images" / f"{name}.jpg",
                                     folder / "ground_truth" / f"GT_{name}.mat"))
        return refs

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, i: int) -> Sample:
        ref = self.refs[i]
        bgr = cv2.imread(str(ref.image_path))
        if bgr is None:
            raise FileNotFoundError(f"cannot read image {ref.image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        points, ids = load_gt(ref.gt_path)
        scale = (1.0, 1.0)
        if self.image_size is not None:
            h, w = image.shape[:2]
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            scale = (self.image_size[0] / w, self.image_size[1] / h)
        return Sample(image, points, ids, scale, ref)


def draw_points(sample: Sample, radius: int = 5) -> np.ndarray:
    """BGR copy of the sample image with GT points drawn (mapped to the image's size)."""
    out = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
    for x, y in sample.points:
        cv2.circle(out, (round(x * sample.scale[0]), round(y * sample.scale[1])), radius,
                   (0, 0, 255), -1)
    label = f"{sample.ref.clip_id} frame {sample.ref.frame_id} n={len(sample.points)}"
    cv2.putText(out, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    return out


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def make_split(cfg: dict) -> dict:
    """Build the manifest from the official lists and write it to ``cfg['manifest']``."""
    root = Path(cfg["root"])
    manifest = build_manifest(read_clip_list(root / cfg["train_list"]),
                              read_clip_list(root / cfg["test_list"]),
                              int(cfg["n_val"]), int(cfg["seed"]))
    check_manifest(manifest)
    out = Path(cfg["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def dataset_from_config(cfg: dict, split: str, **kwargs) -> DroneCrowdDataset:
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    return DroneCrowdDataset(Path(cfg["root"]), manifest, split, cfg["train_dir"],
                             cfg["test_dir"], **kwargs)


def clip_gt_frames(cfg: dict, split: str, clip: int,
                   frame_ids: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """GT of one clip as ``{frame_id: (points (N, 2) x, y px, person_ids (N,))}``.

    Reads only the ``.mat`` files, never the images, so it works when the train images are
    on a drive that is not plugged in. ``frame_ids`` are 0-based (file number minus 1).
    """
    folder = Path(cfg["root"]) / (cfg["test_dir"] if split == "test" else cfg["train_dir"])
    out = {}
    for fid in frame_ids:
        path = folder / "ground_truth" / f"GT_img{clip:03d}{fid + 1:03d}.mat"
        out[int(fid)] = load_gt(path)
    return out


def draw_check(cfg: dict, splits: tuple[str, ...] = SPLITS, per_split: int = 2) -> list[Path]:
    """Save a few frames per split with GT drawn, for visual alignment checks."""
    out_dir = Path(cfg["check_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(cfg["seed"]))
    saved = []
    for split in splits:
        ds = dataset_from_config(cfg, split)
        for i in sorted(rng.sample(range(len(ds)), per_split)):
            sample = ds[i]
            path = out_dir / f"{split}_{sample.ref.clip_id}_f{sample.ref.frame_id:03d}.jpg"
            cv2.imwrite(str(path), draw_points(sample))
            saved.append(path)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("command", choices=["make-split", "draw"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.command == "make-split":
        m = make_split(cfg)
        print(f"wrote {cfg['manifest']}")
        print(f"train {len(m['train'])} clips, val {len(m['val'])}: {m['val']}, "
              f"test {len(m['test'])} clips")
    else:
        for path in draw_check(cfg):
            print(f"saved {path}")


if __name__ == "__main__":
    main()
