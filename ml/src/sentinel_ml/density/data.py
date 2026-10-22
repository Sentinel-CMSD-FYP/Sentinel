"""DroneCrowd frames for DM-Count training and eval (Tier 1, Model A).

A thin layer over :mod:`sentinel_ml.localization.data` (Ahmed's loader, shared by the
team): same frozen split manifest, same (image, head points) samples. It only adds what
the density side needs:

  * the split folders are found by name, so the same config works on the local copy
    (``train_data-002/train_data``) and the Kaggle copy (``train_data/train_data``);
  * a frame stride per split, read from ``ml/configs/density_data.yaml``.

The official ``val_data`` folder is a copy of test frames and is never used (decision
0002). ``val`` here is the manifest's held out train clips.
"""

from __future__ import annotations

import json
from pathlib import Path

from sentinel_ml.localization.data import SPLITS, DroneCrowdDataset

SPLIT_FOLDERS = {"train": "train_data", "test": "test_data"}  # val clips live in train_data


def find_split_dirs(root: Path) -> dict[str, str]:
    """Map ``train``/``test`` to the folder (relative to ``root``) that holds
    ``images/`` and ``ground_truth/``. Raises if either is missing."""
    root = Path(root)
    found: dict[str, str] = {}
    for img_dir in sorted(root.rglob("images")):
        d = img_dir.parent
        for split, name in SPLIT_FOLDERS.items():
            if d.name == name and (d / "ground_truth").is_dir() and split not in found:
                found[split] = d.relative_to(root).as_posix()
    missing = set(SPLIT_FOLDERS) - set(found)
    if missing:
        raise FileNotFoundError(
            f"no {sorted(missing)} folder with images/ and ground_truth/ under {root}")
    return found


def load_split(split: str, root: Path, manifest_path: Path, frame_stride: int = 1,
               **kwargs) -> DroneCrowdDataset:
    """One split of DroneCrowd. ``kwargs`` go to :class:`DroneCrowdDataset`
    (e.g. ``image_size``)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    dirs = find_split_dirs(root)
    manifest = json.loads(Path(manifest_path).read_text())
    return DroneCrowdDataset(Path(root), manifest, split, dirs["train"], dirs["test"],
                             frame_stride=frame_stride, **kwargs)


def load_split_from_config(cfg: dict, split: str, root: Path | None = None,
                           repo: Path = Path("."), **kwargs) -> DroneCrowdDataset:
    """Like :func:`load_split`, with paths and strides from ``density_data.yaml``.
    ``root`` overrides the config (e.g. ``/kaggle/input``); relative config paths are
    taken from ``repo``. A split missing from ``frame_stride`` uses every frame."""
    root = Path(root) if root is not None else Path(repo) / cfg["root"]
    stride = int((cfg.get("frame_stride") or {}).get(split, 1))
    return load_split(split, root, Path(repo) / cfg["manifest"], stride, **kwargs)
