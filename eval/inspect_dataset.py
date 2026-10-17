"""Read-only inspection of the DroneCrowd dataset (P0-3).

Prints clip/frame counts, resolution, annotation format, points-per-frame stats,
ID availability, camera/altitude metadata hints, split files, and val/test overlap.
Saves sample frames with GT points drawn.

Coordinates: x = column, y = row, original frame pixels.

    python eval/inspect_dataset.py --config ml/configs/inspect_dataset.yaml
"""

from __future__ import annotations

import argparse
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import yaml

FRAME_RE = re.compile(r"^img(\d{3})(\d{3})\.jpg$")
SPLITS = {
    "train": "train_data-002/train_data",
    "test": "test_data-001/test_data",
    "val": "val_data/val_data",
}


def load_gt(path: Path) -> np.ndarray:
    """Load one ``GT_img*.mat``. Returns (N, 3) float array of (x, y, id), pixels."""
    info = sio.loadmat(str(path))["image_info"][0, 0][0, 0]
    pts = np.asarray(info["location"], dtype=np.float64).reshape(-1, 3)
    return pts


def declared_number(path: Path) -> int:
    """The ``number`` field stored in a GT file (people count claimed by the file)."""
    info = sio.loadmat(str(path))["image_info"][0, 0][0, 0]
    return int(np.asarray(info["number"]).ravel()[0])


def list_frames(split_dir: Path) -> dict[int, list[int]]:
    """Map sequence id -> sorted frame ids found in ``split_dir/images``."""
    seqs: dict[int, list[int]] = defaultdict(list)
    for p in (split_dir / "images").iterdir():
        m = FRAME_RE.match(p.name)
        if m:
            seqs[int(m.group(1))].append(int(m.group(2)))
    return {k: sorted(v) for k, v in sorted(seqs.items())}


def frame_paths(split_dir: Path, seq: int, fr: int) -> tuple[Path, Path]:
    name = f"img{seq:03d}{fr:03d}"
    return split_dir / "images" / f"{name}.jpg", split_dir / "ground_truth" / f"GT_{name}.mat"


def stats(values: list[int]) -> str:
    a = np.asarray(values)
    return f"min={a.min()} median={np.median(a):.0f} mean={a.mean():.1f} max={a.max()}"


def read_list(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()]


def draw_points(img: np.ndarray, pts: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    for x, y in pts[:, :2]:
        cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
    cv2.putText(out, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    return out


def inspect_split(name: str, split_dir: Path, results: dict) -> None:
    """Print per-split stats and keep per-frame data in ``results[name]``."""
    seqs = list_frames(split_dir)
    n_frames = sum(len(v) for v in seqs.values())
    print(f"\n[{name}] {split_dir}")
    print(f"  clips={len(seqs)} frames={n_frames}")
    print(f"  clip ids: {list(seqs)[:5]}...{list(seqs)[-2:]}")
    print(f"  frames per clip: {stats([len(v) for v in seqs.values()])}")
    gaps = [s for s, v in seqs.items() if v != list(range(v[0], v[0] + len(v)))]
    print(f"  clips with non-consecutive frame ids: {gaps or 'none'}")
    first_frame = {s: v[0] for s, v in seqs.items()}
    print(f"  first frame id per clip: {sorted(set(first_frame.values()))}")

    sizes: dict[tuple[int, int], int] = defaultdict(int)
    brightness: dict[int, float] = {}
    for s in seqs:
        img = cv2.imread(str(frame_paths(split_dir, s, first_frame[s])[0]))
        sizes[(img.shape[1], img.shape[0])] += 1
        brightness[s] = float(img.mean())
    print(f"  resolution (w,h) -> clips (first frame of each clip): {dict(sizes)}")
    b = sorted(brightness.items(), key=lambda kv: kv[1])
    print(f"  mean brightness of first frames (hint only, 0-255): "
          f"darkest clip {b[0][0]}={b[0][1]:.0f}, brightest clip {b[-1][0]}={b[-1][1]:.0f}")

    counts: list[int] = []
    declared_mismatch = 0
    missing_gt = 0
    ids_per_clip: dict[int, set[int]] = defaultdict(set)
    n_pts_per_frame: dict[tuple[int, int], int] = {}
    for s, frs in seqs.items():
        for f in frs:
            gt_path = frame_paths(split_dir, s, f)[1]
            if not gt_path.exists():
                missing_gt += 1
                continue
            pts = load_gt(gt_path)
            counts.append(len(pts))
            n_pts_per_frame[(s, f)] = len(pts)
            ids_per_clip[s].update(int(i) for i in pts[:, 2])
            if declared_number(gt_path) != len(pts):
                declared_mismatch += 1
    print(f"  frames missing a GT file: {missing_gt}")
    print(f"  points per frame: {stats(counts)}")
    print(f"  frames where file's 'number' != row count: {declared_mismatch}")
    ids = [len(v) for v in ids_per_clip.values()]
    print(f"  unique person ids per clip: {stats(ids)}")
    results[name] = {"seqs": seqs, "counts": n_pts_per_frame}


def check_xml_metadata(root: Path) -> None:
    """Look for altitude / camera fields in the annotation XMLs and README."""
    ann = root / "annotations" / "annotations"
    xmls = sorted(ann.glob("*.xml"))
    print(f"\n[annotations] {ann}: {len(xmls)} xml, {len(list(ann.glob('*.mat')))} mat, "
          f"{len(list(ann.glob('*_clean.txt')))} _clean.txt")
    tags: set[str] = set()
    attrs: set[str] = set()
    for x in xmls:
        for el in ET.parse(x).getroot().iter():
            tags.add(el.tag)
            attrs.update(el.attrib)
    print(f"  xml tags: {sorted(tags)}")
    print(f"  xml attributes: {sorted(attrs)}")
    hits = [w for w in ("altitude", "height", "gps", "camera", "attribute", "weather", "night")
            if any(w in t.lower() for t in tags | attrs)]
    print(f"  altitude/camera/lighting-like xml fields: {hits or 'none'}")
    readme = (root / "README.md").read_text(encoding="utf-8", errors="ignore").lower()
    words = [w for w in ("altitude", "height", "gps", "night", "illumination", "attribute")
             if w in readme]
    print(f"  words found in dataset README: {words or 'none'}")
    exts = defaultdict(int)
    for p in root.rglob("*"):
        if p.is_file():
            exts[p.suffix.lower()] += 1
    print(f"  file extensions in dataset: {dict(exts)}")


def check_splits(root: Path, results: dict) -> None:
    print("\n[split files]")
    for fname, key in (("trainlist.txt", "train"), ("testlist.txt", "test")):
        ids = read_list(root / fname)
        on_disk = set(results[key]["seqs"])
        print(f"  {fname}: {len(ids)} ids; on disk in {key}/: {len(on_disk)}")
        print(f"    in list not on disk: {sorted(set(ids) - on_disk) or 'none'}")
        print(f"    on disk not in list: {sorted(on_disk - set(ids)) or 'none'}")
    train, test = set(results["train"]["seqs"]), set(results["test"]["seqs"])
    print(f"  train/test clip overlap: {sorted(train & test) or 'none'}")

    val, test_seqs = results["val"]["seqs"], results["test"]["seqs"]
    print(f"  val clips: {list(val)}; all in test clips: {set(val) <= set(test_seqs)}")
    same_name = sum(1 for s, v in val.items() for f in v if f in test_seqs.get(s, []))
    n_val = sum(len(v) for v in val.values())
    print(f"  val frames that also exist in test (same clip+frame id): {same_name}/{n_val}")
    identical = 0
    for s, frs in val.items():
        for f in frs:
            a = load_gt(frame_paths(root / SPLITS["val"], s, f)[1])
            b = frame_paths(root / SPLITS["test"], s, f)[1]
            if b.exists() and np.array_equal(a, load_gt(b)):
                identical += 1
    print(f"  of those, GT points identical to test: {identical}/{n_val}")
    print("  => val is NOT independent of test" if identical else "  => val looks separate")


def save_samples(root: Path, results: dict, out_dir: Path, n: int, seed: int) -> None:
    """Save ``n`` frames (spread over train/test/val) with GT points drawn."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    order = ["train", "test", "val"]
    picks: list[tuple[str, int, int]] = []
    for i in range(n):
        name = order[i % 3]
        seqs = results[name]["seqs"]
        s = rng.choice(list(seqs))
        picks.append((name, s, rng.choice(seqs[s])))
    for name, s, f in picks:
        img_p, gt_p = frame_paths(root / SPLITS[name], s, f)
        pts = load_gt(gt_p)
        label = f"{name} clip{s:03d} fr{f:03d} n={len(pts)}"
        out = draw_points(cv2.imread(str(img_p)), pts, label)
        dest = out_dir / f"{name}_{s:03d}_{f:03d}.jpg"
        cv2.imwrite(str(dest), out)
        print(f"  saved {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root, out_dir = Path(cfg["root"]), Path(cfg["out_dir"])

    results: dict = {}
    for name, rel in SPLITS.items():
        inspect_split(name, root / rel, results)
    check_splits(root, results)
    check_xml_metadata(root)
    print("\n[sample frames]")
    save_samples(root, results, out_dir, int(cfg["n_samples"]), int(cfg["seed"]))


if __name__ == "__main__":
    main()
