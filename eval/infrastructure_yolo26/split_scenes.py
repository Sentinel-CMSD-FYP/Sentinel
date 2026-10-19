"""Split the Model C labelled images into train/val by scene group (step 14).

    python eval/infrastructure_yolo26/split_scenes.py [--labels data/infra_labels/labels]
        [--groups data/manifests/infra_scene_groups.yaml] [--out data/manifests/infra_split.json]
        [--seed 0] [--val-fraction 0.2] [--tries 20000]

A scene group (clips showing the same place) goes wholly to train or to val, so val measures
unseen places. With a fixed seed, ``tries`` random group orders are drawn; each takes groups
until ``val_fraction`` of the images is reached. Kept is the draw that holds at least one
instance of every class and one night clip, and whose share of images and of each class's
instances is closest to ``val_fraction`` (sum of absolute differences). Image stems are
``img{clip:03d}{frame:03d}``; label files come from ``ls_to_yolo.py``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "ml" / "configs" / "infrastructure_yolo26.yaml"
MANIFESTS = REPO / "data" / "manifests"


def clip_of(stem: str) -> int:
    return int(stem[3:6])


def load_counts(labels: Path, n_classes: int) -> dict[str, np.ndarray]:
    """Instances per class for every label file, keyed by image stem."""
    counts = {}
    for p in sorted(labels.glob("*.txt")):
        ids = [int(line.split()[0]) for line in p.read_text().splitlines() if line.strip()]
        counts[p.stem] = np.bincount(ids, minlength=n_classes)
    return counts


def group_stats(groups: list[list[int]], counts: dict[str, np.ndarray]) -> np.ndarray:
    """(groups, 1 + classes) array: images, then instances per class, per group."""
    where = {c: i for i, g in enumerate(groups) for c in g}
    if len(where) != sum(len(g) for g in groups):
        raise ValueError("a clip is listed in more than one group")
    n_classes = len(next(iter(counts.values())))
    stats = np.zeros((len(groups), 1 + n_classes), dtype=int)
    for stem, c in counts.items():
        if clip_of(stem) not in where:
            raise ValueError(f"clip of {stem} is in no group")
        stats[where[clip_of(stem)]] += np.concatenate(([1], c))
    return stats


def pick_val(groups: list[list[int]], stats: np.ndarray, night: list[int], frac: float,
             seed: int, tries: int) -> list[int]:
    """Indices of the val groups (see the module docstring)."""
    rng = random.Random(seed)
    total = stats.sum(axis=0)
    night_groups = {i for i, g in enumerate(groups) if set(g) & set(night)}
    best, best_score = None, float("inf")
    for _ in range(tries):
        chosen, vec = [], np.zeros_like(total)
        for i in rng.sample(range(len(groups)), len(groups)):
            if vec[0] >= frac * total[0]:
                break
            chosen.append(i)
            vec = vec + stats[i]
        if (vec[1:][total[1:] > 0] == 0).any() or not night_groups & set(chosen):
            continue
        score = np.abs(vec[total > 0] / total[total > 0] - frac).sum()
        if score < best_score:
            best, best_score = sorted(chosen), score
    if best is None:
        raise ValueError("no draw met the constraints")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, default=REPO / "data" / "infra_labels" / "labels")
    ap.add_argument("--groups", type=Path, default=MANIFESTS / "infra_scene_groups.yaml")
    ap.add_argument("--out", type=Path, default=MANIFESTS / "infra_split.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--tries", type=int, default=20000)
    a = ap.parse_args()

    classes = yaml.safe_load(CONFIG.read_text())["classes"]
    spec = yaml.safe_load(a.groups.read_text())
    groups = spec["groups"]
    counts = load_counts(a.labels, len(classes))
    stats = group_stats(groups, counts)
    val_groups = pick_val(groups, stats, spec["night"], a.val_fraction, a.seed, a.tries)
    val_clips = sorted(c for i in val_groups for c in groups[i])
    split = {"train": [s for s in counts if clip_of(s) not in val_clips],
             "val": [s for s in counts if clip_of(s) in val_clips]}
    table = {}
    for name, stems in split.items():
        vec = sum((np.concatenate(([1], counts[s])) for s in stems), np.zeros(1 + len(classes)))
        table[name] = dict(zip(["images", *classes], map(int, vec)))
    out = {"seed": a.seed, "val_fraction": a.val_fraction, "tries": a.tries,
           "groups_file": a.groups.resolve().relative_to(REPO).as_posix(),
           "val_groups": [groups[i] for i in val_groups], "counts": table, **split}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for name, row in table.items():
        print(f"{name:5s} " + "  ".join(f"{k} {v}" for k, v in row.items()))
    print(f"val clips {val_clips} -> {a.out}")


if __name__ == "__main__":
    main()
