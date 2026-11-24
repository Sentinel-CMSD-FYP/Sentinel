"""Sample density (Model A) and infrastructure (Model C) layers for one DroneCrowd clip (step 16).

    python eval/make_layer_samples.py [--clip 11] [--out data/outputs/layers/seq011] [--device cpu]

Runs the two wrappers (``sentinel_ml.density.density_layer``, ``sentinel_ml.infrastructure.
infrastructure_layer``) with their configs over all 300 frames of the clip; each wrapper picks
its own frames (``every``). Writes ``density.json``, ``infrastructure.json`` and a preview jpg
(density zones + polygons) for every frame that has both. ``clip_id`` is ``seq<clip:03d>`` as
in the layer contract (docs/layers on main). Frame ids are 0 based (file frame number - 1).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from sentinel_ml.density import (
    DensityLayerConfig,
    DMCount,
    DMCountConfig,
    density_layer,
)
from sentinel_ml.density.data import find_split_dirs
from sentinel_ml.infrastructure import InfraConfig, InfraModel, infrastructure_layer

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "ml" / "configs"
COLOURS = {"gate_exit": (0, 255, 0), "barrier": (0, 255, 255), "wall": (255, 0, 0),
           "obstruction": (255, 0, 255)}  # BGR


def clip_frames(root: Path, clip: int) -> list[Path]:
    for d in find_split_dirs(root).values():
        paths = sorted((root / d / "images").glob(f"img{clip:03d}[0-9][0-9][0-9].jpg"))
        if paths:
            return paths
    raise SystemExit(f"clip {clip} not found under {root}")


def read(paths: list[Path]) -> Iterator[tuple[int, np.ndarray]]:
    for fid, p in enumerate(paths):  # files are numbered 001.., so the index is the frame id
        yield fid, cv2.imread(str(p))


def preview(frame: np.ndarray, dens: dict, infra: dict, grid: dict) -> np.ndarray:
    """People per zone as numbers on a grid, infrastructure polygons outlined per class."""
    out = frame.copy()
    h, w = frame.shape[:2]
    for o in infra:
        pts = np.round(np.array(o["polygon"])).astype(np.int32)
        cv2.polylines(out, [pts], True, COLOURS[o["cls"]], 2)
    rows, cols = grid["rows"], grid["cols"]
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * w // cols, r * h // rows
            cv2.rectangle(out, (x0, y0), ((c + 1) * w // cols, (r + 1) * h // rows), (200, 200, 200), 1)
            v = dens["density"][r][c]
            if v >= 0.5:
                cv2.putText(out, f"{v:.0f}", (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
    total = sum(map(sum, dens["density"]))
    cv2.putText(out, f"count {total:.1f}, {len(infra)} infrastructure objects", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", type=int, default=11)
    ap.add_argument("--root", type=Path, default=REPO / "data" / "dronecrowd")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    clip_id = f"seq{a.clip:03d}"
    out = a.out or REPO / "data" / "outputs" / "layers" / clip_id
    out.mkdir(parents=True, exist_ok=True)
    paths = clip_frames(a.root, a.clip)

    dcfg = DMCountConfig.from_yaml(CONFIGS / "density_dmcount.yaml")
    dmodel = DMCount(DMCountConfig(weights=str(REPO / dcfg.weights), device=a.device))
    icfg = InfraConfig.from_yaml(CONFIGS / "infrastructure_yolo26.yaml")
    imodel = InfraModel(InfraConfig(**{**icfg.__dict__, "weights": str(REPO / icfg.weights),
                                       "device": a.device}))

    t = time.perf_counter()
    dens = density_layer(read(paths), dmodel, clip_id,
                         DensityLayerConfig.from_yaml(CONFIGS / "density_layer.yaml"))
    print(f"density: {len(dens['frames'])} samples in {time.perf_counter() - t:.0f} s")
    t = time.perf_counter()
    infra = infrastructure_layer(read(paths), imodel, clip_id)
    print(f"infrastructure: {len(infra['frames'])} samples in {time.perf_counter() - t:.0f} s")

    for name, layer in (("density", dens), ("infrastructure", infra)):
        (out / f"{name}.json").write_text(json.dumps(layer, separators=(",", ":")) + "\n")
    for fid in sorted(set(dens["frames"]) & set(infra["frames"]), key=int):
        img = preview(cv2.imread(str(paths[int(fid)])), dens["frames"][fid], infra["frames"][fid],
                      dens["grid"])
        cv2.imwrite(str(out / f"preview_{int(fid):03d}.jpg"), img)
    for fid, e in dens["frames"].items():
        print(f"frame {fid:>3}: count {sum(map(sum, e['density'])):.1f}"
              + (f", {len(infra['frames'][fid])} objects" if fid in infra["frames"] else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
