"""Convert a Label Studio JSON export (polygons) to YOLO segmentation label files.

    python eval/infrastructure_yolo26/ls_to_yolo.py export.json [--out data/infra_labels/labels]
        [--overlays data/infra_labels/overlays]

Writes one ``<image stem>.txt`` per annotated image, one line per polygon:
``class_id x1 y1 x2 y2 ...`` with coordinates normalised to 0..1 (Ultralytics format).
Class ids follow ``classes`` in ``ml/configs/infrastructure_yolo26.yaml``. An image that
was submitted with no polygons gets an empty file (a valid background image); images
that were never submitted or were skipped get no file. The newest submitted annotation
per image is used.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "ml" / "configs" / "infrastructure_yolo26.yaml"
DEFAULT_OUT = REPO / "data" / "infra_labels" / "labels"
UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")  # Label Studio adds this to uploaded files


def image_name(task: dict) -> str:
    """Original file name of a task image (local storage ``?d=...`` path or upload URL)."""
    url = task["data"]["image"]
    name = unquote(url).replace("\\", "/").rsplit("/", 1)[-1]
    return name if url.startswith("/data/local-files") else UPLOAD_PREFIX.sub("", name)


def polygon_lines(result: list[dict], class_ids: dict[str, int]) -> list[str]:
    lines = []
    for region in result:
        if region.get("type") != "polygonlabels":
            continue
        value = region["value"]
        (label,) = value["polygonlabels"]
        if label not in class_ids:
            raise ValueError(f"unknown label {label!r}, expected one of {list(class_ids)}")
        points = value["points"]
        if len(points) < 3:
            raise ValueError(f"polygon with {len(points)} points (label {label}, id {region.get('id')})")
        coords = " ".join(f"{min(max(v / 100, 0.0), 1.0):.6f}" for p in points for v in p)
        lines.append(f"{class_ids[label]} {coords}")
    return lines


def convert(tasks: list[dict], classes: list[str]) -> dict[str, list[str]]:
    """Map image stem to its label lines. Raises on duplicate image names."""
    class_ids = {c: i for i, c in enumerate(classes)}
    out: dict[str, list[str]] = {}
    for task in tasks:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:
            continue
        stem = Path(image_name(task)).stem
        if stem in out:
            raise ValueError(f"image {stem} appears in more than one task")
        newest = max(done, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        out[stem] = polygon_lines(newest["result"], class_ids)
    return out


def draw_labels(img, lines: list[str], classes: list[str]):
    """Return ``img`` (BGR) with the YOLO seg label ``lines`` drawn as filled polygons."""
    import cv2
    import numpy as np

    colours = [(83, 200, 0), (0, 171, 255), (255, 98, 41), (0, 0, 213)]  # BGR, as in the LS config
    img = img.copy()
    h, w = img.shape[:2]
    fill = img.copy()
    for line in lines:
        cid, *xy = line.split()
        pts = (np.array(xy, np.float32).reshape(-1, 2) * (w, h)).round().astype(np.int32)
        colour = colours[int(cid) % len(colours)]
        cv2.fillPoly(fill, [pts], colour)
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
        cv2.putText(img, classes[int(cid)], tuple(int(v) for v in pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return cv2.addWeighted(fill, 0.35, img, 0.65, 0)


def draw_overlays(labels_dir: Path, images_dir: Path, out: Path, classes: list[str]) -> None:
    """Draw the YOLO label files back onto their images, to check the export by eye."""
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    for txt in sorted(labels_dir.glob("*.txt")):
        img = cv2.imread(str(images_dir / f"{txt.stem}.jpg"))
        cv2.imwrite(str(out / f"{txt.stem}.jpg"),
                    draw_labels(img, txt.read_text().splitlines(), classes))
    print(f"overlays -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path, help="Label Studio export, JSON format")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overlays", type=Path, help="also draw the labels onto the images here")
    ap.add_argument("--images", type=Path, default=DEFAULT_OUT.parent / "images")
    args = ap.parse_args()

    classes = yaml.safe_load(args.config.read_text())["classes"]
    labels = convert(json.loads(args.export.read_text(encoding="utf-8")), classes)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = [0] * len(classes)
    for stem, lines in labels.items():
        (args.out / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines))
        for line in lines:
            counts[int(line.split()[0])] += 1
    print(f"{len(labels)} images, {sum(counts)} polygons -> {args.out}")
    for name, n in zip(classes, counts):
        print(f"  {name}: {n}")
    if args.overlays:
        draw_overlays(args.out, args.images, args.overlays, classes)


if __name__ == "__main__":
    main()
