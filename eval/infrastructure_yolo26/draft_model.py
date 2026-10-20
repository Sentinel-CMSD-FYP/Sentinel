"""Draft Model C polygons with a YOLO26 seg model trained on the labels so far.

    python eval/infrastructure_yolo26/draft_model.py train --out runs/infra_draft/r1
    python eval/infrastructure_yolo26/draft_model.py predict runs/infra_draft/r1/weights/best.pt \
        [img003150 ...] --name batch2

``train`` fine tunes the COCO ``yolo26s-seg`` on every image in ``<data>/images`` that has a
label file in ``<data>/labels`` (written by ``ls_to_yolo.py``). It validates on the same
images: this model only makes drafts, the real train/val split comes in step 14.
``predict`` (all images without a label file when no stems are given) writes Label Studio
prediction results to ``<data>/images/_drafts/<name>.json`` (same format as
``sam_draft.py run``) plus overlays in ``<data>/drafts_overlay/``. ``<data>`` defaults to
``data/infra_labels``; the Kaggle kernel ``kaggle/infra_draft`` runs both commands on GPU.
Drafts are never final labels: a person reviews and submits each image (LABELLING.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from sam_draft import COLOURS, LABEL_DIR, REPO, ls_result, mask_polygon

CONFIG = REPO / "ml" / "configs" / "infrastructure_yolo26.yaml"
MIN_SIDE = 15  # px, LABELLING.md skips smaller objects


def split_images(data: Path) -> tuple[list[Path], list[Path]]:
    """(labelled, unlabelled) images in ``data/images``, by presence of a label file."""
    images = sorted((data / "images").glob("*.jpg"))
    has = [(data / "labels" / f"{p.stem}.txt").is_file() for p in images]
    return [p for p, h in zip(images, has) if h], [p for p, h in zip(images, has) if not h]


def train(data: Path, out: Path, weights: Path, epochs: int, imgsz: int, batch: int,
          device: str) -> None:
    from ultralytics import YOLO

    cfg = yaml.safe_load(CONFIG.read_text())
    images = split_images(data)[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("".join(f"{p.resolve().as_posix()}\n" for p in images))
    spec = {"path": out.resolve().as_posix(), "train": "train.txt", "val": "train.txt",
            "names": dict(enumerate(cfg["classes"]))}
    (out / "data.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"{len(images)} labelled images")
    weights.parent.mkdir(parents=True, exist_ok=True)
    YOLO(str(weights)).train(  # Ultralytics downloads known asset names to this path
        data=str(out / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        project=str(out.parent.resolve()), name=out.name, exist_ok=True, workers=4, plots=False)


def predict(data: Path, weights: Path, stems: list[str], name: str, imgsz: int,
            conf: float) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images, overlay_dir = data / "images", data / "drafts_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    stems = stems or [p.stem for p in split_images(data)[1]]
    drafts = {}
    for stem in stems:
        img = cv2.imread(str(images / f"{stem}.jpg"))
        h, w = img.shape[:2]
        res = model(img, imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)[0]
        results, fill = [], img.copy()
        if res.masks is not None:
            for m, c in zip(res.masks.data.cpu().numpy(), res.boxes.cls.int().tolist()):
                pts = mask_polygon(cv2.resize(m.astype(np.uint8), (w, h),
                                              interpolation=cv2.INTER_NEAREST))
                if pts is None or max(np.ptp(pts, axis=0)) < MIN_SIDE:
                    continue
                label = model.names[c]
                results.append(ls_result(pts, label, w, h))
                cv2.fillPoly(fill, [pts], COLOURS[label])
                cv2.polylines(img, [pts], True, COLOURS[label], 2, cv2.LINE_AA)
        drafts[f"{stem}.jpg"] = results
        cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), cv2.addWeighted(fill, 0.35, img, 0.65, 0))
        print(f"{stem}: {len(results)} polygons")
    out = images / "_drafts" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drafts))
    print(f"drafts -> {out}, overlays -> {overlay_dir}")


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("args", nargs="*", help="predict: weights, then image stems (default all "
                                            "unlabelled images)")
    ap.add_argument("--data", type=Path, default=LABEL_DIR, help="folder with images/, labels/")
    ap.add_argument("--out", type=Path, help="train: run folder")
    ap.add_argument("--weights", type=Path, default=REPO / cfg["base_weights"],
                    help="train: base model")
    ap.add_argument("--name", help="predict: drafts file name, e.g. batch2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.data, a.out, a.weights, a.epochs, a.imgsz, a.batch, a.device)
    else:
        predict(a.data, Path(a.args[0]), a.args[1:], a.name, a.imgsz, a.conf)


if __name__ == "__main__":
    main()
