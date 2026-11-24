"""Fine tune YOLO26 seg for Model C on the scene split (step 14).

    python eval/infrastructure_yolo26/train.py --out runs/infra_train/r1 [--data data/infra_labels]
        [--split data/manifests/infra_split.json] [--weights data/weights/yolo26/yolo26s-seg.pt]
        [--epochs 150] [--patience 50] [--imgsz 1280] [--batch 4] [--device cpu]

Trains on the ``train`` stems of the split (``split_scenes.py``) and validates on its ``val``
stems; images in ``<data>/images``, YOLO seg labels in ``<data>/labels``. Ultralytics keeps the
best epoch by its fitness (box + mask, mostly mAP50-95) and writes ``<out>/weights/best.pt``,
``results.csv`` and the curve plots. The Kaggle kernel ``kaggle/infra_train`` runs it on GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "ml" / "configs" / "infrastructure_yolo26.yaml"


def write_data_yaml(data: Path, split_file: Path, out: Path, classes: list[str]) -> Path:
    """Write train.txt, val.txt and data.yaml for the split into ``out``."""
    split = json.loads(split_file.read_text())
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val"):
        paths = [data / "images" / f"{s}.jpg" for s in split[name]]
        if missing := [p.name for p in paths if not p.is_file()]:
            raise FileNotFoundError(f"{name}: {missing[:5]} ...")
        (out / f"{name}.txt").write_text("".join(f"{p.resolve().as_posix()}\n" for p in paths))
        print(f"{name}: {len(paths)} images")
    spec = {"path": out.resolve().as_posix(), "train": "train.txt", "val": "val.txt",
            "names": dict(enumerate(classes))}
    (out / "data.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return out / "data.yaml"


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="run folder")
    ap.add_argument("--data", type=Path, default=REPO / "data" / "infra_labels",
                    help="folder with images/, labels/")
    ap.add_argument("--split", type=Path, default=REPO / "data" / "manifests" / "infra_split.json")
    ap.add_argument("--weights", type=Path, default=REPO / cfg["base_weights"], help="base model")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = write_data_yaml(a.data, a.split, a.out, cfg["classes"])
    a.weights.parent.mkdir(parents=True, exist_ok=True)
    YOLO(str(a.weights)).train(  # Ultralytics downloads known asset names to this path
        data=str(data_yaml), epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch,
        device=a.device, project=str(a.out.parent.resolve()), name=a.out.name, exist_ok=True,
        workers=4, seed=0, plots=True)


if __name__ == "__main__":
    main()
