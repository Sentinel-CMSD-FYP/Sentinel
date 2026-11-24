"""mAP of Model C on the val images without and with the wrapper's class wise NMS (step 16).

    python eval/infrastructure_yolo26/nms_check.py [--out data/outputs/infra_bench/nms_check]

YOLO26 is NMS free, so Ultralytics val skips NMS. For the second run the validator's
postprocess is patched to apply ``sentinel_ml.infrastructure.class_nms`` (IoU ``nms_iou`` from
the config) per image, as ``InfraModel.predict`` does. Writes ``nms_check.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from benchmark import val_metrics
from sentinel_ml.infrastructure import class_nms
from train import CONFIG, REPO, write_data_yaml


def patch_validator(iou: float) -> None:
    from ultralytics.models.yolo.detect.val import DetectionValidator

    original = DetectionValidator.postprocess

    def postprocess(self, preds):
        out = original(self, preds)
        for p in out:
            if len(p["conf"]):
                keep = class_nms(p["bboxes"].cpu().numpy(), p["conf"].cpu().numpy(),
                                 p["cls"].cpu().numpy().astype(np.int64), iou)
                for k in p:
                    p[k] = p[k][keep]
        return out

    DetectionValidator.postprocess = postprocess


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "data" / "outputs" / "infra_bench" / "nms_check")
    ap.add_argument("--data", type=Path, default=REPO / "data" / "infra_labels")
    ap.add_argument("--split", type=Path, default=REPO / "data" / "manifests" / "infra_split.json")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(REPO / cfg["weights"]))
    n_images = len(json.loads(a.split.read_text())["val"])
    data_yaml = write_data_yaml(a.data, a.split, a.out / "data", cfg["classes"])
    kw = {"imgsz": cfg["imgsz"], "device": a.device}
    runs = {"no_nms": val_metrics(model, data_yaml, n_images, kw)}
    patch_validator(cfg["nms_iou"])
    runs[f"class_nms_{cfg['nms_iou']}"] = val_metrics(model, data_yaml, n_images, kw)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "nms_check.json").write_text(json.dumps(runs, indent=1) + "\n")
    lines = ["| run | class | mask mAP50 | mask mAP50-95 | box mAP50 | box mAP50-95 |",
             "|---|---|---|---|---|---|"]
    for name, r in runs.items():
        for row in r["rows"]:
            lines.append(f"| {name} | {row['class']} | {row['mask'][2]:.3f} | {row['mask'][3]:.3f}"
                         f" | {row['box'][2]:.3f} | {row['box'][3]:.3f} |")
    (a.out / "nms_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
