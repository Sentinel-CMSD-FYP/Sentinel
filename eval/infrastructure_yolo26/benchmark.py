"""Benchmark Model C on the held out val images: mAP per class, FPS, GT vs prediction images (step 15).

    python eval/infrastructure_yolo26/benchmark.py --out runs/infra_bench/cpu [--weights W]
        [--imgsz 1280] [--device cpu] [--half] [--frames 300] [--warmup 20] [--overlays]

mAP: Ultralytics val on the ``val`` stems of ``data/manifests/infra_split.json`` (box and mask,
per class). FPS: batch 1, end to end ``predict`` (preprocess, network, NMS, masks) on the val
images held in memory and cycled until ``--frames`` timed frames, after ``--warmup`` untimed ones,
with ``torch.cuda.synchronize`` per frame. Writes ``<out>/bench.json`` and ``bench.md``;
``--overlays`` saves GT (left) next to the prediction at the config ``conf`` (right) per image.
The Kaggle kernel ``kaggle/infra_bench`` runs it on GPU in fp32 and fp16.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ls_to_yolo import draw_labels
from train import CONFIG, REPO, write_data_yaml


def val_metrics(model, data_yaml: Path, n_images: int, kw: dict) -> dict:
    """Box and mask P, R, mAP50, mAP50-95 for all classes and per class."""
    m = model.val(data=str(data_yaml), split="val", batch=1, plots=False, **kw)
    rows = [{"class": "all", "images": n_images, "instances": int(m.nt_per_class.sum()),
             "box": [m.box.mp, m.box.mr, m.box.map50, m.box.map],
             "mask": [m.seg.mp, m.seg.mr, m.seg.map50, m.seg.map]}]
    for i, c in enumerate(m.ap_class_index):
        rows.append({"class": m.names[int(c)], "images": int(m.nt_per_image[c]),
                     "instances": int(m.nt_per_class[c]),
                     "box": list(m.box.class_result(i)), "mask": list(m.seg.class_result(i))})
    for r in rows:
        r["box"] = [round(float(v), 4) for v in r["box"]]
        r["mask"] = [round(float(v), 4) for v in r["mask"]]
    return {"rows": rows, "speed_ms": {k: round(v, 1) for k, v in m.speed.items()}}


def fps(model, images: list[np.ndarray], kw: dict, frames: int, warmup: int) -> dict:
    """Time ``frames`` end to end batch 1 predictions; ms stats, FPS and stage means."""
    cuda = torch.cuda.is_available() and str(kw["device"]) != "cpu"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times, stages = [], {"preprocess": [], "inference": [], "postprocess": []}
    for n in range(warmup + frames):
        t0 = time.perf_counter()
        res = model.predict(images[n % len(images)], verbose=False, **kw)[0]
        if cuda:
            torch.cuda.synchronize()
        if n >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
            for k, v in stages.items():
                v.append(res.speed[k])
    times.sort()
    return {"frames": frames, "warmup": warmup, "mean_ms": round(statistics.fmean(times), 1),
            "median_ms": round(statistics.median(times), 1),
            "p95_ms": round(times[int(0.95 * (len(times) - 1))], 1),
            "fps": round(1000 / statistics.fmean(times), 2),
            "stage_ms": {k: round(statistics.fmean(v), 1) for k, v in stages.items()},
            "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
            "gpu": torch.cuda.get_device_name() if cuda else None}


def save_overlays(model, stems: list[str], images: list[np.ndarray], labels: Path, out: Path,
                  classes: list[str], kw: dict) -> None:
    """GT polygons (left) next to the predicted masks (right), half size, one jpg per image."""
    out.mkdir(parents=True, exist_ok=True)
    for stem, img in zip(stems, images):
        gt = draw_labels(img, (labels / f"{stem}.txt").read_text().splitlines(), classes)
        pred = model.predict(img, verbose=False, **kw)[0].plot()
        pair = cv2.resize(np.hstack([gt, pred]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        for x, text in ((10, "GT"), (pair.shape[1] // 2 + 10, "prediction")):
            cv2.putText(pair, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                        cv2.LINE_AA)
        cv2.imwrite(str(out / f"{stem}.jpg"), pair)
    print(f"overlays -> {out}")


def to_markdown(b: dict) -> str:
    s = b["settings"]
    lines = [(f"# Model C benchmark: {Path(s['weights']).name}, imgsz {s['imgsz']}, "
              f"{'fp16' if s['half'] else 'fp32'}, {b['fps']['gpu'] or 'cpu'}"), "",
             ("| class | images | instances | box P | box R | box mAP50 | box mAP50-95 "
              "| mask P | mask R | mask mAP50 | mask mAP50-95 |"), "|" + "---|" * 11]
    for r in b["val"]["rows"]:
        lines.append(f"| {r['class']} | {r['images']} | {r['instances']} | "
                     + " | ".join(f"{v:.3f}" for v in r["box"] + r["mask"]) + " |")
    f = b["fps"]
    lines += ["", (f"FPS (batch 1, end to end, {f['frames']} frames after {f['warmup']} warmup): "
                   f"**{f['fps']}** (mean {f['mean_ms']} ms, median {f['median_ms']} ms, "
                   f"p95 {f['p95_ms']} ms)"),
              "Stages (ms): " + ", ".join(f"{k} {v}" for k, v in f["stage_ms"].items()),
              f"Peak GPU memory: {f['peak_gpu_mem_mb']} MB" if f["gpu"] else "Device: cpu", ""]
    return "\n".join(lines)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=REPO / "data" / "infra_labels",
                    help="folder with images/, labels/")
    ap.add_argument("--split", type=Path, default=REPO / "data" / "manifests" / "infra_split.json")
    ap.add_argument("--weights", type=Path, default=REPO / cfg["weights"])
    ap.add_argument("--imgsz", type=int, default=cfg["imgsz"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--half", action="store_true", help="fp16 (GPU only)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--overlays", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(a.weights))
    kw = {"imgsz": a.imgsz, "device": a.device, "quantize": 16 if a.half else None}  # 16 = fp16
    stems = json.loads(a.split.read_text())["val"]
    images = [cv2.imread(str(a.data / "images" / f"{s}.jpg")) for s in stems]

    data_yaml = write_data_yaml(a.data, a.split, a.out / "data", cfg["classes"])
    bench = {"settings": {"weights": str(a.weights), "imgsz": a.imgsz, "half": a.half,
                          "device": a.device, "conf_overlays": cfg["conf"]},
             "val": val_metrics(model, data_yaml, len(stems), kw),
             "fps": fps(model, images, {**kw, "conf": cfg["conf"]}, a.frames, a.warmup)}
    (a.out / "bench.json").write_text(json.dumps(bench, indent=2))
    (a.out / "bench.md").write_text(to_markdown(bench))
    print(to_markdown(bench))
    if a.overlays:
        save_overlays(model, stems, images, a.data / "labels", a.out / "overlays", cfg["classes"],
                      {**kw, "conf": cfg["conf"]})


if __name__ == "__main__":
    main()
