"""Draft Model C polygons with SAM 2.1 for human review in Label Studio.

    python eval/infrastructure_yolo26/sam_draft.py tiles img001150 ...
    python eval/infrastructure_yolo26/sam_draft.py run data/infra_labels/prompts/batch1.yaml

``tiles`` saves each image as 4 overlapping quadrant crops with a pixel grid (full res
coordinates every 100 px), so prompts can be read off by eye. ``run`` reads a prompts file
(image stem -> list of objects) and writes Label Studio prediction results to
``data/infra_labels/images/_drafts/<name>.json`` (served by Label Studio's local files, so
the page can import them) plus overlays in ``data/infra_labels/drafts_overlay/``.

Each object is one of
    {c: wall, box: [x1, y1, x2, y2]}         SAM mask from the box, kept inside the box
    {c: barrier, line: [[x, y], ...], w: 6}  open polyline drawn w px wide (thin railings);
                                             a closed loop would fill its inside, so split it
    {c: obstruction, poly: [[x, y], ...]}    polygon as given
    {c: wall, rect: [x1, y1, x2, y2]}        rectangle as given
Coordinates are full resolution pixels. Drafts are never final labels: a person reviews
and submits each image (LABELLING.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
LABEL_DIR = REPO / "data" / "infra_labels"
WEIGHTS = REPO / "data" / "weights" / "sam" / "sam2.1_b.pt"
COLOURS = {"gate_exit": (83, 200, 0), "barrier": (0, 171, 255), "wall": (255, 98, 41),
           "obstruction": (0, 0, 213)}
BOX_MARGIN = 8     # px a SAM mask may spill outside its box
SIMPLIFY_PX = 1.5  # polygon simplification tolerance


def make_tiles(stems: list[str], images: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        img = cv2.imread(str(images / f"{stem}.jpg"))
        h, w = img.shape[:2]
        for x in range(0, w, 100):
            cv2.line(img, (x, 0), (x, h), (255, 255, 255) if x % 500 else (0, 255, 255), 1)
        for y in range(0, h, 100):
            cv2.line(img, (0, y), (w, y), (255, 255, 255) if y % 500 else (0, 255, 255), 1)
        for x in range(100, w, 200):
            for y in range(100, h, 200):
                cv2.putText(img, f"{x},{y}", (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{stem}_full.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        tw, th = w * 3 // 5, h * 3 // 5  # quadrants overlap by a fifth
        for name, (x0, y0) in {"tl": (0, 0), "tr": (w - tw, 0), "bl": (0, h - th),
                               "br": (w - tw, h - th)}.items():
            cv2.imwrite(str(out / f"{stem}_{name}.jpg"), img[y0:y0 + th, x0:x0 + tw],
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"tiles -> {out}")


def mask_polygon(mask: np.ndarray) -> np.ndarray | None:
    """Largest outer contour of a binary mask, simplified, as (N, 2) int points."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    pts = cv2.approxPolyDP(max(contours, key=cv2.contourArea), SIMPLIFY_PX, True)[:, 0]
    return pts if len(pts) >= 3 else None


def line_polygon(line: list, width: int, shape: tuple[int, int]) -> np.ndarray | None:
    mask = np.zeros(shape, np.uint8)
    cv2.polylines(mask, [np.array(line, np.int32)], False, 1, max(int(width), 2))
    return mask_polygon(mask)


def ls_result(pts: np.ndarray, label: str, w: int, h: int) -> dict:
    return {"from_name": "label", "to_name": "image", "type": "polygonlabels",
            "original_width": w, "original_height": h, "image_rotation": 0,
            "value": {"points": [[round(float(x) / w * 100, 4), round(float(y) / h * 100, 4)]
                                 for x, y in pts],
                      "polygonlabels": [label], "closed": True}}


def run(prompts_path: Path, images: Path) -> None:
    from ultralytics import SAM

    prompts = yaml.safe_load(prompts_path.read_text(encoding="utf-8-sig"))
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    model = SAM(str(WEIGHTS))
    drafts, overlay_dir = {}, LABEL_DIR / "drafts_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for stem, objects in prompts.items():
        img = cv2.imread(str(images / f"{stem}.jpg"))
        h, w = img.shape[:2]
        boxes = [o["box"] for o in objects if "box" in o]
        masks = []
        if boxes:
            res = model(img, bboxes=boxes, verbose=False)[0]
            masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                     for m in res.masks.data.cpu().numpy()]
        results, polys, k = [], [], 0
        for o in objects:
            if "box" in o:
                x1, y1, x2, y2 = o["box"]
                keep = np.zeros((h, w), np.uint8)
                keep[max(y1 - BOX_MARGIN, 0):y2 + BOX_MARGIN,
                     max(x1 - BOX_MARGIN, 0):x2 + BOX_MARGIN] = 1
                pts = mask_polygon(masks[k] & keep)
                k += 1
            elif "line" in o:
                pts = line_polygon(o["line"], o.get("w", 6), (h, w))
            elif "rect" in o:
                x1, y1, x2, y2 = o["rect"]
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
            else:
                pts = np.array(o["poly"], np.int32)
            if pts is None:
                print(f"  {stem}: empty mask for {o}")
                continue
            results.append(ls_result(pts, o["c"], w, h))
            polys.append((o["c"], pts))
        drafts[f"{stem}.jpg"] = results
        fill = img.copy()
        for label, pts in polys:
            cv2.fillPoly(fill, [pts], COLOURS[label])
            cv2.polylines(img, [pts], True, COLOURS[label], 2, cv2.LINE_AA)
        cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), cv2.addWeighted(fill, 0.35, img, 0.65, 0))
        print(f"{stem}: {len(results)} polygons")
    out = images / "_drafts" / f"{prompts_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drafts))
    print(f"drafts -> {out}, overlays -> {overlay_dir}")


def vehicles(export_path: Path, images: Path, extra: Path | None = None,
             min_cover: float = 0.5) -> None:
    """Draft `obstruction` polygons for vehicles the labels miss: COCO model detections plus
    SAM masks from hand boxes in ``extra`` (image stem -> list of [x1, y1, x2, y2]).

    A vehicle whose mask is already covered (``min_cover``) by an obstruction polygon of the
    newest annotation, or by one added before it, is skipped. Output:
    ``_drafts/vehicles.json`` + overlays."""
    from ultralytics import SAM, YOLO

    cfg = yaml.safe_load((REPO / "ml" / "configs" / "infrastructure_yolo26.yaml").read_text())
    model = YOLO(str(REPO / cfg["base_weights"]))
    boxes = yaml.safe_load(extra.read_text(encoding="utf-8-sig")) if extra else {}
    sam = SAM(str(WEIGHTS)) if boxes else None
    # bicycle, car, motorcycle, bus, truck; seen from straight above COCO calls cars "cell phone"
    coco_vehicles = [1, 2, 3, 5, 7, 67]
    drafts, overlay_dir = {}, LABEL_DIR / "vehicles_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for task in json.loads(export_path.read_text(encoding="utf-8")):
        name = task["data"]["image"].replace("%5C", "/").rsplit("/", 1)[-1]
        img = cv2.imread(str(images / name))
        h, w = img.shape[:2]
        have = np.zeros((h, w), np.uint8)
        ann = max(task["annotations"], key=lambda a: a.get("updated_at") or "")
        for r in ann["result"]:
            if r["value"]["polygonlabels"] == ["obstruction"]:
                pts = (np.array(r["value"]["points"]) / 100 * (w, h)).round().astype(np.int32)
                cv2.fillPoly(have, [pts], 1)
        res = model(img, imgsz=1920, conf=0.2, classes=coco_vehicles, retina_masks=True,
                    verbose=False)[0]
        masks = list(res.masks.data.cpu().numpy()) if res.masks is not None else []
        stem_boxes = boxes.get(Path(name).stem) or []
        if stem_boxes:
            sam_res = sam(img, bboxes=stem_boxes, verbose=False)[0]
            for b, m in zip(stem_boxes, sam_res.masks.data.cpu().numpy()):
                keep = np.zeros(m.shape, np.uint8)
                keep[max(b[1] - BOX_MARGIN, 0):b[3] + BOX_MARGIN,
                     max(b[0] - BOX_MARGIN, 0):b[2] + BOX_MARGIN] = 1
                masks.append(m.astype(np.uint8) & keep)
        results = []
        for m in masks:
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            if m.sum() == 0 or (m & have).sum() / m.sum() >= min_cover:
                continue
            pts = mask_polygon(m)
            if pts is not None and max(np.ptp(pts, axis=0)) >= 15:  # LABELLING.md min size
                results.append(ls_result(pts, "obstruction", w, h))
                cv2.fillPoly(have, [pts], 1)
                cv2.polylines(img, [pts], True, COLOURS["obstruction"], 2, cv2.LINE_AA)
        drafts[name] = results
        cv2.imwrite(str(overlay_dir / name), img)
        print(f"{name}: {len(results)} new vehicles")
    out = images / "_drafts" / "vehicles.json"
    out.write_text(json.dumps(drafts))
    print(f"drafts -> {out}, overlays -> {overlay_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["tiles", "run", "vehicles"])
    ap.add_argument("args", nargs="+",
                    help="tiles: image stems; run: prompts yaml; vehicles: LS export json")
    ap.add_argument("--images", type=Path, default=LABEL_DIR / "images")
    ap.add_argument("--extra", type=Path, help="vehicles: yaml of hand boxes per image stem")
    a = ap.parse_args()
    if a.cmd == "tiles":
        make_tiles(a.args, a.images, LABEL_DIR / "tiles")
    elif a.cmd == "vehicles":
        vehicles(Path(a.args[0]), a.images, a.extra)
    else:
        run(Path(a.args[0]), a.images)


if __name__ == "__main__":
    main()
