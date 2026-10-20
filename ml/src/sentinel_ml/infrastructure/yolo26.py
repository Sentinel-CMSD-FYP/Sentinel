"""YOLO26 infrastructure segmentation (Tier 1, Model C) as a ``polygons`` layer.

``polygons`` is a fourth layer type proposed to Mahid next to grid, points and tracks
(``docs/layers/layers.md`` on main); see ``docs/layers_density_infrastructure.md``. Each frame
entry is a list of objects ``{"cls", "score", "box": [x0, y0, x1, y1], "polygon": [[x, y], ...]}``
in original frame pixels.

    model = InfraModel(InfraConfig.from_yaml("ml/configs/infrastructure_yolo26.yaml"))
    layer = infrastructure_layer(frames, model, "seq011")

YOLO26 is NMS free, which left overlapping duplicate wall masks on one building (step 15), so a
class wise NMS on the boxes runs after the model. fp16 is used on CUDA only (same mAP, step 15).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class InfraConfig:
    weights: str  # fine tuned Model C, e.g. data/weights/yolo26/model_c_best.pt
    classes: tuple[str, ...]  # order = class id
    device: str = "auto"  # cuda if available, else cpu
    imgsz: int = 1280
    conf: float = 0.25
    half: bool = True  # fp16, applied on CUDA only
    nms_iou: float | None = 0.5  # class wise NMS on boxes; None turns it off
    polygon_eps_px: float = 1.0  # polygon simplification tolerance (Douglas Peucker)
    every: int = 125  # run on frames whose id is a multiple of this (5 s at 25 fps)
    max_age_frames: int = 125
    decimals: int = 1  # rounding of pixel coordinates in the JSON

    def __post_init__(self) -> None:
        object.__setattr__(self, "classes", tuple(self.classes))
        if min(self.every, self.max_age_frames) < 1:
            raise ValueError("every and max_age_frames must be at least 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> InfraConfig:
        """Reads the known keys; the shared config also holds training keys (base_weights)."""
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**{f.name: raw[f.name] for f in fields(cls) if f.name in raw})


def class_nms(boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, iou: float) -> np.ndarray:
    """Indices kept by NMS within each class, highest score first."""
    import torch
    from torchvision.ops import batched_nms

    keep = batched_nms(torch.as_tensor(boxes, dtype=torch.float32),
                       torch.as_tensor(scores, dtype=torch.float32),
                       torch.as_tensor(cls, dtype=torch.int64), iou)
    return keep.numpy()


def largest_part(mask: np.ndarray) -> np.ndarray:
    """Outline ``(K, 2)`` of the largest connected part of a binary mask (by area).

    Ultralytics' ``masks.xy`` joins all parts of a mask into one polygon with bridge lines, which
    draws false edges across buildings; one object = its main part is cleaner.
    """
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return np.zeros((0, 2), np.float32)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def simplify_polygon(xy: np.ndarray, eps: float, frame_size: tuple[int, int]) -> np.ndarray | None:
    """Simplified polygon clipped into the frame, or None when fewer than 3 points remain."""
    if len(xy) < 3:
        return None
    w, h = frame_size
    pts = cv2.approxPolyDP(np.asarray(xy, np.float32).reshape(-1, 1, 2), eps, True).reshape(-1, 2)
    if len(pts) < 3:
        return None
    return np.clip(pts, 0, [w - 1, h - 1])


class InfraModel:
    """Frame in, list of infrastructure objects out."""

    def __init__(self, config: InfraConfig) -> None:
        import torch
        from ultralytics import YOLO

        self.config = config
        dev = config.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(config.weights))
        self.half = config.half and dev.startswith("cuda")
        # quantize=16 is fp16 (Ultralytics 8.4 deprecates half=True).
        self._kw = {"imgsz": config.imgsz, "conf": config.conf, "device": dev,
                    "quantize": 16 if self.half else None, "verbose": False}

    def predict(self, frame_bgr: np.ndarray) -> list[dict]:
        import torch
        from ultralytics.utils.ops import scale_coords

        cfg = self.config
        res = self.model.predict(frame_bgr, **self._kw)[0]
        if res.masks is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        order = (class_nms(boxes, scores, cls, cfg.nms_iou) if cfg.nms_iou is not None
                 else np.argsort(-scores))
        frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        masks = res.masks.data[torch.as_tensor(order, dtype=torch.long)].cpu().numpy()
        out = []
        for i, mask in zip(order, masks, strict=True):
            # Masks come at the network's input size; scale_coords maps back to frame pixels.
            xy = scale_coords(mask.shape, largest_part(mask), res.orig_shape, normalize=False)
            poly = simplify_polygon(xy, cfg.polygon_eps_px, frame_size)
            if poly is None:
                continue
            out.append({"cls": cfg.classes[cls[i]], "score": round(float(scores[i]), 3),
                        # float64 first, or float32 noise survives the rounding in the JSON
                        "box": np.round(boxes[i].astype(float), cfg.decimals).tolist(),
                        "polygon": np.round(poly.astype(float), cfg.decimals).tolist()})
        return out


class InfraModule:
    """Model C behind the common interface (``sentinel_ml.pipeline.Module``).

    ``process`` runs the model only on frame ids that are a multiple of ``model.config.every``.
    """

    name = "infrastructure"

    def __init__(self, model: InfraModel) -> None:
        self.model = model
        self._frame_size: list[int] | None = None
        self._frames: dict[str, list] = {}

    def process(self, frame_id: int, frame: np.ndarray) -> None:
        if frame_id % self.model.config.every:
            return
        size = [frame.shape[1], frame.shape[0]]
        if self._frame_size is None:
            self._frame_size = size
        elif size != self._frame_size:
            raise ValueError(f"frame {frame_id} is {size}, earlier frames are {self._frame_size}")
        self._frames[str(frame_id)] = self.model.predict(frame)

    def layer(self, clip_id: str) -> dict:
        cfg = self.model.config
        if self._frame_size is None:
            raise ValueError(f"no frame id is a multiple of every={cfg.every}")
        return {"schema_version": 1, "clip_id": clip_id, "layer": self.name,
                "type": "polygons", "frame_size": self._frame_size,
                "classes": list(cfg.classes), "max_age_frames": cfg.max_age_frames,
                "frames": self._frames}


def infrastructure_layer(frames: Iterable[tuple[int, np.ndarray]], model: InfraModel,
                         clip_id: str) -> dict:
    """Run Model C on ``(frame_id, BGR frame)`` pairs and return the polygons layer as a dict.

    Only frames whose id is a multiple of ``model.config.every`` are run (0 based clip indices).
    """
    module = InfraModule(model)
    for fid, frame in frames:
        module.process(fid, frame)
    return module.layer(clip_id)
