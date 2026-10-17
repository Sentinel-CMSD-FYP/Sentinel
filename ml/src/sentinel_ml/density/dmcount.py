"""DM-Count crowd density estimation (Tier 1, Model A).

The network is ported from https://github.com/cvlab-stonybrook/DM-Count
(commit cc5f213, MIT License, Copyright (c) 2020 CVLab@StonyBrook). Layer names
match the original so the released ``model_*.pth`` files load with
``strict=True``. Only inference lives here; training is added in step 8.

The model maps an RGB frame to a density map at 1/8 of the input resolution.
The map sums to the predicted head count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

# VGG19 configuration "E" without the last max pool, as in DM-Count.
VGG19_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
             512, 512, 512, 512, "M", 512, 512, 512, 512]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DOWNSAMPLE = 8  # input pixels per density map cell, per side


def _vgg_features(cfg: list[int | str]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)


class DMCountNet(nn.Module):
    """VGG19 backbone + regression head. Returns ``(density, normed_density)``."""

    def __init__(self) -> None:
        super().__init__()
        self.features = _vgg_features(VGG19_CFG)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        # Same as the original F.upsample_bilinear (align_corners=True).
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        mu = self.density_layer(self.reg_layer(x))
        mu_sum = mu.flatten(1).sum(1).view(-1, 1, 1, 1)
        return mu, mu / (mu_sum + 1e-6)


@dataclass(frozen=True)
class DMCountConfig:
    # None builds a randomly initialised net (tests only).
    weights: str | None = None
    # "auto" picks CUDA when available, else CPU.
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> DMCountConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DensityMap:
    """Predicted density for one frame, at 1/``DOWNSAMPLE`` of its resolution."""

    density: np.ndarray  # (h, w) float32, heads per cell
    frame_shape: tuple[int, int]  # (H, W) of the input frame

    @property
    def count(self) -> float:
        return float(self.density.sum())


class DMCount:
    def __init__(self, config: DMCountConfig | None = None) -> None:
        self.config = config or DMCountConfig()
        dev = self.config.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        self.model = DMCountNet()
        if self.config.weights is not None:
            state = torch.load(self.config.weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 ``(H, W, 3)`` to a normalised ``(1, 3, H, W)`` tensor."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) BGR frame, got {frame_bgr.shape}")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> DensityMap:
        mu, _ = self.model(self.preprocess(frame_bgr))
        density = mu[0, 0].float().cpu().numpy()
        return DensityMap(density=density, frame_shape=frame_bgr.shape[:2])


def overlay(frame_bgr: np.ndarray, dm: DensityMap, alpha: float = 0.5) -> np.ndarray:
    """Jet heat map of ``dm`` blended over the frame, with the count drawn on top."""
    h, w = frame_bgr.shape[:2]
    d = dm.density
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    out = cv2.addWeighted(frame_bgr, 1.0 - alpha, heat, alpha, 0.0)
    cv2.putText(out, f"count {dm.count:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return out
