"""Tests for sentinel_ml.density.train (crops, discrete maps, Sinkhorn, OT loss, resume)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sentinel_ml.density.train import (
    CropDataset,
    OTLoss,
    collate,
    discrete_map,
    sinkhorn,
    train_epoch,
)


def frame(points, h=64, w=96):
    rng = np.random.default_rng(0)
    return SimpleNamespace(image=rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
                           points=np.asarray(points, np.float32).reshape(-1, 2))


def test_discrete_map_rounds_and_clamps_like_dmcount():
    m = discrete_map(np.array([[0.4, 0.4], [7.6, 0.0], [31.9, 31.9], [10.0, 20.0]]), 32, 8)
    assert m.shape == (4, 4) and m.sum() == 4
    assert m[0, 0] == 1 and m[0, 1] == 1  # 7.6 rounds to pixel 8, the next cell
    assert m[3, 3] == 1 and m[2, 1] == 1  # 31.9 rounds to 32, clamped to 31


def test_crop_keeps_only_points_inside_and_map_matches():
    pts = np.random.default_rng(1).uniform([0, 0], [96, 64], (50, 2))
    ds = CropDataset([frame(pts)], crop=32, cell=8)
    for _ in range(20):
        img, p, m = ds[0]
        assert img.shape == (3, 32, 32) and m.shape == (1, 4, 4)
        assert ((p >= 0) & (p <= 32)).all()
        assert m.sum().item() == len(p)


def test_crop_rejects_small_frames():
    with pytest.raises(ValueError):
        CropDataset([frame([], h=16, w=16)], crop=32)[0]


def test_sinkhorn_meets_both_marginals():
    a = torch.tensor([0.5, 0.5])
    b = torch.tensor([0.25, 0.25, 0.25, 0.25])
    cost = torch.tensor([[0.0, 1, 4, 9], [9, 4, 1, 0]])
    plan, beta = sinkhorn(a, b, cost, reg=1.0, max_iter=500)
    torch.testing.assert_close(plan.sum(1), a, atol=1e-4, rtol=0)
    torch.testing.assert_close(plan.sum(0), b, atol=1e-4, rtol=0)
    assert beta.shape == b.shape


def test_ot_loss_prefers_density_on_the_points():
    loss_fn = OTLoss(crop=32, cell=8, reg=10.0, iters=100)
    pts = [torch.tensor([[4.0, 4.0], [28.0, 28.0]])]
    near = torch.full((1, 1, 4, 4), 1e-3)
    near[0, 0, 0, 0] = near[0, 0, 3, 3] = 1.0
    far = torch.full((1, 1, 4, 4), 1e-3)
    far[0, 0, 0, 3] = far[0, 0, 3, 0] = 1.0
    _, wd_near = loss_fn(near / near.sum(), near, pts)
    _, wd_far = loss_fn(far / far.sum(), far, pts)
    assert wd_near < wd_far


def test_ot_loss_skips_empty_images_and_backprops():
    loss_fn = OTLoss(crop=32, cell=8)
    d = torch.rand(2, 1, 4, 4, requires_grad=True)
    loss, _ = loss_fn(d / d.sum((1, 2, 3), keepdim=True), d, [torch.zeros(0, 2),
                                                              torch.tensor([[5.0, 9.0]])])
    loss.backward()
    assert torch.isfinite(loss) and d.grad[0].abs().sum() == 0 and d.grad[1].abs().sum() > 0


class TinyNet(torch.nn.Module):
    """Stand in for DMCountNet: 1/8 density + its normalised copy."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 8, stride=8)

    def forward(self, x):
        mu = torch.relu(self.conv(x)) + 1e-3
        return mu, mu / mu.sum((1, 2, 3), keepdim=True)


def test_train_epoch_runs_and_returns_losses():
    pts = np.random.default_rng(2).uniform([0, 0], [96, 64], (30, 2))
    loader = torch.utils.data.DataLoader(CropDataset([frame(pts)] * 4, crop=32), batch_size=2,
                                         collate_fn=collate)
    model = TinyNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = {"wot": 0.1, "wtv": 0.01}
    stats = train_epoch(model, loader, opt, OTLoss(32, 8), cfg, torch.device("cpu"), 1)
    assert set(stats) == {"loss", "wd", "count", "tv", "train_mae"}
    assert all(np.isfinite(v) for v in stats.values())


def test_repo_root_points_at_configs():
    from sentinel_ml.density.train import TRAIN_CONFIG
    assert Path(TRAIN_CONFIG).exists()
