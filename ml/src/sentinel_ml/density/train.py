"""Fine tune DM-Count on DroneCrowd (step 8, Tier 1 Model A).

Losses, Sinkhorn and the crop pipeline are ported from https://github.com/cvlab-stonybrook/DM-Count
(commit cc5f213, MIT License, Copyright (c) 2020 CVLab@StonyBrook): ``losses/ot_loss.py``,
``losses/bregman_pytorch.py`` (plain Sinkhorn Knopp only), ``datasets/crowd.py`` and
``train_helper.py``. Loss = wot * OT + count L1 + wtv * TV, as in the paper.

Checkpoints are selected on our held out val clips only (never on test). Every epoch
writes ``<out>/last.pt`` (model, optimiser, epoch, best val), so a run that finds
``last.pt`` in its out folder resumes from it (Kaggle sessions stop at 12 h).
``<out>/best.pth`` is a plain state dict that :class:`DMCount` loads directly.

    python -m sentinel_ml.density.train --out runs/dmcount_train --epochs 2
    python -m sentinel_ml.density.train --out runs/tmp --epochs 1 --limit 4 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sentinel_ml.density.data import load_split_from_config
from sentinel_ml.density.dmcount import DOWNSAMPLE, IMAGENET_MEAN, IMAGENET_STD, DMCountNet
from sentinel_ml.density.targets import clip_points

REPO = Path(__file__).resolve().parents[4]
TRAIN_CONFIG = REPO / "ml" / "configs" / "density_train.yaml"
DATA_CONFIG = REPO / "ml" / "configs" / "density_data.yaml"
M_EPS = 1e-16
EARLY_LOG_STEPS = (1, 5, 10)  # timing shows up before the first log_every
cv2.setNumThreads(0)  # OpenCV threads inside forked DataLoader workers can deadlock
HISTORY_FIELDS = ["epoch", "loss", "wd", "count", "tv", "train_mae", "val_mae", "val_rmse",
                  "val_mean_rel_pct", "train_s", "val_s"]


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) to an ImageNet normalised (3, H, W) float tensor."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def discrete_map(points: np.ndarray, size: int, cell: int) -> np.ndarray:
    """(size // cell, size // cell) head counts per cell; a head at (x, y) sits in pixel
    (round(x), round(y)), clamped into the crop, as DM-Count's ``gen_discrete_map``."""
    n = size // cell
    p = np.clip(np.round(points).astype(int), 0, size - 1) // cell
    out = np.zeros((n, n), np.float32)
    np.add.at(out, (p[:, 1], p[:, 0]), 1.0)
    return out


class CropDataset(Dataset):
    """Random ``crop`` x ``crop`` crops with a random horizontal flip, from frames that
    give ``.image`` (RGB uint8) and ``.points`` ((N, 2) x, y in pixels) per index.
    Returns (image tensor, crop points (M, 2), discrete map (1, crop/8, crop/8))."""

    def __init__(self, frames, crop: int, cell: int = DOWNSAMPLE) -> None:
        if crop % cell:
            raise ValueError(f"crop {crop} is not a multiple of {cell}")
        self.frames, self.crop, self.cell = frames, crop, cell

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        s = self.frames[i]
        h, w = s.image.shape[:2]
        c = self.crop
        if min(h, w) < c:
            raise ValueError(f"frame {w}x{h} is smaller than crop {c}")
        top, left = random.randint(0, h - c), random.randint(0, w - c)
        img = s.image[top:top + c, left:left + c]
        pts = clip_points(s.points, w, h) - np.array([left, top], np.float32)
        pts = pts[(pts >= 0).all(1) & (pts <= c).all(1)]
        if random.random() < 0.5:
            img = img[:, ::-1]
            pts[:, 0] = c - pts[:, 0]  # Crowd_qnrf convention, stays in [0, c]
        return (to_tensor(img), torch.from_numpy(pts.copy()),
                torch.from_numpy(discrete_map(pts, c, self.cell))[None])


def collate(batch):
    images, points, maps = zip(*batch, strict=True)
    return torch.stack(images), list(points), torch.stack(maps)


class FullFrames(Dataset):
    """Whole frames for validation: (image tensor, GT count, clip, frame_id)."""

    def __init__(self, frames) -> None:
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        s = self.frames[i]
        return to_tensor(s.image), len(s.points), s.ref.clip, s.ref.frame_id


def sinkhorn(a: torch.Tensor, b: torch.Tensor, cost: torch.Tensor, reg: float,
             max_iter: int, stop_thr: float = 1e-9, eval_freq: int = 10):
    """Entropic OT between ``a`` (na,) and ``b`` (nb,) with cost (na, nb).
    Returns (transport plan, beta = dual potential of ``b``)."""
    K = torch.exp(-cost / reg)
    u = torch.ones_like(a) / len(a)
    v = torch.ones_like(b) / len(b)
    for it in range(1, max_iter + 1):
        u_prev, v_prev = u, v
        v = b / (u @ K + M_EPS)
        u = a / (K @ v + M_EPS)
        if not (torch.isfinite(u).all() and torch.isfinite(v).all()):
            u, v = u_prev, v_prev
            break
        if it % eval_freq == 0 and ((b - (u @ K) * v) ** 2).sum().item() < stop_thr:
            break
    return u[:, None] * K * v[None, :], reg * torch.log(v + M_EPS)


class OTLoss(nn.Module):
    """DM-Count's OT loss over a fixed crop size. ``forward`` returns (surrogate loss, summed
    Wasserstein distance). The surrogate <grad, density> is 0 by construction; only its
    gradient (the OT gradient w.r.t. the density) matters, so log the distance instead."""

    def __init__(self, crop: int, cell: int = DOWNSAMPLE, reg: float = 10.0,
                 iters: int = 100) -> None:
        super().__init__()
        self.reg, self.iters = reg, iters
        # Cell centres in crop pixels.
        self.register_buffer("cood", torch.arange(0, crop, cell, dtype=torch.float32) + cell / 2)

    def forward(self, normed: torch.Tensor, density: torch.Tensor, points: list[torch.Tensor]):
        n = len(self.cood)
        if normed.shape[-2:] != (n, n):
            raise ValueError(f"density {tuple(normed.shape)} does not match the {n}x{n} grid")
        loss = density.new_zeros(())
        wd = 0.0
        c = self.cood[None]
        for i, p in enumerate(points):
            if len(p) == 0:
                continue
            x, y = p[:, :1], p[:, 1:]
            dx = (x - c) ** 2  # (#gt, n)
            dy = (y - c) ** 2
            dis = (dy[:, :, None] + dx[:, None, :]).reshape(len(p), -1)  # (#gt, n * n), row major
            src = normed[i, 0].reshape(-1).detach()
            tgt = torch.full((len(p),), 1.0 / len(p), device=p.device)
            plan, beta = sinkhorn(tgt, src, dis, self.reg, self.iters)
            dens = density[i, 0].reshape(-1).detach()
            total = dens.sum()
            sq = total * total + 1e-8
            grad = beta * total / sq - (dens * beta).sum() / sq
            loss = loss + (density[i, 0] * grad.view(n, n)).sum()
            wd += (dis * plan).sum().item()
        return loss, wd


def train_epoch(model, loader, opt, ot_loss: OTLoss, cfg: dict, device, log_every: int) -> dict:
    model.train()
    sums = dict.fromkeys(["loss", "wd", "count", "tv", "train_mae"], 0.0)
    seen, wait_s = 0, 0.0
    t0 = t_fetch = time.perf_counter()
    for step, (images, points, maps) in enumerate(loader, 1):
        wait_s += time.perf_counter() - t_fetch  # time spent waiting for the loader
        images, maps = images.to(device, non_blocking=True), maps.to(device)
        points = [p.to(device) for p in points]
        gt = torch.tensor([len(p) for p in points], dtype=torch.float32, device=device)
        density, normed = model(images)
        ot, wd = ot_loss(normed, density, points)
        pred = density.sum((1, 2, 3))
        count = (pred - gt).abs().mean()
        tv = ((normed - maps / (gt.view(-1, 1, 1, 1) + 1e-6)).abs().sum((1, 2, 3)) * gt).mean()
        loss = cfg["wot"] * ot + count + cfg["wtv"] * tv
        opt.zero_grad()
        loss.backward()
        opt.step()

        b = len(points)
        seen += b
        sums["wd"] += wd
        for k, v in (("loss", loss), ("count", count), ("tv", cfg["wtv"] * tv),
                     ("train_mae", (pred - gt).abs().mean())):
            sums[k] += v.item() * b
        if step % log_every == 0 or step == len(loader) or step in EARLY_LOG_STEPS:
            avg = {k: v / seen for k, v in sums.items()}
            el = time.perf_counter() - t0
            print(f"  step {step}/{len(loader)}  loss {avg['loss']:.3f}  wd {avg['wd']:.1f}  "
                  f"count {avg['count']:.2f}  tv {avg['tv']:.3f}  mae {avg['train_mae']:.2f}  "
                  f"{el / step:.2f} s/step (data wait {wait_s / step:.2f})  "
                  f"epoch eta {el / step * (len(loader) - step) / 60:.0f} min", flush=True)
        t_fetch = time.perf_counter()
    return {k: v / max(seen, 1) for k, v in sums.items()}


@torch.inference_mode()
def validate(model, loader, device) -> tuple[dict, list[dict]]:
    """Full frame counts on val: (metrics, per frame rows clip,frame_id,pred,gt)."""
    model.eval()
    rows = []
    for image, gt, clip, frame_id in loader:
        density, _ = model(image.to(device))
        rows.append({"clip": int(clip), "frame_id": int(frame_id),
                     "pred": density.sum().item(), "gt": int(gt)})
    err = np.array([r["pred"] - r["gt"] for r in rows])
    gt = np.array([r["gt"] for r in rows], dtype=np.float64)
    return {"val_mae": float(np.abs(err).mean()), "val_rmse": float(np.sqrt((err ** 2).mean())),
            "val_mean_rel_pct": float((np.abs(err) / np.maximum(gt, 1)).mean() * 100)}, rows


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def out_of_patience(epoch: int, best_epoch: int, patience: int) -> bool:
    """True when ``patience`` (> 0) epochs have passed since the best val MAE."""
    return patience > 0 and epoch - best_epoch >= patience


def train(cfg: dict, data_cfg: dict, out: Path, epochs: int, root: Path | None = None,
          init_weights: Path | None = None, device: str = "auto", limit: int | None = None,
          time_limit_h: float | None = None) -> None:
    """Train up to epoch ``epochs`` (1 based, total), resuming from ``out/last.pt`` if there."""
    t_start = time.time()
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else "cpu" if device == "auto" else device)
    torch.backends.cudnn.benchmark = True
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    train_ds = load_split_from_config(data_cfg, "train", root=root, repo=REPO)
    val_ds = load_split_from_config(data_cfg, "val", root=root, repo=REPO)
    if limit:
        train_ds.refs, val_ds.refs = train_ds.refs[:limit], val_ds.refs[:limit]
    workers = int(cfg["num_workers"])
    train_dl = DataLoader(CropDataset(train_ds, int(cfg["crop_size"])),
                          batch_size=int(cfg["batch_size"]), shuffle=True,
                          num_workers=workers, collate_fn=collate,
                          pin_memory=dev.type == "cuda", persistent_workers=workers > 0,
                          timeout=600 if workers else 0)  # a stuck worker raises, not hangs
    val_dl = DataLoader(FullFrames(val_ds), batch_size=1, num_workers=workers,
                        timeout=600 if workers else 0)

    model = DMCountNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                           weight_decay=float(cfg["weight_decay"]))
    ot_loss = OTLoss(int(cfg["crop_size"]), DOWNSAMPLE, float(cfg["ot_reg"]),
                     int(cfg["ot_iters"])).to(dev)

    out.mkdir(parents=True, exist_ok=True)
    last, history = out / "last.pt", out / "history.csv"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from {last}: epoch {ck['epoch']} done, best val MAE "
              f"{best['val_mae']:.2f} at epoch {best['epoch']}", flush=True)
    else:
        weights = init_weights or REPO / cfg["init_weights"]
        model.load_state_dict(torch.load(weights, map_location=dev, weights_only=True), strict=True)
        start, best = 1, {"epoch": 0, "val_mae": float("inf")}
        print(f"initialised from {weights}", flush=True)
    if not history.exists():
        with history.open("w", newline="") as f:
            csv.writer(f).writerow(HISTORY_FIELDS)

    print(f"device {dev}, train {len(train_ds)} frames ({len(train_dl)} steps), "
          f"val {len(val_ds)} frames, epochs {start}..{epochs}, ready after "
          f"{time.time() - t_start:.0f} s", flush=True)
    epoch_s = 0.0
    for epoch in range(start, epochs + 1):
        if time_limit_h and time.time() - t_start + epoch_s > time_limit_h * 3600:
            print(f"stopping before epoch {epoch}: time limit {time_limit_h} h", flush=True)
            break
        t0 = time.time()
        print(f"epoch {epoch}/{epochs}", flush=True)
        stats = train_epoch(model, train_dl, opt, ot_loss, cfg, dev, int(cfg["log_every"]))
        t1 = time.time()
        val, rows = validate(model, val_dl, dev)
        t2 = time.time()
        stats.update(val, epoch=epoch, train_s=round(t1 - t0), val_s=round(t2 - t1))
        epoch_s = t2 - t0
        print(f"epoch {epoch} done: loss {stats['loss']:.3f}, train MAE {stats['train_mae']:.2f}, "
              f"val MAE {val['val_mae']:.2f}, RMSE {val['val_rmse']:.2f}, "
              f"mean rel {val['val_mean_rel_pct']:.1f}%, {t1 - t0:.0f} s + {t2 - t1:.0f} s",
              flush=True)

        if val["val_mae"] < best["val_mae"]:
            best = {"epoch": epoch, **val}
            save_atomic(model.state_dict(), out / "best.pth")
            with (out / "best_val_preds.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["clip", "frame_id", "pred", "gt"])
                w.writeheader()
                w.writerows({**r, "pred": f"{r['pred']:.3f}"} for r in rows)
            print(f"  new best (epoch {epoch})", flush=True)
        with history.open("a", newline="") as f:
            csv.writer(f).writerow([f"{stats[k]:.4f}" if isinstance(stats[k], float) else stats[k]
                                    for k in HISTORY_FIELDS])
        save_atomic({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                     "best": best, "config": cfg}, last)
        if out_of_patience(epoch, best["epoch"], int(cfg.get("patience", 0))):
            print(f"early stop after epoch {epoch}: no val MAE gain for {cfg['patience']} epochs",
                  flush=True)
            break
    print(f"best val MAE {best['val_mae']:.2f} at epoch {best['epoch']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="run folder (resumes from last.pt)")
    ap.add_argument("--epochs", type=int, required=True, help="train up to this epoch (total)")
    ap.add_argument("--config", type=Path, default=TRAIN_CONFIG)
    ap.add_argument("--data-config", type=Path, default=DATA_CONFIG)
    ap.add_argument("--root", type=Path, help="dataset root, overrides the data config")
    ap.add_argument("--init-weights", type=Path, help="overrides init_weights in the config")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, help="first N train and val frames only (quick checks)")
    ap.add_argument("--time-limit-h", type=float, help="no new epoch past this many hours")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train(cfg, yaml.safe_load(args.data_config.read_text()), args.out, args.epochs, args.root,
          args.init_weights, args.device, args.limit,
          args.time_limit_h if args.time_limit_h is not None else cfg.get("time_limit_h"))


if __name__ == "__main__":
    main()
