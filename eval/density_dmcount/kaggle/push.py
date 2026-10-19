"""Push our code to Kaggle and run kernels there with the kaggle CLI.

The repo is private, so kernels cannot clone it. Instead the ``ml`` package, the
density eval scripts and ``data/samples/`` are uploaded as the private Kaggle
Dataset ``<user>/sentinel-ml-src``, which kernels attach as an input.

    python eval/density_dmcount/kaggle/push.py src -m "what changed"
    python eval/density_dmcount/kaggle/push.py kernel dmcount_smoke
    python eval/density_dmcount/kaggle/push.py kernel eval/infrastructure_yolo26/kaggle/yolo26_smoke
    python eval/density_dmcount/kaggle/push.py ckpt runs/kaggle/dmcount_train/v4 -m "epoch 2"
    python eval/density_dmcount/kaggle/push.py labels -m "batch 1 labels + batch 2 images"
    kaggle kernels status <user>/sentinel-dmcount-smoke
    kaggle kernels output <user>/sentinel-dmcount-smoke -p runs/kaggle/dmcount_smoke

Needs the API token in ``~/.kaggle/access_token`` (or a legacy ``kaggle.json``) and
``KAGGLE_USERNAME`` set to the account name.

A kernel cannot list its own output as an input (Kaggle rejects it, and such versions sat
queued for hours), so training resumes through the private dataset ``<user>/sentinel-dmcount-ckpt``:
``ckpt`` uploads the ``run/`` folder of a downloaded kernel output. ``optional_dataset_sources``
in a kernel-metadata.json are attached only if they exist.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SRC_SLUG = "sentinel-ml-src"
CKPT_SLUG = "sentinel-dmcount-ckpt"
LABELS_SLUG = "sentinel-infra-labels"
LABELS_DIR = REPO / "data" / "infra_labels"
# Copied into the dataset, relative to the repo root.
SRC_PATHS = ["ml/pyproject.toml", "ml/src", "ml/configs", "eval/density_dmcount",
             "eval/infrastructure_yolo26", "data/samples", "data/manifests"]
SKIP = shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc", ".pytest_cache", "kaggle")


def kaggle_user() -> str:
    """Username from KAGGLE_USERNAME, else from a legacy kaggle.json."""
    if user := os.environ.get("KAGGLE_USERNAME"):
        return user
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.exists() and token.stat().st_size:
        return json.loads(token.read_text())["username"]
    raise SystemExit("set KAGGLE_USERNAME (the API token in ~/.kaggle/access_token has no username)")


def run(*cmd: str) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def dataset_exists(ref: str) -> bool:
    return run("kaggle", "datasets", "status", ref).returncode == 0


def upload_dataset(stage: Path, slug: str, message: str) -> None:
    """Create or version the private dataset ``<user>/<slug>`` from folder ``stage``."""
    user = kaggle_user()
    meta = {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    if dataset_exists(f"{user}/{slug}"):
        res = run("kaggle", "datasets", "version", "-p", str(stage), "-m", message,
                  "--dir-mode", "zip")
    else:
        # New datasets are private unless --public is passed.
        res = run("kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip")
    print(res.stdout, res.stderr)
    if res.returncode:
        raise SystemExit(res.returncode)


def push_src(message: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "src"
        for rel in SRC_PATHS:
            src, dst = REPO / rel, stage / rel
            if src.is_dir():
                shutil.copytree(src, dst, ignore=SKIP)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        upload_dataset(stage, SRC_SLUG, message)


def push_ckpt(output_dir: Path, message: str) -> None:
    """Upload ``<output_dir>/run`` (a downloaded training kernel output) as the ckpt dataset."""
    run_dir = output_dir / "run"
    if not (run_dir / "last.pt").exists():
        raise SystemExit(f"no run/last.pt in {output_dir}")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "ckpt"
        shutil.copytree(run_dir, stage / "run")
        upload_dataset(stage, CKPT_SLUG, message)


def push_labels(message: str) -> None:
    """Upload the Model C images and YOLO labels (``data/infra_labels``) as the labels dataset."""
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "labels"
        for sub, pattern in (("images", "*.jpg"), ("labels", "*.txt")):
            (stage / sub).mkdir(parents=True)
            for p in (LABELS_DIR / sub).glob(pattern):
                shutil.copy2(p, stage / sub / p.name)
        upload_dataset(stage, LABELS_SLUG, message)


def push_kernel(name: str) -> None:
    """Push a kernel folder: a name next to this script, or a path to a folder."""
    user = kaggle_user()
    src = Path(name) if Path(name).is_dir() else HERE / name
    name = src.name
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / name
        shutil.copytree(src, stage)
        meta_path = stage / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text().replace("{user}", user))
        for ref in meta.pop("optional_dataset_sources", []):
            if dataset_exists(ref):
                meta["dataset_sources"].append(ref)
        meta_path.write_text(json.dumps(meta, indent=2))
        res = run("kaggle", "kernels", "push", "-p", str(stage))
        print(res.stdout, res.stderr)
        if res.returncode:
            raise SystemExit(res.returncode)
    print(f"status:  kaggle kernels status {meta['id']}")
    print(f"output:  kaggle kernels output {meta['id']} -p runs/kaggle/{name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("src", help="create or update the private code dataset")
    s.add_argument("-m", "--message", default="update")
    k = sub.add_parser("kernel", help="push and run a kernel folder (name here, or a path)")
    k.add_argument("name")
    c = sub.add_parser("ckpt", help="upload a downloaded training run folder as the ckpt dataset")
    c.add_argument("output_dir", type=Path, help="folder holding run/ (kaggle kernels output)")
    c.add_argument("-m", "--message", default="update")
    lb = sub.add_parser("labels", help="upload Model C images + YOLO labels as a dataset")
    lb.add_argument("-m", "--message", default="update")
    args = ap.parse_args()
    if args.cmd == "src":
        push_src(args.message)
    elif args.cmd == "labels":
        push_labels(args.message)
    elif args.cmd == "ckpt":
        push_ckpt(args.output_dir, args.message)
    else:
        push_kernel(args.name)


if __name__ == "__main__":
    main()
