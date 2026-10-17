"""Push our code to Kaggle and run kernels there with the kaggle CLI.

The repo is private, so kernels cannot clone it. Instead the ``ml`` package, the
density eval scripts and ``data/samples/`` are uploaded as the private Kaggle
Dataset ``<user>/sentinel-ml-src``, which kernels attach as an input.

    python eval/density_dmcount/kaggle/push.py src -m "what changed"
    python eval/density_dmcount/kaggle/push.py kernel dmcount_smoke
    kaggle kernels status <user>/sentinel-dmcount-smoke
    kaggle kernels output <user>/sentinel-dmcount-smoke -p runs/kaggle/dmcount_smoke

Needs the API token in ``~/.kaggle/access_token`` (or a legacy ``kaggle.json``) and
``KAGGLE_USERNAME`` set to the account name.
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
# Copied into the dataset, relative to the repo root.
SRC_PATHS = ["ml/pyproject.toml", "ml/src", "ml/configs", "eval/density_dmcount", "data/samples"]
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


def push_src(message: str) -> None:
    user = kaggle_user()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "src"
        for rel in SRC_PATHS:
            src, dst = REPO / rel, stage / rel
            if src.is_dir():
                shutil.copytree(src, dst, ignore=SKIP)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        meta = {"title": SRC_SLUG, "id": f"{user}/{SRC_SLUG}", "licenses": [{"name": "other"}]}
        (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        exists = run("kaggle", "datasets", "status", f"{user}/{SRC_SLUG}").returncode == 0
        if exists:
            res = run("kaggle", "datasets", "version", "-p", str(stage), "-m", message,
                      "--dir-mode", "zip")
        else:
            # New datasets are private unless --public is passed.
            res = run("kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip")
        print(res.stdout, res.stderr)
        if res.returncode:
            raise SystemExit(res.returncode)


def push_kernel(name: str) -> None:
    user = kaggle_user()
    src = HERE / name
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / name
        shutil.copytree(src, stage)
        meta_path = stage / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text().replace("{user}", user))
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
    k = sub.add_parser("kernel", help="push and run a kernel folder next to this script")
    k.add_argument("name")
    args = ap.parse_args()
    push_src(args.message) if args.cmd == "src" else push_kernel(args.name)


if __name__ == "__main__":
    main()
