"""Put every Tier 1 model file where the configs in ml/configs/ expect it, checked by md5.

* DMCount ``finetune_e4.pth`` and Model C ``model_c_best.pt``: assets of the GitHub release
  ``TAG`` of our (private) repo, downloaded with the GitHub CLI (``gh auth login`` first).
* Model B: the authors' code (``up-count/uav-dot`` at ``UAV_DOT_COMMIT``) cloned into
  ``ml/third_party/uav-dot`` and their checkpoint from Google Drive (``pip install gdown``).
  Its MiT b2 encoder (94 MB) is downloaded by torch on the first build of the points module.
* D (flow), E (tracks) and geometry have no weights.

    python eval/pipeline/fetch_weights.py            # fetch what is missing
    python eval/pipeline/fetch_weights.py --check    # only report what is there

A file that is present with the right md5 is left alone, so running it again is safe.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASE_REPO = "Sentinel-CMSD-FYP/Sentinel"
TAG = "tier1-models"
# (release asset, path in the repo, md5)
RELEASE = [
    ("finetune_e4.pth", "data/weights/dmcount/finetune_e4.pth",
     "725694bafc292950a0c62ac43d855d22"),
    ("model_c_best.pt", "data/weights/yolo26/model_c_best.pt",
     "12213bdea884c2f7413ea53456be259d"),
]
UAV_DOT_URL = "https://github.com/up-count/uav-dot"
UAV_DOT_COMMIT = "a6d3664"
UAV_DOT_DIR = "ml/third_party/uav-dot"
B_CKPT = ("ml/third_party/uav-dot/checkpoints/dot_pd_dronecrowd_51.00.ckpt",
          "1wYa01jGYfrAun3SfxuWzcKAni3hzBOMV", "fcf6762ad393717741ffc7e6c467f021")


def md5(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def ok(path: Path, want: str) -> bool:
    return path.is_file() and md5(path) == want


def run(*cmd: str) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_release(asset: str, dest: Path, want: str, tag: str) -> None:
    if shutil.which("gh") is None:
        sys.exit("the GitHub CLI (gh) is needed: https://cli.github.com, then gh auth login")
    with tempfile.TemporaryDirectory() as tmp:
        run("gh", "release", "download", tag, "-R", RELEASE_REPO, "-p", asset, "-D", tmp)
        got = Path(tmp) / asset
        if md5(got) != want:
            sys.exit(f"{asset}: md5 {md5(got)} is not {want}; the release holds another file")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(got, dest)


def fetch_uav_dot() -> None:
    code = REPO / UAV_DOT_DIR
    if not (code / ".git").is_dir() and not code.is_dir():
        run("git", "clone", UAV_DOT_URL, str(code))
    if (code / ".git").is_dir():
        run("git", "-C", str(code), "checkout", "--quiet", UAV_DOT_COMMIT)
    path, drive_id, want = B_CKPT
    dest = REPO / path
    if ok(dest, want):
        return
    try:
        import gdown
    except ImportError:
        sys.exit("pip install gdown (downloads Model B's checkpoint from Google Drive)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=drive_id, output=str(dest), quiet=False)
    if not ok(dest, want):
        sys.exit(f"{dest}: md5 does not match {want}; delete it and try again")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only report, download nothing")
    ap.add_argument("--tag", default=TAG, help=f"release to download from (default {TAG})")
    a = ap.parse_args()

    files = [(REPO / dest, want) for _, dest, want in RELEASE]
    files.append((REPO / B_CKPT[0], B_CKPT[2]))
    if a.check:
        missing = 0
        for path, want in files:
            state = "ok" if ok(path, want) else ("wrong md5" if path.is_file() else "missing")
            missing += state != "ok"
            print(f"{state:9} {path.relative_to(REPO).as_posix()}")
        sys.exit(1 if missing else 0)

    for asset, dest, want in RELEASE:
        if not ok(REPO / dest, want):
            fetch_release(asset, REPO / dest, want, a.tag)
    fetch_uav_dot()
    print("all Tier 1 weights are in place")


if __name__ == "__main__":
    main()
