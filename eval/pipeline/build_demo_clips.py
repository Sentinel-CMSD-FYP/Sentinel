"""Build the Tier 1 demo clips on this machine through the real backend commands.

For each clip: ``manage.py ingest_clip`` (DroneCrowd frames to video.mp4 + clip.json), then
``manage.py run_models <clip> --modules all --device <device>`` (all five models in one decode),
then ``docs/layers/validate.py`` on the clip folder. Same steps as the Kaggle kernel
``eval/pipeline/kaggle/demo_clips`` (step 25). Needs the weights (``fetch_weights.py``) and
DroneCrowd under ``--root`` (any layout with ``train_data/images`` and ``test_data/images``).

    python eval/pipeline/build_demo_clips.py --root data/dronecrowd --device cuda

A clip that already exists is skipped; ``--rebuild`` deletes it first. Clips go to the
backend's ``CLIPS_ROOT`` (default ``data/samples``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sentinel_ml.density.data import find_split_dirs

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "apps" / "backend"
# (clip id, DroneCrowd split, clip number, name shown in the viewer); val clip 71 is in train
DEMO = [("seq011", "test", "011", "DroneCrowd test 11 (day)"),
        ("seq023", "test", "023", "DroneCrowd test 23 (dense)"),
        ("seq070", "test", "070", "DroneCrowd test 70 (night)"),
        ("seq071", "train", "071", "DroneCrowd val 71 (dusk)")]


def run(*cmd: str, cwd: Path = REPO) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=REPO / "data" / "dronecrowd",
                    help="DroneCrowd folder (default data/dronecrowd)")
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    ap.add_argument("--clips", help="comma separated subset, e.g. seq011,seq070")
    ap.add_argument("--rebuild", action="store_true", help="delete existing clips first")
    a = ap.parse_args()

    clips_root = Path(os.environ.get("CLIPS_ROOT", REPO / "data" / "samples"))
    wanted = set(a.clips.split(",")) if a.clips else None
    unknown = (wanted or set()) - {c[0] for c in DEMO}
    if unknown:
        sys.exit(f"unknown clips {sorted(unknown)}; demo clips: {[c[0] for c in DEMO]}")
    splits = find_split_dirs(a.root.resolve())
    py = sys.executable
    for clip_id, split, seq, name in DEMO:
        if wanted and clip_id not in wanted:
            continue
        folder = clips_root / clip_id
        if folder.exists():
            if not a.rebuild:
                print(f"{clip_id}: exists, skipped (--rebuild to make it again)")
                continue
            shutil.rmtree(folder)
        start = time.monotonic()
        images = a.root.resolve() / splits[split] / "images"
        run(py, "manage.py", "ingest_clip", str(images), clip_id, "--seq", seq, "--fps", "25",
            "--name", name, cwd=BACKEND)
        run(py, "manage.py", "run_models", clip_id, "--modules", "all", "--device", a.device,
            cwd=BACKEND)
        run(py, str(REPO / "docs" / "layers" / "validate.py"), str(folder))
        print(f"{clip_id}: done in {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
