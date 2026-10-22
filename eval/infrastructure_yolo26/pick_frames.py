"""Pick DroneCrowd frames for Model C labelling.

    python eval/infrastructure_yolo26/pick_frames.py sheet
    python eval/infrastructure_yolo26/pick_frames.py copy 100:150 70:1 ...

``sheet`` saves contact sheets of one frame per clip (all 112 clips, labelled with clip
number and split) so frames can be chosen by eye. ``copy`` copies the chosen frames,
given as ``clip:frame`` (1 based frame, as in the file names), into the labelling image
folder. Files keep their DroneCrowd names (``img{clip:03d}{frame:03d}.jpg``).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from sentinel_ml.density.data import find_split_dirs

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "data" / "dronecrowd"
DEFAULT_OUT = REPO / "data" / "infra_labels"
THUMB = (384, 216)
COLS, ROWS = 4, 4


def frame_path(split_dirs: dict[str, Path], clip: int, frame: int) -> Path:
    name = f"img{clip:03d}{frame:03d}.jpg"
    for d in split_dirs.values():
        p = d / "images" / name
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def make_sheets(split_dirs: dict[str, Path], out: Path, frame: int) -> None:
    clips = sorted({(int(p.name[3:6]), split) for split, d in split_dirs.items()
                    for p in (d / "images").glob(f"img???{frame:03d}.jpg")})
    out.mkdir(parents=True, exist_ok=True)
    per_sheet = COLS * ROWS
    for s in range(0, len(clips), per_sheet):
        sheet = np.zeros((ROWS * THUMB[1], COLS * THUMB[0], 3), np.uint8)
        for i, (clip, split) in enumerate(clips[s:s + per_sheet]):
            img = cv2.resize(cv2.imread(str(frame_path(split_dirs, clip, frame))), THUMB)
            cv2.putText(img, f"{clip} {split}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255), 2, cv2.LINE_AA)
            r, c = divmod(i, COLS)
            sheet[r * THUMB[1]:(r + 1) * THUMB[1], c * THUMB[0]:(c + 1) * THUMB[0]] = img
        path = out / f"sheet_{s // per_sheet + 1}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["sheet", "copy"])
    ap.add_argument("frames", nargs="*", help="copy: clip:frame pairs, e.g. 100:150")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frame", type=int, default=150, help="sheet: frame shown per clip")
    args = ap.parse_args()

    split_dirs = {k: args.root / v for k, v in find_split_dirs(args.root).items()}
    if args.cmd == "sheet":
        make_sheets(split_dirs, args.out / "sheets", args.frame)
        return
    dst = args.out / "images"
    dst.mkdir(parents=True, exist_ok=True)
    for item in args.frames:
        clip, frame = (int(x) for x in item.split(":"))
        src = frame_path(split_dirs, clip, frame)
        shutil.copy2(src, dst / src.name)
    print(f"{len(args.frames)} frames in {dst} ({len(list(dst.glob('*.jpg')))} total)")


if __name__ == "__main__":
    main()
