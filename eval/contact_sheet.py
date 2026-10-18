"""Contact sheets for tagging clips by lighting, without any scores on them.

One frame per clip, labelled with the clip number only (no GT, no predictions, no
metrics), in clip-number order. Tag the clips by eye in ``data/manifests/lighting_tags.yaml``.

    python eval/contact_sheet.py --config ml/configs/dronecrowd_data.yaml --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sentinel_ml.localization import data as D

THUMB = (640, 360)  # width, height of one thumbnail
COLS = 3
ROWS = 5            # 15 clips per sheet


def middle_frame_index(refs: list[D.FrameRef], clip: int) -> int:
    """Index in ``refs`` of the middle frame of ``clip``."""
    idx = [i for i, r in enumerate(refs) if r.clip == clip]
    return idx[len(idx) // 2]


def make_sheet(thumbs: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Grid of (clip, BGR thumbnail) pairs, COLS columns, labelled with the clip number."""
    rows = (len(thumbs) + COLS - 1) // COLS
    sheet = np.zeros((rows * THUMB[1], COLS * THUMB[0], 3), np.uint8)
    for k, (clip, img) in enumerate(thumbs):
        r, c = divmod(k, COLS)
        tile = cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"clip {clip:03d}", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (0, 0, 0), 6)
        cv2.putText(tile, f"clip {clip:03d}", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 2)
        sheet[r * THUMB[1]:(r + 1) * THUMB[1], c * THUMB[0]:(c + 1) * THUMB[0]] = tile
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the dronecrowd_data.yaml")
    ap.add_argument("--split", choices=D.SPLITS, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/outputs/contact_sheets"))
    args = ap.parse_args()

    ds = D.dataset_from_config(D.load_config(args.config), args.split)
    clips = sorted({r.clip for r in ds.refs})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = COLS * ROWS
    for s in range(0, len(clips), per_sheet):
        chunk = clips[s:s + per_sheet]
        thumbs = [(c, cv2.cvtColor(ds[middle_frame_index(ds.refs, c)].image, cv2.COLOR_RGB2BGR))
                  for c in chunk]
        path = args.out_dir / f"{args.split}_sheet{s // per_sheet + 1}.jpg"
        cv2.imwrite(str(path), make_sheet(thumbs), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"saved {path}: clips {chunk}")


if __name__ == "__main__":
    main()
