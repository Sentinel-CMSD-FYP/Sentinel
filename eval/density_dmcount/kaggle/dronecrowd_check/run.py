"""Kaggle kernel (CPU): fingerprint the public DroneCrowd copy onvithong/dronecrowddata.

Prints where Kaggle mounted it, then runs ``dataset_fingerprint.py make`` over
/kaggle/input (the attached sentinel-ml-src has no images/ or ground_truth/
folders, so only the DroneCrowd splits are found). Download ``fp_kaggle.json``
and compare it with the fingerprint of our local official copy.
"""

import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def main() -> None:
    for p in sorted(INPUT.rglob("*")):
        depth = len(p.relative_to(INPUT).parts)
        if depth <= 5 and p.name not in ("images", "ground_truth") and p.parent.name not in ("images", "ground_truth"):
            print("dir: " if p.is_dir() else "file:", p.relative_to(INPUT))
    script = next(INPUT.rglob("dataset_fingerprint.py"))
    subprocess.run([sys.executable, str(script), "make", str(INPUT),
                    "--out", str(WORK / "fp_kaggle.json")], check=True)


if __name__ == "__main__":
    main()
