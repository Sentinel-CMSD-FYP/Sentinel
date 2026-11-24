"""Kaggle kernel (CPU): step 4 GT check on the public DroneCrowd copy onvithong/dronecrowddata.

Runs ``gt_check.py`` from the attached sentinel-ml-src (with ``ml/src`` on PYTHONPATH)
over /kaggle/input, so the split folders and the frozen manifest are resolved the same
way training will. Split sizes should match the local run; drawn frames land in
/kaggle/working/gt_check.
"""

import os
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def main() -> None:
    script = next(INPUT.rglob("gt_check.py"))
    src = script.parents[2]
    env = {**os.environ, "PYTHONPATH": str(src / "ml" / "src")}
    subprocess.run([sys.executable, str(script), "--root", str(INPUT),
                    "--out", str(WORK / "gt_check")], check=True, env=env)


if __name__ == "__main__":
    main()
