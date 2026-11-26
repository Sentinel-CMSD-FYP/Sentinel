"""Worker-side live latency (capture to sample ready) for 1, 3 and 4 streams on this machine.

Reuses ``make_sources`` and ``run_streams`` of the Kaggle kernel
``eval/pipeline/kaggle/live_latency/run.py`` unchanged, with local folders and a local DroneCrowd.
Output: ``<out>/result.json``, ``latency.md``, ``streams_<n>/``. Needs the weights
(``fetch_weights.py``) and a GPU.

    python eval/pipeline/latency_workers_local.py --data-root /path/to/dronecrowd
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "dronecrowd")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "outputs" / "latency" / "worker")
    ap.add_argument("--streams", type=int, nargs="+", help="stream counts (default 1 3 4)")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("kg", HERE / "kaggle/live_latency/run.py")
    kg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kg)
    sys.path.insert(0, str(HERE))
    from live_latency import table

    out = args.out.resolve()
    if (out / "result.json").exists():
        sys.exit(f"{out / 'result.json'} exists: results are not overwritten, pick another --out")
    kg.WORK, kg.FRAMES, kg.CLIPS = out, out / "frames", out / "clips"
    kg.LOG, kg.DEADLINE = out / "eval.log", time.monotonic() + kg.WATCHDOG_H * 3600
    images = {s: next(args.data_root.resolve().glob(f"**/{s}/images"))
              for s in ("test_data", "train_data")}
    shutil.rmtree(kg.FRAMES, ignore_errors=True)
    kg.CLIPS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(REPO / "ml" / "src"), "CLIPS_ROOT": str(kg.CLIPS),
           "CUDA_VISIBLE_DEVICES": "0", "PIPELINE_DEVICE": "cuda", "MAX_STREAMS": "4",
           "YOLO_CONFIG_DIR": str(out / "ultralytics")}
    sources = kg.make_sources(images)
    results = []
    for n in args.streams or kg.COUNTS:
        results.append(kg.run_streams(n, sources, env, REPO / "apps" / "backend"))
        (out / "result.json").write_text(json.dumps(results, indent=2))
    (out / "latency.md").write_text(table(results) + "\n")
    print(table(results))


if __name__ == "__main__":
    main()
