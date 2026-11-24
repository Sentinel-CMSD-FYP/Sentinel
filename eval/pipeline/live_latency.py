"""Step 29: latency of live streams from the worker side, capture to sample ready, per layer.

Reads the stream folders a run left behind (``<CLIPS_ROOT>/.streams/<id>/samples.jsonl`` and
``status.json``) and keeps only the samples captured while every given stream was live, so N
streams are compared under the load of N. The browser side (received, drawn) is measured in the
page (``eval/pipeline/browser_latency.js``).

    python eval/pipeline/live_latency.py <stream folder> [<stream folder> ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(folder: Path) -> tuple[list[dict], dict]:
    lines = (folder / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    return [json.loads(x) for x in lines if x.strip()], status


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)

    def q(p: float) -> float:
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    return {"n": len(xs), "median": round(q(0.5), 2), "p90": round(q(0.9), 2),
            "max": round(xs[-1], 2)}


def summarize(folders: list[Path]) -> dict:
    """Per layer ``t_ready - t_capture`` over the time all streams were live, plus per stream
    frames run and skipped."""
    runs = [load(f) for f in folders]
    windows = []
    for samples, _ in runs:
        t = [s["t_capture"] for s in samples if s["t_capture"] is not None]
        windows.append((min(t), max(t)))
    lo = max(w[0] for w in windows)
    hi = min(w[1] for w in windows)
    ready: dict[str, list[float]] = {}
    for samples, _ in runs:
        for s in samples:
            if s["t_capture"] is not None and lo <= s["t_capture"] <= hi:
                ready.setdefault(s["layer"], []).append(s["t_ready"] - s["t_capture"])
    return {
        "streams": len(folders),
        "overlap_s": round(hi - lo, 1),
        "ready": {k: stats(v) for k, v in sorted(ready.items())},
        "workers": [{"id": st["stream_id"], "state": st["state"], "frames_read": st["frames_read"],
                     "ran": st["ran"], "skipped": st["skipped"], "late_s": st["late_s"],
                     "errors": st["errors"]} for _, st in runs],
    }


def table(results: list[dict]) -> str:
    """Markdown: one row per stream count, median / p90 seconds per layer."""
    layers = sorted({k for r in results for k in r["ready"]})
    head = "| streams | overlap s | " + " | ".join(f"{k} med / p90" for k in layers) + " |"
    rows = [head, "|" + " --- |" * (len(layers) + 2)]
    for r in results:
        cells = []
        for k in layers:
            s = r["ready"].get(k, {"n": 0})
            cells.append(f"{s['median']} / {s['p90']} (n {s['n']})" if s["n"] else "none")
        rows.append(f"| {r['streams']} | {r['overlap_s']} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, help="also write the summary here")
    a = ap.parse_args(argv)
    result = summarize(a.folders)
    if a.json:
        a.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(table([result]))


if __name__ == "__main__":
    main()
