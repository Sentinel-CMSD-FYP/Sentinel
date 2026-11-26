"""Run ``browser_latency.js`` in headless Chromium, so a latency run can be repeated by script.

The measurement itself is the existing ``eval/pipeline/browser_latency.js``; this only opens the
Live page, calls ``latency.run(streams, clip)`` and saves what it returns. With ``--raw`` it also
saves every record of the first stream and prints per-minute figures (for long runs, to see drift).
The backend and the viewer must already be running (``eval/pipeline/run_latency.sh`` does that).
Needs ``pip install playwright && playwright install chromium``. Units: seconds.

    python eval/pipeline/latency_headless.py --streams 3 --runs 3 --clip live_demo48 \
        --out data/outputs/latency
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "browser_latency.js"


def per_minute(drawn: list[dict]) -> dict[str, list[dict]]:
    """Capture to drawn delay per layer and minute: ``{layer: [{minute, n, median, max}]}``.

    ``drawn`` holds records with ``layer``, ``tCapture`` and ``tDrawn`` (Unix seconds). Minute 0
    starts at the first drawn picture; samples captured before it are in minute -1.
    """
    pictures = [r["tCapture"] for r in drawn if r["layer"] == "picture"]
    if not pictures:
        return {}
    t0 = min(pictures)
    groups: dict[tuple[str, int], list[float]] = {}
    for r in drawn:
        key = (r["layer"], int((r["tCapture"] - t0) // 60))
        groups.setdefault(key, []).append(r["tDrawn"] - r["tCapture"])
    out: dict[str, list[dict]] = {}
    for (layer, minute), values in sorted(groups.items()):
        values.sort()
        out.setdefault(layer, []).append(
            {"minute": minute, "n": len(values),
             "median": round(values[len(values) // 2], 2), "max": round(values[-1], 2)})
    return out


def wait_for(url: str, seconds: int = 90) -> None:
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(1)
    sys.exit(f"{url} not reachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--clip", default="live_demo48", help="stored clip to replay")
    ap.add_argument("--out", type=Path, default=Path("data/outputs/latency"))
    ap.add_argument("--raw", action="store_true",
                    help="also save every record of the first stream and per-minute figures")
    ap.add_argument("--viewer", default="http://127.0.0.1:5173/")
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright  # imported here so tests need no browser

    wait_for(f"{args.backend}/api/v1/health")
    wait_for(args.viewer)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.clip}_{args.streams}_streams"
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for k in range(args.runs):
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            errors: list[str] = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            page.goto(args.viewer)
            page.get_by_role("button", name="Live", exact=True).click()
            page.add_script_tag(content=SCRIPT.read_text())
            res = json.loads(page.evaluate("([n, c]) => window.latency.run(n, c)",
                                           [args.streams, args.clip]))
            res["page_errors"] = errors
            res["browser"] = "headless Chromium " + browser.version
            if args.raw:
                drawn = page.evaluate("(id) => window.sentinelLatency[id] || []",
                                      res["workers"][0]["id"])
                res["per_minute"] = per_minute(drawn)
                (args.out / f"{tag}_run{k + 1}_raw.json").write_text(json.dumps(drawn))
            results.append(res)
            print(json.dumps(res), flush=True)
            page.close()
        browser.close()
    (args.out / f"{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out / f'{tag}.json'}")


if __name__ == "__main__":
    main()
