# 0015 End-to-end latency on the RTX 4000 Ada (1, 3, 4 streams and a 10 minute run)
Date: 2026-11-26   Owner: Mahid (team target; run in Ahmed's folder)
Context: the latency target (<= 3 s from capture to overlay) was "Not measured". One live stream is the current goal; several streams are future work.
Decision: report latency as met for 1 stream on a replayed clip, with the limits below.
Evidence:
- Code version: the 1-stream browser runs and the worker-only runs used `4f5e4bb` plus the then uncommitted fp16 change (default off, later `60f2860`); the 3 and 4 stream browser runs and the 10 minute run used `8fbec2a`.
- Scripts: the runs were driven by local scripts under `data/outputs/latency/`. Tidied versions are now in git: `eval/pipeline/run_latency.sh` (servers + memory sampling + `latency_headless.py`) and `eval/pipeline/latency_workers_local.py`. They were smoke-run once on 2026-11-26 (1 stream: browser points median 1.15 s, worker points median 1.08 s); the tables here come from the original runs.
- To repeat: `ingest_clip` a source into `data/samples`, then `bash eval/pipeline/run_latency.sh --streams 3 --runs 3 --clip live_demo48` (add `--raw` for per-minute figures), under `infra/pc/gpu-job.sh`.
- Setup: RTX 4000 Ada (no compute process listed before the runs), all five models, default fp32 pipeline (fp16 not used), torch 2.11.0+cu128, venv `.venv`, source `live_demo48` = DroneCrowd test 11, 23, 70 and val 71 back to back (1200 frames, 25 fps, 1920x1080), replayed at real time. Team code unchanged.
- Weights: the `tier1-models` GitHub release was "release not found"; DM-Count and YOLO26 weights copied from Ahmar's folder, md5 matching `eval/pipeline/fetch_weights.py` (725694ba..., 12213bde...).
- Browser: `eval/pipeline/browser_latency.js` run in headless Chromium 159 on the same PC (driver `data/outputs/latency/browser_run.py`), 1 stream, 3 runs. Raw: `data/outputs/latency/browser_1_streams.json`.

Capture to drawn on screen, 1 stream, 3 runs (seconds):

| Layer | median (range over runs) | worst seen |
|---|---|---|
| picture | 0.03 | 0.06 |
| density | 0.21 to 0.24 | 0.39 |
| infrastructure | 0.30 to 0.31 | 1.67 |
| flow | 0.47 to 0.50 | 0.67 |
| points (B) | 1.15 to 1.16 | 1.28 |
| tracks (E) | 1.15 to 1.16 | 1.29 |

- Points ran on 462 to 477 of 1200 frames (the rest skipped). The browser drew about 190 point/track updates in 47.9 s (about 4 per second); the picture about 24.7 to 24.9 per second.
- Worker side (capture to result ready, team's step 29 method, `data/outputs/latency/worker/result.json`), median (p90), seconds:

| Streams | density | flow | infrastructure | points / tracks | GPU memory peak |
|---|---|---|---|---|---|
| 1 | 0.07 (0.10) | 0.32 (0.33) | 0.16 (1.67) | 1.08 (1.10) | 2.2 GB |
| 3 | 0.23 (0.34) | 0.34 (0.37) | 0.40 (0.49) | 1.24 (1.40) | 6.7 GB |
| 4 | 0.29 (0.43) | 0.37 (0.43) | 0.55 (0.78) | 1.34 (1.52) | 8.9 GB |

  Worst single sample: 3 streams 2.69 s, 4 streams 4.6 s (both infrastructure). Points processed per 1200 frames: 179 to 186 (3 streams), 122 to 129 (4 streams). Browser side for 3 and 4 streams was run later the same day (below).
Alternatives & why not: measuring in a visible Chrome window was not done (headless is repeatable); drawing time on another screen or GPU may differ.
Open issues:
- Not included: drone to PC video link, network, any real flight. One PC, replayed recordings.
- The picture is drawn about 0.03 s after capture but point and track overlays about 1.15 s after, so dots belong to a frame about 1.1 s older than the picture under them. The visual effect was not judged.
- Likely reason for about 1.1 s on points (from reading `pipeline/live.py`, not tested): a frame may wait up to MAX_LAG_S = 1 s in the queue before being dropped. Lowering it is a possible change for the pipeline owner.
- Tracking accuracy at this live frame rate (about 2 frames in 5) is not measured.

## 3 and 4 streams, capture to drawn (added 2026-11-26)
Same setup and code (commit 8fbec2a), `browser_latency.js`, 3 runs each, headless Chromium; raw `browser_3_streams.json`, `browser_4_streams.json`. Median range over runs, seconds:

| Layer | 3 streams | 4 streams |
|---|---|---|
| picture | 0.03 to 0.04 | 0.04 |
| density | 0.36 to 0.37 | 0.42 to 0.45 |
| infrastructure | 0.50 to 0.56 | 0.64 to 0.70 |
| flow | 0.50 to 0.56 | 0.61 to 0.62 |
| points | 1.38 to 1.40 | 1.52 to 1.54 |
| tracks | 1.39 to 1.41 | 1.53 to 1.55 |
| worst single sample, any layer | 1.79 | 2.07 |

- No module or page errors. Points/tracks ran on 178 to 204 of 1200 frames per stream (3 streams) and 124 to 136 (4 streams). Picture drawn per stream: 22.9 to 24.5 per second (3), 18.8 to 19.1 (4). GPU memory peak 8893 MB, GPU utilisation reached 100 %.
- The earlier worker-only run had one 4.6 s infrastructure sample with 4 streams; it did not recur in these three runs.

## 10 minute run, 1 stream
Source `live_long`: `live_demo48` frames looped to 15,000 frames (600 s at 25 fps), so the content repeats. Same setup; raw `long_summary.json`, `long_raw.json`, `long_mem.csv` (every 10 s).
- Capture to drawn: points median 1.15 s (max 1.28), tracks 1.15 (1.37), density 0.23 (0.38), flow 0.48 (0.63), infrastructure 0.29 (0.47), picture 0.03 (0.07). One infrastructure sample at the start took 2.03 s (before the first full minute).
- No drift: the per-minute median for points was 1.15 to 1.16 s in each of the 10 minutes; picture 0.03 s throughout. Points processed 565 to 582 frames per minute; 5,729 of 15,000 frames over the run. No errors; reader at most 0.15 s behind.
- Memory grew steadily: stream worker RSS 3740 MB (about 1 min in) to 4020 MB (about 10 min in), roughly 31 MB per minute; headless browser 706 to 920 MB, roughly 24 MB per minute (the latency script itself stores every record in the page, so part of the browser growth is the measurement). GPU memory flat at 2243 MB.
- Likely cause of the worker growth (from reading `pipeline/live.py`, not tested): the modules keep their samples to build the saved clip at the end. If the rate held for an hour that would be about 1.9 GB more; this is an extrapolation, not measured.
