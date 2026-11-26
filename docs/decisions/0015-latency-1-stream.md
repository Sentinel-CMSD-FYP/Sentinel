# 0015 End-to-end latency on the RTX 4000 Ada
Date: 2026-11-26   Owner: Mahid (team target; run in Ahmed's folder)
Context: the latency target (<= 3 s from capture to overlay) was "Not measured". One live stream is the current goal; several streams are future work.
Decision: report latency as met for 1 stream on a replayed clip, with the limits below.
Evidence:
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

  Worst single sample: 3 streams 2.69 s, 4 streams 4.6 s (both infrastructure). Points processed per 1200 frames: 179 to 186 (3 streams), 122 to 129 (4 streams). Browser side was not run for 3 or 4 streams.
Alternatives & why not: measuring in a visible Chrome window was not done (headless is repeatable); drawing time on another screen or GPU may differ.
Open issues:
- Not included: drone to PC video link, network, any real flight. One PC, replayed recordings.
- The picture is drawn about 0.03 s after capture but point and track overlays about 1.15 s after, so dots belong to a frame about 1.1 s older than the picture under them. The visual effect was not judged.
- Likely reason for about 1.1 s on points (from reading `pipeline/live.py`, not tested): a frame may wait up to MAX_LAG_S = 1 s in the queue before being dropped. Lowering it is a possible change for the pipeline owner.
- Tracking accuracy at this live frame rate (about 2 frames in 5) is not measured.
