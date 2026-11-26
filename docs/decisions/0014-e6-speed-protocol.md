# 0014 E-6 speed work: protocol and results
Date: 2026-11-26   Owner: Ahmed (run by Mahid)
Context: B+E misses 5 FPS per stream with 3 and 4 streams (4.43 and 3.32, decision 0012). E-6: one change at a time, re-measure FPS and F1.
Rules, written before any E-6 measurement:
- Order: (1) fp16 for Model B, (2) frame_stride 2, (3) smaller net_size. Changes are stacked only after each is measured on its own step.
- Speed: `eval/benchmark_be.py`, same command, clips and settings as decision 0012, on the RTX 4000 Ada with no other compute job; a fresh fp32 baseline in the same session.
- Accuracy: F1 at 10 px, threshold 0.2 (fixed in B-4, not re-tuned), validation split only, fp32 and the changed setting both run on this PC. Test is not touched. Val is optimistic (the released weights trained on these clips), so only the difference between settings is meaningful.
- Proposed acceptance: keep a change if it speeds up B+E and val F1 drops by at most 0.01. This tolerance is a proposal for Ahmed/Mahid to confirm; the table reports the raw numbers either way.
- Speed target: lowest stream avg >= 5 FPS with 3 and with 4 streams.
Results: (below, added after each step)

### Step 1: fp16 for Model B (2026-11-26)
Command: `bash infra/pc/gpu-job.sh docker compose ... run --rm -T ml bash data/outputs/e6/step1.sh` (script and log under `data/outputs/e6/`). RTX 4000 Ada, torch 2.1.0+cu121, no compute process listed before the run.
Speed (DroneCrowd test clips 11, 15, 16, 17; 250 timed frames per stream after 10 warm-up; frame_stride 1; net 1920x1088; `benchmark/e6_fp32_baseline.json`, `benchmark/e6_fp16.json`):

| Streams | fp32 lowest avg / p95 | fp16 lowest avg / p95 | all streams fp32 -> fp16 | fp16 vs 5 FPS (avg) |
|---|---|---|---|---|
| 1 | 10.01 / 9.94 | 12.98 / 12.86 | 10.01 -> 12.98 | met |
| 3 | 4.50 / 3.72 | 5.65 / 4.90 | 13.81 -> 17.08 | met (p95 just under) |
| 4 | 3.33 / 2.72 | 4.35 / 3.84 | 13.70 -> 17.96 | NOT met |

Accuracy (DroneCrowd val, 10 clips, 3000 frames, 10 px, threshold 0.2; optimistic, only the difference matters):

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| fp32 | 406,436 | 68,406 | 60,100 | 0.8559 | 0.8712 | 0.8635 |
| fp16 | 406,519 | 69,210 | 60,017 | 0.8545 | 0.8714 | 0.8629 |

- The fp32 run gives the same TP, FP and FN as B-4 (decision 0004), so the baseline reproduces.
- F1 difference: -0.0006, within the proposed 0.01 tolerance. **Keep fp16** (pending Ahmed's/Mahid's confirmation of the tolerance).
- Not measured: tracking metrics with fp16; fp16 on the test split (the reported test F1 0.689 is fp32); a run with teammates' jobs; the team pipeline with fp16 (its default is unchanged, fp32).
- 4 streams need about 1.15x more than fp16 gives (5 / 4.35).

Note before step 2: `eval/benchmark_be.py` counts processed frames. Processing every second frame does not raise processed frames per second; it only lowers how many are needed per second of video. Counting skipped frames as FPS would change the meaning of the target, so it needs a decision first.
