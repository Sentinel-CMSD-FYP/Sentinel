# 0004 Model B results (point localisation)
Date: 2026-10-14   Owner: Ahmed
Context: B-4. Choose the score threshold for the released DOT + Pixel Distill DroneCrowd checkpoint on validation only, then score the test split once. The Phase 0 sheet has no numeric F1 target, so the number below is reported, not marked pass or fail. The team may set a target.

## Setup
- Model: released checkpoint `dot_pd_dronecrowd_51.00.ckpt` (repo commit a6d3664), no training by us (option A, Ahmed 2026-10-14).
- Predictions: `sentinel_ml.localization.infer` on Kaggle T4, points saved down to score 0.05 (decision 0003).
- Metric: `eval/point_metrics.py`: one-to-one nearest-first matching, 10 px radius (inclusive), counts summed over all frames. Sweep script: `eval/threshold_sweep.py`.
- Validation split: 10 clips held out from the official train clips (seed 0): 6, 47, 52, 59, 67, 71, 73, 84, 85, 89. 3000 frames, 466,536 GT points.

## Validation sweep (DroneCrowd val split, 10 clips, 3000 frames, 10 px)
| thr | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.10 | 426134 | 206847 | 40402 | 0.6732 | 0.9134 | 0.7751 |
| 0.20 | 406436 | 68400 | 60100 | 0.8560 | 0.8712 | 0.8635 |
| 0.30 | 355371 | 29765 | 111165 | 0.9227 | 0.7617 | 0.8345 |
| 0.40 | 284945 | 13330 | 181591 | 0.9553 | 0.6108 | 0.7451 |
| 0.50 | 186421 | 4717 | 280115 | 0.9753 | 0.3996 | 0.5669 |
| 0.60 | 90683 | 1116 | 375853 | 0.9878 | 0.1944 | 0.3248 |
| 0.70 | 32126 | 159 | 434410 | 0.9951 | 0.0689 | 0.1288 |
| 0.80 | 7321 | 20 | 459215 | 0.9973 | 0.0157 | 0.0309 |
| 0.90 | 179 | 0 | 466357 | 1.0000 | 0.0004 | 0.0008 |

Finer steps near the peak (same val data): 0.14 F1 0.8226, 0.16 0.8396, 0.18 0.8547, **0.20 0.8635**, 0.22 0.8637, 0.24 0.8602, 0.26 0.8537.
Per clip at thr 0.2: F1 from 0.842 (clip 84) to 0.914 (clip 47); none below 0.84.

## Chosen threshold: 0.2 (agreed by Ahmed, 2026-10-14)
Reason: F1 is highest at 0.20 to 0.22 and the top is flat (0.8635 vs 0.8637, a gap too small to matter). 0.2 is also the repo's own default and gives slightly more recall than 0.22 (0.871 vs 0.850), which suits an early-warning use where missing people costs more than a few extra points. Going lower than 0.2 adds many false points (FP 68k at 0.2, 207k at 0.1).

## Caveats (must travel with every number)
- **The released weights were trained on the official train clips, which include our 10 val clips.** The val F1 above is therefore probably optimistic, and the threshold may sit a little differently on unseen footage. (Ahmed chose to accept this and label it, 2026-10-14.)
- The released checkpoint was probably selected using test frames (authors' val_data is a copy of test frames), so the test F1 may also be slightly optimistic (decision 0003).
- Ground truth is noisy: some people are unlabelled and some labels sit off the person (decision 0002), which lowers precision and recall for any model.
- DroneCrowd only. Not validated on other footage, other drones or multi-drone setups.

## Test result (run once, threshold 0.2 fixed on val beforehand)
**DroneCrowd official test split, 30 clips, 9000 frames, 10 px, thr=0.2, released DOT+PD checkpoint: precision 0.7180, recall 0.6616, F1 0.6887** (TP 817,116, FP 320,864, FN 417,962).
- Predictions made on Kaggle T4: 9000 frames in 1881.8 s = 4.78 frames/s for B alone (not the B+E benchmark, not the 4070). All 30 files passed the contract check and matched the point counts Kaggle printed (2,710,706 points in total).
- Source files: `data/outputs/points/test/`, scores in `data/outputs/b4/test/` (git-ignored).
- **Status: no numeric F1 target exists in the Phase 0 sheet, so this is reported, not marked pass or fail.** Ahmed and the team can set a target.

### The test F1 is far below the val F1 (0.6887 vs 0.8635)
Facts from the per-clip table at thr 0.2 (diagnosis only, the threshold was not changed):
- Test clip F1 ranges from 0.290 (clip 74) to 0.838 (clip 75), median 0.699. 8 of 30 clips are below 0.6.
- The worst clips are mostly low in recall (clip 74 R 0.214, clip 17 R 0.209, clip 18 R 0.233), i.e. the model misses most people there. Clip 70, the dark clip, has F1 0.472.
- On val, every clip was between 0.842 and 0.914.

Likely contributors (not verified, to investigate if needed):
- The released weights were trained on our val clips, so val was optimistic (known, see caveats). Test clips are unseen.
- Some test scenes may differ from the training scenes (altitude, density, lighting). We have not looked at the frames of the worst clips.
- GT noise (decision 0002).
Not comparable to the authors' L-AP@10 57.06 or L-mAP 51.0: those are average-precision metrics from a different protocol, not F1 at one threshold.

## Alternatives & why not
- Train our own model on the 72 train clips only: cleaner, but needs GPU hours we have not measured; not done for now.

## Open issues
- Why test is so much lower than val: not investigated yet (see above). Options: look at the worst clips' frames (74, 17, 18, 35, 69), or train our own model.
- Whether the team wants a numeric F1 target.
