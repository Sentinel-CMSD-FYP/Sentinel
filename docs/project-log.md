# Project log: point localisation (Model B) and tracking (Model E)

Working log for my part of Sentinel (Tier 1 perception). Dated entries, newest last. Every number
names the data it came from. Result files are under `data/outputs/` (not in git); the scripts that
produce them are in `eval/` and `ml/src/sentinel_ml/`.

## Summary so far (18 Oct 2026)
| Item | Result | Source |
|---|---|---|
| Point localisation F1 at 10 px | 0.6887 (precision 0.7180, recall 0.6616) | DroneCrowd official test, 30 clips, score threshold 0.2 chosen on val |
| Same on validation | 0.8635 (optimistic, see caveats) | 10 held-out train clips |
| Tracking count error, no filter | 48.5% mean over clips (median 48.6%, 1.33 ids per person) | DroneCrowd test, tracker frozen from val, no CMC, no altitude scaling |
| Tracking count error, >=45-frame filter (offline) | 38.7% mean (median 30.4%) | same run |
| B speed | 4.79 frames/s (val), 4.78 (test), model only | Kaggle T4, not the 4070, not B+E |
| E speed, B+E benchmark | not measured yet | |
No numeric F1 target exists; the count targets in the draft Phase 0 sheet (unsigned) are treated as
provisional and nothing is marked pass or fail.

## 11 Oct: kickoff, dataset, split
- Agreed with Mahid and Ahmar: DroneCrowd primary, UP-COUNT secondary. The JSON formats were accepted
  as drafted. GPU work on the 4070, with Kaggle as the fallback. Night footage is not excluded: if the
  dataset has it, it is used for training and testing. Split and F1 target left open.
- Downloaded DroneCrowd to `data/dronecrowd` (11 GB) and inspected it with `eval/inspect_dataset.py`:
  112 clips of 300 frames, 1920x1080; 82 train and 30 test clips; ground truth per frame in `.mat`
  files as (x, y, person id). Points per frame: median 131 in train, 113 in test, maximum 455. The
  published frame rate is 25 fps (DroneCrowd paper, arXiv 2105.02440, p.3).
- Problems found:
  - The official `val_data` folder is 12 frames per test clip with identical ground truth, so it is a
    copy of test and cannot be used for tuning.
  - The points are the centres of 20x20 boxes, not separate head markings.
  - No altitude or camera metadata in the dataset.
  - Night clips exist (clips 70 and 71 are very dark).
  - Labels are noisy: in test clip 112, frame 273, some dots are off the person and some people have
    none (checked by eye on 12 sample frames).
- Decisions: own validation = 10 clips held out of the 82 train clips with seed 0 (clips 6, 47, 52, 59,
  67, 71, 73, 84, 85, 89); the official test set only for final numbers; altitude as one config constant
  (60 m, an assumption, midpoint of 40-80 m); no separate "without night" score, report all clips.

## 11-14 Oct: Model B environment (released DOT + Pixel Distill checkpoint)
- Model B code: `up-count/uav-dot` (Apache-2.0, commit a6d3664), with the authors' released DroneCrowd
  DOT + Pixel Distill checkpoint. No training of my own (option A, chosen because it is the fastest).
  Everything GPU-side runs on Kaggle (Tesla T4).
- Problems on Kaggle and what fixed them:
  - The uploaded dataset was mounted under `/kaggle/input/datasets/<user>/...`, not at the top level.
  - Recent PyTorch (2.6 and later) refuses to load the checkpoint in its safe mode. Allowing types one
    by one failed because safe mode cannot restore a `defaultdict`. Listed the 14 types stored in the file
    (all plain data or config types) and loaded it with a restricted unpickler that blocks anything else.
  - `segmentation-models-pytorch` had to be pinned to 0.3.4 (with `timm` 0.9.7); the newer version renamed
    an argument the repo uses. The model also downloads encoder weights at start-up (internet on).
  - The model stretches 1080 to 1088 pixels; it does not pad (the repo's config comment says "padded").
- Caveats recorded: the released weights were trained on the official train clips, so my 10 validation
  clips were in their training data and validation numbers are optimistic. The authors also chose the
  checkpoint using `val_data`, which is a copy of test frames, so test numbers may be slightly optimistic.
- Demo check: 30 frames of test clip 011 gave 6,703 points; the circles sat on people.

## 14 Oct: output contracts (v1.0, final)
- Agreed with the team: `*_points.json` and `*_tracks.json` with header `schema_version`, `clip_id`,
  `frame_size`, `fps` (25), `frame_stride` (1), `altitude_m` (60.0, assumed) and `altitude_source`.
  Coordinates are original-frame pixels (x = column, y = row), never padded or resized. Contract and
  samples (clip 011, frames 35-39, built from ground truth) are in `docs/contracts.md` and `docs/samples/`.

## 14 Oct: Model B results
- Inference on Kaggle (scores saved down to 0.05, nothing thresholded when saving): 3,000 val frames in
  626.1 s and 9,000 test frames in 1,881.8 s, i.e. 4.79 and 4.78 frames/s for the model alone.
- Threshold chosen on val only. F1 at 10 px peaks at 0.20 to 0.22 (0.8635 at 0.20, precision 0.856,
  recall 0.871); picked 0.2 (the repo default, slightly more recall).
- Test, scored once at 0.2: precision 0.7180, recall 0.6616, F1 0.6887 (TP 817,116, FP 320,864,
  FN 417,962). Per-clip F1 ranges from 0.29 (clip 74) to 0.84 (clip 75), median 0.70; 8 of 30 clips
  are below 0.6.
- Why test is far below val: the weakest clips are night (69, 70, 74) and dusk or large-scale scenes
  (17, 18), plus clip 35. In the frames, clip 74 has tiny people in dark streets (26 of 110 found) and
  clip 17 has larger people with long shadows (about a third found). Lowering the threshold does not fix
  it: at the lowest saved score (0.05) overall F1 is 0.511 (precision 0.372, recall 0.816), and on clips
  74, 17 and 18 recall rises to 0.56, 0.74 and 0.71 but precision is only 0.23, 0.24 and 0.21. That
  0.05 run was a diagnostic; nothing was selected from it. Val hides this: val clip 71 is dark and
  scores well only because the model trained on it.
- No F1 target is set. The idea of F1 >= 0.70 was dropped because 8 of 30 clips are already below 0.6,
  mostly for reasons outside my pipeline.
- Point count error per frame (score >= 0.2), protocol fixed before running: bins by ground-truth count
  (sparse < 100, moderate 100-200, dense > 200), reported for all clips and for day-only clips. Test,
  30 clips: 21.6% overall (median 15.4%, bias -9.6%, i.e. under-counting); sparse 24.2%, moderate 26.3%,
  dense 7.4%; the three night clips 45.4% with bias -45.4%. Val: 6.7% (contaminated). Count error can
  hide bad points because misses and false points cancel, so it is read together with F1.
- Lighting tags (by me, not blind because I had seen results): test clips 69, 70, 74 night; val clips
  71 and 73 dusk; all others day. Dusk is not covered by the Phase 0 sheet. Open question.

## 15 Oct: disk space
- C: was nearly full (9.2 GB free). Deleted the Kaggle upload packs (about 7 GB, rebuilt with
  `eval/pack_for_kaggle.py` when needed) and purged the pip cache. Moved the 24,600 train images (7.7 GB)
  to a USB drive (`D:\dronecrowd\train_images`, FAT32), with a Windows junction at the old path, so
  nothing in the code changed. Checked: 24,600 files, none failed, names and sizes identical, 60 random
  files with matching checksums, tests pass. C: free went to 25.0 GB.
- The USB drive holds the only copy of those images on this laptop. Anything that reads train or val
  images needs it plugged in. The ground-truth `.mat` files stay on C: (the train GT folder has too many
  long file names for FAT32), so scripts that only need GT, predictions or test images work without it.
  If the drive letter changes, recreate the junction (`mklink /J`); never delete the junction with a
  recursive delete.

## 16 Oct: Model E, authors' tracker and the baseline
- Reviewed `up-count/uav-dot-track` (Apache-2.0, commit 505a728) against the summary I had been given:
  the full method needs a fourth flag (`--track-use-pointflow`, the DDCF step that takes ID switches from
  about 6,600 to 388); on DroneCrowd the "altitude" is a hard-coded 50 or 100 proxy for object size, not
  real altitude, and it only sets a box size (`size = int(100/alt*20)`); the tracker needs video frames and
  the network's decoder output, not just points; its numbers without DDCF are 0.30 to 0.32 count error
  (their DroneCrowd table), above the 25% figure; the submodule URLs are SSH and `eval/TrackEval` is a third
  submodule. Their GT person ids match mine (269 and 128 for clips 011 and 075).
- Decision (option C): build our own point tracker first, add DDCF only if the count error needs it, and
  keep to the points-only contract.
- Baseline tracker (SORT on points, no camera-motion compensation, no altitude scaling): Kalman filter on
  (x, y, vx, vy), 20 px gate, Hungarian matching, 3 consecutive hits to confirm, dies after more than 5 misses
  or within 10 px of the frame border; noise values are the authors' defaults, not tuned. Tests use
  synthetic moving points with known identities. A quirk found while testing: the border rule means heads
  within 10 px of the frame border never get a confirmed track (kept as the visible setting
  `border_margin_px`).
- Watched the 5-second trail videos of clips 011 and 075 (see 18 Oct for my observations).
- Investigation of the heavy fragmentation on test clips 011 and 075 (diagnosis only, nothing tuned;
  these clips were therefore seen): clip 011 has 1,058 ids for 269 people and clip 075 has 441 for 128.
  By ground-truth identity, 56-58% of the excess ids sit on non-GT points (false detections plus unlabelled
  people, which I cannot separate without images) and 44-46% are extra ids for already-tracked people.
  73% of output ids end by expiring; for 74% (clip 011) and 53% (clip 075) of those, a detection was
  within the gate at the first missed frame, i.e. another track had taken it. B's dropouts longer than 5
  frames: 355 and 95. Camera motion is about zero: optical-flow estimate 0.10 and 0.25 px per frame median,
  no GT step above the 20 px gate. The border rule changes ids by 15 and 0. Count error against GT people:
  no filter 293.3% and 244.5%; keeping only ids output in 45 or more frames (the authors' evaluation drops
  shorter tracks, offline only) 46.5% and 23.4%.
- Decisions: measure first (the count-error metric) before any tracker change; do not copy the authors'
  `warp[:2,2]/2` in camera-motion compensation (their downscale is already undone earlier, so the halving
  gives half the true translation); a `score_thr` sweep on val belongs in the fragmentation experiments.

## 16-18 Oct: tracking count error (E-3)
- Definition (fixed before computing, GT definition confirmed 18 Oct): per clip |unique track ids -
  unique ground-truth person ids| / unique ground-truth person ids, then the mean over clips. Primary: every
  id counts and all clips; secondary: ids with 45 or more output frames, and day-only clips. Test scored once.
- Baseline on val (defaults, score 0.2, min_hits 3, max_age 5): 205.2% no filter (3.05 ids per person),
  25.9% with the 45-frame filter. This is the reference for the experiments.

## 18 Oct: trail videos, experiments, camera check, test
- My observations of the clip 011 and 075 videos (my scale: "few" = a couple of cases, "some" = more than
  a couple, roughly 2 in 10 people): in clip 011 a few colour changes while a person moves; two people close
  together or passing share one colour for some frames and have two in others, and after a close pass a dot
  can disappear and come back with a new colour; some people near buildings or in shade (partly visible) never
  get a colour. In clip 075 the problems are at the borders and in shade (a few colour changes near the
  border, some flicker under shade or next to buildings). A lone trail with no dot turned out to be a bug in
  the video renderer (it never forgot a finished track); fixed and tested.
- Experiment protocol written before running, on val only, with two caveats: clips 011 and 075 were seen
  in the investigation, and val is optimistic because B trained on it. Step 1 swept the tracker's score
  threshold (0.10 to 0.50); step 2 swept `min_hits` {3, 5, 8, 12} and `max_age` {5, 10, 15, 25} at the step-1
  winner. Selection rule fixed in advance: lowest primary count error among configurations that still cover
  at least 90% of the ground-truth people, ties within 1 point broken towards lower latency.
- Step 1 (val): count error falls from 411% at 0.10 to 130% at 0.35 while spurious ids fall from 5,857 to
  636; coverage of real people 0.961 to 0.919 (0.878 at 0.40, below the floor). Extra ids for already-tracked
  people stay at about 2,000 to 2,100 whatever the threshold, so thresholding does not fix identity breaks.
  Selected 0.35.
- Step 2 (val, at 0.35): a longer `max_age` cuts the extra ids (2,114 at 5 frames to 829 at 25 frames);
  larger `min_hits` removes spurious ids but lowers coverage (0.919 to 0.845), and the 0.90 floor decided the
  choice. Selected `min_hits` 3, `max_age` 25: 54.7% on val (1.55 ids per person). `max_age` 25 is the top
  of the grid, so the trend beyond it was not explored.
- Camera motion on all 30 test clips (measurement only): largest optical-flow translation 1.19 px per frame
  (clip 17), 6.0% of the gate; largest GT common-mode motion 1.41 px; at most 0.057% of GT steps above the
  gate in any clip. Slow drift over many frames was not measured.
- Test result, scored once with the frozen config (score 0.35, min_hits 3, max_age 25, no CMC, no altitude
  scaling, DroneCrowd only): mean count error 48.5% over 30 clips (median 48.6%, 1.33 ids per person); with
  the 45-frame filter 38.7% (median 30.4%); day-only 52.0% and night (3 clips) 16.8%. Only 6 of 30 clips are
  at or below the provisional 25% figure. The errors have both signs: 6,828 ids for 5,092 people overall,
  but 5 clips under-count (clip 17: 51 ids for 175 people; clip 108: 83 for 239; also 18, 35, 74) and 12 clips
  have more than 1.5 ids per person. The under-counting clips are the ones where B is weakest and
  under-confident, so the threshold chosen on val (0.35) was too high for them: the val optimism caveat
  showed up in practice. Nothing was changed after seeing the test result, and no test baseline with the
  default settings exists.

## 18 Oct (later): decisions after the E-3 baseline
- E-3 accepted as "baseline measured, target not met". CI passed on the merged pull request (#7).
- Target: 25% tracking count error stays as a stretch goal only. Provisional sign-off target I can
  defend from the evidence: tracking count error <= 35% with the >=45-frame filter, DroneCrowd test,
  all clips, with IDF1 reported alongside (baseline: 38.7% filtered, 48.5% unfiltered; the authors'
  own no-DDCF DroneCrowd rows are 0.30 to 0.32, 0.23 with DDCF). Still provisional: the Phase 0 sheet is
  an unsigned draft. DDCF is decided only after E-2 and the extended max_age results on val.
- Scope: night is out of scope; dusk is reported separately with no target; night results are reported as
  a limitation, not hidden (test night clips: 69, 70, 74; dusk: val clips 71, 73, none in test). Wording for
  the sheet: "Night is out of scope. Dusk is reported separately, with no target. Night results are
  reported as a limitation."
- Plan additions: IDF1 and HOTA through TrackEval (converted MOT text against the authors' MOT-format GT,
  same ids as my .mat GT; TrackEval imports on the current numpy, not yet run on real tracks). CMC will take
  a flow field in and give a camera translation out, behind a small interface, so Mahid's flow code (now on
  main under `eval/optical_flow_farneback/`) or another model can be swapped in; the authors' halved
  translation is not copied.
- Second look at test: allowed only if the comparison and config are fixed on val in writing first, both
  test runs are reported (never the better one chosen on test), the second run is labelled "second look,
  not blind", and my own footage is preferred as the next clean test.
- E-2a built: seconds for the miss limit (`max_age_s`, converted with the clip's fps and frame_stride) and
  an altitude-scaled gate (authors' rule `size = int(100/alt * 20)`), both off by default, so the baseline
  is unchanged (148 tests pass). A gate in metres is not possible: it needs the camera field of view or
  intrinsics, which DroneCrowd does not give. DroneCrowd benchmarking uses the authors' object-size proxy
  (50 for large, 100 for small clips; their list covers the test clips only), the pipeline keeps 60 m.
  The val clips need small/large tags from me before altitude can be compared on val.
- Extended max_age on val first (25 frames was the edge of the grid): step 3 sweeps 1.0 to 4.0 s at the
  frozen score 0.35 and min_hits 3, with a new identity-purity metric so a long max_age cannot win by
  merging different people (purity floor 0.935 = the step-2 winner's 0.955 minus 0.02, declared before the
  run).

- Step 3 result (val, 10 clips, B trained on these clips): count error without the filter falls from 54.7%
  at a 1.0 s miss limit to 39.4% at 4.0 s (extra ids for tracked people 829 to 596), the filtered view stays
  flat (11.2% to 11.6%), and purity stays at 0.945 or higher. Coverage falls from 0.904 to 0.887, and the
  0.90 floor declared in advance excludes every value above 1.0 s (first step: 0.895). By the rule as
  written the selection stays at 1.0 s. I am not changing the rule after seeing this; relaxing the floor
  would be a post-hoc decision and would have to be written down as one before the second look at test.

## Open items
- E-2a (altitude-scaled gate) and E-2b (camera-motion compensation, full translation, toggles), each
  tested on its own. Camera motion is small on all test clips, so a large effect is not expected.
- E-4 (ID-stability videos, best and worst clips, my judgement) and E-5 (B+E speed on the 4070 with nothing
  else running, target 5 frames/s per stream; Kaggle only as fallback).
- Decide whether DDCF is needed for the count target, and whether to keep the 25% figure (the draft Phase 0
  sheet is unsigned and says E reaches it only with DDCF in the authors' results). Tell the supervisor.
- Decide dusk and night scope (not covered by the Phase 0 sheet).
- Ask Mahid whether camera-motion compensation can reuse his optical-flow output.
- CI has not run on this branch yet (it runs on pushes to `main` and on pull requests); open a pull request.
- Whether B should be trained on the 72 train clips only, to remove the val contamination; not done, GPU
  hours not measured.
