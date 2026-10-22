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
| Tracking count error, run 2 (frozen E-2 config) | 35.2% unfiltered, 37.9% filtered (>=45 frames), mean over clips | DroneCrowd test, second look, not blind, no CMC, no altitude scaling |
| IDF1 / HOTA, run 2 | 0.388 / 0.300 (run 1: 0.384 / 0.298) | TrackEval IoU 0.5, DroneCrowd test, mean over clips |
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

- Post-hoc decision (made after seeing step 3 on val, written here before any second look at test): I relaxed
  the step-3 coverage floor from 0.90 to 0.88 and picked a 3.0 s miss limit. With 0.88 all five values are
  eligible and the tie rule (within 1 point, take the smaller value) selects 3.0 s on its own: 40.39% against
  39.43% at 4.0 s, coverage 0.8897, purity 0.9463. The rule as first written selected 1.0 s; both outcomes are
  recorded. It is post-hoc, on optimistic data (B trained on the val clips). The second look at test will be
  labelled "second look, not blind" and both DroneCrowd test runs will be reported: run 1 (score 0.35,
  min_hits 3, 25 frames, no CMC, no altitude scaling: 48.5% unfiltered, 38.7% filtered) and run 2 (the final
  E-2 config, fixed on val in writing first); I will not choose the better one on test.

- Altitude comparison on val (my size tags: small = clips 6, 47, 52, 59, 67, 71, 73; large = 84, 85, 89;
  only those three clips get a different gate): scaling on gives 36.72% against 40.39% off (3.66 points), but
  the filtered view is slightly worse (12.24% against 11.62%) and both floors are missed by hair margins
  (coverage 0.8784 against 0.88, purity 0.9344 against 0.935). By the rule fixed in advance (at least 1.0
  point gain and both floors held) altitude scaling stays off. I did not relax the floors a second time.
  It should be re-tested on my own footage with real altitude.

- Altitude scaling stays off by the rule fixed in advance. It is untested on real altitude (the proxy is the
  authors' object-size stand-in, and only 3 of 10 val clips got a different gate) and should be re-tested on my
  own footage.
- CMC interface (built, 177 tests pass, not yet evaluated): a flow field goes in, a camera shift comes out.
  `FlowInput(u, v, downscale, in_original_pixels)` carries the flow grid's downscale explicitly and says
  which pixels the vectors are in; `estimate_camera_shift` returns the median flow as (dx, dy) in original
  pixels (previous to current frame). A flow provider is any callable (prev frame, frame) -> FlowInput, so
  another flow model can be swapped in; `FarnebackProvider` wraps Mahid's Farneback model (his `FlowField`
  hides its downscale, so the provider passes it on from his config; carrying it in the object would make the
  interface self-contained). In the tracker, `use_cmc` (off by default) adds the full shift to the predicted
  positions after the Kalman prediction and before the border check; velocity and covariance are untouched.
  Tests cover units, robustness to moving people, the sign convention end to end through Farneback, and a
  30 px camera jump that breaks tracks without CMC and not with it.
- E-2b protocol (fixed before computing any val shift): Farneback with Mahid's default config (downscale 0.25),
  median shift over the whole grid, no exclusion mask; candidate config (score 0.35, min_hits 3, max_age 3.0 s,
  altitude off) with CMC off against on; on only if the primary count error improves by at least 1.0 point and
  coverage >= 0.88 and purity >= 0.935 hold. Expectation stated in advance: a small effect, since camera
  motion is small on all 30 test clips. The val frames are on the USB drive, which has to be plugged in.

- E-2b result on val (10 clips, B trained on these clips): camera shifts from Mahid's Farneback (default config)
  are tiny (per-clip median 0.02 to 0.06 px per frame, largest 0.39 px). CMC on gives 40.44% against 40.39% off,
  a gain of -0.06 points (the rule needs +1.0), with the floors held; the filtered view is 12.68% against 11.62%.
  By the rule fixed in advance CMC stays off, as I expected beforehand: the footage barely moves. The mechanism
  itself works (a 30 px camera jump is handled in a test), so the code stays, off by default, to be re-tested on
  my own footage. Candidate config after E-2: score 0.35, min_hits 3, max_age 3.0 s, altitude off, CMC off.

- E-2 closed: CMC and altitude scaling both stay off by the rules fixed in advance. CMC is untested beyond
  DroneCrowd val and altitude is untested on real altitude; both are to be re-tested on my own footage. The
  video check for E-2 is waived because the estimated camera shifts are tiny (val per-clip median 0.02 to
  0.06 px per frame, at most 1.19 px per frame on the 30 test clips).
- IDF1 and HOTA with TrackEval (tool `eval/trackeval_eval.py`, 186 tests pass). Sanity checks first, on
  3 val clips (553 GT people): ground truth as prediction scores 1.0 on everything with 0 switches; a consistent
  relabelling changes nothing; shuffled ids keep DetA 0.999 but give HOTA 0.091, IDF1 0.033 and 138,270 ID
  switches; dropping half the detections gives IDF1 0.662 (expected 0.667). One finding: at the metric
  threshold 0.0 used in the authors' evaluation script, shuffled ids keep IDF1 1.0 and show 106 switches, so
  IDF1 and ID switches stop measuring identity at that setting (HOTA is unaffected). I did not run their
  pipeline (this is from reading their script and running TrackEval at 0.0 on our GT), so their ID-switch
  figures should not be compared with mine. I use the standard IoU 0.5, with 0.3 (about a 10 px offset) as
  secondary.
- Val results (10 clips, 1964 GT people, IoU 0.5; B trained on these clips, so optimistic). E-2 candidate
  (score 0.35, max_age 3.0 s, no altitude scaling, no CMC): HOTA 0.523, DetA 0.496, AssA 0.553, IDF1 0.706,
  MOTA 0.619, 2,366 ID switches; with the >=45-frame filter IDF1 0.707 and 1,518 switches. The E-1 defaults
  give HOTA 0.542, IDF1 0.698 and 4,193 switches, so the new settings cut switches by 44% and move IDF1 only
  slightly; HOTA drops a little because the higher score threshold lowers DetA. At IoU 0.3 the candidate's
  IDF1 is 0.739.
- DDCF evidence on val: a perfect-association oracle on B's points (score 0.35) reaches IDF1 0.784, HOTA 0.564
  and 352 switches against the candidate's 0.706, 0.523 and 2,366, so better association has about 0.08 IDF1
  of headroom, and even perfect association leaves DetA at 0.503 (B's detections are the limit). The
  authors' own HOTA gain from DDCF is 0.01 (0.53 to 0.54). DDCF would also add frames and decoder output to E
  and an unmeasured speed cost. Recommended decision (mine to confirm): do not build DDCF now; revisit only
  if test run 2 gives a >=45-frame-filtered count error above the provisional 35% sign-off target.

- Config frozen in writing before test run 2 (DDCF: not building it now, revisit only if run 2's >=45-frame
  filtered count error exceeds 35%; I do not compare my ID switches with the authors' because of their 0.0
  threshold). Frozen config `ml/configs/tracker_e2_candidate.yaml` (sha256 59c6ebea4a1e3348...): score 0.35,
  min_hits 3, max_age 3.0 s (75 frames), gate 20 px, border margin 10 px, authors' Kalman noise, no altitude
  scaling, no CMC. It was chosen on val (optimistic: B trained on those clips), with the post-hoc coverage floor
  0.88 in step 3. Inputs: B's 30 test points files from the Kaggle run of 2026-10-14 (combined sha256
  1e3f32527c56f22f...), tracker code at commit 4104a95. Run 2 is a second look at DroneCrowd test and is
  labelled "second look, not blind": the test split was already scored by run 1 (48.5% unfiltered, 38.7%
  filtered), clips 011 and 075 were seen in the investigation, and the config was chosen knowing run 1.
  Both runs are reported, per clip and overall, with count error, IDF1, HOTA, MOTA and ID switches; the better
  one is never chosen on test. Sign-off reading: filtered count error <= 35% on test with IDF1 reported, from
  run 2's filtered mean next to run 1's, nothing called pass or fail until the Phase 0 sheet is signed.

- Test run 2 ("second look, not blind", frozen config) next to run 1, DroneCrowd test, 30 clips, no CMC, no
  altitude scaling. Count error, mean over clips: unfiltered 35.2% (run 1: 48.5%), >=45-frame filtered 37.9%
  (run 1: 38.7%), day-only filtered 35.2%, night clips unfiltered 18.2% and filtered 62.1%. TrackEval at IoU 0.5
  (mean over clips): HOTA 0.300 (run 1: 0.298), DetA 0.251, AssA 0.382, IDF1 0.388 (0.384), MOTA 0.236,
  5,439 ID switches (5,644); IDF1 0.448 at IoU 0.3. The longer miss limit mainly lowers the unfiltered count
  (fewer ids created: 5,871 against 6,828 for 5,092 people); identity quality is essentially unchanged. Sign-off
  reading as fixed before the run: filtered count error 37.9% is above the provisional 35% target (day-only
  35.2%, also just above) with IDF1 reported; nothing is called pass or fail until the sheet is signed. The
  DDCF trigger I set (filtered error above 35%) is met. The filtered view undercounts (3,455 ids for 5,092
  people) while the unfiltered one overcounts (5,871), so an absolute-value mean hides opposite signs. Test
  IDF1 is far below val (0.388 against 0.706) and DetA halves (0.251 against 0.496), in line with B's val-to-test
  gap; per-clip IDF1 is lowest (0.04 to 0.07) on the clips where B finds few people (17, 18, 74, 108) and
  highest (0.63 to 0.71) on 22, 23, 62, 75, 24. Per-clip table: `data/outputs/run_comparison/test_per_clip.csv`.

- Decision rule written before running the oracle on test (measurement only): perfect association on B's
  test points at score 0.35, frozen config unchanged (hash verified). DDCF is built only if the oracle's mean
  IDF1 (IoU 0.5, 30 test clips) is at least 0.15 above run 2's 0.388; otherwise not. The oracle uses
  ground-truth identities, so it is not a tracker and is labelled "oracle on test, measurement only". It is
  the third scoring of the test split (run 1, run 2, oracle) and nothing is changed afterwards.

- Oracle on test, measurement only (third scoring of the test split; the oracle uses ground-truth identities, is
  not a tracker, and is not a result of mine). Perfect association on B's test points at score 0.35, frozen
  config unchanged: mean IDF1 0.454 against run 2's 0.388 (+0.066), HOTA 0.335 against 0.300, AssA 0.456 against
  0.382, DetA 0.257 against 0.251, ID switches 1,371 against 5,439; at IoU 0.3 IDF1 0.512 against 0.448. The rule
  I set before running (DDCF built only if the gain is at least 0.15 IDF1) says do not build: no clip reaches
  +0.15 (largest +0.130, median +0.060), and on clips 17, 18, 74 and 108 even the oracle scores IDF1 0.045 to
  0.090 because B finds almost nobody there. Association was never the bottleneck on test; Model B's detections
  are (DetA 0.25 on test against 0.50 on val). DDCF is not built.
- Where this leaves the work: accept and report the limitation for the tracker (run 2 filtered count error 37.9%
  against the provisional 35%, IDF1 0.388, HOTA 0.300, the shortfall set by B's detections on night, dusk and
  large-scale clips), and treat improving B as the next investment, scoped before any GPU hours are spent
  (training on the 72 train clips would also give a clean val). E-4 (ID-stability clips) and E-5 (B+E speed on
  the 4070) are still to do, then integration.

- Decision after the oracle: DDCF is closed and not built (the rule I set before the run said no). For the
  tracker I accept and report the limitation (filtered count error 37.9% against the provisional 35%, IDF1
  0.388, HOTA 0.300, second look, not blind; a gap, not a pass). Improving B is the next investment, but a
  scoping note with a success rule and a Kaggle-hours cap comes first, before any training. Night stays out of
  scope; dusk is reported separately with no target. Next: E-5 (B+E speed, average and p95 frames/s per
  stream with 3 to 4 streams), then E-4 (ID-stability clips), then the B note.

- E-5 (B+E speed) is written and tested on CPU, but not measured: the 4070 is not available right now. No
  Kaggle interim number (a T4 miss would say little about the 4070). Status in every table: "not measured
  (4070 unavailable)". I will book the 4070 and run it myself. Script: `eval/benchmark_be.py`.
- E-4 (ID-stability clips): picked the clips by a rule written before rendering, so I do not cherry-pick: rank
  the 30 test clips by run-2 IDF1, take the 4 highest, the 4 lowest and 4 random (seed 0) from the rest; 10 s
  of footage each, from the frozen run-2 tracks. File names carry only the clip number, and the group and
  numbers are in a separate `index.csv` that I open after watching. My judgement is recorded below once I have
  watched them.

## Open items
- E-2a (altitude-scaled gate) and E-2b (camera-motion compensation, full translation, toggles), each
  tested on its own. Camera motion is small on all test clips, so a large effect is not expected.
- E-4 (ID-stability videos, best and worst clips, my judgement) and E-5 (B+E speed on the 4070 with nothing
  else running, target 5 frames/s per stream; Kaggle only as fallback).
- (Closed 18 Oct: DDCF not built, see above.) Earlier question: whether DDCF is needed for the count target, and whether to keep the 25% figure (the draft Phase 0
  sheet is unsigned and says E reaches it only with DDCF in the authors' results). Tell the supervisor.
- Decide dusk and night scope (not covered by the Phase 0 sheet).
- Ask Mahid whether camera-motion compensation can reuse his optical-flow output.
- CI has not run on this branch yet (it runs on pushes to `main` and on pull requests); open a pull request.
- Whether B should be trained on the 72 train clips only, to remove the val contamination; not done, GPU
  hours not measured.
