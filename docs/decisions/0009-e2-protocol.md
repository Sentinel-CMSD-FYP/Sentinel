# 0009 E-2 protocol and decisions after the E-3 baseline (fixed before the E-2 experiments run)
Date: 2026-10-18   Owner: Ahmed
Context: E-3 is approved as "baseline measured, target not met" (test count error 48.5%, decision 0008). CI passed on the merged PR #7. This file records the decisions that follow and fixes the E-2 experiments before they run.

## Decisions (Ahmed, 2026-10-18)
1. **Tracking-count target.** Keep 25% as a stretch goal only. The provisional sign-off target is a number that can be defended from the evidence: **tracking count error <= 35% with the >=45-frame filter, DroneCrowd test, all clips, and IDF1 reported alongside.** For reference, the baseline is 38.7% filtered (48.5% unfiltered); the authors' own no-DDCF DroneCrowd rows are 0.30 to 0.32 and 0.23 with DDCF. The Phase 0 sheet is an unsigned draft outside the repo; this target stays provisional and nothing is marked pass or fail until it is signed. DDCF is decided only after E-2 and the extended max_age results on val.
2. **Night and dusk.** Night is out of scope; dusk is reported separately with no target. Night clips are reported as a limitation, not hidden (test: clips 69, 70, 74; val has none). Dusk clips are those tagged dusk (val: 71, 73; test: none). Wording for the Phase 0 sheet: "Night is out of scope. Dusk is reported separately, with no target. Night results are reported as a limitation."
3. **CMC.** After E-2a, read `eval/optical_flow_farneback/` (Mahid's scripts, merged to main) and design CMC to take a flow field in and give a camera translation out, behind a small interface, so another flow model can be swapped in without rewriting CMC. The authors' halved translation is not copied (decision 0006).
4. **IDF1 and HOTA** are added to the plan: convert tracks to MOT text, evaluate with the TrackEval copy in `ml/third_party/uav-dot-track/eval/TrackEval` (imports on the current numpy: HOTA, CLEAR and Identity load; not yet run on real tracks), against the authors' MOT-format GT (same ids as our .mat GT).
5. **E-2a settings.** Gate and max_age are expressed in seconds or altitude-scaled pixels, with the toggles off by default. DroneCrowd benchmarking uses the 50/100 per-clip proxy; the pipeline uses the contract's 60 m (assumed). The max_age grid is extended on val first, because 25 frames was the edge of the step-2 grid.
6. **Second test scoring rule.** A second scoring of the DroneCrowd test split is allowed only under these conditions: (a) the E-2 comparison and the config are fixed on val, in writing, first; (b) both test runs are reported, and the better one is never chosen on test; (c) the second run is labelled "second look, not blind"; (d) own footage is the preferred next clean test, so the DroneCrowd test split is not used a third time.

## Limits to state
- Gate in metres is not possible here: it needs the ground sampling distance (camera field of view or intrinsics), which DroneCrowd does not provide. The gate is therefore expressed as pixels at a reference altitude and scaled with 1/altitude (the authors' rule `size = int(100/alt * 20)`), and the miss limit in seconds (`max_age_s`, converted with the clip's fps and frame_stride).
- The 50/100 proxy is the authors' object-size stand-in, not a measured altitude, and exists only for the test clips (authors' list in `data/manifests/object_size_tags.yaml`). The val clips need tags assigned by looking at one frame each (small or large), before any altitude result is computed. Until they are filled, altitude scaling cannot be compared on val.
- B's val points are optimistic (B trained on those clips) and clips 011 and 075 were seen in the investigation. Same caveats as decision 0008.

## Step 3: extended max_age on val (runs first)
Frozen: score_thr 0.35, min_hits 3 (the step-2 winner, `ml/configs/tracker_selected.yaml`), gate 20 px, border margin 10 px, no altitude scaling.
- Grid: `max_age_s` in {1.0, 1.5, 2.0, 3.0, 4.0} (25, 38, 50, 75, 100 frames at 25 fps). 1.0 s equals the step-2 winner and must reproduce its numbers (count error 54.7%, coverage 0.904, purity 0.955). Cap: 4 s, declared now.
- New metric, identity purity: over the rows of real tracks that sit on a GT head, the share that belong to the track's dominant GT person. 1.0 means no track ever jumped to another person. A long max_age can lower the id count by linking different people; the coverage floor cannot see that.
- Selection rule (fixed now): among configurations with mean coverage >= 0.90 and mean purity >= 0.935 (the step-2 winner's purity, measured descriptively at 0.955, minus 0.02), choose the lowest primary count error (no filter); configurations within 1 percentage point of the best are tied, and ties go to the smaller max_age_s. Full table reported.
- `eval/tracker_experiments.py --step 3` refuses the test split and writes `ml/configs/tracker_selected_e2.yaml`.

## Altitude comparison on val (after step 3, needs the val size tags)
Compare, at the step-3 config, `use_altitude` off against on with the per-clip proxy (large 50 m, small 100 m). Rule (fixed now): turn altitude scaling on only if the primary count error improves by at least 1 percentage point and the coverage and purity floors still hold; otherwise it stays off. The comparison is not run until Ahmed has filled the val size tags.

## Order of work
1. Step 3 on val (now). 2. Val size tags (Ahmed), then the altitude comparison. 3. Read Mahid's flow code and design the CMC interface and E-2b (tested on val with a synthetic-shift unit test; camera motion is small on all test clips, so a large effect is not expected). 4. IDF1/HOTA tooling, run on val. 5. DDCF decision on val evidence. 6. Freeze the final config in writing, then one second look at test, labelled "second look, not blind", with both test runs reported; own footage as the next clean test when available.

## Step 3 result: extended max_age on val (10 clips; B trained on these clips)
Run 2026-10-18, files in `data/outputs/tracker_experiments/step3/`. Frozen: score_thr 0.35, min_hits 3. The 1.0 s row reproduces the step-2 winner exactly (54.7%, coverage 0.904, purity 0.955).
| max_age_s | frames at 25 fps | count error, no filter | >=45-frame filter | ids per person | coverage | purity | spurious ids | extra ids |
|---|---|---|---|---|---|---|---|---|
| **1.0** | 25 | **54.7%** | 11.2% | 1.55 | 0.904 | 0.955 | 391 | 829 |
| 1.5 | 38 | 45.8% | 11.5% | 1.46 | 0.895 (below 0.90) | 0.948 | 355 | 689 |
| 2.0 | 50 | 43.1% | 11.4% | 1.43 | 0.893 (below) | 0.948 | 342 | 651 |
| 3.0 | 75 | 40.4% | 11.6% | 1.40 | 0.890 (below) | 0.946 | 328 | 611 |
| 4.0 | 100 | 39.4% | 11.6% | 1.39 | 0.887 (below) | 0.945 | 324 | 596 |
**Selected by the rule as written: max_age_s 1.0** (unchanged from step 2). Every longer value falls below the 0.90 coverage floor; all five pass the purity floor (0.935).
Facts:
- The primary count error falls steadily with a longer miss limit, from 54.7% to 39.4% at 4 s (15 points), and extra ids for tracked people from 829 to 596. The filtered view stays flat (11.2% to 11.6%).
- Purity stays at 0.945 or higher, so longer miss limits are not buying their gain by merging different people (on val).
- Coverage falls by 0.017 (0.904 to 0.887); the floor was set at 0.90 in advance and the first step below it is 0.895. The floor decided the outcome by a margin of 0.005 at 1.5 s.
- The rule is not changed after seeing this. Any change (for example a lower floor for step 3) would be a post-hoc decision and must be written down as one before the second look at test.
Selected config file written by the script: `ml/configs/tracker_selected_e2.yaml` (identical settings to `tracker_selected.yaml`, with the miss limit given as 1.0 s).

## POST-HOC decision (Ahmed, 2026-10-18, made after seeing the step-3 results on val): relax the step-3 coverage floor to 0.88, select max_age_s 3.0
- **What changed:** the coverage floor for step 3 only, from 0.90 to 0.88 (`selection.step3_min_coverage` in `ml/configs/tracker_experiments.yaml`, labelled post-hoc there). Nothing else in the rule changed: purity floor 0.935, lowest primary count error among eligible configurations, ties within 1 percentage point go to the smaller max_age_s.
- **Why:** step 3 showed a 15-point drop in the primary count error (54.7% at 1.0 s to 39.4% at 4.0 s) for 0.017 less coverage (0.904 to 0.887), with purity staying at 0.945 or higher. The 0.90 floor was set before any result and excluded every longer value by 0.005 at the first step.
- **What the amended rule selects:** with 0.88, all five values are eligible; the lowest error is 39.43% (4.0 s); 3.0 s (40.39%) is within 1 point of it, so the tie goes to the smaller value: **max_age_s 3.0** (coverage 0.8897, purity 0.9463). This is what the tool outputs (`ml/configs/tracker_selected_e2.yaml`); it was checked against the recorded numbers by a test.
- **Status of the evidence:** the floor was chosen after seeing step 3 on val, so this is a post-hoc choice on optimistic data (B trained on those clips). The step-3 selection as first written was 1.0 s; both outcomes are recorded here.
- **Before any second look at test:** this decision is logged here and in `docs/project-log.md`. The second look will be labelled "second look, not blind", and both DroneCrowd test runs will be reported: run 1 (score_thr 0.35, min_hits 3, max_age 25 frames, no CMC, no altitude scaling: 48.5% unfiltered, 38.7% filtered) and run 2 (the final E-2 config, fixed on val in writing first). The better run is never chosen on test. The altitude and CMC decisions are still open and must be settled on val before run 2.

## Altitude comparison on val: exact settings (fixed 2026-10-18, before the run; val size tags filled by Ahmed first)
- Val size tags (Ahmed): small = clips 6, 47, 52, 59, 67, 71, 73; large = clips 84, 85, 89 (proxy altitude 100 m and 50 m, gate 20 px and 40 px). Only three of ten val clips get a different gate, so the comparison is thin evidence.
- Configs compared: the step-3 config (`tracker_selected_e2.yaml`: score_thr 0.35, min_hits 3, max_age_s 3.0, gate 20 px) with `use_altitude` off, against the same with `use_altitude` on and the per-clip proxy altitude.
- Rule (as fixed earlier): altitude scaling is turned on only if the primary count error (no filter, mean over the 10 clips) improves by at least 1.0 percentage point and the floors still hold: coverage >= 0.88 and purity >= 0.935 (the step-3 floors, the coverage one being the post-hoc value). Otherwise it stays off.
- The comparison is val only (`eval/tracker_experiments.py --step 4`, refuses the test split) and writes the candidate config `ml/configs/tracker_e2_candidate.yaml`.

## Altitude comparison result on val (2026-10-18; 10 clips, B trained on these clips; only clips 84, 85, 89 get a different gate)
Config: step-3 config (score_thr 0.35, min_hits 3, max_age_s 3.0). Files: `data/outputs/tracker_experiments/step4/results.json`.
| | count error, no filter | >=45-frame filter | ids per person | coverage | purity | spurious ids | extra ids |
|---|---|---|---|---|---|---|---|
| altitude scaling off (gate 20 px everywhere) | 40.39% | 11.62% | 1.40 | 0.8897 | 0.9463 | 328 | 611 |
| altitude scaling on (size proxy: 20 px or 40 px) | 36.72% | 12.24% | 1.37 | 0.8784 | 0.9344 | 311 | 572 |
Gain on the primary error: +3.66 points (the rule asks for at least 1.0). Floors: coverage 0.8784 is below 0.88 (by 0.0016) and purity 0.9344 is below 0.935 (by 0.0006), so **by the rule as fixed in advance, altitude scaling stays OFF.** Candidate config written: `ml/configs/tracker_e2_candidate.yaml` (`use_altitude: false`).
Facts:
- The primary count error falls by 3.66 points with scaling on, but the >=45-frame view gets slightly worse (11.62% to 12.24%), and both floors are missed by hair margins, in the same direction as in step 3 (wider matching trades a little coverage and purity for fewer ids).
- Only three of ten clips changed their gate, so this is thin evidence either way, and the proxy is the authors' size stand-in, not a measured altitude.
- No floor was changed after seeing this result. A second post-hoc relaxation, directly after the first (step 3), would weaken the point of fixing rules in advance; if one is wanted it must be recorded as post-hoc before any second look at test.
- Altitude scaling is therefore not part of run 2 unless that changes; it should be re-tested on own footage with real altitude when available.

## Altitude scaling: status (Ahmed, 2026-10-18)
Approved: altitude scaling stays OFF by the pre-declared rule (val gain 3.66 points, but coverage 0.8784 and purity 0.9344 just under the floors 0.88 and 0.935). **It is untested on real altitude** (the proxy is the authors' object-size stand-in, and only 3 of 10 val clips got a different gate). To be re-tested on own footage with real altitude.

## CMC interface and E-2b protocol (fixed 2026-10-18, before any val shift is computed)
Design (`ml/src/sentinel_ml/tracking/cmc.py`, tested in `ml/tests/test_cmc.py`):
- Flow field in, camera shift out. `FlowInput(u, v, downscale, in_original_pixels)`: the downscale of the flow grid is an explicit field, and the flag says whether the vectors are in original-frame pixels (Mahid's `FlowField` convention) or in grid pixels (raw OpenCV output, divided by the downscale). `estimate_camera_shift(flow)` returns `CameraShift(dx, dy, spread, cells)`: how far a static point moved in the image, in original pixels, from the previous frame to the current one.
- A flow provider is any callable `(prev_frame, frame) -> FlowInput`, so another flow model can be swapped in without changing the CMC or the tracker. `FarnebackProvider` wraps Mahid's `sentinel_ml.flow.farneback.FarnebackFlow`; his `FlowField` hides the downscale, so the provider passes it on from his config.
- Estimator: median of the flow over the whole grid (robust to moving people while they cover less than half of it). Optional exclusion of cells around given points exists but is NOT used in the experiment.
- Tracker: `use_cmc` (off by default). The shift is added to the predicted track positions after the Kalman prediction and before the border check, as the FULL translation (the authors' halving is not copied, decision 0006). Velocity and covariance are untouched. `track_clip` needs the per-frame shifts when `use_cmc` is on.
- Note for Mahid's module (not edited by us): `FlowField` does not store its downscale; carrying it in the object would make the interface self-contained.

E-2b experiment on val (`eval/tracker_experiments.py --step 5`, refuses the test split):
- Flow model: Mahid's Farneback with his default config (`ml/configs/flow_farneback.yaml`: downscale 0.25, OpenCV default parameters), computed between consecutive frames of each val clip. Nothing is tuned.
- Configs compared: `tracker_e2_candidate.yaml` (score_thr 0.35, min_hits 3, max_age_s 3.0, altitude off) with `use_cmc` off against on.
- Rule (fixed now, same as for altitude): CMC is turned on only if the primary count error (no filter, mean over the 10 clips) improves by at least 1.0 percentage point and coverage >= 0.88 and purity >= 0.935 still hold; otherwise it stays off. The result is written to the candidate config.
- Expectation, stated in advance: camera motion is small on all 30 test clips (flow estimate at most 1.19 px per frame, GT common-mode motion at most 1.41 px, against a 20 px gate), so a large effect is not expected.
- The val frames (train folder) are on the USB drive, so it must be plugged in for this run.

After E-2b: IDF1 and HOTA tooling (TrackEval), run on val; the DDCF decision; then the final config is written down, and only then test run 2 ("second look, not blind").

## E-2b result on val: camera-motion compensation off vs on (2026-10-18; 10 clips, B trained on these clips)
Shifts: Mahid's Farneback, default config (downscale 0.25), median over the grid, computed between consecutive frames of each val clip (`eval/compute_camera_shifts.py`). The estimated shifts are tiny: per-clip median 0.02 to 0.06 px per frame, largest single frame 0.39 px. Config: candidate (score_thr 0.35, min_hits 3, max_age_s 3.0, altitude off). Files: `data/outputs/tracker_experiments/step5/results.json`.
| | count error, no filter | >=45-frame filter | ids per person | coverage | purity | spurious ids | extra ids |
|---|---|---|---|---|---|---|---|
| CMC off | 40.39% | 11.62% | 1.40 | 0.8897 | 0.9463 | 328 | 611 |
| CMC on | 40.44% | 12.68% | 1.40 | 0.8913 | 0.9456 | 322 | 624 |
Gain on the primary error: -0.06 points (the rule needs at least +1.0). Floors held (coverage 0.8913, purity 0.9456), but there is no gain, so **CMC stays OFF** by the rule fixed in advance. This matches the expectation written beforehand: camera motion is a fraction of a pixel per frame in these clips (and at most 1.19 px per frame on all 30 test clips), far below the 20 px gate.
Candidate config after E-2: `ml/configs/tracker_e2_candidate.yaml` = score_thr 0.35, min_hits 3, max_age_s 3.0, `use_altitude` false, `use_cmc` false.
Facts and limits:
- The unit test with a 30 px camera jump shows the CMC mechanism works (tracks survive with it, break without it). The footage simply does not move enough for it to matter. It would matter for footage with a shaking or panning drone, so the code stays, off by default, and should be re-tested on own footage.
- The filtered view is slightly worse with CMC on (12.68% against 11.62%), within noise for a 0.05 px correction.
- The interface was designed to take any flow model's field; only Mahid's Farneback was evaluated.
Remaining before the final config is fixed: IDF1/HOTA tooling and its val numbers, and the DDCF decision on val evidence.

## E-2 closed (Ahmed, 2026-10-18)
CMC stays off (val gain -0.06 points, below the +1.0 rule); it is untested beyond DroneCrowd val and is to be re-tested on own footage. The roadmap's video check for E-2 is **waived** because the estimated camera shifts are tiny (per-clip median 0.02 to 0.06 px per frame on val, at most 1.19 px per frame on the 30 test clips); there is nothing visible to compare with CMC on and off. Altitude scaling stays off (untested on real altitude, re-test on own footage). The E-2 candidate config is `tracker_e2_candidate.yaml`: score_thr 0.35, min_hits 3, max_age_s 3.0, no altitude scaling, no CMC.

## IDF1 / HOTA with TrackEval (val, 2026-10-18; B trained on these clips, so every number is val-optimistic)
Tool: `eval/trackeval_eval.py` (tests in `ml/tests/test_trackeval_eval.py`, 186 pass). Tracks and GT are written as MOT text with 20x20 boxes; GT comes from the .mat files (same ids as the authors' MOT GT; our boxes agree with the authors' on 94% of rows within 0.5 px and the rest are heads at the frame border, up to 6.5 px).
**Sanity checks first** (3 val clips: 6, 47, 52; 553 GT people):
| prediction | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
|---|---|---|---|---|---|---|
| GT as prediction (IoU 0.5) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| consistent relabel of ids | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| shuffled ids, IoU 0.5 | 0.091 | 0.999 | 0.009 | 0.033 | 0.021 | 138,270 |
| half the detections dropped, IoU 0.5 | 0.496 | 0.495 | 0.497 | 0.662 | 0.495 | 18 |
| **shuffled ids, threshold 0.0** | 0.091 | 0.999 | 0.009 | **1.000** | 0.942 | **106** |
All behave as they should (half dropped: IDF1 0.662 against the expected 2*1*0.5/1.5 = 0.667) except the last row. **Finding:** at the metric threshold 0.0, which the authors' `eval/track_eval.py` sets, TrackEval counts every pair of boxes as a potential match, so IDF1 and ID switches stop measuring identity (shuffled ids keep IDF1 1.0 and show about 1000 times too few switches). I did not run the authors' pipeline; this comes from reading their script and from running the same TrackEval code at 0.0 on our GT. Their ID-SW figures (6,645 without and 388 with DDCF) should therefore not be compared with ours or taken as a size of the gain. HOTA does not use this threshold. We use the standard IoU 0.5 as primary and 0.3 (about a 10 px offset, like B's F1 matching) as secondary.
**Val results** (10 clips, 1964 GT people; mean over clips; ids = tracker ids; filtered = ids output in >=45 frames, offline):
| tracker | IoU | HOTA | DetA | AssA | IDF1 | MOTA | ID switches | ids |
|---|---|---|---|---|---|---|---|---|
| E-1 defaults (score 0.2, max_age 5 frames) | 0.5 | 0.542 | 0.546 | 0.541 | 0.698 | 0.642 | 4193 | 5970 |
| E-1 defaults, filtered | 0.5 | 0.551 | 0.541 | 0.563 | 0.723 | 0.658 | 1870 | 2498 |
| step-2 selected (score 0.35, max_age 25 frames) | 0.5 | 0.523 | 0.496 | 0.554 | 0.702 | 0.618 | 2369 | 2977 |
| step-2 selected, filtered | 0.5 | 0.523 | 0.483 | 0.570 | 0.705 | 0.608 | 1297 | 1781 |
| **E-2 candidate (score 0.35, max_age 3.0 s)** | 0.5 | 0.523 | 0.496 | 0.553 | 0.706 | 0.619 | 2366 | 2664 |
| **E-2 candidate, filtered** | 0.5 | 0.523 | 0.487 | 0.566 | 0.707 | 0.612 | 1518 | 1755 |
| E-2 candidate | 0.3 | 0.523 | 0.496 | 0.553 | 0.739 | 0.670 | 2468 | 2664 |
| E-2 candidate, filtered | 0.3 | 0.523 | 0.487 | 0.566 | 0.740 | 0.659 | 1523 | 1755 |
Facts:
- Moving from the E-1 defaults to the E-2 candidate raises IDF1 only slightly (0.698 to 0.706) and cuts ID switches by 44% (4193 to 2366; filtered 1870 to 1518). HOTA falls from 0.542 to 0.523 because the higher score threshold (0.35) lowers DetA (0.546 to 0.496) while AssA rises a little (0.541 to 0.553).
- Filtering removes short ids and moves IDF1 by about 0.001 to 0.025; it does not change the picture.
- Matching at IoU 0.3 gives IDF1 0.739 against 0.706 at 0.5: B's points are within about 10 px but not within about 6.7 px of the GT centres for a part of the people.

## DDCF decision evidence on val (2026-10-18)
Question: would the authors' DDCF step (deep correlation-filter re-association; needs the video frames and B's raw decoder output) be worth building?
**Oracle experiment** (`--oracle`): B's detections with perfect association (each detection that matches a GT head within 10 px gets that head's true id; false detections get unique ids). It is the ceiling for any association method, DDCF included, on the same detections:
| val, IoU 0.5 | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
|---|---|---|---|---|---|---|
| E-2 candidate | 0.523 | 0.496 | 0.553 | 0.706 | 0.619 | 2366 |
| oracle association, score >= 0.35 | 0.564 | 0.503 | 0.633 | 0.784 | 0.626 | 352 |
| oracle association, score >= 0.2 | 0.604 | 0.544 | 0.671 | 0.819 | 0.631 | 516 |
At IoU 0.3 the same oracle gives IDF1 0.815 (score 0.35) and 0.870 (0.2), against 0.739 for the candidate.
Facts:
- The headroom for better association on these detections is about +0.08 IDF1, +0.04 HOTA, +0.08 AssA (IoU 0.5, score 0.35). Even perfect association leaves IDF1 at 0.784 and DetA at 0.503: the rest is limited by B's detections (missed people, false points).
- The authors report HOTA 0.53 without and 0.54 with DDCF on DroneCrowd test (their table), a gain of 0.01; their count-error gain (0.30 to 0.23) is larger, and their ID-switch gain is measured with the threshold that cannot count switches (see above).
- DDCF would need frames and decoder output inside E (the contract only covers outputs; the pipeline passes points), runs a correlation-filter tracker per unmatched track, and its speed on top of B is not measured against the 5 FPS target.
- B's weakest test clips (night, dusk, large-scale) are detection failures; DDCF recovers some missed positions, but the oracle shows the detection ceiling is where most of the loss sits.
**Recommended decision (Ahmed to confirm): do not build DDCF now.** Association has at most about 0.08 IDF1 of headroom on val, DDCF's own HOTA gain in the authors' results is small, and it adds frames and decoder output to E and an unmeasured speed cost. Revisit only if test run 2 (the final E-2 config, "second look, not blind") gives a >=45-frame-filtered count error above the provisional 35% sign-off target; if it does, DDCF (or a B improvement for night and dusk) is the next lever to compare.

## FROZEN: final E-2 config, written before test run 2 (Ahmed, 2026-10-18)
DDCF decision confirmed by Ahmed: **not building it now**; revisit only if run 2's >=45-frame-filtered count error exceeds 35%. Ahmed's rule stands: our ID switches are not compared with the authors' (their metric threshold of 0.0 cannot count switches).
**The frozen config** is `ml/configs/tracker_e2_candidate.yaml` (sha256 `59c6ebea4a1e3348d49fbf24cb2d9e97a756bc34ac4dc5b8b33a17494f121f3c`): score_thr 0.35, min_hits 3, max_age_s 3.0 (75 frames at 25 fps), gate size_px 20, border_margin_px 10, Kalman noise p_pos 10, p_vel 10000, q_pos 1, q_vel 0.5, r_meas 1, `use_altitude` false, `use_cmc` false. No value in it changes after this point, whatever run 2 shows.
**How it was chosen (all on val, optimistic, B trained on those clips):** score_thr from step 1 (0.35); min_hits 3 from step 2; max_age_s 3.0 from step 3 with the post-hoc coverage floor 0.88 (the rule as first written gave 1.0 s); altitude scaling and CMC left off by the rules fixed in advance (val gains 3.66 points with floors missed, and -0.06 points).
**Inputs:** B's test points, 30 files in `data/outputs/points/test/` from the Kaggle run of 2026-10-14 (released DOT+PD checkpoint, score floor 0.05); combined sha256 over file names and contents `1e3f32527c56f22fb826ef95d503919fcc3780fb3583b4248e0a0eb226a9bc5b`. Tracker code is committed at HEAD `4104a95` (uncommitted edits at this moment are only in evaluation tooling, labels and overwrite guards, and tests).
**Run 1** (already scored once, 2026-10-18): `ml/configs/tracker_selected.yaml` (sha256 `ecc35b2b2a6f7e8f5fdd845033778bf386292faf3812ab1638831a6e38edeeae`): score 0.35, min_hits 3, max_age 25 frames. Count error 48.5% unfiltered and 38.7% filtered.
**Run 2** = the frozen config, a **second look at the DroneCrowd test split, "second look, not blind"**: the test split was already scored by run 1, clips 011 and 075 were seen in the investigation, and the config was chosen with knowledge of run 1's result and of val. The better of the two runs is never chosen on test; both are reported.
**What is reported for both runs, per clip and overall:** the E-3 count error (mean over the 30 clips; unfiltered primary, >=45-frame filter secondary, day-only secondary; night limitation), and IDF1, HOTA (with DetA and AssA), MOTA and ID switches from TrackEval at IoU 0.5 (primary) and 0.3 (secondary), unfiltered and filtered. Commands: `sentinel_ml.tracking.run_tracker --config ml/configs/tracker_e2_candidate.yaml --split test --out-dir data/outputs/tracks_final/test`; `eval/track_count_error.py --split test --tracks-dir data/outputs/tracks_final/test --tracker-config ml/configs/tracker_e2_candidate.yaml --out-dir data/outputs/track_count_error_run2/test --note "run 2, second look, not blind"`; `eval/trackeval_eval.py --config ml/configs/trackeval_test.yaml --split test --allow-test`.
**Sign-off reading rule (fixed now):** the provisional sign-off target is a filtered count error of at most 35% on DroneCrowd test, all clips, with IDF1 reported. It is read from run 2's filtered mean and stated next to run 1's; nothing is called pass or fail until the Phase 0 sheet is signed. The DDCF trigger is the same number (filtered > 35%).

## RESULTS: test run 2 ("second look, not blind") next to run 1 (2026-10-18)
DroneCrowd official test split, 30 clips, 9000 frames, B's released checkpoint with the score floor 0.05 (Kaggle run of 2026-10-14), no CMC, no altitude scaling, DroneCrowd only. Run 1 = score 0.35, min_hits 3, max_age 25 frames. Run 2 = the frozen config (score 0.35, min_hits 3, max_age 3.0 s). Clips 011 and 075 were seen in the investigation; the test split had already been scored by run 1; B's released weights were probably selected using test frames. Per-clip table: `data/outputs/run_comparison/test_per_clip.csv` (tool `eval/compare_runs.py`). The files: `data/outputs/track_count_error*/test/`, `data/outputs/trackeval/test/results.json`.
**Count error** (E-3, mean over the 30 clips, unfiltered is primary):
| view | run 1 | run 2 (frozen) |
|---|---|---|
| unfiltered, all clips | 48.5% (median 48.6%, 1.33 ids per person) | **35.2%** (median 33.6%, 1.15 ids per person) |
| >=45-frame filter, all clips | 38.7% (median 30.4%) | **37.9%** (median 30.0%) |
| day-only (27 clips), unfiltered / filtered | 52.0% / 35.6% | 37.1% / 35.2% |
| night (3 clips: 69, 70, 74), unfiltered / filtered | 16.8% / 66.2% | 18.2% / 62.1% |
**IDF1, HOTA, MOTA, ID switches** (TrackEval, IoU 0.5, mean over clips; ids unfiltered; the ID-switch counts are ours only and are not to be compared with the authors'):
| | run 1 | run 2 (frozen) | run 1, filtered | run 2, filtered |
|---|---|---|---|---|
| HOTA | 0.298 | 0.300 | 0.297 | 0.299 |
| DetA | 0.250 | 0.251 | 0.235 | 0.239 |
| AssA | 0.377 | 0.382 | 0.408 | 0.406 |
| IDF1 | 0.384 | 0.388 | 0.378 | 0.382 |
| MOTA | 0.235 | 0.236 | 0.229 | 0.231 |
| ID switches | 5644 | 5439 | 3215 | 3611 |
| tracker ids (GT people 5092) | 6828 | 5871 | 3470 | 3455 |
At IoU 0.3: IDF1 0.443 (run 1) and 0.448 (run 2), HOTA the same as above.
Facts:
- The longer miss limit lowers the unfiltered count error by 13 points (48.5% to 35.2%) because fewer ids are created (6828 to 5871 for 5092 people), but identity quality is essentially unchanged: IDF1 0.384 to 0.388, HOTA 0.298 to 0.300, AssA 0.377 to 0.382. The >=45-frame filtered error barely moves (38.7% to 37.9%).
- **Sign-off reading** (fixed before the run): the provisional target is a filtered count error of at most 35% on DroneCrowd test with IDF1 reported. Run 2's filtered mean is **37.9%, above 35%** (day-only filtered 35.2%, also just above), with IDF1 0.382 to 0.388. The 25% stretch is also above. Nothing is called pass or fail until the Phase 0 sheet is signed. **The DDCF trigger set in advance (filtered error above 35%) is met.**
- The filtered view undercounts: 3455 ids against 5092 GT people (-32%), so its error is mostly people who are never tracked or tracked for fewer than 45 frames; the unfiltered view overcounts (+15%). An absolute-value mean hides the opposite signs.
- The gap to val is large: on val the candidate had IDF1 0.706, HOTA 0.523, DetA 0.496; on test the same config has IDF1 0.388, HOTA 0.300, DetA 0.251. Detection quality (DetA) halves, which matches Model B's val-to-test F1 gap (0.86 to 0.69) and its weakness on night, dusk and large-scale clips.
- Per clip, IDF1 runs from 0.041 to 0.709 (run 2). The lowest are the clips where B finds few people: clip 17 (IDF1 0.041, 37 ids for 175 people), 18 (0.070), 74 (0.063), 108 (0.060, 73 ids for 239 people), then 35, 69, 34, 70, 105, 111, 112 (about 0.20 to 0.25). The best are 22, 23, 62, 75, 24 (0.63 to 0.71). On the clips where B finds the people, both runs score alike; on the clips where it does not, neither run can track what was never detected.

## DECISION RULE for the oracle on test (Ahmed, 2026-10-18, written before the oracle is run)
Context: run 2's filtered count error (37.9%) is above the provisional 35% target, so the DDCF trigger is met. Before building anything, Ahmed chose to measure how much better association could possibly do on the test detections, **as a measurement only**.
- **What is run:** the perfect-association oracle (`eval/trackeval_eval.py --oracle`) on B's 30 test points files (same inputs as run 2, digest `1e3f3252...`), at score >= 0.35 (the frozen config's score threshold): each detection that matches a GT head within 10 px gets that head's true id, every false detection gets a unique id. It uses ground-truth identities, so it is not a tracker and its numbers are never reported as a result of ours. Label: **"oracle on test, measurement only"**.
- **Frozen config unchanged:** `ml/configs/tracker_e2_candidate.yaml` (sha256 `59c6ebea4a1e3348d49fbf24cb2d9e97a756bc34ac4dc5b8b33a17494f121f3c`, verified before this note was written). Nothing in the tracker is touched.
- **The rule (fixed now):** gain = mean-over-clips IDF1 of the oracle minus mean-over-clips IDF1 of run 2 (unfiltered, 0.388), TrackEval IoU 0.5, DroneCrowd test, 30 clips. **DDCF is built only if the gain is at least 0.15 IDF1.** Below 0.15 it is not built. The gain is an upper bound for any association method (DDCF would reach only part of it), which makes the 0.15 bar generous for DDCF, not strict.
- **Also reported, not part of the rule:** HOTA, DetA, AssA, MOTA and ID switches at IoU 0.5, IDF1 at IoU 0.3, and the per-clip IDF1 and gain for every clip.
- Counts as a further look at test (the third scoring of this split: run 1, run 2, oracle); no config, threshold or rule is changed afterwards.
- After the number: a recommendation among option 1 (build DDCF), option 3 (put the effort into B for night, dusk and large-scale footage) and option 4 (accept and report the limitation). The rule decides option 1 only.

## RESULT: oracle on test, measurement only (2026-10-18)
Frozen config unchanged (`tracker_e2_candidate.yaml`, sha256 verified again before the run: `59c6ebea...121f3c`). Perfect-association oracle on B's 30 test points files at score >= 0.35, scored with `eval/trackeval_eval.py --oracle`; verdict from `eval/oracle_gap.py`; per clip: `data/outputs/run_comparison/test_oracle_gap_per_clip.csv`. **The oracle uses ground-truth identities; it is not a tracker and its numbers are not a result of ours.** This is the third scoring of the DroneCrowd test split (run 1, run 2, oracle).
| TrackEval IoU 0.5, mean over 30 clips | run 2 (frozen) | oracle, measurement only | gain |
|---|---|---|---|
| IDF1 | 0.388 | 0.454 | **+0.066** |
| HOTA | 0.300 | 0.335 | +0.035 |
| AssA | 0.382 | 0.456 | +0.074 |
| DetA | 0.251 | 0.257 | +0.006 |
| MOTA | 0.236 | 0.239 | +0.003 |
| ID switches | 5439 | 1371 | |
At IoU 0.3: IDF1 0.448 (run 2) against 0.512 (oracle), +0.064.
**Rule (fixed before the run): DDCF is built only if the oracle gains at least 0.15 IDF1 over run 2. The gain is +0.066, so the rule says DO NOT BUILD.**
Facts:
- No clip reaches the bar: the largest gain is +0.130 (clip 063), the median +0.060, and 0 of 30 clips gain 0.15 or more. The largest gains are on clips 63, 43, 24, 42, 15 (+0.11 to +0.13).
- Even with perfect association the mean IDF1 is only 0.454 and DetA 0.257. The limit is the detections: of the GT person-frames, the oracle's detections miss 770k of about 1.24M at IoU 0.5 (run 2: 778k), and 161k detections are false (run 2: 153k). Association was never the bottleneck on test.
- On clips 17, 18, 74 and 108 the oracle's IDF1 is only 0.045 to 0.090 (run 2: 0.041 to 0.070): B finds almost nobody there, so nothing can be tracked.
- Association itself is not perfect in run 2 (AssA 0.382 against 0.456, 5439 ID switches against 1371), but fixing all of it would add 0.066 IDF1, not the 0.15 that was set as the bar.

## Recommendation (Claude, for Ahmed to decide)
- **Option 1, build DDCF: no.** The rule says no, and the evidence agrees on val (+0.08 headroom) and on test (+0.066), where the authors' own HOTA gain was 0.01. It would also add frames and decoder output to E and an unmeasured speed cost.
- **Option 4, accept and report the limitation: yes, now.** Run 2's filtered count error is 37.9% against the provisional 35%, with IDF1 0.388, HOTA 0.300; the oracle shows the shortfall is set by Model B's detections, and the report should say so. The 25% stretch and the 35% target both stay provisional until the sheet is signed.
- **Option 3, improve B for night, dusk and large-scale footage: yes, as the next investment, scoped before any GPU hours are spent.** It is the only lever the numbers point at (DetA 0.25 on test against 0.50 on val; four clips at IDF1 below 0.1). It has costs not yet measured: training our own model on the 72 train clips (which would also remove the val contamination and give a clean val), GPU hours on Kaggle (one-epoch timing first), and the risk that it does not generalise. A time-boxed first step is a one-epoch timing run and a small fine-tune, evaluated on the 10 val clips that the new model has not seen, plus own footage when it exists.
- Whatever is chosen, E still needs its remaining roadmap items: E-4 (ID-stability clips, Ahmed's judgement) and E-5 (the B+E speed benchmark on the 4070, target 5 frames/s), then I-1 (integration).

## DECISION (Ahmed, 2026-10-18): DDCF closed, option 4 for E, B is the next investment
- **DDCF is closed and not built.** The rule fixed before the oracle run said do not build (gain +0.066 against the 0.15 bar). It is not reopened unless new evidence arrives (for example much better detections from a retrained B, or own footage where association is the visible failure).
- **Option 4 (accept and report the limitation) is the decision for Model E.** The frozen config `tracker_e2_candidate.yaml` is final for E. Reported as: run 2 filtered count error 37.9% against the provisional 35% (25% stretch also not met), IDF1 0.388, HOTA 0.300, DroneCrowd test, second look, not blind; the shortfall is set by B's detections (oracle IDF1 0.454, DetA 0.257). It is a gap, stated plainly, not a pass.
- **B is the next investment, scoped first.** A scoping note with a success rule and a Kaggle-hours cap is written before any training or GPU hour is spent on it (see `docs/decisions/0010-b-scoping.md` when written).
- **Night is out of scope.** Dusk is reported separately, with no target. Night results are reported as a limitation. (The Phase 0 sheet is not in the repo; Ahmed to paste its wording.)
- **Order of work:** E-5 (B+E speed on the 4070, average and p95 frames/s per stream, warm-up skipped, 3 to 4 concurrent streams), then E-4 (ID-stability clips, Ahmed's judgement), then the B scoping note.
