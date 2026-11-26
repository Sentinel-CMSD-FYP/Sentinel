# 0008 Tracker experiments on val: protocol (fixed before any experiment runs)
Date: 2026-10-18   Owner: Ahmed
Context: after E-1 (baseline tracker) and the E-3 val baseline (decision 0006), Ahmed asked for val experiments first (a `score_thr` sweep, then the confirmation and max-age protocol) and one final test scoring with the chosen config. This file fixes the grids, the selection rule and the caveats before any result exists.

## Facts and caveats that go with every number below
1. **Clips 011 and 075 were seen in the investigation** (`eval/diagnose_tracking.py`, decision 0006) and in the E-1 trail videos. Both are test clips. They were used for diagnosis only and nothing was tuned on them, but the final test number is not from clips nobody has looked at. Report it that way.
2. **Val is optimistic because Model B was trained on the val clips** (released checkpoint, official train folder). B's detections are cleaner on val than on unseen footage, so values chosen on val may suit cleaner detections than test will have (for example a lower `score_thr` or smaller `min_hits` than test would want).
3. No tuning on test. Test is scored once, with the config frozen from val.
4. GT person count = distinct person ids in the clip's GT .mat files (confirmed by Ahmed, 2026-10-18; the authors' MOT-format GT has the same ids).
5. CI note: `.github/workflows/ci.yml` runs on pushes to `main` and on any pull request, so a PR from the feature branch triggers it (a draft PR is enough).

## Ahmed's observations of the E-1 trail videos (clips 011 and 075, 5 s each, 2026-10-18)
Scale used by Ahmed: "few" = a couple of cases; "some" = more than a couple, roughly 2 in 10 people.
- Clip 011: a few colour changes while a person is moving; when two people move close together or pass each other they share one colour for some frames and have two colours in others, and after two people pass very close one dot disappears and later reappears with a new colour. Some people standing near buildings or under shade (partially visible) never get a colour for the whole clip.
- Clip 075: the only problems are near the border and in shade: a few people change colour close to the border; some flicker in colour when moving partly under shade or very close to a building.
- Ahmed also saw a lone purple trail with no dot. **That was a bug in `eval/render_trails.py`** (a track's history was never forgotten, so after a track ended its last trail stayed on screen). Fixed 2026-10-18: a trail is now drawn only while its track is output, with a test. The "disappear and reappear with a new colour" and "never marked" observations are not renderer artifacts.
- These match the investigation: track competition and B dropouts in crowded or shaded spots (decision 0006). The ID colour changes near borders are consistent with the 10 px border rule (a few ids), though the rule's measured effect on counts was small.

## Experiment design (val only; `eval/tracker_experiments.py` refuses the test split)
Everything else stays at the baseline defaults: gate `size_px` 20, Kalman noise values, `border_margin_px` 10. E-2a (altitude gate) and E-2b (CMC) are separate tasks.
- **Step 1, `score_thr` sweep** (tracker-level threshold on B's points, saved down to 0.05): 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, with `min_hits` 3 and `max_age` 5.
- **Step 2, confirmation and max-age grid** at the `score_thr` chosen in step 1: `min_hits` in {3, 5, 8, 12} x `max_age` in {5, 10, 15, 25}. Caps declared now: `min_hits` <= 12 (0.48 s at 25 fps, the extra reporting delay for a new track) and `max_age` <= 25 (1 s).
- Metrics per configuration, mean over the 10 val clips: E-3 count error without filter (primary) and with the 45-frame filter (secondary), ids per person, GT-people coverage (fraction of GT people covered by at least one real id, matched within 10 px), spurious ids, extra ids for already-tracked people.
- **Selection rule (fixed now):** among configurations with mean coverage >= 0.90, choose the one with the lowest primary count error. Any configuration within 1 percentage point of the minimum counts as tied. Tie-break in step 2: the smaller `min_hits`, then the smaller `max_age`. Tie-break in step 1 (added before any result existed, because step 1 varies only `score_thr`): the `score_thr` closest to 0.20, the value chosen for B on val in decision 0004, then the lower one. The coverage floor stops the rule from rewarding a configuration that only lowers the id count by dropping people. The full tables are reported, not just the winner.
- After selection the chosen values are written to `ml/configs/tracker_selected.yaml` and not changed again. The final test run uses that file once.

## Open questions
- E-2a and E-2b (altitude gate, CMC) would change the tracker; if they are to be part of the final config they must happen before the single test scoring.
- The 25% tracking-count figure comes from the unsigned draft Phase 0 sheet; nothing is called pass or fail.

Evidence: Ahmed's message in session on 2026-10-18; `ml/configs/tracker_experiments.yaml` (the grids above).

## Step 1 result: score_thr sweep (val, 10 clips, 3000 frames, min_hits 3, max_age 5; B trained on these clips)
Run 2026-10-18, files in `data/outputs/tracker_experiments/step1/`. The 0.20 row reproduces the E-3 baseline (205.2%), which checks that the runner matches the metric.
| score_thr | count error, no filter | count error, >=45-frame filter | ids per person | GT-people coverage | spurious ids | extra ids for tracked people |
|---|---|---|---|---|---|---|
| 0.10 | 411.2% | 65.7% | 5.11 | 0.961 | 5857 | 1977 |
| 0.15 | 281.3% | 43.9% | 3.81 | 0.960 | 3551 | 2028 |
| 0.20 | 205.2% | 25.9% | 3.05 | 0.961 | 2065 | 2019 |
| 0.25 | 167.3% | 14.6% | 2.67 | 0.951 | 1341 | 2094 |
| 0.30 | 147.4% | 7.2% | 2.47 | 0.939 | 947 | 2120 |
| **0.35** | **130.0%** | 10.1% | 2.30 | 0.919 | 636 | 2114 |
| 0.40 | 116.6% | 19.8% | 2.17 | 0.878 (below the 0.90 floor) | 439 | 2108 |
| 0.50 | 90.3% | 42.2% | 1.90 | 0.746 (below the floor) | 172 | 2062 |

**Selected by the pre-declared rule: score_thr = 0.35** (lowest primary error among configurations with coverage >= 0.90; no ties within 1 point).
What the table shows (facts, not verdicts):
- Raising `score_thr` removes spurious ids fast (5857 to 636) but costs coverage of real people (0.961 to 0.919 at the selected value, 0.746 at 0.50).
- Extra ids for already-tracked people stay at about 2000 to 2100 whatever the threshold (about 1.07 per GT person; the 10 val clips have 1964 GT people). So thresholding does not fix identity breaks; step 2 (min_hits, max_age) is where that must be addressed.
- The primary error is still 130.0% at the selected value (2.30 ids per person). The secondary (>=45-frame) view is lowest at 0.30 (7.2%), not at the selected 0.35 (10.1%); the 0.35 choice follows the pre-declared primary rule.

## Step 2 result: min_hits x max_age grid at score_thr 0.35 (val, 10 clips; B trained on these clips)
Run 2026-10-18, files in `data/outputs/tracker_experiments/step2/`.
| min_hits | max_age | count error, no filter | >=45-frame filter | ids per person | coverage | spurious ids | extra ids |
|---|---|---|---|---|---|---|---|
| 3 | 5 | 130.0% | 10.1% | 2.30 | 0.919 | 636 | 2114 |
| 3 | 10 | 86.5% | 10.2% | 1.87 | 0.912 | 505 | 1378 |
| 3 | 15 | 68.7% | 10.8% | 1.69 | 0.909 | 450 | 1075 |
| **3** | **25** | **54.7%** | 11.2% | 1.55 | 0.904 | 391 | 829 |
| 5 | 5 | 94.1% | 10.6% | 1.94 | 0.900 | 422 | 1644 |
| 5 | 10 | 62.1% | 10.9% | 1.62 | 0.892 (below floor) | 349 | 1095 |
| 5 | 15 | 49.8% | 11.5% | 1.50 | 0.890 (below floor) | 319 | 884 |
| 5 | 25 | 39.1% | 11.8% | 1.39 | 0.885 (below floor) | 284 | 692 |
| 8 | 5 | 66.9% | 11.7% | 1.67 | 0.881 (below floor) | 278 | 1275 |
| 8 | 10 | 44.9% | 12.7% | 1.45 | 0.872 (below floor) | 247 | 896 |
| 8 | 15 | 36.1% | 13.1% | 1.36 | 0.868 (below floor) | 232 | 744 |
| 8 | 25 | 28.0% | 13.0% | 1.28 | 0.864 (below floor) | 212 | 589 |
| 12 | 5 | 48.9% | 14.1% | 1.49 | 0.858 (below floor) | 197 | 1022 |
| 12 | 10 | 31.3% | 14.8% | 1.31 | 0.851 (below floor) | 184 | 716 |
| 12 | 15 | 24.6% | 14.8% | 1.25 | 0.848 (below floor) | 173 | 602 |
| 12 | 25 | 20.4% | 14.2% | 1.19 | 0.845 (below floor) | 163 | 490 |

**Selected by the pre-declared rule: score_thr 0.35, min_hits 3, max_age 25** (lowest primary error among configurations with coverage >= 0.90; no ties within 1 point). Written to `ml/configs/tracker_selected.yaml` and frozen.
What the table shows (facts, not verdicts):
- Longer `max_age` is what reduces identity breaks: extra ids for tracked people fall from 2114 (max_age 5) to 829 (max_age 25) at min_hits 3. The selected `max_age` 25 is the top of the grid (the cap declared in advance, 1 s at 25 fps), so the trend was not explored further.
- Larger `min_hits` removes spurious ids but costs coverage of real people (0.919 down to 0.845). The 0.90 coverage floor decides the selection here: configurations below it have lower primary error (down to 20.4% at min_hits 12, max_age 25) but track fewer of the GT people. The floor was declared before any result and is not changed afterwards.
- The primary error of the selected configuration is still 54.7% (1.55 ids per person) on val, where B's detections are optimistic.
- The secondary (>=45-frame) view is nearly flat across the grid (10.1% to 14.8%).

## Camera motion on all 30 test clips (measurement only, 2026-10-18, `eval/camera_motion_check.py`, 300 frames per clip)
Optical-flow camera translation estimate (the authors' method, half-resolution): across the 30 test clips the largest value is 1.19 px per frame (clip 17) = 6.0% of the 20 px gate; per-clip medians are 0.05 to 0.59 px. Largest GT common-mode motion (median step of all GT people per frame pair): 1.41 px. Share of GT per-person steps above the 20 px gate: at most 0.057% in any clip. So camera motion is small compared with the gate in every test clip. This is a measurement of the footage, nothing was tuned on it. It does not measure slow drift accumulated over many frames.
Output: `data/outputs/camera_motion/test.json`.

## Final test result (single scoring, 2026-10-18)
**DroneCrowd official test split, 30 clips, 9000 frames, tracker on B points (score_thr 0.35, min_hits 3, max_age 25), no CMC, no altitude scaling, DroneCrowd only.** Config frozen from val (`ml/configs/tracker_selected.yaml`), scored once. Clips 011 and 075 were seen in the investigation, and the val selection is optimistic because B trained on the val clips (see the caveats at the top). Files: `data/outputs/track_count_error/test/`; tracks in `data/outputs/tracks_selected/test/`.
| view | clips | mean error over clips | median | ids per person |
|---|---|---|---|---|
| all clips, no filter (primary) | 30 | **48.5%** | 48.6% | 1.33 |
| all clips, >=45-frame filter (secondary, offline) | 30 | 38.7% | 30.4% | |
| day-only, no filter / filtered | 27 | 52.0% / 35.6% | 51.0% / 30.0% | 1.36 |
| night, no filter / filtered | 3 | 16.8% / 66.2% | 11.4% / 65.9% | 1.09 |
The provisional 25% tracking-count figure (unsigned draft Phase 0 sheet) is shown for reference only; nothing is marked pass or fail. 6 of 30 clips have a primary error of 25% or less. Same tracker on val: 54.7% (the step-2 row), so the primary error is not lower on test.
Facts about the result:
- Errors have both signs. Total output ids 6828 against 5092 GT people (net +1736), but 5 clips under-count: clip 17 (51 ids for 175 people, 0.29 ids per person), 18 (0.49), 35 (0.70), 74 (0.89) and 108 (83 ids for 239 people, 0.35). 12 clips have more than 1.5 ids per person, and 3 clips are within 1.0 to 1.1 (34, 70, 103). An absolute-value mean hides the opposite signs.
- The under-counting clips match the Model B findings (decision 0004): B is under-confident on night and dusk or large-scale scenes (clips 17, 18, 74 and 35 were among its weakest). At score_thr 0.35 many of their people are dropped, because the threshold was selected on val clips where B is confident. The val optimism caveat materialised: the val-selected 0.35 did not transfer to these clips. Nothing was changed after seeing this (no tuning on test).
- In the filtered view the under-counting clips look much worse (for example clip 17: 92.0%), because the filter removes even more ids there.
- The baseline defaults (score_thr 0.2, min_hits 3, max_age 5) were scored on val only (205.2%), not on test, so no test baseline comparison exists. A second test scoring of another config would break the score-once rule.
- E-2a (altitude gate) and E-2b (CMC) are not part of this result. The camera-motion check (above) found camera motion small on all 30 test clips, so a large effect from CMC is not expected, but that was not measured.
