# The tracking count-error target: what needs deciding

Prepared 26 November 2026 for the project supervisor. Sentinel, Tier 1, Model E (tracking). Decision support only.

## 1. Decisions needed
1. **Which tracking count-error target goes into the signed Phase 0 sheet: 25%, or 35% with 25% as a stretch goal?** We miss both.
2. **Do you approve the scope: night out of scope, dusk reported separately with no target?**
3. **Will you sign the Phase 0 sheet?** It is still an unsigned draft, so every target in it is provisional.

## 2. What "count error" means
The tracker gives each person it follows an ID number, so the number of IDs is its count of people. For one video clip:

> error = |number of IDs - real number of people| / real number of people

The errors are averaged over all test clips. Lower is better.

Example (clip 22, all IDs): 342 IDs for 283 people gives (342 - 283) / 283 = 20.8%.

The **filtered** figure counts only IDs seen in at least 45 frames (under 2 seconds at 25 frames per second), which removes short-lived tracks.

## 3. What we measured
DroneCrowd official test split, 30 clips, 5,092 people. Model B (head detection) uses the authors' released weights.

| | Run 1 | Run 2 (current) |
|---|---|---|
| Count error, filtered (45 frames or more) | 38.7% | **37.9%** |
| Count error, every ID | 48.5% | 35.2% |
| Total IDs, every ID | 6,828 | 5,871 |
| IDF1 (identity quality, 0 to 1) | 0.384 | 0.388 |
| HOTA (overall tracking quality, 0 to 1) | 0.298 | 0.300 |

Run 2 used settings frozen after run 1. It is a **second look at the test split, not a blind test**.

Further facts about run 2:
- **Against the targets:** 37.9% misses 35% by 2.9 points and 25% by 12.9 points.
- **The error runs both ways.** Filtered, the tracker reports 3,455 IDs for 5,092 people (too few). Unfiltered it reports 5,871 (too many). The formula ignores the sign.
- **Without night clips the target is still missed:** day-only filtered error is 35.2%. Night clips (69, 70, 74) alone are 62.1% filtered and 18.2% unfiltered.
- **Clips differ a lot:** with the filter, 17 of 30 clips are at or below 35% and 12 of 30 at or below 25%.

| Clip | People | IDs | Error, every ID | Error, filtered | IDF1 |
|---|---|---|---|---|---|
| 23 | 296 | 384 | 29.7% | 0.7% | 0.709 |
| 22 | 283 | 342 | 20.8% | 8.1% | 0.692 |
| 74 (night) | 123 | 88 | 28.5% | 81.3% | 0.063 |
| 17 | 175 | 37 | 78.9% | 93.1% | 0.041 |
| 108 | 239 | 73 | 69.5% | 94.1% | 0.060 |

**A low count error does not mean identities are correct.** Clip 23 has 0.7% filtered error, yet our later analysis found about 2 identity switches per person on it. We judge identity stability as "met by eye only".

## 4. How the target changed
| Date | Event |
|---|---|
| 25 September | Draft Phase 0 sheet: tracking count error at most **25%** (unsigned). |
| Test run 1 | Filtered error **38.7%**; only 6 of 30 clips at or below 25% on the unfiltered measure. |
| 18 October | **After seeing run 1**, 25% was moved to a stretch goal and **35%** (filtered, IDF1 also reported) became the provisional target. |
| Test run 2 | Frozen settings: **37.9%**, still above 35%. |

So the 35% figure was **not** fixed before the first test result. It was fixed before run 2.

Scope also changed: at the 11 October team kickoff night footage was not excluded. The log of 18 October puts night out of scope.

## 5. Why we miss
- **The detector is the main limit.** We gave the tracker perfect identities using the true answers (an "oracle": a measurement, not a usable tracker). IDF1 rose only from 0.388 to 0.454 (+0.066). Better matching alone cannot close the gap in identity quality. The oracle was scored on IDF1 and HOTA, not shown here as a count-error result.
- **Detection is much weaker on test than on validation:** detection score (DetA) 0.251 on test against 0.496 on validation. Validation is optimistic because the released detector was trained on those clips.
- **Weak clips:** on night, dusk and large-scale scenes the detector finds few people (for example clips 17, 74 and 108 above).

## 6. Reference figure from other work
The model authors report count error of 30% to 32% without an extra re-matching step (called DDCF), and 23% with it. These figures are as quoted in our project log; they were not re-checked for this document. Their setup differs from ours, so this is not a like-for-like comparison. We did not build DDCF, because the oracle test showed too small a possible gain by our rule (set before running it).

## 7. Options
| Option | What it means | Point to weigh |
|---|---|---|
| A. Keep **25%** | Report a clear gap (37.9%). | It is the original figure. The gap is large. |
| B. Adopt **35%**, 25% as stretch | Report a smaller gap (2.9 points). The report already uses this. | Must be stated as set after a first test result. It is still missed. |
| C. Fix a new rule now | Choose the target and conditions today and apply them to a future evaluation on data not yet looked at. | Cleanest evidence, but gives no pass or fail for Tier 1. |

Whichever is chosen, the current result is reported as measured.

## 8. Plan
Improve the head detector in Tier 2 by training on our own data split. Before any training we will write down a success rule and a cap on GPU hours on Kaggle. Neither is written yet.

## 9. Supervisor's answer
- Target (25% / 35% with 25% stretch / other): ______________________
- Night out of scope, dusk reported separately (approve / change): ______________________
- Phase 0 sheet signed (yes / no): ______________________
- Comments: ______________________
- Name and date: ______________________

## Sources
`docs/project-log.md` (entries of 11 and 18 October); `docs/decisions/0005-count-error-protocol.md` (draft sheet targets), `0009-e2-protocol.md`, `0013-id-switch-analysis.md`; per-clip values from `data/outputs/run_comparison/test_per_clip.csv` (local file); FYP-1 report, Table 7.5.
