# 0005 Count-error protocol (fixed before any count result)
Date: 2026-10-14   Owner: Ahmed
Context: after the B-4 test result, Ahmed set how the point-count error is measured, so nobody can adjust the definition after seeing numbers. Written before `eval/count_error.py` was run on real predictions.

## Decisions
1. **What is measured:** per frame, B's point count at score >= 0.2 against the GT count. Relative error = |predicted - GT| / GT; signed bias = (predicted - GT) / GT (negative = under-counting). Frames with GT = 0 are skipped and counted. This is B's point count, not Model E's tracking count error.
2. **Density bins, from each frame's GT count:** sparse < 100, moderate 100 to 200 (inclusive), dense > 200. Fixed now; not to be changed after results.
3. **Threshold:** 0.2 (decision 0004). The threshold sweep is closed. The val sweep is contaminated by training on the val clips, so it only tells us 0.2 is the repo default. The 0.05 run on test was a diagnostic only, nothing was selected from it.
4. **No tuning on test.** There is no unseen data besides test and no own footage yet, so ideas to split test into tuning and reporting halves, or to tune on own footage later, are withdrawn.
5. **Run order:** val first, then test once. `count_error.py` refuses to overwrite an existing test result.
6. **Lighting tags:** every clip is tagged day, dusk or night by looking at one frame, without scores (contact sheets from `eval/contact_sheet.py`, tags in `data/manifests/lighting_tags.yaml`), ideally by someone who has not seen the results, or by two people independently. Claude has seen scores and some frames of the test clips, so Claude does not tag. Tags are not changed after results exist.
7. **Reporting:** every table shows the numbers two ways: all clips (primary, nothing hidden) and day-only clips (secondary). Clips are never dropped after seeing results. Tables also show clips and frames per bin, and clips per lighting tag.
8. **Targets:** the draft Phase 0 sheet (`sentinel-phase0-target-sheet.md`, section 4.2, an unsigned draft by Ahmed dated 25 Sept 2026, not in the repo) lists count error <= 15% for sparse/moderate, <= 25% for dense, and <= 25% for tracking. Until signed, they are shown as "provisional, draft Phase 0 sheet" and nothing is called pass or fail.
9. **No F1 target.** The earlier suggestion (F1 >= 0.70, no clip below 0.5) is withdrawn: 8 of 30 test clips are already below 0.6, mostly night and dusk, so it would fail for reasons unrelated to our pipeline. Revisit when we have our own footage.

## Lighting tags used
Tagged by Ahmed on 2026-10-14 from the contact sheets. Ahmed had already seen results, so these tags are NOT blind (item 6 above asked for a tagger who had not). Val: clips 71 and 73 dusk, the other 8 day. Test: clips 69, 70, 74 night, the other 27 day, none dusk. Claude had described clips 17 and 18 as looking like dusk in `0004`; Ahmed's tags say day and were not changed.

## Results (B point count, score >= 0.2, run 2026-10-14, files in `data/outputs/count_error/`)
Mean absolute relative error per frame; bias is the mean signed error (negative = under-counting). Provisional figures are from the unsigned draft Phase 0 sheet s4.2; nothing is marked pass or fail. A clip counts in a bin if any of its frames falls in it, so clip counts add up to more than the number of clips.

**DroneCrowd test split, 30 clips, 9000 frames, all clips (primary)**
| group | clips | frames | mean err | median err | bias | provisional figure |
|---|---|---|---|---|---|---|
| overall | 30 | 9000 | 21.6% | 15.4% | -9.6% | |
| sparse (GT < 100) | 13 | 3427 | 24.2% | 22.2% | -14.9% | <= 15% |
| moderate (100-200) | 17 | 3689 | 26.3% | 22.5% | -8.6% | <= 15% |
| dense (> 200) | 9 | 1884 | 7.4% | 6.9% | -1.7% | <= 25% |
| day | 27 | 8100 | 18.9% | 12.8% | -5.6% | |
| night | 3 | 900 | 45.4% | 48.8% | -45.4% | |

**Day-only (secondary, 27 clips, 8100 frames):** overall 18.9% (bias -5.6%); sparse 20.1% (10 clips, bias -8.6%); moderate 24.3% (16 clips, bias -5.3%); dense 7.4% (9 clips, bias -1.7%).

**DroneCrowd val split, 10 clips, 3000 frames, all clips (contaminated: the model trained on these clips):** overall 6.7% (bias +3.3%); sparse 6.3%; moderate 6.1%; dense 8.5%; day 7.4%; dusk 4.0%. Not informative for generalisation.

What the numbers say (facts, not verdicts):
- On unseen test clips the model under-counts: -9.6% overall, -14.9% in sparse frames, and -45.4% on the three night clips. Val shows a small over-count (+3.3%) instead.
- Sparse and moderate bins are above the provisional 15% figure (24.2% and 26.3%); the dense bin is below the provisional 25% figure (7.4%).
- Count error can hide localisation errors: missed people and false points cancel in the count. It must be read together with the F1 results in `0004`.
- The mean (21.6%) is well above the median (15.4%): a few weak clips pull it up.

## Open items
- **Dusk is not addressed in the Phase 0 sheet** (it says daytime only is in scope and night is an open decision). Clips 17 and 18 look like dusk. Needs a decision: in scope or not.
- Night is still an open decision (kickoff 0001 said night clips in DroneCrowd are used for training and testing).
- The draft Phase 0 sheet is not in the repo; targets above are provisional until it is signed.

Evidence: Ahmed's answers in session on 2026-10-14. No count results existed when this was written.

Alternatives & why not: bins by clip instead of by frame (clips mix densities over time); dropping dark clips (hides the weakness).
