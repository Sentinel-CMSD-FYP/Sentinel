# 0011 E-4 ID-stability judgement (Ahmed, by eye)
Date: 2026-10-18   Owner: Ahmed

Context: ROADMAP E-4. 12 videos of 10 s each (first 250 frames, 25 fps, 1280x720) rendered by
`eval/render_demo.py` from the frozen run-2 tracks (`tracker_e2_candidate.yaml`, score 0.35, min_hits 3,
max_age 3.0 s, no CMC, no altitude scaling) on DroneCrowd TEST clips. Clips were chosen by a rule written
before rendering: the 4 highest and 4 lowest run-2 IDF1 plus 4 random (seed 0); file names carry only the clip
number; the group and metrics (`data/outputs/demo/index.csv`) were opened only after watching. The test split
has been scored three times already; this is a qualitative look, not a new test.

Scale used by Ahmed: "few" = a couple of people; "some" = about 2 to 3 out of 10 in the whole clip.

## Judgement (Ahmed's words, lightly formatted)
- clip_016: few missing problem, few id switch close to buildings, 1 or 2 fragmentation, overall stable
- clip_017: only few people were marked, rest of the whole footage was not marked
- clip_018: only few people were marked, rest of the whole footage was not marked, few id switch issue
- clip_022: few wrong merge issues and a few to some untracked, overall stable
- clip_023: very few untracked, very few wrong merge, overall stable
- clip_024: very few untracked, very few wrong merge, overall stable
- clip_062: few people untracked, few people lost their dots occasionally
- clip_063: few people untracked, few wrong merge
- clip_074: some people got tracked only, id switch, wrong merge
- clip_082: almost 35% of people did not get tracked because they were in dark areas of the footage
- clip_101: some moving people got occasionally untracked but retained their colours, overall stable
- clip_108: few people got tracked only
- **Best:** clips 23, 24. **Worst:** clips 17, 18, 108. (Copies of these five are in `data/outputs/demo/best/` and `worst/`; the originals stay in `demo/`.)

## Evidence: Ahmed's picks against the numbers (opened after watching; whole 300-frame clips, run 2, IoU 0.5)
| clip | group | IDF1 | HOTA | tracker ids / GT people | ID switches |
|---|---|---|---|---|---|
| 16 | random | 0.353 | 0.302 | 270 / 175 | 295 |
| 17 | worst | 0.041 | 0.062 | 37 / 175 | 6 |
| 18 | worst | 0.070 | 0.100 | 61 / 153 | 9 |
| 22 | best | 0.692 | 0.513 | 342 / 283 | 510 |
| 23 | best | 0.709 | 0.534 | 384 / 296 | 579 |
| 24 | best | 0.635 | 0.469 | 324 / 253 | 558 |
| 62 | best | 0.649 | 0.476 | 271 / 169 | 275 |
| 63 | random | 0.573 | 0.425 | 264 / 155 | 355 |
| 74 | worst (night) | 0.063 | 0.071 | 88 / 123 | 32 |
| 82 | random | 0.464 | 0.346 | 153 / 108 | 82 |
| 101 | random | 0.373 | 0.325 | 339 / 245 | 104 |
| 108 | worst | 0.060 | 0.088 | 73 / 239 | 17 |
Source: `data/outputs/demo/index.csv` (from `data/outputs/run_comparison/test_per_clip.csv`).
- Ahmed's best clips (23, 24) are both in the metric-best group; his worst clips (17, 18, 108) are three of the four metric-worst (the fourth, 74, he described as "some tracked, id switch, wrong merge").
- The failures he saw in the worst clips are missing detections (B finds few people), not identity errors; clip 82 is the same, in dark areas.

## Decision
- E's ID stability is **qualitatively acceptable where B finds the people**: by Ahmed's eye the best and mid clips are "overall stable" over 10 s, with a few wrong merges, fragmentations and untracked people (his scale: a couple of people). Where B finds few people (17, 18, 74, 108, and dark areas in 82), the problem is detection, not ID stability. This matches the oracle result (association is not the bottleneck on test).
- This is a human, qualitative reading on the first 10 s of 12 clips of one dataset (DroneCrowd test, daytime except clip 74). It is not a metric and not a real-world validation. The roadmap's "visibly stable over 10 s" criterion is met on this reading for the clips where people are tracked, and not assessable on the clips where B finds almost nobody.

## Open issues
- Metric and eye differ in tone: clips 22 to 24 have 510 to 579 ID switches (TrackEval, IoU 0.5, 300 frames, about 280 people), yet were judged "overall stable" with "few wrong merges". Not investigated: the switch counts include flicker between matches at 20 px boxes that are hard to see by eye, and the video covers only the first 10 s. Kept as an observation, not resolved.
- The same 12 clips are not a sample of own-drone footage (still to be tested when it exists).
