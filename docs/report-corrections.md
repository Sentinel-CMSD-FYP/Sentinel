# Report corrections (replacement sentences)

Report: `docs/F26_083_i230828_p230686_i230644.pdf` (not edited). Plain wording: "Tier 1 delivered, with the gaps listed in Table B.1".
Speed figures below are from `docs/decisions/0012-e5-speed.md` (RTX 4000 Ada, DroneCrowd test clips 11, 15, 16, 17, 250 frames per stream, fp32, GPU otherwise idle). Counts: DroneCrowd test, 30 clips, 5,092 people.

## Table B.1 (Appendix B)

| Row | Old | New |
|---|---|---|
| 3 (B) | Measured `F1 0.689 (P 0.718, R 0.662)`, Status `Reported` | Measured `F1 0.689 (P 0.718, R 0.662), threshold 0.2 chosen on validation, released weights`. Status `Met (reported; no F1 target)` |
| 4 (E) | Measured `37.9% filtered; IDF1 0.388, HOTA 0.300`, Status `Reported` | Measured `37.9% filtered (35.2% unfiltered); IDF1 0.388, HOTA 0.300`. Status `Gap: target 35% (stretch 25%) missed` |
| 6 (B+E speed) | Measured `4.9-5.3 on a Kaggle T4`, Status `At target` | Measured `RTX 4000 Ada: 10.2 FPS for 1 stream; 4.4 for 3 streams and 3.3 for 4 streams (lowest stream). Earlier Kaggle T4: 4.9-5.3`. Status `Met for 1 stream; Gap for 3 to 4 streams` |
| 10 (latency, streams) | Measured `First single-stream check 1.4 s`, Status `Full test at Tier 2 start` | Measured `Latency not measured (a first single-stream check showed 1.4 s). 3 to 4 streams: see row 6`. Status `Not measured (latency)` |
| 2 (A) | Status `Refined in Tier 2` | Status `Gap (25.1% and 28.0% against 15%)` |

## Abstract (speed claim and "complete")
- Old: "Tier 1, perception, is complete: five models ... run in one pipeline"
- New: "Tier 1, perception, is delivered, with the gaps listed in Table B.1: five models ... run in one pipeline"
- Old: "and B with E runs at about 5 frames per second on one GPU."
- New: "and B with E runs at 10.2 frames per second for one stream on an RTX 4000 Ada, and below 5 per stream with three or four streams at once (4.4 and 3.3)."

## Executive Summary
- Old: "Tier 1 delivered. All five perception models are built, evaluated and integrated: ..."
- New: "Tier 1 delivered, with the gaps listed in Table B.1. All five perception models are built, evaluated and integrated: ..."
- Old: "tracking IDF1 0.388 and HOTA 0.300; ... B with E runs at about 5 frames per second on a Kaggle T4."
- New: "tracking IDF1 0.388 and HOTA 0.300, with a count error of 37.9% against the 35% target (a gap); ... B with E runs at 10.2 frames per second for one stream on an RTX 4000 Ada, but below 5 per stream with three or four streams. Density count error in sparse and moderate frames (25.1%, 28.0%) is also above its 15% target."
- Old: "Next. Tier 2 starts with the full latency and multi-stream tests..."
- New: "Next. Tier 2 starts with the speed fix for three to four streams, the full latency test, ..."

## Section 3.7, Table 3.3
- Old: row "Head points and tracking | Done | B F1 0.689; E IDF1 0.388, HOTA 0.300"
- New: row "Head points and tracking | Done, with a gap | B F1 0.689; E IDF1 0.388, HOTA 0.300; tracking count error 37.9% against 35% (gap)"
- Add a row: "B+E speed | Partly | 10.2 FPS for 1 stream; 4.4 (3 streams) and 3.3 (4 streams) per stream, below the 5 FPS target"
- Old (other rows): "Done" for density and "Done" for the proof of concept. New: density "Done; sparse and moderate bands are a gap (25.1%, 28.0% against 15%)".

## Section 7.5.7 (speed) and 8.1
- Old: "B with E runs at 4.9 to 5.3 frames per second per clip, at the 5 frames per second target"
- New: "On an RTX 4000 Ada, B with E runs at 10.2 frames per second for one stream (p95 10.1), meeting the 5 FPS target. With three streams the slowest stream runs at 4.4 FPS (p95 3.6) and with four at 3.3 FPS (p95 2.7), below the target. The Kaggle T4 figure of 4.9 to 5.3 FPS was measured earlier on a different GPU."
- Section 8.1, old: "Tier 1 is complete." New: "Tier 1 is delivered, with the gaps listed in Table B.1."

## Section 7.5.5 (tracking, optional but recommended)
- Add after the count-error table: "The count error of 37.9% (filtered) misses the 35% target and the 25% stretch goal. It is a gap."
- ID stability sentence: keep "stable by eye where B detects the people", and add "TrackEval counts 510 to 579 ID switches on clips 22 to 24 (about 2 per person; each person receives about 2 tracks), mostly exchanges between neighbouring heads about 7 px apart and flips back to an earlier id, so identities are stable only at overview scale." Source: decision 0013 (IM-4).

## Conclusion (Chapter 10)
- Old: "Iteration 1 completed Tier 1."
- New: "Iteration 1 delivered Tier 1, with the gaps listed in Table B.1."
- Old: "a point tracker (IDF1 0.388, close to the best possible association on these detections, and visibly stable over 10 s)"
- New: "a point tracker (IDF1 0.388 and a count error of 37.9% against a 35% target, close to the best possible association on these detections, and visibly stable over 10 s where Model B finds the people)"
- Old: "With Tier 1 complete, FYP-2 covers ..." (Chapter 9 intro). New: "With Tier 1 delivered, FYP-2 first closes the gaps in Table B.1 (multi-stream speed, latency, sparse and moderate density, tracking count error), then covers ..."

## Not changed
- Validation F1 0.864 appears in the report (Section 7.5.4 threshold table). Keep it only with the sentence that the released weights saw the validation clips; the cited result is 0.689.
