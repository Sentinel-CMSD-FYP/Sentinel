# 0013 ID-switch analysis on clips 22 to 24
Date: 2026-11-24   Owner: Ahmed (run by Mahid)
Context: clips 22, 23, 24 were judged "overall stable" by eye (decision 0011) but TrackEval counts 510, 579 and 558 ID switches. The report claims "ID stability: Met". This analysis tests that claim. It is measurement only: frozen run-2 tracks, DroneCrowd test clips already scored, no tracker setting changed, nothing tuned.
Decision: the "stable by eye" judgement holds only as a statement about overview-scale trails where B finds the people. It is **not supported quantitatively**: each person gets about 2 tracks and about 2 TrackEval ID switches. Report ID stability as "Met by eye only; TrackEval shows about 1.8 to 2.2 ID changes per person in dense clips". The 0.5 IoU threshold does not explain the count.
Evidence (script `eval/analyse_id_switches.py`, config `ml/configs/id_switches.yaml`, output `data/outputs/id_switches/analysis.json`; source label: DroneCrowd test, run-2 frozen E-2 config, all tracks, IoU 0.5 on 20x20 boxes, 300 frames each, 25 fps):
- The script reproduces TrackEval's count exactly: 510, 579, 558 (checked in the output, `reproduces_trackeval: true` for all three).
- TrackEval's rule counts a switch whenever a person's matched id differs from the last id matched to them, after any gap. A track that dies and a new one that starts on the same person is one switch.

| | clip 22 | clip 23 | clip 24 |
|---|---|---|---|
| GT people / predicted ids | 283 / 342 | 296 / 384 | 253 / 324 |
| tracks per matched person (mean, median, max) | 2.20, 2, 10 | 2.13, 2, 8 | 2.32, 2, 7 |
| purity (matched frames, weighted) | 0.871 | 0.890 | 0.870 |
| ID switches, IoU 0.5 (TrackEval's) | 510 | 579 | 558 |
| switches per GT person (arithmetic) | 1.80 | 1.96 | 2.21 |
| in the first 10 s (250 of 300 frames) | 398 | 446 | 478 |
| real swap (id taken from another person within 5 frames) | 200 | 224 | 212 |
| flip-back (A, B, A on the same person) | 118 | 169 | 157 |
| new id on the next frame, same person | 58 | 64 | 43 |
| re-matched after a gap with a new id | 134 | 122 | 146 |
| switches with gap 1 / 2 to 25 / over 25 frames | 224 / 197 / 89 | 300 / 193 / 86 | 257 / 224 / 77 |
| IDSW at IoU 0.3 / 0.1 | 470 / 460 | 455 / 448 | 474 / 497 |
| share of matches with IoU in 0.5 to 0.6 | 5.7% | 6.1% | 6.5% |
| same share among the switch frames | 11.8% | 13.5% | 9.3% |

- First 10 s: 77% to 86% of switches fall in the first 10 s, about in proportion to its share of the clip (250 of 300 frames = 83%). Stability does not improve with time.
- Real swap vs flip-back: about 38% to 39% of switches are a real id exchange between two people, 23% to 29% are the same person flipping back to an earlier id. Both are present in large numbers; neither is a small tail.
- Threshold: lowering the IoU threshold from 0.5 to 0.3 removes only 8% to 21% of switches (510 to 470, 579 to 455, 558 to 474), and 0.1 does not remove many more. Few matches sit near the 0.5 boundary (about 6%), and switch frames are only about twice as likely to be there. The threshold is a minor factor, not the explanation.
- Box size: ground-truth heads are very close. Median distance to the nearest other head is 7.0 to 7.2 px, and 76% of heads have a neighbour closer than 10 px (about 90% closer than 20 px). Two 20 px boxes 7 px apart have IoU 0.48, so a track between two heads can match either one. The 20 px box is larger than the head spacing, which makes the matching ambiguous in these dense clips. This is a plausible reason why many "real swaps" are tiny positional moves between adjacent heads, and why they are hard to see by eye. I did not test that claim (for example by re-scoring with a smaller box); it is a hypothesis.
Alternatives & why not: rescoring with another box size or changing the tracker now would be tuning on test; both are written as proposals for validation-only testing, below.
Open issues:
- Proposal (validation only, not run): score the val clips with a smaller box (for example 10 px) and with the tracker unchanged, to see how many switches depend on the box size. Test is not touched.
- Proposal (validation only, not run): look at the track-switch rate against the tracker's own gate (`size_px: 20`) in dense clips; a smaller gate or an appearance cue could reduce neighbour swaps. Needs a decision first.
- For the report: change ID stability from "Met" to "Met by eye only; per-person TrackEval switches 1.8 to 2.2 in dense clips". Done in `docs/tier1-status.md` and `docs/report-corrections.md`. The final wording is Ahmed's call.
