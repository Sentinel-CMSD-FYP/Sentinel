# Tier 1 status (honest table)

Date: 2026-11-23. Status is one of **Met / Gap / Not measured**. Every row has a number and a source.
Sources: report = `docs/F26_083_i230828_p230686_i230644.pdf` (Table 1.4, 7.5, B.1); log = `docs/project-log.md`;
dec = `docs/decisions/`. Where a value is not in those files it says "not found, ask Ahmed".
All DroneCrowd rows are the official test split (30 clips, 9,000 frames) unless stated.

| # | Target (Table 1.4) | Measured | Status | Source |
|---|---|---|---|---|
| 1 | A: density count error, dense (>200) <= 25% | 8.0% (fine-tuned DM-Count) | **Met** | report 7.5.6, B.1 row 1 (owner Ahmar) |
| 2 | A: sparse (<100) and moderate (100-200) <= 15% | 25.1% and 28.0% (all clips); day-only sparse 10.2% | **Gap** (owner: Ahmar) | report 7.5.6, B.1 row 2 |
| 3 | B: points within 10 px, report P / R / F1 (no numeric F1 target exists) | F1 **0.689** (P 0.718, R 0.662), thr 0.2 chosen on val, released DOT+PD weights | **Met** (reported; nothing to pass or fail) | dec/0004; log summary |
| 3a | B on validation | F1 0.864 | **Seen by released weights; do not cite as a result** | report 7.5.4 note; dec/0004 |
| 4 | E: tracking count error <= 35% with 45-frame filter (25% stretch) | 37.9% filtered, 35.2% unfiltered (run 2 frozen config, no CMC, no altitude, 5,092 people, "second look, not blind") | **Gap** (35% missed by 2.9 points; 25% stretch also missed) | log; report Table 7.5 |
| 5 | E: IDF1 reported | IDF1 0.388, HOTA 0.300 (TrackEval, IoU 0.5, mean over 30 clips); run 1: 0.384 / 0.298 | **Met** (reported; no numeric target) | log; report Table 7.5 |
| 6 | E: ID stability, trajectories visibly stable over 10 s | Stable by eye on clips 16, 22, 23, 24, 101 where B finds the people; 17, 18, 108 fail because B misses people. TrackEval counts 510, 579, 558 ID switches on clips 22 to 24 (1.8 to 2.2 per person; about 2 tracks per person). IM-4 reproduced the counts and found the 0.5 IoU threshold is not the cause | **Met by eye only; not supported by TrackEval in dense clips** (Ahmed decides the final label) | dec/0011, dec/0013 |
| 7 | B+E speed >= 5 FPS per stream, 1 stream | 10.20 FPS avg, 10.12 p95 | **Met** | dec/0012: RTX 4000 Ada, test clips, 250 frames, fp32, alone |
| 8 | B+E speed >= 5 FPS per stream, 3 and 4 concurrent streams | 3 streams: lowest 4.43 avg (p95 3.59); 4 streams: lowest 3.32 avg (p95 2.74) | **Gap** (measured on RTX 4000 Ada, alone; not run with teammates' jobs) | dec/0012 |
| 9 | End-to-end latency <= 3 s | not measured (a single-stream first check showed 1.4 s; note only) | **Not measured** | report 7.5.7 |
| 10 | C: infrastructure segmentation | mask mAP50 0.778 obstructions, 0.492 walls (30 held-out images) | No numeric target found, ask Ahmed. Not scored | report 7.5.6 (owner Ahmar) |
| 11 | D: flow against head motion | recovers up to 0.88 of annotated head speed | No numeric target found, ask Ahmed. Not scored | report B.1 row 8 (owner Mahid) |
| 12 | Geometry: object of known size | 4.5 m car within 2 to 3% | No numeric target found, ask Ahmed. Not scored | report B.1 row 9 (owner Mahid) |

## Notes
- **Speed history:** the report's "4.9 to 5.3 FPS" was measured on a Kaggle T4, not on the target GPU. It is a note only; rows 7 and 8 use the RTX 4000 Ada run. Model B alone was 4.78 FPS on the T4.
- **Tracking target (pending sign-off):** the target is **35% (25% stretch), pending sign-off**. `CLAUDE.md` used to say 25% (now reads "35% (25% stretch), pending sign-off"); the report says 35% with 25% as stretch; the Phase 0 sheet is unsigned. The note for the supervisor is `docs/supervisor-note-tracking-target.md`; the answer is still open.
- **Why E misses:** the oracle on test shows perfect association adds only 0.066 IDF1 (0.388 to 0.454), so the limit is Model B's detections (dec and log, 18 Oct).
- **Rows 10 to 12:** the report's own labels ("Delivered", "Met") have no numeric target behind them in Table 1.4, so they are not given Met or Gap here.
- Multi-drone, own-footage and real-world validation: none done.
