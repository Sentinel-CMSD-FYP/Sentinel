# Tier 1 tick list (sprint [3-5]) and Tier 2 handover

Date: 2026.10.21   Owner: Ahmar   Status: DRAFT for team review
Context: the sprint asks to go through Phase 0 sheet s4.1 to 4.2 row by row, put the measured number
beside each target and mark pass, fail or gap. The Phase 0 sheet itself is not in the repo and is
unsigned, so the rows below are rebuilt from the sprint task targets (`guide/Sentinel Tier 1 Sprint.html`),
decision 0005 (count protocol), 0006 (geometry), 0007 (Models A and C) and Ahmed's `docs/project-log.md`.
Each row names its source. Check the rows against the real sheet when it is signed.

Hardware: no RTX 4070 was available. GPU numbers are a Kaggle Tesla T4, CPU numbers this PC (Ryzen 5
3600, no CUDA). Rows that need the 4070 read "gap: Kaggle T4 number, 4070 pending" and are filled on
the GPU machine (step 31b, `docs/run_tier1.md` section 7).

Verdicts: pass, fail or gap. "(provisional)" = the target figure is unsigned or provisional, so the
verdict is a reading of the draft, not a sign off (keeps decision 0005). Fails are listed, not hidden.
Latency is judged against the sprint target of 3 s only (team choice; the SRS asks for 2 s at p95).

## s4.1 Quantitative targets

DroneCrowd official test split (30 clips, 9000 frames, all clips incl. night) unless stated.

| # | Target | Source | Measured | Verdict |
|---|---|---|---|---|
| 1 | A density, sparse (< 100 people) count error ≤ 15% (MAE main number) | sprint [2-5], 0005 | 25.1%, MAE 17.3 (day only 10.2%) | fail (provisional), night drives it |
| 2 | A density, moderate (100 to 200) ≤ 15% | sprint [2-5], 0005 | 28.0%, MAE 39.9 (day only 24.6%) | fail (provisional) |
| 3 | A density, dense (> 200) ≤ 25% | sprint [2-5], 0005 | 8.0%, MAE 18.6 | pass (provisional) |
| 4 | System count per bucket with the chosen split (B points for sparse and moderate, A for dense; step 10) | sprint [2-7] | sparse 24.2% (B), moderate 26.3% (B), dense 8.0% (A) | sparse and moderate fail, dense pass (provisional) |
| 5 | B point localisation F1 at 10 px | sprint [0-3], [0-4] | 0.689 (P 0.718, R 0.662) | gap: no numeric target set (log 11 Oct) |
| 6 | E tracking count error ≤ 25% (stretch); provisional sign off ≤ 35% with the 45 frame filter | sprint [0-7], project log 18 Oct | 37.9% filtered, 35.2% unfiltered; IDF1 0.388, HOTA 0.300 | fail (provisional) on both figures |
| 7 | B + E ≥ 5 FPS per stream on the RTX 4070 | sprint [0-9] | T4 in the pipeline: 4.9 to 5.3 FPS per clip (points 56 to 57 s + tracks 0.3 to 4 s per 300 frames, step 25); CPU 0.9 FPS | gap: Kaggle T4 number, 4070 pending |
| 8 | A density keeps its 1 sample per second per stream (no numeric FPS target) | sprint [2-11] | T4 fp16 140 ms per frame (7.1 FPS), 295 ms under 4 streams | gap: Kaggle T4 number, 4070 pending (T4 meets it) |
| 9 | C infrastructure mAP (no numeric target) | sprint [2-9] | mask mAP50 0.318, `mAP50-95` 0.185 (30 val images); obstruction 0.778, wall 0.492, barrier 0.002, gate_exit 0 | gap: no numeric target; barrier and gate_exit not learned |
| 10 | C speed: runs every few seconds per stream | sprint [2-11] | T4 fp16 41 ms per frame (24.5 FPS), runs every 125 frames (5 s) | gap: Kaggle T4 number, 4070 pending (T4 meets it) |
| 11 | D flow fits the ≥ 5 FPS / 3 s budget when sharing the GPU | sprint [1-8] | Farneback on CPU: 180 ms per sample (5.6 samples/s) on this PC; live capture to ready 1.3 s at 1 stream, 4.2 s at 3, 5.5 s at 4 (T4 box) | gap: Kaggle T4 number, 4070 pending (T4 box fails at 3 and 4 streams) |
| 12 | D flow accuracy against the DroneCrowd trajectory proxy | sprint [1-7], [1-12] | not recorded on this branch (Mahid's D results table) | gap: ask Mahid |
| 13 | Geometry px to m on an object of known size | sprint [1-6], 0006 | unit test: 4.5 m car back within 2 to 3%; seq011 parked cars imply 86 m (assumed 60 m), so DroneCrowd altitude is unknown | pass (unit test); real footage gap (row 15) |
| 14 | Walking speed comes out near 1.2 m/s | sprint [1-10] | seq011 tracks, moving median 1.03 m/s (p90 1.30) at the car implied altitude; flow cell means 0.05 m/s (cells mix people and still ground) | pass for tracks (altitude assumed); flow cells far below |
| 15 | Own 40 to 80 m footage through the whole pipeline, geometry checked | sprint [1-9], [3-n2] | software ready (upload with camera fields, `geometry_check.py`); no footage exists | gap: no footage (0006 open issue 2) |
| 16 | End to end latency ≤ 3 s, 1 stream | sprint [3-4] | T4 box, median capture to ready: density 0.17, points/tracks 1.4, infra 0.32, flow 1.3 s, + 0.1 to 0.2 s browser; reader 19.5 of 25 fps. CPU: density 8.7 s, points 4.7 s | gap: Kaggle T4 number, 4070 pending (T4 passes on medians; CPU fails) |
| 17 | End to end latency ≤ 3 s with 3 and 4 concurrent streams | sprint [3-4] | T4 box, 3 streams: flow 4.2 s, others ≤ 2.2 s, reader 7.2 fps per stream; 4 streams: flow 5.5 s, points 2.5 s, reader 5.1 fps. CPU: 3 streams 23 to 29 s, 4 streams out of memory | gap: Kaggle T4 number, 4070 pending (T4 box fails: CPU side caps all streams at ~20 frames/s) |
| 18 | 3 to 4 concurrent streams on one GPU | sprint [0-11], [2-11] | A + C only: 4 streams keep up (30.3 video FPS min, 7.5 GB peak). All five models live: no stream reads at 25 fps; 8.5 GB at 4 streams | gap: Kaggle T4 number, 4070 pending |

## s4.2 Qualitative checks

| # | Check | Source | Result | Verdict |
|---|---|---|---|---|
| 19 | Track IDs do not flicker or swap on the best and worst clips | sprint [0-8] ("qualitative by design") | clips rendered by Ahmed's fixed rule (E-4); his judgement not yet recorded. On seq011, 43% of matched people keep one id (step 24) | gap: Ahmed's E-4 judgement |
| 20 | Combined overlay: points, tracks, flow arrows, density, infrastructure, with toggles | sprint [3-3] | all five layers on the 4 demo clips, single and split views, faint density, coarse arrows, polygon hover (step 25) | pass |
| 21 | One stream shown live in the frontend | sprint [3-n3] | live mode with MJPEG picture, samples drawn as they arrive, grid of 1 to 4 streams (steps 28, 29) | pass |
| 22 | Night clips | project log 18 Oct (out of scope) | A 85.0% (bias -85%), B 45.4%, seq070 density ~5 vs GT 42 | reported, no target (out of scope) |
| 23 | Dusk clips | project log 18 Oct (no target) | A val dusk 19.3% (bias -19.3%) | reported, no target |

Summary:
* Pass: #3, #20, #21; in part #4 (dense), #13 (unit test), #14 (tracks).
* Fail (all provisional): #1, #2, #6; in part #4 (sparse, moderate).
* Gap, 4070 pending: #7, #8, #10, #11, #16, #17, #18. On the T4, #8, #10 and #16 meet the target, #7 sits
  at the line, and #11, #17 and #18 miss it.
* Gap, no target or no data: #5, #9, #12, #15. Waiting for Ahmed: #19.

Sources for every number: 0007 (A, C, A + C streams), `docs/pipeline.md` "Live latency" (rows 16 and 17),
`docs/project-log.md` summary table (B, E), `docs/pipeline_models.md` (D timing),
`data/outputs/geometry_check/seq011_60m/` (rows 13, 14), `data/samples/seq0*/clip.json` `produced_by`
(row 7).

## What Tier 2 consumes

Every output is a layer in a clip folder (`<CLIPS_ROOT>/<clip_id>/layers/<name>.json`, contract
`docs/layers/`, `schema_version` 1, string frame keys, `frame_size` [1920, 1080] for DroneCrowd), served
by `GET /api/v1/clips/<id>/layers/<name>`; live streams send the same sample shape through
`GET /api/v1/streams/<id>/samples?after=`. A sample older than `max_age_frames` is a gap, never a value.

| Layer | Type | Content | Rate (25 fps) | `max_age_frames` | Model |
|---|---|---|---|---|---|
| `density` | grid 9x16 (120 px zones) | field `density`, `people_per_cell`; zones sum to the frame count | every 25 frames (1 Hz) | 25 | A, `finetune_e4` fp16 |
| `flow` | grid 6x10 | fields `u`, `v`, `speed`; `px_per_frame`, or `m_per_s` when `clip.json` has a `camera` block | every 25 frames, pair gap 5 | 25 | D, Farneback, camera motion removed |
| `infrastructure` | polygons | per object `cls` (gate_exit, barrier, wall, obstruction), `score`, `box`, `polygon` | every 125 frames (5 s) | 125 | C, `model_c_best.pt` fp16 + NMS |
| `points` | points | rows `[x, y, score]`, score ≥ 0.2 | every frame | 1 | B, `dot_pd_dronecrowd_51.00` |
| `tracks` | tracks | rows `[track_id, x, y, score]`, score ≥ 0.35; `camera_motion_compensated` false | every frame B runs on | 1 | E, `tracker_e2_candidate.yaml` |

Metadata Tier 2 can rely on: `clip.json` `fps`, `frame_size`, optional `camera` (`altitude_m`, `hfov_deg`,
`altitude_source` assumed, stated or telemetry), `produced_by` per layer (model, weights md5, every,
device, seconds). Density for dense zones comes from A and counts for sparse and moderate zones from B
(step 10 decision); the > 200 rule is per frame today, not per zone.

## Gaps carried into Tier 2

* Accuracy: night undercount (A 85%, B 45%); low altitude overcount by A (clips 17, 18, ~70%); moderate
  bucket above 15% for A and B; E count error 37.9% and IDF1 0.388, set mainly by B's misses; C barrier and
  gate_exit not learned (58 and 10 labels), 120 of 150 C images never reviewed by a person.
* Speed: all GPU numbers are T4, none 4070 (step 31b). The CPU side of a live stream (decode, x264
  recording, preview JPEG, Farneback) caps throughput, so live real time holds for about 1 stream; ideas:
  x264 ultrafast or recording in its own thread, a read rate option, flow on a smaller frame.
* Live: the tracker assumes B runs every frame, so when live frames are skipped it associates over longer
  gaps than tuned; points and tracks of a saved stream flicker on replay (`max_age_frames` 1 with skipped
  frames).
* Data: no own 40 to 80 m footage, so geometry and m/s rest on assumed altitudes; the per zone dense
  threshold; the Phase 0 sheet is unsigned, so every verdict above stays provisional.
* Team: D proxy error table (Mahid), E-4 judgement and B+E on the 4070 (Ahmed), Mahid to accept the
  `polygons` type and the frontend changes flagged in `progress.md`.
