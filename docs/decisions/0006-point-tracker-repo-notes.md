# 0006 Point tracker repo (uav-dot-track): what it is, checked against the code
Date: 2026-10-14   Owner: Ahmed
Context: E-1. Ahmed was given a summary of `up-count/uav-dot-track` (the Ptak and Kraft tracker, IEEE Access 2026) and asked Claude to check it. Claude read the README, the authors' tracking page (up-count.github.io/tracking) and the code. Nothing was run; no tracking numbers of ours exist yet.

## Repo facts (verified)
- URL https://github.com/up-count/uav-dot-track, cloned to `ml/third_party/uav-dot-track/` (git-ignored), commit `505a7289b9602524383ae4260912f8e783000ccf` (2026-04-28). Licence Apache-2.0.
- Submodules (pinned): `uavdot` 40eb790 (DOT repo, Apache-2.0), `pyeco` 3e44bb4 (DDCF, MIT), `eval/TrackEval` 5477c17 (MIT). The `.gitmodules` URLs are SSH (`git@github.com:...`), so `git clone --recurse-submodules` fails without GitHub SSH keys. Worked around locally with `git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --depth 1`.
- Checkpoints: the same four DOT checkpoints as the B repo, placed in `./checkpoints/`. The trajectory classifier weights ship in the repo: `classification/best_model_epoch_dronecrowd.pth` and `..._upcount.pth` (about 17 KB each).
- Input is a video file (`main.py --video`). The detector runs inside `main.py`; `src/tracking/tracker.py` is called as `tracker(frame, decoder_output, frame_index, detections)`.
- Dependencies beyond B: `opencv-contrib-python`, `filterpy`, `zarr`.
- Authors' DroneCrowd results (their table, their pipeline): Baseline HOTA 0.53, ID-SW 6290, Tr-nMAE 0.32; + CMC 6231, 0.31; + CMC + ALT 6771, 0.30; + CMC + ALT + CLS 6645, 0.30; **+ DDCF (full) HOTA 0.54, ID-SW 388, Tr-nMAE 0.23**.
- Tr-nMAE is computed in `eval/track_eval.py` as |GT unique IDs - predicted unique IDs| / GT unique IDs per sequence, then averaged. This is the same definition as ROADMAP E-3.

## Discrepancies with the summary Ahmed was given
1. **Flags for the full method are incomplete.** The summary lists `--track-cmc-flow`, `--track-use-alt`, `--track-use-add-cls`. DDCF is switched on by a fourth flag, `--track-use-pointflow` (`pointflow_method='deep_eco'`, from `pyeco`), which is the step that takes ID switches from about 6,600 to 388. With only the three listed flags you get the "+CMC+ALT+CLS" row (ID-SW 6645, Tr-nMAE 0.30). There is also `--track-use-cutoff`. The README does not say which flags the "full" table row used.
2. **Altitude on DroneCrowd is not real altitude and not 60.** In `src/utils/video_reader.py`, for DroneCrowd the code hard-codes 50 for clips in a "large objects" list (17, 18, 75, 82, 88, 95, 101, 103, 105, 108, 111, 112) and 100 for all others, a proxy for object size. Real per-frame altitude (a GPS text file) exists only for UP-COUNT. `alt` is used only in `detection_result.py` to size a box around each point: `size = int(100 / alt * 20)`, so alt 100 gives 20 px and alt 50 gives 40 px (alt 60 would give 33 px); alt < 0 means a fixed size. A constant 60 would work but would differ from what the authors' numbers used.
3. **The tracker is not a points-only function.** It needs the video frame (optical-flow CMC, classifier crops, DDCF) and `decoder_output`, the network's raw output used in association. Our contract passes only points (docs/contracts.md), so DDCF cannot be fed from the points files alone; a thin wrapper is not enough.
4. **"Margin is thin" understates it.** The 0.23 needs DDCF. Without it the authors' own numbers on DroneCrowd are 0.30 to 0.32, above the 25% target. The roadmap E as written (point tracker + CMC + altitude, no DDCF) matches the rows with 0.30 and about 6,300 to 6,800 ID switches.
5. **The submodule list in the README is incomplete:** it names `pyeco` and `uavdot`; `eval/TrackEval` is a third submodule.
6. **Unknown: whether the reported numbers use the track-length filter.** `scripts/filter_trackings.py` drops tracks shorter than 45 frames (1.8 s at 25 fps) and `track_eval.py` handles `filtered_` results separately. The README does not say which was used; the filter lowers the ID count, so it directly affects Tr-nMAE.
7. Also: the authors' CMC uses sparse optical flow (`sparse_optical_flow.py`), which overlaps with Mahid's flow module; evaluation needs videos (`./datasets/dronecrowd/test_videos/*.avi`), we only have frames.

## Not discrepancies (confirmed)
Repo URL, licence, IEEE Access 2026, the four DOT checkpoints, `main.py --dataset dronecrowd --video <path> --task pred`, the evaluation script, and the DroneCrowd full-method figures (HOTA 0.54, ID-SW 388, Tr-nMAE 0.23).

## Open decision (Ahmed)
How to build E:
- A. Write our own point tracker (SORT-style, points, CMC, altitude-scaled matching), as ROADMAP E-1 and E-2 describe, using this repo as a reference. Fits the points-only contract and the speed target. Expect count error near the authors' no-DDCF rows, i.e. probably above 25%.
- B. Reuse their `Tracker` code including DDCF. Likely closest to 0.23 but needs frames, decoder output and the optical flow, does not fit the points contract, and its speed with B is unknown.
- C. Start with A, then add DDCF only if the count error misses the target.

Evidence: README, authors' tracking page, and the code in `ml/third_party/uav-dot-track/` at commit 505a728.

## E-1 notes (2026-10-16): our baseline tracker
Decision (Ahmed, option C): build our own point tracker first (E-1 to E-3), add DDCF only if the count error needs it. Build to the FINAL points/tracks contract; run the authors' pipeline later to cross-check.
What was built: `sentinel_ml/tracking/{kalman,tracker,run_tracker}.py`, `ml/configs/tracker.yaml`, `eval/render_trails.py`, `ml/tests/test_tracking.py`. It follows the authors' no-DDCF, no-CMC, no-altitude baseline: fixed 20 px gate, similarity `1 - min(dist, gate)/gate` with Hungarian matching, 3 consecutive hits to confirm, dies after more than 5 misses or near the border, only confirmed tracks matched in the frame are reported. Noise values are the authors' defaults, not tuned. Differences from their code: plain numpy Kalman over (x, y, vx, vy); we report the matched detection, they report the filter's prediction.
Found while testing: the authors' border rule kills any track whose predicted position is within `size/2` (10 px here) of the frame border, so heads that close to the border never get a confirmed track (and a head in the first frame is reported for one frame only). We kept the rule as a visible setting `border_margin_px` (10 = authors', 0 = only kill tracks that leave).
Also seen in their code: the CMC step halves the translation (`warp[:2,2]/2`). Not copied; to be handled in E-2.
Informational sanity check, test clips 011 and 075 (B points at 0.2, nothing tuned; NOT the E-3 metric):
- clip 011: GT 269 unique person ids, our tracker 1058 ids (394 with a track length of at least 45 frames); median track length 22 frames, 63% of ids shorter than 45 frames.
- clip 075: GT 128 ids, our tracker 441 ids (158 with at least 45 frames); median length 22, 64% shorter than 45.
So the raw ID count is about 3.9 times the GT count. The authors drop tracks shorter than 45 frames in their evaluation scripts, which would change the count a lot; E-3 must report with and without that filter and say which one is used.

## Investigation of the E-1 fragmentation and the agreed plan (2026-10-16)
Ahmed asked for a read-only investigation of clips 011 and 075 (test clips, diagnosis only, nothing tuned). The analysis is now `eval/diagnose_tracking.py` (tested; reads GT from the .mat files only). Findings, approximate shares of the excess ids (ids minus GT people):
- Ids sitting on non-GT points (B false points at 0.2, plus people the GT does not label; cannot be told apart without images): about 56 to 58%. Real people who received extra ids: about 44 to 46%.
- Expired tracks (73% of output ids): at the first missed frame a detection was within the 20 px gate for 74% (clip 011) and 53% (clip 075) of them, so another track had taken it (track competition). B's detection dropouts longer than max_age: 355 (clip 011) and 95 (clip 075).
- Camera motion: about 0. GT common-mode motion max 1 px per frame; optical-flow estimate median 0.10 and 0.25 px per frame; no GT step above the 20 px gate. Only checked on these two clips.
- Border rule: margin 10 vs 0 changes ids by 15 and 0; count error by at most about 5 points unfiltered and 1 point filtered.
- Count error against GT unique ids (269 and 128; the authors' MOT-format GT has the same ids): no filter 293.3% and 244.5%; with the 45-frame filter 46.5% and 23.4%.
- The authors' `warp[:2,2]/2` is not a unit conversion (the downscale is already undone in `sparse_optical_flow.py:46-48`), so it halves a correct translation. **Decision: do not copy the halving; apply the full translation in E-2.**

Agreed order (Ahmed, 2026-10-16): 1) E-3 measurement first, before any tracker change; 2) check camera motion on all 30 test clips; 3) E-2a altitude-scaled gate (config toggle); 4) E-2b CMC with the full translation and a synthetic-shift test (config toggle); 5) fragmentation experiments as a separate task with a protocol written before running, on val (B's val points are contaminated by training, so label the bias), **including a `score_thr` sweep on val**, then decide on DDCF. No tuning on test.

## E-3 protocol (fixed before any tracking result)
- Metric per clip: |unique track ids - unique GT person ids| / unique GT person ids, then the mean over clips. GT person count = distinct person ids in the clip's GT .mat files. (Assumption for Ahmed to confirm; the authors' GT has the same ids.)
- Primary view: every output id counts, all clips. Secondary views: only ids output in at least 45 frames (offline, post-hoc: needs the whole clip), and day-only clips using the lighting tags. No clip is dropped after seeing results. Also reported: signed ids-per-person ratio, median and max.
- Tracker settings are the unmodified defaults in `ml/configs/tracker.yaml` (B points at score >= 0.2). Val first, then test once; the script refuses to overwrite a test result.
- The 25% tracking-count figure is from the unsigned draft Phase 0 sheet, shown for reference only; nothing is called pass or fail.

## E-3 validation result (val split, 10 clips, 3000 frames; B was trained on these clips)
Mean error over clips: no filter 205.2% (median 201.5%, 3.05 ids per person); with the 45-frame filter 25.9% (median 24.4%). Day-only (8 clips): 204.2% and 28.2%. Dusk (2 clips): 209.3% and 16.6%. Per clip, no filter: 178% to 255%; filtered: 9.4% (clip 71) to 45.3% (clip 84). Files in `data/outputs/track_count_error/val/`.
