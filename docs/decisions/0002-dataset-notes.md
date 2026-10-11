# 0002 DroneCrowd dataset notes
Date: 2026-10-11   Owner: Ahmed
Context: P0-3. Read-only inspection of `data/dronecrowd/` (11.04 GB) with `eval/inspect_dataset.py --config ml/configs/inspect_dataset.yaml` (CPU, about 4m40s). Sample frames with GT drawn are in `data/outputs/inspect/`.

Decision / findings (all from the script output unless marked "visual"):
- **Size:** 112 clips x 300 frames, all 1920x1080. Train 82 clips (24,600 frames), test 30 clips (9,000), val 30 clips x 12 frames (360). Frame names `img{clip:03d}{frame:03d}.jpg`, frame ids start at 1.
- **Split files:** `trainlist.txt` (82 ids) and `testlist.txt` (30 ids) match the folders exactly. Train and test clips do not overlap.
- **GT format:** `ground_truth/GT_img*.mat` -> `image_info[0,0][0,0]` with `location` = (N,3) rows of `(x, y, person_id)` and `number` = N. `number` equals the row count in every frame. No frames lack a GT file.
- **Points per frame:** train min 25 / median 131 / max 455. Test min 40 / median 113 / max 274.
- **IDs:** person IDs exist. Unique IDs per clip: train median 164, test median 154. IDs have gaps (e.g. id 71 missing in clip 001 frame 1). That is fine, because people leave the frame.
- **Points are box centres.** In clip 001 the first point is (747, 592) and the XML box is x 737-757, y 582-602 (20x20), so the point is the box centre. It is not a separate head marking.
- **Val is a copy of test.** All 360 val frames exist in test with identical GT points. The dataset README also says val is sampled from test. Val cannot be used for tuning.
- **No altitude or camera metadata.** XML only has `box`, `track`, `frame`, `id`, `xtl/ytl/xbr/ybr`, `occluded`, `outside`. Dataset README has no altitude.
- **Night footage exists (visual, Ahmed):** `val_070_004` is night. By brightness hint, clips 70 and 71 are very dark (mean 25 and 40 of 255). Per the kickoff decision (0001), it stays in train/test.
- **GT quality (visual, Ahmed + Claude on `test_112_273`):** most dots are on people in the other 11 frames Ahmed looked at. In `test_112_273` some dots are off the person and some people have no dot (many on the left side of the street, near stalls and shadow). So GT is not perfect. F1 will be capped by label noise, and an F1 of 1.0 should not be expected.

Evidence: script output on 2026-10-11, Ahmed's review of the 12 images, Claude's view of `test_112_273.jpg`. Numbers come from the script run only.

Alternatives & why not: not applicable yet.

Open issues (need Ahmed's decision):
1. **Validation split.** Val is not independent. Proposal: hold out about 10 of the 82 train clips (fixed seed) as our own validation. Official test used only for final numbers. Pending.
2. **Altitude for E.** The dataset has none. Options: one config constant (e.g. 60 m, mid of 40-80 m) or a per-clip value set by Ahmed. Pending.
3. **Night clips.** Decide whether to report F1 with and without the dark clips. Pending.
4. Only 12 frames were looked at, so the "dots on heads" check is a spot check.
