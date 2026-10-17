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

From the DroneCrowd paper (Wen et al., arXiv 2105.02440, read 2026-10-14): videos are recorded at **25 fps** at 1920x1080. Its tables also list per-video attributes including "Night", but those attribute labels are not in the files we have. The 455 max people per frame in the paper matches our train maximum (455).

Decisions made by Ahmed (2026-10-11):
1. **Validation split:** hold out about 10 of the 82 train clips (fixed seed) as our own validation. The official test set is used only for final numbers. The exact clip list is still to be written (B-1).
2. **Altitude for E:** one config constant (proposed 60 m, middle of the 40-80 m range). Not a measured value, so label it as an assumption in results.
3. **Night clips:** no separate "without night" F1. Report on all clips.

Open issues:
- Only 12 frames were looked at, so the "dots on heads" check is a spot check.
- Exact validation clip list and seed to be chosen and recorded in B-1.

## Storage (2026-10-15): train images live on a USB drive
Context: the laptop's C: drive was nearly full (9.2 GB free). Claude's Kaggle packs had added about 7 GB (deleted, they can be rebuilt with `eval/pack_for_kaggle.py`) and the dataset is 11 GB.
Decision: the 24,600 train images (7.7 GB, 8,263,638,526 bytes on disk) were moved to `D:\dronecrowd\train_images` (USB, FAT32, 14.6 GB). A Windows junction at the old path `data\dronecrowd\train_data-002\train_data\images` points to it, so no config or code changed.
Evidence: robocopy copied 24,600 files, 0 failed; names and sizes identical; 60 random files had matching SHA-256; after the swap the loader read a real frame from train, val and test, and all 80 tests passed. C: free went from 9.2 GB to 25.0 GB.
Stays on C:: test images (Model E needs frames), all ground truth `.mat` files (tiny; the train GT folder has too many long file names for FAT32), annotations, predictions and results.
Things to know:
- Anything that reads train or val images (val overlays, the val contact sheet, rebuilding the val Kaggle pack, training our own model) needs the USB drive plugged in. Test images, GT and predictions work without it.
- If the drive gets another letter or the junction is lost, recreate it (PowerShell, adjust the drive letter):
  `cmd /c rmdir "C:\dev\fyp\data\dronecrowd\train_data-002\train_data\images"` then
  `cmd /c mklink /J "C:\dev\fyp\data\dronecrowd\train_data-002\train_data\images" "D:\dronecrowd\train_images"`
  Do not use `Remove-Item -Recurse` on the junction path: it can delete the files on the drive.
- FAT32 has a 4 GB file limit and about 65,000 entries per folder; the train image folder (24,600 files) fits, a bigger folder may not.
- The USB copy is now the only copy of these images on this machine (the official download is the other source). Do not format the drive.
