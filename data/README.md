# Data

This folder is gitignored except for this README and `manifests/`.
Download datasets here (VisDrone, DroneCrowd, etc.).

## DroneCrowd (primary)
- Source: https://github.com/VisDrone/DroneCrowd (link taken from the dataset's own README; Ahmed to confirm it is where he downloaded from).
- Licence: academic and non-commercial use only (per the dataset README).
- Location: `data/dronecrowd/` (about 11 GB). Unzipped as downloaded, names unchanged.
- Layout found on disk:
  - `train_data-002/train_data/{images,ground_truth}` (24,600 files each)
  - `test_data-001/test_data/{images,ground_truth}` (9,000 files each)
  - `val_data/val_data/{images,ground_truth}` (360 files each)
  - `annotations/annotations/` (`.xml` and `.mat` per sequence, some `_clean.txt`)
  - `trainlist.txt` (82 clips), `testlist.txt` (30 clips)
- Dataset README says: the val set is sampled from the test set. See `docs/project-log.md`.
