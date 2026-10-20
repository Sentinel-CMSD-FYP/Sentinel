# DMCount: crowd density model (Sentinel Tier 1, Model A)

Branch `density-dmcount`. Owner: Ahmar. Status on 2026.10.19: set up, fine tuned on DroneCrowd, evaluated on the full test split, density decision made. Step 16 (2026.10.20) added the layer output wrapper; backend and FPS are still to come.

## 1. Summary

* Model: DM-Count (VGG19 backbone, distribution matching / optimal transport loss), ported into `ml/src/sentinel_ml/density/` from github.com/cvlab-stonybrook/DM-Count (commit cc5f213, MIT).
* Weights: started from the released UCF-QNRF weights, fine tuned on the DroneCrowd train split. The Tier 1 model is **`finetune_e4`** (best epoch 4 by val MAE).
* Test result (DroneCrowd test, all 9000 frames, mean abs relative count error): **22.7% overall** (zero shot 75.0%, Model B 21.6%), **dense 8.0%** (target 25%), moderate 28.0%, sparse 25.1% (day only 10.2%), night 85.0%.
* Decision (user, step 10): **DMCount is the density head for dense zones (> 200 people)**, as in the Phase 0 sheet. Model B points count sparse and moderate zones. No retraining based on test findings (decision 0005: no tuning on test).

## 2. Using the model (integration reference)

### Input and output

```python
import cv2
from sentinel_ml.density import DMCount, DMCountConfig, overlay

model = DMCount(DMCountConfig(weights="path/to/best.pth", device="auto"))  # auto = cuda if available
frame = cv2.imread("frame.jpg")            # BGR uint8, shape (H, W, 3), any size
dm = model.predict(frame)                  # DensityMap
dm.count                                   # float, predicted people in the frame (= dm.density.sum())
dm.density                                 # np.float32 array (h, w), people per cell
dm.frame_shape                             # (H, W) of the input frame
cv2.imwrite("overlay.jpg", overlay(frame, dm))  # jet heat map + count, for visual checks
```

* **Input:** one BGR uint8 frame (OpenCV order), any resolution. The model converts to RGB and applies ImageNet normalisation itself. Run it at full resolution: all numbers below are at 1920x1080, fp32. No resizing or batching is done inside.
* **Output:** `DensityMap(density, frame_shape)`.
  * `density` has shape `(2 * floor(H / 16), 2 * floor(W / 16))`: **(134, 240) for 1920x1080**, (90, 160) for 1280x720. Each cell covers about 8x8 input pixels (`DOWNSAMPLE = 8`). For 1080 input the grid has 134 rows, not 135, so the last ~8 pixel rows have no cell; scale cell coordinates by `H / h` and `W / w` to map back to pixels.
  * Values are people per cell (non negative, from a ReLU). **The sum over any region is the predicted count of that region**, so a per zone count is just a sum over the zone's cells:

```python
h, w = dm.density.shape
H, W = dm.frame_shape
def zone_count(x0, y0, x1, y1):            # zone in input pixels
    r0, r1 = int(y0 * h / H), int(y1 * h / H)
    c0, c1 = int(x0 * w / W), int(x1 * w / W)
    return float(dm.density[r0:r1, c0:c1].sum())
```

* The output is a density map, not points: there are no per person positions or ids. Tracking and person points come from Model B.
* Backend output (step 16): `sentinel_ml.density.density_layer` turns frames into a `grid` layer `density` (people per 120 px zone, 9x16 by default, 1 Hz) as in Mahid's layer contract. Format, sample and API: `docs/layers_density_infrastructure.md`, sample `docs/samples/seq011/density.json`.

### Weights

| weights | where | notes |
|---|---|---|
| fine tuned `finetune_e4` (use this) | local `data/weights/dmcount/finetune_e4.pth` (config default) = `runs/kaggle/dmcount_train/v7/run/best.pth` (86 MB, md5 725694bafc292950a0c62ac43d855d22); Kaggle private dataset `ahmar004/sentinel-dmcount-ckpt` (file `best.pth`) | plain state dict, loads with `strict=True` |
| released QNRF (zero shot baseline) | local `data/weights/dmcount/model_qnrf.pth`; GitHub raw `cvlab-stonybrook/DM-Count/raw/cc5f213.../pretrained_models/model_qnrf.pth` | start point of fine tuning |

* `ml/configs/density_dmcount.yaml` points `weights:` at `data/weights/dmcount/finetune_e4.pth` (local copy of `best.pth`, same md5) since step 16. Pass `--weights data/weights/dmcount/model_qnrf.pth` for the zero shot model.
* Weights are gitignored (`*.pth`). Teammates get them from the Kaggle dataset (ask Ahmar for access) or from Ahmar directly.

### Speed (measured so far)

| where | resolution | time per frame |
|---|---|---|
| Kaggle Tesla T4, fp32 | 1920x1080 | 370 ms (about 2.7 FPS), over 9000 test frames, includes JPEG decode |
| local CPU (no CUDA, AMD RX 560 PC) | 1920x1080 | about 5.4 s |

Proper FPS, fp16, lower resolution and concurrent stream tests are CLAUDE.md step 19. These are Kaggle numbers, not RTX 4070 numbers.

## 3. Data

* **Dataset:** DroneCrowd (academic, non commercial licence). 112 clips x 300 frames, 1920x1080, head points `(x, y, id)` per frame in `GT_img*.mat`. Frames are named `img{clip:03d}{frame:03d}.jpg` (frame is 1 based in the file name, 0 based `frame_id` in our code).
* **Local copy:** `data/dronecrowd/{train_data-002/train_data, test_data-001/test_data, val_data/val_data, annotations, ...}` (12 GB, gitignored). **Kaggle:** public dataset `onvithong/dronecrowddata`, byte identical to the official release for every GT file and all sampled images (checked by `eval/density_dmcount/dataset_fingerprint.py`).
* **Split** (team frozen split, Ahmed, decision 0002, `data/manifests/dronecrowd_split.json`): train 72 clips, val 10 clips held out from train (seed 0), test 30 clips. The official `val_data` is a copy of test frames, so it is never used.
* **Frame strides** (`ml/configs/density_data.yaml`): train every 5th frame (4,320 frames), val every 10th (300 frames), test all frames (9,000).
* **Lighting tags** (`data/manifests/lighting_tags.yaml`, Ahmed): test night 69, 70, 74; val dusk 71, 73; the rest day.
* **Loader:** `sentinel_ml.density.data.load_split_from_config(cfg, split, root=...)` gives a `DroneCrowdDataset` (wraps Ahmed's `sentinel_ml.localization.data`, copied unchanged). Samples: RGB uint8 image, points `(N, 2)` float32 `(x, y)` in pixels, 0 based `frame_id`. `find_split_dirs` finds `train_data` / `test_data` by name, so local and Kaggle nesting both work.
* **GT targets** (step 5, `ml/src/sentinel_ml/density/targets.py`, `eval/density_dmcount/make_targets.py`): point `.npy` files and Gaussian GT density maps (fixed sigma 4 px, 240x135 grid of 8x8 cell sums, sum = number of points exactly), in `data/dmcount/` (gitignored). Training does not need them (it reads points from the loader); they are for visualisation and cell level checks. 7 border heads in 5 of 33,600 frames are clipped into the image (`clip_points`).

## 4. Training (fine tune)

* Code: `ml/src/sentinel_ml/density/train.py` (port of DM-Count's OT loss, Sinkhorn, count + TV loss, crop pipeline), config `ml/configs/density_train.yaml`.
* Settings (DM-Count QNRF defaults): random 512 crop + horizontal flip, batch 10, Adam lr 1e-5, weight decay 1e-4, OT weight 0.1, TV weight 0.01, Sinkhorn reg 10 x 100 iterations, fp32. Val every epoch on the 300 val frames; best checkpoint by val MAE; early stop patience 4; stop before an epoch that would pass the time limit.
* Command: `python -m sentinel_ml.density.train --out RUN_DIR --epochs N [--root DIR --init-weights PTH --device cuda --limit N --time-limit-h H]`. A run folder holds `last.pt` (model + optimiser + epoch, resumes automatically), `best.pth`, `best_val_preds.csv`, `history.csv`, `train.log`; `eval/density_dmcount/plot_history.py history.csv` draws `curve.png`.
* Run history (Kaggle T4, about 12 min per epoch): epochs 1 and 2 (smoke run, step 8), epochs 3 to 8 (step 9, early stop). Val MAE by epoch: 24.8, 21.1, 26.1, **15.4**, 16.8, 23.0, 16.2, 18.6 (noisy). Train loss fell steadily 8.9 → 3.4. Best = epoch 4.

## 5. Evaluation protocol

* Follows team decision 0005 (Ahmed): per frame relative error |pred − GT| / GT and bias (pred − GT) / GT; density bins by the frame's GT count: **sparse < 100, moderate 100 to 200, dense > 200**; provisional targets (draft Phase 0 sheet s4.2) sparse 15%, moderate 15%, dense 25%, shown for reference, never pass/fail; all clips (primary) and day only (secondary); **test is scored once per run name**, and no tuning or checkpoint choice on test.
* Scripts (all in `eval/density_dmcount/`):
  * `predict_split.py --split val|test --out DIR --weights PTH [--root --device --overlays]` writes `<out>/<split>/preds.csv` (`clip,frame_id,pred,gt`).
  * `count_error.py --split val|test --preds CSV --name RUN [--root --out]` writes `count_error.{json,csv,md}` and `frames.csv` to `data/outputs/density_count_error/<RUN>/<split>/`; refuses to overwrite a test result.
  * `compare_runs.py --split val|test RUN...` writes the side by side table `data/outputs/density_count_error/compare_<split>.md` (includes Model B's test numbers and per clip error).

## 6. Results

### Test (30 clips, 9000 frames, all clips): MAE / mean error / bias

| group | frames | zero shot QNRF | fine tuned `finetune_e4` | Model B (err / bias) | provisional target |
|---|---|---|---|---|---|
| overall | 9000 | 95.9 / 75.0% / -74.7% | 26.8 / 22.7% / -4.2% | 21.6% / -9.6% | |
| sparse | 3427 | 65.2 / 85.3% / -85.3% | 17.3 / 25.1% / -14.9% | 24.2% / -14.9% | 15% |
| moderate | 3689 | 98.1 / 71.6% / -71.3% | 39.9 / 28.0% / +0.2% | 26.3% / -8.6% | 15% |
| dense | 1884 | 147.2 / 63.2% / -62.1% | 18.6 / 8.0% / +6.5% | 7.4% / -1.7% | 25% |
| day | 8100 | 98.8 / 72.9% / -72.5% | 22.9 / 15.8% / +4.7% | 18.9% / -5.6% | |
| night | 900 | 69.6 / 93.9% / -93.9% | 62.3 / 85.0% / -85.0% | 45.4% / -45.4% | |

Day only (27 clips): fine tuned sparse 10.2%, moderate 24.6%, dense 8.0%, overall 15.8% (Model B 20.1 / 24.3 / 7.4 / 18.9%).

### Val (10 held out train clips, 300 frames)

| group | zero shot | fine tuned |
|---|---|---|
| overall | MAE 135.9 / 84.4% | MAE 15.37 / 13.8% / +1.7% |
| sparse / moderate / dense | 83.9% / 80.2% / 95.0% | 24.9% / 9.7% / 7.7% |
| dusk | 99.8% | 19.3% (bias -19.3%) |

### What the numbers say

* Zero shot undercounts everywhere: QNRF is ground level photos, and drone people are a few pixels wide. 14 of 30 test clips got under 10% of GT.
* Fine tuning fixed the high altitude views: 19 of 30 test clips are now under 15% error (clips 22 to 24 went from about 100% to 5 to 9%).
* Weakest bucket: **moderate (28.0%)**, from two scene types missing in training/val:
  * Clips 17 and 18: low altitude, large people with long shadows. The model puts 2 or 3 blobs on each person and **overcounts by about 70%** (GT checked complete). Without them moderate is 19.9%.
  * Night clips 69, 70, 74: still **undercount by 74 to 94%**. Without 17, 18 and night, moderate is 15.0%.
* Sparse misses the target only because of night (day only 10.2%). Dense meets it (8.0%).
* Val (13.8%) is better than test (22.7%) because the val clips hold no night and no low altitude oblique scenes.

### Where the result files are (all gitignored, on Ahmar's PC)

| what | path |
|---|---|
| side by side tables | `data/outputs/density_count_error/compare_test.md`, `compare_val.md` |
| fine tuned results + notes | `data/outputs/density_count_error/finetune_e4/{val,test}/` (`count_error.*`, `frames.csv`, `preds.csv`), `finetune_e4/notes.md` |
| fine tuned overlays | `finetune_e4/test/overlays/` (densest, per bin median, night clips, clip 17 overcount + its GT points) |
| zero shot results + notes | `data/outputs/density_count_error/zeroshot_qnrf/{val,test}/`, `zeroshot_qnrf/notes.md` |
| training run | `runs/kaggle/dmcount_train/v7/run/` (`best.pth`, `last.pt`, `history.csv`, `curve.png`, `train.log`) |
| raw Kaggle outputs | `runs/kaggle/<kernel>/` (eval kernel log: `runs/kaggle/finetune_eval/eval.log`) |

## 7. Kaggle setup

* All GPU work runs on Kaggle (no CUDA locally). Weekly quota 30 GPU h; DMCount used about 4 h 18 min in total for steps 7 to 10.
* Code reaches Kaggle as the private dataset `ahmar004/sentinel-ml-src` (the repo is private): `python eval/density_dmcount/kaggle/push.py src -m "msg"`. Kernels: `push.py kernel <folder>`; checkpoints: `push.py ckpt <output dir>` (uploads `run/` as `ahmar004/sentinel-dmcount-ckpt`).
* Kernel folders in `eval/density_dmcount/kaggle/`: `dmcount_smoke`, `dronecrowd_check`, `gt_check`, `zeroshot_eval`, `dmcount_train` (resumable training), `finetune_eval` (test eval with watchdog and `eval.log`).
* Environment (PowerShell): `$env:PATH="$PWD\.venv\Scripts;$env:PATH"; $env:PYTHONUTF8="1"; $env:KAGGLE_USERNAME="ahmar004"`. Status / output: `kaggle kernels status ahmar004/<kernel>`, `kaggle kernels output ahmar004/<kernel> -p runs/kaggle/<name>`.
* Lessons:
  1. The API token is the KGAT type in `~/.kaggle/access_token` with no username, so set `KAGGLE_USERNAME`. On Windows set `PYTHONUTF8=1` or the CLI crashes. Kernels need a phone verified account for internet.
  2. Kernels put `ml/src` on `PYTHONPATH` instead of pip installing (torch, cv2, yaml are preinstalled). Kaggle has torch 2.11 cu128, so the P100 is unusable; use T4.
  3. After `push.py src`, wait until `kaggle datasets status` says ready before pushing a kernel, or it runs on the old code.
  4. Never list a kernel in its own `kernel_sources` (it never starts). Resume goes through the ckpt dataset instead.
  5. QUEUED = waiting for a GPU (no quota used; waits of 17 min to 4 h seen, shortest at night). The log page shows nothing until a run ends.
  6. Runs sometimes hang in RUNNING with no output (train v4, eval v1). **Every GPU kernel needs a watchdog** and a log file in `/kaggle/working` so it always ends with its log. Cancel a stuck run via View versions → ... → Cancel; the cancelled time was refunded.

## 8. Code map

| path | what |
|---|---|
| `ml/src/sentinel_ml/density/dmcount.py` | `DMCountNet`, `DMCount` (inference), `DensityMap`, `overlay` |
| `ml/src/sentinel_ml/density/data.py` | DroneCrowd split loading with strides |
| `ml/src/sentinel_ml/density/targets.py` | `clip_points`, Gaussian GT `density_map` |
| `ml/src/sentinel_ml/density/train.py` | fine tune (OT loss, Sinkhorn, resume, early stop) |
| `ml/configs/density_dmcount.yaml`, `density_data.yaml`, `density_train.yaml` | model, data/eval, training settings |
| `ml/tests/test_density_*.py` | unit tests (73 pass with the localization tests: `python -m pytest ml/tests`) |
| `eval/density_dmcount/` | smoke test, dataset checks, GT check, targets, predict, count error, compare, plot |
| `ml/src/sentinel_ml/localization/`, `data/manifests/` | copied unchanged from Ahmed's branch; changes go through Ahmed |

Local setup: `python -m venv .venv`, then `pip install -e "ml[dev,eval]" kaggle` (Python 3.13, CPU torch).

## 9. Open items for integration and Tier 2

* Step 16 done (2026.10.20): grid layer wrapper, config switched to `finetune_e4`. Still open: Mahid / Ahmed to confirm the format in `docs/layers_density_infrastructure.md`; per zone dense thresholds (the > 200 rule is per frame) are a Tier 2 risk choice.
* Step 19: FPS on Kaggle at full res vs fp16 vs lower resolution, and 3 to 4 concurrent streams with memory.
* Tier 2 gaps: scale augmentation or multi scale training for low altitude views (clips 17/18), low light augmentation or data for night, LR decay or a larger/averaged val (noisy val curve), own drone footage at 40 to 80 m for visual checks.
* Deferred: DOT density head as an alternative to benchmark against DMCount.
