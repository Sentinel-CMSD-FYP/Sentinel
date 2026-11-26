# 0003 Model B environment setup (DOT + Pixel Distill)
Date: 2026-10-11   Owner: Ahmed
Context: P0-4. Ahmed did not have the repo link and asked Claude to find and clone it.

Decision / findings (in progress; GPU demo not run yet):
- **Repo:** https://github.com/up-count/uav-dot (found by web search; its README states it is the code supplement to the Ptak & Kraft paper, arXiv 2502.04014, and the paper's footnote points to up-count.github.io for code and data).
- **Cloned to:** `ml/third_party/uav-dot/` on the laptop. Commit `a6d3664b50cc0cc6f1a08848eb294b71753b6600` (2025-12-29).
- **Licence:** Apache-2.0 (`LICENSE`).
- **Git ignore:** `ml/third_party/` was not in `.gitignore`, so Claude added it to the local `.git/info/exclude` only. Ahmed should also add it to `.gitignore`.
- **Submodule:** `eval_tool` -> https://github.com/up-count/DroneCrowd-VID-toolkit-python (commit f854679a). Not cloned yet. It is only needed for the repo's own evaluation, not for the demo.
- **Pinned requirements:** torch 2.1.0, torchvision 0.16.0, opencv-python 4.5.5.64, numpy 1.26.4, pytorch-lightning 2.1.3, timm 0.9.7, segmentation-models-pytorch 0.3.4, hydra-core 1.3.2. torch 2.1.0 has no Python 3.12 wheels, so use Python 3.10 or 3.11 on the 4070 PC.
- **Checkpoints:** the authors supply DroneCrowd checkpoints (DOT and DOT+PD) via Google Drive links in the README, to be placed in `checkpoints/`. Their README reports L-mAP 51.00 / L-AP@10 57.06 for DOT+PD on DroneCrowd. These are the authors' numbers, not ours, and use a different metric from our F1.
- **Data layout expected:** `data_path/{train_data,val_data,test_data}/images`, plus `train.txt`/`val.txt`/`test.txt`. Our layout is `train_data-002/train_data/...`, so it needs a mapping (symlinks or a config change).
- **Possible leakage in the released checkpoint (not verified):** in `src/datamodule/dot_datamodule.py`, the default DroneCrowd validation set is `val_data`, which we showed is sampled from the test set (decision 0002). The config monitors `val_f1` for checkpoint choice and early stopping. So the released checkpoint was probably chosen using test frames. Our test F1 with this checkpoint may be slightly optimistic. Report it with this caveat. Training our own model on train only would avoid it, but costs GPU time.
- **Output formats:**
  - `pt_pred.py` saves `x, y, score` per frame as `*_loc.txt`.
  - `infer_video.py` saves only `frame,x,y` with no scores. B-3 should use the `pt_pred.py` route, or our own script, to keep scores.
- **Input size:** frames are **resized (stretched, not padded)** from 1920x1080 to 1920x1088 with `A.Resize` (linear), then normalised with the config's mean/std. The `dronecrowd.yaml` comment says "padded", but the code in `dot_datamodule.py` and `infer_video.py` resizes. `pt_pred.py` scales y back by 1080/1088. (Corrected 2026-10-14 while writing B-3; the first version of this note said "padded".)

Evidence: README, `configs/dronecrowd.yaml`, `pt_pred.py`, `infer_video.py`, `dot_datamodule.py` read at the commit above. No GPU run yet.

Alternatives & why not: UP-COUNT checkpoints exist too, but UP-COUNT footage is 3840x2160 and our primary dataset is DroneCrowd.

## Demo run on Kaggle (2026-10-13), not the 4070 PC
Ahmed chose option A (use the released DOT+PD DroneCrowd checkpoint, no training) and to run inference on Kaggle.
- Setup: GPU Tesla T4 (15360 MiB), Python 3.13, Internet on. Repo cloned at commit a6d3664 in the notebook. Checkpoint `dot_pd_dronecrowd_51.00.ckpt` fetched with gdown from the README's Drive link.
- Input: 30 frames, `img011001.jpg` to `img011030.jpg` (test clip 011), joined into a 1920x1080 mp4 and run with `infer_video.py` (config `dronecrowd.yaml`, DOT+PD).
- Output: 6703 predicted points over 30 frames. Frame 0: 186 points (GT 211). Frame 29: 220 points (GT for `img011030` is 209). Counts are in the same range; this is not a metric.
- Progress bar showed 3.58 frames/s, including video read/write. Not a benchmark (E-5 does that).
- Some predicted points sit at the very top edge (e.g. `0,1056,1`, `0,986,3`, `0,1000,11`). Not yet checked visually whether these are false positives.

Errors hit and fixes:
1. `IndexError` in the frames cell: the dataset was mounted at `/kaggle/input/datasets/<user>/dronecrowd-demo`, not `/kaggle/input/dronecrowd-demo`. Used a recursive glob with the real path.
2. `UnpicklingError: Weights only load failed`: Kaggle's PyTorch is 2.6 or later, where `torch.load` defaults to `weights_only=True`. The checkpoint stores omegaconf config objects, a `defaultdict`, and Python 2-style names (`__builtin__.list`, `long`). Allowlisting types one by one did not work, because safe mode cannot set items on a `defaultdict`.
3. Workaround used: listed every global in the checkpoint with `pickletools` without loading it (14 types, all plain data or config types, none that run code). Then loaded it with a custom unpickler that blocks anything not on that list and `weights_only=False`. Scan reads raw bytes and could miss some names. The file's source is the authors' Drive link; not verified beyond the scan.
4. `TypeError: Conv2dReLU.__init__() got an unexpected keyword argument 'use_batchnorm'`: Kaggle had a newer `segmentation-models-pytorch`. Fixed with `pip install segmentation-models-pytorch==0.3.4 timm==0.9.7` (the repo's pins), then restarting the Python session.
5. Note: the model downloads ImageNet encoder starting weights from the internet at startup (mit_b2). Internet must be on. The checkpoint overwrites them.

## B-3 notes (2026-10-14)
- `sentinel_ml/localization/infer.py` runs the released checkpoint on a split and writes contract v1.0 points files. Score = sigmoid of the raw logit that `postprocessing` returns in its 4th column. Points are saved down to `min_score: 0.05` (the repo's own cut-off is 0.2; B-4 sweeps 0.1-0.9). Not tuned.
- Real-model check on this laptop, CPU, torch 2.14.1, smp 0.3.4: val clip 006, frames 0-1, 13.1 s for 2 frames (includes first-call warm-up), 527 and 535 points per frame at the 0.05 floor, score range 0.05-0.78. Contract check passed.
- Smoke score only (plumbing check, 2 frames of one val clip, 10 px): thr=0.05 P 0.617 R 0.833 F1 0.709; thr=0.2 P 0.901 R 0.785 F1 0.839; thr=0.5 P 0.988 R 0.307 F1 0.468. Not a result.
- **Validation inference on Kaggle (Ahmed, 2026-10-14):** Tesla T4, fp32, batch 1, released DOT+PD DroneCrowd checkpoint, val split (10 clips, 3000 frames), `min_score` 0.05. Total 626.1 s for 3000 frames = **4.79 frames/s** for B alone, including image decode and JSON building (about 62-64 s per 300-frame clip). This is not the B+E benchmark and not the RTX 4070 (E-5 covers that). GPU 2-frame check gave 1062 points, identical to the laptop CPU run.
- Output: 10 contract-valid files, 859,151 points in total (mean 286.4 per frame; per-clip frame medians from 87 to 546), score range 0.05-0.93, median 0.246. All 10 files pass `check_doc`, have frames 0-299, and match the point counts Kaggle printed. Files are in `data/outputs/points/val/` (git-ignored).
- **Our validation clips were in the released checkpoint's training data.** The authors trained on the official `train_data` (every 4th frame), and our `val` clips are held out from that same folder. So the released weights have seen our val clips, and a threshold picked on them may be biased (more confident on seen images). Test clips were not trained on (but see the checkpoint-selection caveat above).

Open issues:
- Val-vs-train overlap above affects B-4 (threshold choice). Options to discuss: accept and label it, or fine-tune/train our own model on the 72 train clips only.
- `docs/contracts.md` says "DOT pads 1080 to 1088"; the code resizes (stretches). Wording only, no format change; waiting for Ahmed's OK to correct it.
- Visual check of the demo frame (are circles on people, are the top-edge points false positives): waiting on Ahmed.
- Kaggle run is not the 4070 environment. The 4070 setup was not done.
- Mapping of our data folder layout to the repo's expected layout.
- Leakage caveat above must appear in the Model B results (0004).
- UP-COUNT has altitude information per the repo/paper. Could supply real altitudes for E if UP-COUNT-TRACK is used.
