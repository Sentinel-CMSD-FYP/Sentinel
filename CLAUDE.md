This file is like a Roadmap, that we have to follow at all costs.

## Info:

1) We are building the perception model for sentinel, or final year project, the goal for this sprint is to setup DMCount and implement it effectively, and ensure that it is working well.

2) Our frontend, that we have built so far lies under @guide/frontend, and the docs dir under frontend also contains our proposal document pdf that you shuold read.

3) frontend is only shared with you for context about the system. Read the Roadmap under guide/frontend for knowing what we built in the frontend.

4) the guide directory under this project, shall be excluded in .git, it is here, just for giving you context about what we are building.

5) the "drone-crowd-dataset-info.txt" file has some info regarding the DroneCrowd dataset that we may use.

6) **Hardware:** the RTX 4070 is NOT available. All GPU work (training, inference, eval, FPS) runs on **Kaggle**: notebooks, ~30 GPU h/week quota, sessions of at most 12 h, so training must save checkpoints and be able to resume. Locally we also have an **AMD Radeon RX 560 4 GB**. It has no CUDA, so it is used for development, CPU side scripts, the labelling tool and light CPU inference only. The sprint's 4070 based targets (FPS, concurrent streams) are measured on Kaggle instead, and the gap is recorded.

## Rules:

1) No commits and push shall be done by claude, leave that to me, I will commit and push the work by myself.

2) Ask questions from user (me) when you have to decide anything or when you hit any confusion, never assume anything by yourself. Ask questions with clarity and detail, so I can understand what's being asked easily.

3) We have to work in the branch: https://github.com/Sentinel-CMSD-FYP/Sentinel/tree/density-dmcount

4) We have to do efficient utilization of tokens, but this does not mean that we do not focus on quality of the work, just optimize the usage of tokens as much as possible, without compromizing on the quality of the work.

5) save progress under @progress.md file, so we can work with newer sessions for each step. keep it precise and helpful for future sessions, and avoid making it too long.

6) do not use hyphens, emojis, italic text anywhere

7) use Claude-in-chrome extension, when needed, it is enabled. For example, when we need to do anything on Kaggle, you can open it and then ask me to login, or if its already logged in, you can perform the required tasks after having a confirmation from me via the terminal.

8) always share the "recommended" in the most feasible option when asking questions from me.

## Steps:

1) (Done) extract what Ahmar needs to do from @guide/Sentinel Tier 1 Sprint.html, and incorporate all tasks here as steps. We only have to setup, implement and test DMCount for now, ignore Dot. We also have to have a dataset like droneCrowd, or have our own dataset, based on footage downloaded from the internet. make the steps small, so we could deliver each step in a seperate session, to avoid longer sessions with high contextual memory sizes (it eats away tokens quickly)

> Source: Ahmar's "A + C + backend" track in the sprint HTML (A = density, C = infrastructure detection, backend scaffold). All of it is included **except DOT**, which is deferred. DOT was only an alternative to benchmark against DMCount, so it can be added later as an extra step.
> Each step = one session. "[ASK]" = a decision to ask the user about at the start of that step. Sprint task ids in [brackets].
> Team tasks Ahmar takes part in (kickoff, GPU/Kaggle schedule, own drone footage at 40 to 80 m, integration session, combined overlay, frontend demo, latency check, Tier 1 target tick list, demo run) are joint work and are not separate steps here. Steps 19 and 20 prepare Ahmar's part of them.

### Part A: Density (DMCount)

2) (Done) **Environment + DMCount smoke test** [2-0, DMCount only]
   * [ASK] How the DMCount code is held in the repo (gitignored `third_party/` clone vs git submodule vs porting the model into `ml/src/sentinel_ml/density/`)? Which released weights (QNRF / NWPU / SHA / SHB)?
   * Local venv on the RX 560 PC (CPU PyTorch) for development. Kaggle notebook with a GPU for real runs.
   * Clone github.com/cvlab-stonybrook/DM-Count, get the released weights (gitignored), run inference on one image, print the count and save a heat map overlay. Do this both locally (CPU) and on Kaggle.
   * Done when: the count is printed and the overlay is saved in both places. Setup errors and fixes are noted for the decisions log.

3) **Dataset acquisition** (DroneCrowd and/or own footage)
   * [ASK] DroneCrowd only / own internet footage only / both (e.g. DroneCrowd for train+test, own clips for qualitative checks)? Own footage needs head point annotation, so decide the scope first. Licence: DroneCrowd is academic and noncommercial.
   * [ASK] How to get DroneCrowd onto Kaggle: upload it as a private Kaggle Dataset (~11 GB), or reuse an existing Kaggle copy if its layout matches.
   * Local copy goes in gitignored `data/dronecrowd/`, with the same layout as the team's.
   * Verify the facts in `guide/drone-crowd-dataset-info.txt`: 112 clips × 300 frames, 1920×1080, 82 train / 30 test (trainlist/testlist), frames named `img{clip:03d}{frame:03d}.jpg`, GT in `GT_img*.mat` as (x, y, id) rows, median ~130 points per frame.
   * `eval/inspect_dataset.py` and `ml/configs/inspect_dataset.yaml` are NOT on this branch. [ASK] Get them from the teammate's branch, or write a minimal check?
   * Rerun the step 2 smoke test on one DroneCrowd frame (copy it to `data/samples/`, push the src dataset, push the kernel). Step 2 itself ran on the DM-Count example image.
   * Done when: the data is on disk and on Kaggle, the counts are verified, and a few sample images have been viewed (including night clips 70/71).

4) **Splits + data loader** [2-2 part 1]
   * [ASK] Validation split: the team proposes ~10 train clips held out with a fixed seed, but the list is NOT agreed yet. Use the team's list and never pick our own. [ASK] Frame subsampling for training (all 300 frames per clip, or every kth)?
   * Rules: `val_data` is a copy of test, so never tune or select checkpoints on it. The test split is only for final numbers. Night clips stay in.
   * Write a loader class in `ml/src/sentinel_ml/density/` that returns (image, head points) per frame. Put the split files under `data/manifests/` (or in a config).
   * Done when: GT points are drawn on 5 images and line up with the people.

5) **GT density maps / DMCount format targets** [2-2 part 2]
   * DMCount trains on points (OT loss), so export per frame point files in the format DMCount expects. Also generate Gaussian GT density maps ([ASK] fixed vs adaptive sigma) for visualisation and eval, and save them to disk once (gitignored).
   * Done when: an assert script confirms sum(map) ≈ number of points on every frame, and sample maps are visualised.

6) **Bucketed count error eval script** [2-3]
   * Count error = |pred − GT| / GT, split into buckets (e.g. <50, 50 to 200, >200). [ASK] Confirm the thresholds and which target applies to which bucket (≤15% / ≤25%). `eval/targets.yaml` is not on this branch.
   * Sparse bucket: MAE is the main number, and frames with <10 people are left out of the percentage. Report MAE (+ RMSE) for every bucket.
   * The script goes under `eval/density_dmcount/`, with unit tests on fake data with known answers in `ml/tests/`.
   * Done when: the tests pass and the script outputs a per bucket table (CSV/markdown).

7) **Zero shot baseline on the test split** [2-5, DMCount only]
   * Run the released weights on Kaggle over all test frames ([ASK] all 9000 frames or a fixed subsample) and log the per bucket errors in a table.
   * Expect poor results, because the weights were trained on ground level photos. This is the baseline, not the final number. Check for undercounting in dense zones.
   * Done when: the baseline table is saved with short notes.

8) **Fine tune pipeline + smoke run (Kaggle)** [2-n1 part 1]
   * Adapt DMCount training to the drone train split (dataset class, crop size, downsample ratio, LR, OT params), starting from the released weights. Select checkpoints only on OUR held out val clips.
   * Make training resumable (save the checkpoint + optimiser state every epoch), because of Kaggle's 12 h session limit.
   * Run 1 or 2 epochs and confirm that the loss goes down and val MAE is computed.
   * Done when: the smoke run logs show the loss decreasing and the full run notebook/command is ready.

9) **Full fine tune run (Kaggle)** [2-n1 part 2]
   * [ASK] Budget for the Kaggle GPU hours (shared weekly quota, also needed for Model C training later).
   * Launch the full run (resuming across sessions if needed), monitor it, and keep the best checkpoint by val MAE (weights gitignored, stored as Kaggle output/dataset).
   * Done when: the best checkpoint and the training curve are saved.

10) **Evaluate fine tuned model + density decision** [2-7]
    * Rerun the bucket eval on test with the fine tuned weights, with zero shot and fine tuned results side by side in one table.
    * Decide whether DMCount meets the bucket targets, and whether to go density only for dense zones (as in the Phase 0 sheet). Note the weakest bucket. [ASK] The final decision is the user's.
    * Done when: the comparison table and the decision are recorded.

### Part C: Infrastructure detection (YOLO26)

11) **YOLO26 env + Model C design decisions** [2-1]
    * [ASK] Class list (3 to 5 classes, e.g. gate, barrier, wall, bridge edge)? [ASK] Boxes vs masks (masks suit long thin objects but take ~2× longer to label)?
    * Install the Ultralytics YOLO26 package locally (CPU) and on Kaggle, then run one pretrained model on one image.
    * Done when: the pretrained model has run and the classes and box/mask choice are recorded.

12) **Labelling setup + rules + first batch (RX 560 PC)** [2-4]
    * [ASK] Labelling tool (CVAT vs Label Studio; both run without a GPU). [ASK] Image sources (dataset frames, public drone clips with gates/barriers, own footage once it is flown).
    * Write the labelling rules (what counts as each class) in a doc so labels stay consistent. Pick diverse frames and label the first batch.
    * Done when: the tool works, the rules are written, and the first batch is labelled.

13) **Continue labelling (~150 to 200 images)** [2-6]
    * Aim for variety in altitude, lighting and scene. Ask a teammate for help if labelling becomes the bottleneck.
    * Done when: ~150 to 200 images are labelled. (This is mostly manual work by the user. The session is for a check in and a QA pass on the labels.)

14) **Export + YOLO26 fine tune (Kaggle)** [2-8]
    * Split train/val ~80/20, export in YOLO format, upload as a Kaggle Dataset, and fine tune from a pretrained checkpoint. Run a few epochs first to confirm the pipeline works, then the full run.
    * Done when: the best weights are saved with the training curves.

15) **Benchmark YOLO26: mAP + FPS** [2-9]
    * mAP (box or mask) on the held out val images, a visual check of the predictions, and inference FPS on Kaggle. If mAP is weak, note the likely cause (too few labels, objects too small).
    * Done when: the metrics table and sample prediction images are saved.

### Integration + wrap up

16) **A and C output formats + module wrappers** [2-10]
    * [ASK] Agree with the team (Ahmed / Mahid) on: the density output (per zone count + density grid, zone size, JSON shape) and the infrastructure output (class, box or polygon, score, per frame).
    * Wrap DMCount in `ml/src/sentinel_ml/density/` and YOLO26 in `ml/src/sentinel_ml/infrastructure/`, each behind one function (frames in → agreed format out), with configs in `ml/configs/` and unit tests.
    * If a density bucket misses its target, revisit the settings here.
    * Done when: both wrappers and their tests work and a sample output file of each is ready to share.

17) **Backend scaffold: layout + common module interface** [2-n2 part 1]
    * [ASK] Framework: the repo already has an `apps/backend` Django skeleton (README: Django + DRF + Channels), but the sprint suggests a small FastAPI service, and the root `requirements.txt` lists FastAPI. Which one, and where should it live?
    * Folders per module, plus one common interface that every model follows (frames in → agreed format output out).
    * Done when: the skeleton and the interface are defined and documented.

18) **Backend: video ingest + one module end to end** [2-n2 part 2]
    * One way to read a video stream or file into frames. Run the density module end to end through it on one clip (CPU locally; GPU on Kaggle if needed).
    * Done when: one clip goes in and density output comes out through the backend.

19) **FPS benchmark + concurrent stream test (Kaggle)** [2-11]
    * Measure FPS for A and C on the Kaggle GPU (skip warmup, call `torch.cuda.synchronize()`, 200+ frames). Optimise if slow (lower resolution, fp16, frame skipping), rechecking accuracy after each change. Model C can run every few seconds, not every frame.
    * Run 3 to 4 streams concurrently and record FPS and memory. [ASK] How to coordinate with the others' B/E/D runs, since Kaggle sessions are separate per account. Record that these are Kaggle numbers, not RTX 4070 numbers.
    * Done when: the FPS and memory numbers (and trade offs) are recorded.

20) **Results table + decisions log entry for A and C** [2-12]
    * `docs/decisions/` entry: dataset notes, setup issues, density per bucket error vs targets (zero shot and fine tuned), YOLO26 mAP, FPS (Kaggle), labelling rules, the choices made, and failures and gaps for Tier 2.
    * Done when: the entry is written and the user has reviewed and committed it.

### Deferred
* **DOT density head** (rest of [2-0], [2-5]): set it up and benchmark it against DMCount later, as extra steps after step 20 (or after step 10 if the user wants the comparison earlier).
