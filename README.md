# Project Sentinel

Sentinel is an early warning system for dangerous crowds. Drones fly over a large event, our models
read the video, and a dashboard shows the people in charge where the crowd is too packed, where it is
moving and where the walls, barriers and exits are. The goal is to spot a crowd crush or a stampede
before it happens, so people on the ground can act in time. Sentinel only advises; it never controls
anything physical.

This is our Final Year Project at FAST NUCES Islamabad.

Note: everything here is tested on public datasets (mainly DroneCrowd) and on simulation. It has not
been tested on real Hajj operations data.

## Contents

1. How it works, in one picture
2. The models
3. Where we stand (Tier 1 results)
4. Run it
5. Try your own video
6. Live mode
7. Folder map
8. Tests
9. Where to read more
10. Team rules

## 1. How it works, in one picture

```
 drone video (file, or a stream played at real time)
        |
        v
 ingest_clip      turns the video into a clip folder: video.mp4 + clip.json (size, fps, camera)
        |
        v
 run_models       reads the video once and gives every frame to every model
        |
        +--> A density        how many people, and where      (heat map)
        +--> B points         one dot per person               (dots)
        +--> C infrastructure walls, barriers, exits, vehicles (outlines)
        +--> D flow           which way the crowd moves        (arrows)
        +--> E tracks         the same person over time        (trails)
        |
        v
 add_layer        checks each result and saves it as a "layer" file in the clip folder
        |
        v
 backend (Django)  serves clips and layers over a web API
        |
        v
 dashboard (React) plays the video and draws the layers on top of it
```

A few words used everywhere in this repo:

* **Clip**: one video plus its facts, stored as one folder (`data/samples/<clip id>/`).
* **Layer**: the output of one model for a whole clip, saved as one JSON file. Every layer has the same
  basic shape, so the dashboard can draw any of them.
* **Sample**: one model result for one frame. Models do not all run on every frame (see the table
  below), so a layer holds samples only for the frames its model ran on.
* **Gap**: when a sample is too old to trust, the dashboard shows nothing instead of an old value.
  Showing nothing is honest; showing old numbers as if they were new is not.

## 2. The models

| | What it finds | Model | Layer the dashboard draws | How often | Owner |
|---|---|---|---|---|---|
| A | Crowd density: people per area | DM Count, fine tuned on DroneCrowd | `density`: heat map, 9 x 16 zones | once per second | Ahmar |
| B | Each person as a point | DOT with Pixel Distill (authors' DroneCrowd weights) | `points`: one dot per person | every frame | Ahmed |
| C | Infrastructure: walls, barriers, exits, obstructions (vehicles, stalls) | YOLO26 segmentation, trained on our own labels | `infrastructure`: outlines with class and score | every 5 seconds | Ahmar |
| D | Crowd movement | Farneback optical flow, with the drone's own motion removed | `flow`: arrows, 6 x 10 zones | once per second | Mahid |
| E | The same person across frames | Our own Kalman point tracker on top of B | `tracks`: id per person, short trails | every frame | Ahmed |
| | Pixels to metres | Geometry from altitude and camera field of view | used by D to give speeds in m/s | when the clip has camera facts | Mahid |

Ahmar also owns the pipeline that connects all of them: the shared module interface, the backend
commands, the upload and live features, and the dashboard drawing for every layer.

How the counts are used: A counts the dense areas (more than 200 people in a frame), B counts the sparse
and moderate ones, because each model is better in its own range.

## 3. Where we stand (Tier 1 results)

All numbers come from the DroneCrowd test split (30 clips, 9000 frames). GPU numbers are from a
Kaggle Tesla T4, because the RTX 4070 we planned to use was not available. The full list of targets,
with pass, fail or gap for each, is in `docs/tier1_ticklist.md`.

| What | Result | Target | Reading |
|---|---|---|---|
| A count error, dense frames (over 200 people) | 8.0% | 25% or less | meets it |
| A count error, sparse frames (under 100) | 25.1% (day only 10.2%) | 15% or less | misses it, night clips pull it down |
| A count error, moderate frames | 28.0% | 15% or less | misses it |
| B finding people (F1 at 10 px) | 0.689 | none set | |
| C outlines (mask mAP50, 30 unseen images) | 0.318 (vehicles 0.78, walls 0.49) | none set | barriers and exits not learned yet |
| E count error from tracks | 37.9% | 35% (provisional) | misses it |
| Speed of A, B and C on a T4 | A 7 FPS, B about 5 FPS, C 24 FPS | B with E 5 FPS on the 4070 | 4070 numbers still to measure |
| Time from frame to dashboard, 1 stream, T4 | 0.2 to 1.4 s per model | 3 s or less | meets it on a GPU |
| Same, 3 or 4 streams at once, T4 | flow 4 to 6 s, others under 3 s | 3 s or less | the CPU side is the bottleneck |

The targets are still provisional, because the team's target sheet is not signed yet.

Known weak spots, carried into Tier 2:

* Night footage: A and B undercount badly.
* Low or slanted camera angles (big people close to the camera): A overcounts or undercounts, because
  the training data is mostly high, straight down views.
* C has very few labels for barriers (58) and exits (10), so it cannot find them yet.
* Running several live streams on one machine: video decoding and recording on the CPU slow
  everything down.

## 4. Run it

Pick the way that fits your machine. All three use the same code; only where the models run changes.

| Way | Your machine | Models run on | Good for |
|---|---|---|---|
| A. Full system on a GPU machine | NVIDIA GPU (for example the RTX 4070 box) | your GPU | the demo, live mode, speed numbers |
| B. CPU only PC | any Windows, Linux or Mac PC | your CPU (slow) | looking at stored clips, small tests |
| C. CPU PC with Kaggle | any PC plus a Kaggle account | a free Kaggle GPU in the cloud | uploading videos without owning a GPU |

The commands below are for Windows PowerShell, run from the repo root (the folder with this README).
On Linux or Mac, activate the environment with `source .venv/bin/activate` and set environment
variables with `export NAME=value` instead of `$env:NAME="value"`.

### Way A: the full system on a GPU machine

This is the complete setup: every model on the GPU, the backend, the dashboard, upload and live mode.
The longer version, with the speed measurements for the RTX 4070, is `docs/run_tier1.md`.

**Step 1. Check what you need.**

* An NVIDIA GPU with a recent driver. Run `nvidia-smi`; it must print your GPU. About 8 GB of GPU
  memory is enough for 4 live streams.
* Python 3.11, 3.12 or 3.13 (`python --version`), Node 20 or newer (`node --version`) and git.
* The GitHub CLI `gh`, logged in to an account that can see this repo: run `gh auth login` once.
  Our trained weights are stored in a release of this repo, and the download needs this login.
* About 15 GB of free disk if you want the demo clips (for the DroneCrowd dataset).

**Step 2. Get the code.**

```powershell
git clone https://github.com/Sentinel-CMSD-FYP/Sentinel.git
cd Sentinel
```

**Step 3. Make one Python environment for the models and the backend.**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. Every later step expects this, so activate it again in every
new terminal. If PowerShell refuses to run the script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**Step 4. Install PyTorch with CUDA, then everything else.**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e "ml[density,infra,points,export,dev]"
pip install -r apps/backend/requirements.txt pytest-django
```

`cu128` must suit your driver; if it fails, pick another one from pytorch.org (for example `cu126`).
Check that PyTorch sees the GPU; it must print `True` and your GPU name:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Step 5. Set up the database (once).**

```powershell
python apps/backend/manage.py migrate
```

It uses a small SQLite file, so you do not need to install a database server.

**Step 6. Download the model weights.** Weights are large files and are not in git.

```powershell
python eval/pipeline/fetch_weights.py
python eval/pipeline/fetch_weights.py --check
```

The first command downloads what is missing: models A and C from the GitHub release `tier1-models`,
and model B from its authors (their code at a fixed version plus their checkpoint). It checks every file
by its md5. The second command only reports; every line should say ok. D, E and geometry need no
weights.

**Step 7. Build the demo clips (optional, but needed for the demo).**

Download DroneCrowd from the official Google Drive (the link is in the README of
github.com/VisDrone/DroneCrowd). Unzip `train_data.zip` and `test_data.zip` anywhere under
`data/dronecrowd/`. Then:

```powershell
python eval/pipeline/build_demo_clips.py --device cuda
```

This makes four clips in `data/samples/`: `seq011` (day), `seq023` (dense crowd), `seq070` (night) and
`seq071` (dusk), each with all five layers. It takes about 2 minutes per clip; the first one is slower,
because model B downloads a 94 MB part of its network. Clips that already exist are skipped; add
`--rebuild` to make them again.

**Step 8. Start the backend (terminal 1).**

```powershell
.venv\Scripts\Activate.ps1
$env:PIPELINE_RUNNER="local"
$env:PIPELINE_DEVICE="cuda"
cd apps/backend
python manage.py runserver
```

Wait for `Starting development server at http://127.0.0.1:8000/` and leave this terminal open. The two
settings tell the backend to run uploaded videos on this machine's GPU (without them it would send them
to Kaggle, see Way C).

**Step 9. Start the dashboard (terminal 2).**

```powershell
cd apps/frontend
npm ci
npm run dev
```

`npm ci` is only needed the first time. Open http://localhost:5173 in your browser.

**Step 10. Use it.**

* **Clips**: choose a clip at the top right. "Single" shows all layers on one video; the boxes on the
  right turn each layer on or off. "Split by model" shows one small video per model, all in step.
  Hover a coloured outline to see its class and score.
* **Upload**: open "Upload a video and run the models", choose a video and press "Upload and run". The
  panel shows the progress; when it says done, the clip has all five layers. See section 5.
* **Live**: open the "Live" tab, pick a clip and press "Go live with this clip". It plays at real time
  like a drone feed, and each result appears as soon as its model is done.

**Step 11. Check that everything works (optional).**

```powershell
python -m pytest ml/tests
cd apps/backend; python -m pytest; cd ../..
cd apps/frontend; npm test; npm run lint; npm run build; cd ../..
```

All three should pass. One backend test runs every model end to end on 10 DroneCrowd frames; it skips
itself when the weights or the dataset are missing.

### Way B: a CPU only PC

Everything works without a GPU, only more slowly. Showing stored clips is instant; running the models
takes about 8 minutes for a 12 second clip, mostly model B.

1. Do steps 2, 3 and 5 of Way A.
2. Install PyTorch for CPU, then the rest:

   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -e "ml[density,infra,points,export,dev]"
   pip install -r apps/backend/requirements.txt pytest-django
   ```

3. Get the weights with step 6 of Way A (needed only if you want to run the models).
4. Get some clips, in one of two ways:
   * copy the clip folders (for example `seq011`) from a teammate into `data/samples/`, or
   * build them yourself with step 7 of Way A, using `--device cpu` (about 8 minutes per clip).
5. Start the backend. The settings make uploads run on this PC:

   ```powershell
   $env:PIPELINE_RUNNER="local"
   $env:PIPELINE_DEVICE="cpu"
   cd apps/backend
   python manage.py runserver
   ```

6. Start the dashboard with step 9 of Way A and open http://localhost:5173.

On a CPU, live mode cannot keep up: density and points arrive 5 to 9 seconds late, so the dashboard shows
them as gaps. Use stored clips for anything you want to show.

### Way C: a CPU PC that sends the work to Kaggle

Kaggle gives free GPU time (about 30 hours a week). With this way, an uploaded video goes to a Kaggle
GPU, the models run there, and the layers come back to your dashboard, about 5 minutes for a 12 second
clip, most of it waiting for Kaggle.

1. Set up the PC as in Way B, steps 1 to 4, and install the Kaggle tool: `pip install kaggle`.
2. Make a Kaggle account and verify your phone number (without it, Kaggle runs get no internet). On
   kaggle.com, open Settings, create an API token, and save it as `~/.kaggle/access_token`
   (on Windows `C:\Users\<you>\.kaggle\access_token`).
3. The Kaggle run needs our code and weights as private datasets in your Kaggle account. On the account
   `ahmar004` they already exist. On another account, create them once with
   `eval/density_dmcount/kaggle/push.py` (`src` for the code, `ckpt` for model A, `uavdot` for model B)
   and train model C there; `apps/backend/README.md` lists what the Kaggle run expects.
4. Each time the code changes, send the new code to Kaggle before a run:

   ```powershell
   $env:KAGGLE_USERNAME="<your kaggle name>"
   $env:PYTHONUTF8="1"
   python eval/density_dmcount/kaggle/push.py src -m "new code"
   ```

5. Start the backend with the Kaggle runner (it is the default, so no `PIPELINE_RUNNER` is needed):

   ```powershell
   $env:KAGGLE_USERNAME="<your kaggle name>"
   $env:PYTHONUTF8="1"
   cd apps/backend
   python manage.py runserver
   ```

6. Start the dashboard with step 9 of Way A, then upload a video. The panel shows each stage, for example
   "Kaggle: waiting for a GPU".

`PYTHONUTF8=1` is needed on Windows, because the Kaggle tool crashes without it.

### If something goes wrong

| What you see | Why | What to do |
|---|---|---|
| `No module named 'django'` | the environment is not active | run `.venv\Scripts\Activate.ps1`, or use `..\..\.venv\Scripts\python` |
| `torch.cuda.is_available()` prints `False` | PyTorch for CPU was installed | repeat step 4 of Way A with the CUDA index |
| `fetch_weights.py` cannot find the release | `gh` is not logged in, or has no access to the repo | `gh auth login` with an account in the team |
| The clip list is empty | no clips in `data/samples/`, or the backend is not running | build or copy clips, start terminal 1 |
| The video does not play | the video tools are missing | `pip install -e "ml[export]"` |
| An upload says a model failed | that model could not run (for example no weights) | the other layers are still added; read the error in the panel |
| Port 8000 or 5173 is already in use | an old server is still running | close the old terminal, or stop the old `python` or `node` process |

## 5. Try your own video

In the dashboard, open "Upload a video and run the models", choose a file (mp4, mov, avi or mkv, up to
1 GB) and press "Upload and run". All five models run, then the layers appear on the video.

If you know the drone's height and camera, fill in the camera fields (altitude, field of view, drone
model). With them, flow speeds are given in metres per second instead of pixels per frame. Without them,
the clip still works.

From the command line, the same thing is:

```bash
cd apps/backend
python manage.py ingest_clip my_video.mp4 my_clip --altitude 60 --dfov 84
python manage.py run_models my_clip --modules all --device cuda   # or cpu (slow)
```

For best results the video should look like our training data: filmed from high up, camera pointing
mostly straight down, 1080p.

## 6. Live mode

The "Live" tab plays a stored clip at real time as a stand in for a drone feed. The models work on each
frame as it arrives, and the dashboard draws every result as soon as it is ready. Up to 4 streams can run
at once. When a stream ends, it is saved as a normal clip, so it can be replayed later.

## 7. Folder map

| Folder | What is inside |
|---|---|
| `apps/frontend` | The dashboard (React, Vite, Tailwind) |
| `apps/backend` | The web API (Django and Django REST Framework) and the pipeline commands |
| `ml/src/sentinel_ml` | All model code, one folder per model, plus `pipeline/` that runs them together |
| `ml/configs` | Settings for every model (weights, how often it runs, thresholds) |
| `ml/tests` | Tests for the model code |
| `eval` | Scripts that measure the models (count error, mAP, speed, latency) and Kaggle runners |
| `docs` | How things work, decisions, results (see section 9) |
| `docs/layers` | The layer format, its JSON schemas, and the tools that check and add layers |
| `data` | Datasets, weights, clips and outputs. Not in git, except `data/manifests/` (dataset splits) |
| `infra` | Docker Compose files for Postgres and Redis |
| `simulation` | Planned: Gazebo, PX4 and pedsim crowd simulation |

## 8. Tests

```bash
python -m pytest ml/tests              # model code
cd apps/backend && python -m pytest    # API and pipeline, includes a full end to end run
cd apps/frontend && npm test && npm run lint && npm run build
```

Tests that need weights or the DroneCrowd dataset skip themselves when those are missing, so they also
pass on GitHub's CI, which has neither. CI runs all three on every pull request.

## 9. Where to read more

| File | Read it when you want to know |
|---|---|
| `docs/pipeline.md` | how a clip travels through the system, how to add a new model, live latency numbers |
| `docs/pipeline_models.md` | facts per model: entry function, weights, config, speed |
| `docs/run_tier1.md` | how to set up a GPU machine from a fresh clone |
| `docs/tier1_ticklist.md` | every Tier 1 target with the measured number, and what Tier 2 receives |
| `docs/demo_checklist.md` | how to run the demo, step by step |
| `docs/layers/README.md` | the exact layer format |
| `docs/decisions/` | why we chose what we chose (0003 layers, 0006 geometry, 0007 models A and C) |
| `docs/project-log.md` | Ahmed's log for models B and E |
| `README-DMCOUNT.md` | model A in detail: training, evaluation, results |
| `eval/infrastructure_yolo26/LABELLING.md` | model C: labelling rules, training and benchmark |
| `apps/backend/README.md` | every backend command and API endpoint |
| `apps/frontend/README.md` | how each layer is drawn |

## 10. Team rules

* Work on a branch, then open a pull request into `main` with passing CI and one review.
* Branch names: `feat/<area>-<name>` or `fix/<area>-<name>`.
* Never commit data, model weights or `.env` files.
* No Tier 3 work until Tier 1 and Tier 2 meet their targets.
* Every model joins the system the same way: one module behind `sentinel_ml.pipeline`, layers written
  only through `add_layer`, and checked in both dashboard views. Details: `docs/pipeline.md`.
