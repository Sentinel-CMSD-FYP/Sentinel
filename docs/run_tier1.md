# Run Tier 1 on a GPU machine

From a fresh clone to the dashboard with every model (A density, B points, C infrastructure,
D flow, E tracks) running on an NVIDIA GPU. Tested so far on Kaggle T4 kernels and on a CPU only
Windows PC; the RTX 4070 numbers of the sprint are still to be measured here (last section).
How a clip flows through the system: `docs/pipeline.md`. Per model facts: `docs/pipeline_models.md`.

## 1. What you need

* NVIDIA GPU with a recent driver (`nvidia-smi` works), about 8 GB GPU memory for 4 streams.
* Python 3.11 to 3.13, Node 20 or newer, git, and the GitHub CLI `gh` logged in to an account
  with access to this repo (`gh auth login`); the trained weights are assets of a release here.
* About 15 GB of disk for DroneCrowd (only to rebuild the demo clips; own footage needs none).

## 2. Install (one Python environment for models and backend)

From the repo root (Windows: `.venv\Scripts\activate`):

```bash
python -m venv .venv && source .venv/bin/activate
# torch with CUDA first (pick the index that matches your driver: cu126, cu128, ...)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e "ml[density,infra,points,export,dev]"
pip install -r apps/backend/requirements.txt pytest-django
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The backend and the models share this environment: `run_models` imports `sentinel_ml` and torch.
The database is SQLite by default (`python apps/backend/manage.py migrate` once); Postgres and
Redis from `infra/docker-compose.yml` are not needed for Tier 1.

## 3. Weights

```bash
python eval/pipeline/fetch_weights.py          # downloads what is missing, checks md5
python eval/pipeline/fetch_weights.py --check  # only reports
```

| model | file | from |
|---|---|---|
| A DMCount | `data/weights/dmcount/finetune_e4.pth` (md5 725694ba...) | release `tier1-models` of this repo |
| C YOLO26 | `data/weights/yolo26/model_c_best.pt` (md5 12213bde...) | release `tier1-models` of this repo |
| B DOT | `ml/third_party/uav-dot` at a6d3664 + `checkpoints/dot_pd_dronecrowd_51.00.ckpt` | authors' GitHub + Google Drive |
| D, E, geometry | none | |

The first time B is built, torch also downloads its MiT b2 encoder (94 MB).

## 4. Demo clips (optional)

Download DroneCrowd from the official Google Drive (link in the README of
github.com/VisDrone/DroneCrowd) and unzip `train_data.zip` and `test_data.zip` under
`data/dronecrowd/` (any nesting; the folders `train_data/images` and `test_data/images` are found
by name). Then:

```bash
python eval/pipeline/build_demo_clips.py --device cuda   # seq011, seq023, seq070, seq071
```

Each clip: `ingest_clip`, `run_models --modules all`, `validate.py`. On a T4 about 1.5 min per clip
(the first one longer: encoder download). Clips go to `data/samples/` (`CLIPS_ROOT`).

## 5. Start the system

```bash
# terminal 1, backend: models run on this machine's GPU
cd apps/backend
export PIPELINE_RUNNER=local PIPELINE_DEVICE=cuda   # PowerShell: $env:PIPELINE_RUNNER="local"; $env:PIPELINE_DEVICE="cuda"
python manage.py runserver

# terminal 2, frontend
cd apps/frontend && npm ci && npm run dev            # http://localhost:5173
```

Without `PIPELINE_RUNNER=local` the backend sends uploads to a Kaggle kernel, which needs the
Kaggle account set up in `apps/backend/README.md`.

* Clips page: pick a clip, Single view (all layers together) or Split by model (one panel each).
* Upload panel: a video (mp4, mov, avi, mkv, up to 1 GB) plus optional camera facts (altitude,
  field of view, drone model; with them flow is in m/s). All models run, then the layers appear.
* Live page: "Go live" replays a stored clip at real time as a stand in for a drone feed.

Own drone footage: upload it with its altitude and FOV, then check the metres with
`eval/pipeline/geometry_check.py` (see `docs/pipeline.md`).

## 6. Tests

```bash
pytest ml/tests                          # repo root
cd apps/backend && pytest                # includes an end to end run on 10 DroneCrowd frames
cd apps/frontend && npm test && npm run lint && npm run build
```

Tests that need weights or DroneCrowd skip themselves when those are missing.

## 7. Measure on this GPU (sprint targets)

Kaggle T4 results are in `docs/decisions/0007-models-a-c-tier1-results.md` and
`docs/pipeline.md` ("Live latency"). On the 4070:

```bash
# FPS of A and C alone, then 1 to 4 concurrent streams (DroneCrowd test frames)
T=data/dronecrowd/test_data-001/test_data/images   # your test_data/images folder
python eval/pipeline/fps_bench.py model density --half --device cuda --test-dir $T --out runs/fps/density_fp16
python eval/pipeline/fps_bench.py model infrastructure --device cuda --test-dir $T --out runs/fps/infra
python eval/pipeline/fps_bench.py streams --n 4 --half --device cuda --test-dir $T --out runs/fps/streams4
python eval/pipeline/fps_bench.py table runs/fps
```

Live latency, capture to overlay drawn (target 3 s or less), with the backend and frontend from
section 5 running: open the Live page, paste `eval/pipeline/browser_latency.js` into the DevTools
console, then `await latency.run(1)`, `await latency.run(3)`, `await latency.run(4)`. Worker side
numbers per stream: `python eval/pipeline/live_latency.py data/samples/.streams/<id> ...`.
