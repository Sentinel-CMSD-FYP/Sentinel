# Team GPU PC guide (Ubuntu + Docker)

One shared PC for running and testing the whole system (ML, backend, frontend, pipeline) in Docker with GPU.
Status: image built, `infra/pc/check_all.sh` run on this PC (results in `docs/decisions/0004-gpu-pc-setup.md`).

## What is on the PC
| Thing | Value |
|---|---|
| OS | Ubuntu 26.04.1 LTS, 28 cores, 30 GiB RAM |
| GPU | NVIDIA RTX 4000 Ada Generation, 20 GB, driver 595.99.02 |
| Docker | 29.9.3, Compose v5.7.0, NVIDIA Container Toolkit |
| ML image | `sentinel-ml-gpu` (Python 3.11, torch 2.1.0+cu121). One tag, built once, shared by everyone |
| Shared data (read-only) | `/srv/sentinel/datasets/dronecrowd` (zips are kept in its `_download/`) |
| Shared checkpoints | `/srv/sentinel/checkpoints` |
| GPU lock | `/srv/sentinel/locks/gpu.lock` (used by `infra/pc/gpu-job.sh`) |
| Your clone | `~/sentinel` in your own home. Your outputs (`data/outputs/`) stay there |

Access: your Linux user must be in groups `docker` and `sentinel`. **Membership in `docker` is root-equivalent on this machine.**
After being added, log out and back in.

## New member setup (about 10 minutes, plus image time if nobody built it yet)
```bash
git clone https://github.com/Sentinel-CMSD-FYP/Sentinel ~/sentinel && cd ~/sentinel
git checkout <your branch>
bash infra/ml/setup_third_party.sh             # uav-dot, uav-dot-track, TrackEval at pinned commits
export UID GID=$(id -g)
export SENTINEL_DATA_DIR=/srv/sentinel/datasets/dronecrowd      # put these 3 lines in ~/.bashrc
docker image inspect sentinel-ml-gpu >/dev/null 2>&1 || docker compose -p sentinel-$USER -f infra/ml/compose.gpu.yaml build
mkdir -p ~/.cache/sentinel-container
cp /srv/sentinel/checkpoints/dot_pd_dronecrowd_51.00.ckpt ml/third_party/uav-dot/checkpoints/ 2>/dev/null \
  || docker compose -p sentinel-$USER -f infra/ml/compose.gpu.yaml run --rm ml bash infra/ml/get_checkpoint.sh
bash infra/pc/check_all.sh
```
- Re-use the image, do not rebuild: `docker run`/`compose run` only builds when the tag `sentinel-ml-gpu` is missing.
  Rebuild (`... build`) only after `infra/ml/Dockerfile` or `requirements-gpu.txt` changed, and tell the team.
- `UID`/`GID` are used at run time, so the container writes files as YOU even if the image was built by someone else.
- Docker may leave an empty root-owned `data/dronecrowd` folder in your clone (mount point). That is normal and git-ignored.

## Run things
| What | Command (from repo root) |
|---|---|
| Shell in the ML container | `docker compose -p sentinel-$USER -f infra/ml/compose.gpu.yaml run --rm ml bash` |
| Everything, small checks | `bash infra/pc/check_all.sh` |
| Model B inference | `... run --rm ml python -m sentinel_ml.localization.infer --config ml/configs/infer_b.yaml --split val` |
| Tracker | `... run --rm ml python -m sentinel_ml.tracking.run_tracker --config ml/configs/tracker_selected_e2.yaml --split val` |
| Postgres + Redis | `docker compose -p sentinel-$USER -f infra/compose.yaml up -d postgres redis` (then `down`) |
| Backend / frontend | Not runnable yet, see "Known gaps" |

## GPU etiquette
1. Always run `nvidia-smi` first. The GPU has 20 GB, so look at what is already running.
2. Post in the team chat before a heavy job (full split, training, benchmark). One heavy job at a time.
3. Heavy jobs: wrap them so the lock shows who is using the GPU:
   `bash infra/pc/gpu-job.sh docker compose -p sentinel-$USER -f infra/ml/compose.gpu.yaml run --rm -T ml python -m ...`
   (`--wait` waits for the lock). The lock is a courtesy: if someone forgot it, `nvidia-smi` still shows their process.
4. Not measured: how many small jobs fit in 20 GB at once. Check `nvidia-smi` memory before starting a second one.

## Clean up
```bash
docker compose -p sentinel-$USER -f infra/compose.yaml down        # app stack
docker compose -p sentinel-$USER -f infra/ml/compose.gpu.yaml down
docker system df                                                    # what Docker uses (shared by everyone!)
```
Do not run `docker system prune -a` or delete `sentinel-ml-gpu`: that affects everybody. Delete only your own `data/outputs`.

## Known gaps (as of this setup)
- `apps/backend` has no `requirements.txt`, `manage.py` is empty, and its Dockerfile CMD points to `apps.backend\manage:app`.
- `apps/frontend` has no `package.json`.
- Density, geometry, risk, features and pipeline have no code yet; YOLO / density libraries are not in `requirements-gpu.txt` because nothing imports them.
- `infra/compose.yaml` publishes Postgres 5432 and Redis 6379 on all interfaces: only one person can run it at a time.

## Notes
- Deviations from the DOT authors' pins: `opencv-python-headless` instead of `opencv-python==4.5.5.64`; `scikit-learn>=1.3` instead of `1.0.2`
  (no Python 3.11 wheels). Neither is on our inference path.
- The checkpoint is pickle-based and comes from the authors' Google Drive link (same trust caveat as before).
- First Model B run downloads ImageNet encoder weights (mit_b2); they are cached in `~/.cache/sentinel-container`.
- DroneCrowd Google Drive often hits a download quota. Use a browser ("Make a copy" if needed). Licence: academic, non-commercial.
