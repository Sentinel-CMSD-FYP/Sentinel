You are Claude Code running on the team's shared GPU PC (Ubuntu). This is the Sentinel FYP team's primary machine for running and testing the WHOLE system: ML (all modules), backend, frontend and the integrated pipeline, not just one person's model. Your job: set the PC up so any team member can clone the repo, get data, and run/test everything here in Docker with GPU. Work step by step, run the real commands, and report real output. Use short, plain English.

## Who you are working for
- The person at the keyboard is the "operator". It may be Ahmed, Mahid or Ahmar. Ask who it is first, and use their name/git identity for anything personal (their clone, their notes). Do not assume.
- The machine and the shared parts (data, image, conventions) belong to the whole team. Do not tie setup to one person's home folder, branch or workflow.
- Team split (so you know whose code is whose): Ahmed = localisation (B) + tracking (E); Mahid = optical flow + geometry; Ahmar = density + YOLO. Backend/frontend/pipeline are shared. Do not edit anyone's module code as part of this setup.

## Facts (already true, do not redo)
- GPU: NVIDIA RTX 4000 Ada Generation, 20 GB VRAM. It is NOT an RTX 4070: ignore any "4070" wording in CLAUDE.md/ROADMAP.md (stale).
- NVIDIA driver 595.99.02 installed; `nvidia-smi` works on the host.
- Docker and the NVIDIA Container Toolkit are installed. `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi` already printed the GPU table.
- Repo: <PASTE REPO URL>   Branch to start from: <PASTE BRANCH, e.g. main or the branch with infra/ml/>
- Existing files from the first setup attempt (written on a laptop, NEVER built or run): `infra/ml/{Dockerfile,requirements-gpu.txt,compose.gpu.yaml,setup_third_party.sh,get_checkpoint.sh}`, `docs/gpu-pc-setup.md`. They cover only Model B + E (torch 2.1.0 + cu121, Python 3.11). Treat them as a draft to fix and extend.
- Repo facts seen so far: `ml/src/sentinel_ml/` has packages localization, tracking, flow (code exists) and density, geometry, risk, features, pipeline, infrastructure (currently empty). `apps/backend` is Django (+ Postgres 16, Redis 7 via `infra/compose.yaml`); `apps/frontend` only has `src/` (no package.json seen). CI (`.github/workflows/ci.yml`) runs: frontend `npm ci/lint/test/build` (Node 20); backend `ruff`, `makemigrations --check`, `pytest` (Python 3.11); ml `ruff check .` and `pytest` from `ml/` with `pip install -e ".[dev]"`. Verify all of this yourself; the repo may have changed.

## Rules
1. Never run `git add`, `git commit`, `git push`, `git reset`, or switch the project repo's branch after cloning. The operator commits. (Cloning, and the pinned checkouts inside third-party clones done by the setup script, are allowed.) `git status` / `git diff` are fine.
2. You cannot type a sudo password. If a command needs `sudo` and asks for one, STOP and ask the operator to run that exact command and paste the result.
3. Never invent metrics or results. Every number comes from a command you ran. Paste real output, including errors. Label results with their source and the GPU.
4. Do not edit `docs/contracts.md` or any module code (`ml/src/sentinel_ml/*`, `apps/*`). You may change `infra/`, `docs/`, and add scripts under `infra/`. If something in apps/ or ml/ is broken for the setup (e.g. a wrong path in a Dockerfile), REPORT it with the exact line; do not fix it without the operator's OK.
5. Do not guess URLs. If you need a link you cannot find in the repo or the dataset's own README, ask the operator.
6. No data, weights, videos or `.env` in git. `data/*` and `ml/third_party/` are git-ignored; keep it that way.
7. Don't create Linux users/groups, change system-wide config, open network ports, or delete anything outside this project without asking the operator first (the machine is shared). Note: membership in the `docker` group is root-equivalent; mention this once.
8. Heavy GPU jobs (full-split inference, training, benchmarks) are NOT part of this setup. Small smoke tests only. Before any GPU use run `nvidia-smi` and tell the operator if another job is running.
9. If a step fails, diagnose the real cause, try a fix, and if it still fails after two attempts stop and report. Never skip a step silently.

## Steps
1. **Check the host and ask.** Run `lsb_release -a`, `nproc`, `free -h`, `nvidia-smi`, `docker --version`, `docker compose version`, `df -h`, `git --version`, `docker ps`, `docker images`, and `nvidia-smi` process list. Report what you find, including free disk (need roughly 60 GB: ~11 GB data, ~5-8 GB images, outputs). Then ask the operator these, and wait for answers:
   a. Who are you (name for git/notes)?
   b. Where should shared things live? Proposal: shared read-only data and checkpoints in `/srv/sentinel/` (e.g. `/srv/sentinel/datasets/dronecrowd`), each person's own clone in their own home folder. If `/srv/sentinel` doesn't exist, ask the operator to create it with the exact sudo commands you give them (owner/group, mode 2775 or similar), instead of doing it yourself.
   c. Is there a team agreement on how people take turns on the GPU? If not, propose the lightweight convention in step 9 and let the operator confirm.
   d. Do they also want the secondary datasets (UP-COUNT / UP-COUNT-TRACK)? Default: no, DroneCrowd only.
2. **Clone the repo** into the operator's home (`~/sentinel`) with their branch. Read `CLAUDE.md`, `ROADMAP.md`, `docs/contracts.md`, `docs/gpu-pc-setup.md`, `docs/decisions/0003-env-setup.md`, `README.md`, `.github/workflows/ci.yml`, `ml/pyproject.toml`, and every `requirements*.txt` / `package.json` under `apps/` and `ml/`. CLAUDE.md is Ahmed's per-person workflow file; follow its honesty and no-git rules, but its task-by-task roadmap is not what you are doing now.
3. **Find all dependencies of the whole system.** Do not guess. Grep each package under `ml/src/sentinel_ml/` and `eval/` for imports, and read teammates' requirement files/READMEs. Build a table: package -> who needs it -> declared where. Examples to check: YOLO (ultralytics?), density model libs, optical-flow libs, Django/FastAPI/Celery, Redis/Postgres clients. Anything that is missing from a requirements file but imported: list it for the operator rather than silently adding it. If a teammate's module is still empty, say so.
4. **Third-party repos and data (shared):**
   - `bash infra/ml/setup_third_party.sh` (pins: uav-dot a6d3664b50cc0cc6f1a08848eb294b71753b6600, uav-dot-track 505a7289b9602524383ae4260912f8e783000ccf, TrackEval 5477c170b741f95a311dcea99a7099fca805e7e2). Verify each with `git -C <dir> rev-parse HEAD`; report mismatches. If you find that other modules need their own third-party repos, ask the operator for URLs.
   - **DroneCrowd (~11 GB)** into the shared location from step 1b, once, for everyone. Read `data/README.md` and https://github.com/VisDrone/DroneCrowd (its README) to find the download links; prefer a command-line download. If it needs a browser login or a manual click, STOP and give numbered steps, then wait for `done`. Expected layout (from `data/README.md`): `train_data-002/train_data/{images,ground_truth}`, `test_data-001/test_data/{images,ground_truth}`, `val_data/val_data/{images,ground_truth}`, `annotations/annotations/`, `trainlist.txt`, `testlist.txt`. Verify real counts: images 24,600 (train), 9,000 (test), 360 (val), same for ground_truth; trainlist 82 lines, testlist 30. Report differences; do not "fix" data. Keep original names; do not delete the zip.
   - Make the shared data read-only for normal use (ask the operator to run any needed sudo/chmod). Each person's outputs (`data/outputs/`) stay in their own clone and are never in the shared folder.
5. **Make the Docker setup serve the whole team and the whole pipeline.**
   - Keep one GPU image for the ML stack (the base is `infra/ml/`). Extend `requirements-gpu.txt` / Dockerfile with the packages found in step 3 so that ALL ML modules and `eval/` scripts import and test inside it. Pin versions you actually resolved; note every change (file, old, new, why). Keep the authors' DOT pins that matter (torch 2.1.0, torchvision 0.16.0, numpy 1.26.4, timm 0.9.7, segmentation-models-pytorch 0.3.4, pytorch-lightning 2.1.3). If another module needs a conflicting torch/numpy version, STOP and report the conflict; propose a second image rather than breaking the first.
   - In `compose.gpu.yaml`: bind-mount the shared dataset read-only into the container at the path the configs expect (`/workspace/data/dronecrowd`, see `ml/configs/dronecrowd_data.yaml`), while the rest of `data/` (outputs, manifests) comes from the person's own clone. Make the dataset location configurable with an env var (default to the in-repo path so a laptop still works).
   - Containers must not collide between people: use per-user compose project names (e.g. `docker compose -p sentinel-$USER ...`), no fixed container names, no fixed host ports for ML containers. The image is built once and shared (same tag), so say how a teammate re-uses it without rebuilding.
   - Bring up the app stack too: Postgres + Redis + backend through `infra/compose.yaml` (profile `app`). Report what happens. The backend Dockerfile CMD had a suspicious entry point (`apps.backend\manage:app`, with a backslash). If it fails, report the exact error and the line; don't fix it without OK. Check frontend: if `apps/frontend/package.json` exists, run `npm ci`, lint/test/build inside a `node:20` container; if there is none, say "frontend has no package.json yet".
6. **Build and check the image:**
   ```
   export UID GID=$(id -g)
   docker compose -f infra/ml/compose.gpu.yaml build 2>&1 | tee /tmp/build.log
   ```
   Known risks: `antlr4-python3-runtime` and `efficientnet-pytorch` build from source; old pins; missing apt packages. Fix minimally and record every change. Then fetch the checkpoint: `docker compose -f infra/ml/compose.gpu.yaml run --rm ml bash infra/ml/get_checkpoint.sh`; verify `ml/third_party/uav-dot/checkpoints/dot_pd_dronecrowd_51.00.ckpt` and report its size. Put the checkpoint in the shared location too if the operator agrees to share it (it's a large git-ignored file). If Google Drive rate-limits, ask the operator to download it manually.
7. **One test entry point for everyone.** Create `infra/pc/check_all.sh` (run from the repo root on the host) that runs, in order, and prints PASS/FAIL per stage with a final summary and non-zero exit if anything failed:
   1. GPU: `nvidia-smi`, and in the container `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` (expect `2.1.0+cu121 True NVIDIA RTX 4000 Ada Generation`).
   2. ML in the container: `ruff check .` and `pytest ml/tests -q`, plus any `eval/` tests that exist.
   3. Model B smoke on GPU, tiny: `python -m sentinel_ml.localization.infer --config ml/configs/infer_b.yaml --split val --max-clips 1 --max-frames 5 --out-dir data/outputs/smoke` (plumbing check only; no F1 claims; first run downloads ImageNet encoder weights so it needs internet).
   4. Tracking smoke on those points if the tracker CLI allows a tiny run (read `ml/src/sentinel_ml/tracking/run_tracker.py` and the configs first; don't invent flags).
   5. Other modules' smoke tests only if they exist (flow, density, YOLO, pipeline). If a module has no tests/entry point yet, print SKIP with the reason.
   6. Backend: with Postgres + Redis up, `ruff check .`, `python manage.py makemigrations --check --dry-run`, `pytest` in `apps/backend` (match CI).
   7. Frontend: `npm ci`, lint/test/build if `package.json` exists, otherwise SKIP.
   8. Pipeline end-to-end: if `ml/src/sentinel_ml/pipeline/` has an entry point, run it on a few frames; otherwise SKIP and say it doesn't exist yet. Do not write pipeline code.
   Run it once and report the real result. The script must not touch teammates' data or run anything heavy.
8. **Team documentation.** Update `docs/gpu-pc-setup.md` into the team guide: what is on the PC, shared paths, how a new member sets up in about 10 minutes (clone, `setup_third_party.sh`, build or reuse the image, `check_all.sh`), how to run each part, how to run a GPU job, how to clean up (`docker compose down`, `docker system df`). Remove any 4070 or single-person wording. Write a decisions note `docs/decisions/NNNN-gpu-pc-setup.md` (next free number; use the template in CLAUDE.md; owner = the operator): GPU/driver/OS facts, commit hashes, every Dockerfile/requirements change, real check results, open issues, and the sharing conventions chosen.
9. **GPU etiquette (lightweight, confirm with the operator before adding).** Propose, don't impose: (a) always run `nvidia-smi` before starting; (b) post in the team chat before a heavy job (the existing rule is one heavy job at a time); (c) optionally a tiny wrapper `infra/pc/gpu-job.sh` that takes a `flock` lock on a file in the shared folder before running the given command and prints who holds the lock if busy. Only create (c) if the operator says yes. Note the GPU has 20 GB, so two small jobs may fit at once, but do not claim a specific number without measuring.

## Final report (end your last message with this)
1. What you changed (file list; `git status` output, including untracked files).
2. How you verified (commands + real output, trimmed to what matters).
3. Results: torch/CUDA line; `check_all.sh` summary table (PASS / FAIL / SKIP with reasons); pytest and ruff counts per area; smoke output. (The laptop's last report had 186 ML tests passing; report your own numbers.)
4. Dependency table from step 3 and anything missing or conflicting.
5. Anything uncertain, failed or deferred, plainly, including broken things you found in apps/ or ml/ and did NOT fix.
6. Disk usage: `du -sh` of the shared data folder and each clone, and `docker system df`.
7. A short "how a teammate starts" block with the exact commands.
8. End with: **"Approve? Reply `approved` to continue. Commit it yourself."**
Then wait. Do not start any roadmap task, full-split inference, training or benchmark until the operator says so.
