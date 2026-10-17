# Progress: DMCount (Model A, density)

Roadmap and step details: `CLAUDE.md` → Steps. Branch: `density-dmcount`.

## Done
* **Step 1** (2026.10.17): Extracted all of Ahmar's sprint tasks (A density/DMCount, C YOLO26, backend scaffold) from `guide/Sentinel Tier 1 Sprint.html` into CLAUDE.md steps 2 to 20. Only DOT is deferred.
  * Hardware: no RTX 4070. GPU work runs on Kaggle, and the local AMD RX 560 4 GB (no CUDA) is for dev, CPU work and labelling (CLAUDE.md Info 6).
  * `guide/` is now listed in `.git/info/exclude`, so git ignores it on this machine. The shared `.gitignore` is unchanged.
* **Rule 6 cleanup** (2026.10.17): removed hyphens, dashes and emojis from CLAUDE.md and progress.md. Bullets now use `*`, the question mark emoji became [ASK], the tick emoji became (Done). Kept as is: file paths, URLs, branch name and sprint task ids like [2-0], since changing them would break references.
* **Step 2** (2026.10.17): DMCount env + smoke test done, locally (CPU) and on Kaggle (T4). Both give count 172.8 on the DM-Count example image `data/samples/dmcount_example_1.png` (no DroneCrowd yet, so the DroneCrowd frame rerun moved to step 3).
  * Choices (user): model ported into `ml/src/sentinel_ml/density/dmcount.py` (MIT notice, DM-Count commit cc5f213, same layer names so weights load strict), QNRF weights, code reaches Kaggle as private dataset `ahmar004/sentinel-ml-src`, Kaggle driven by the kaggle CLI.
  * Files: `ml/configs/density_dmcount.yaml`, `ml/tests/test_density_dmcount.py` (6 pass), `eval/density_dmcount/smoke_test.py`, `eval/density_dmcount/kaggle/push.py` (`src` uploads code dataset, `kernel <name>` pushes a kernel folder), kernel `eval/density_dmcount/kaggle/dmcount_smoke/`.
  * Weights: in the DM-Count GitHub repo itself (not Google Drive). Local copy `data/weights/dmcount/model_qnrf.pth`; the kernel downloads it from GitHub raw.
  * Timing (1695x1271, inference only): CPU 3.9 s, Kaggle T4 315 ms.
  * Setup issues for the decisions log: (1) Kaggle API token is the new KGAT type in `~/.kaggle/access_token`, no username in it, so set `KAGGLE_USERNAME=ahmar004`. (2) Kaggle CLI on Windows crashes writing logs (charmap), so set `PYTHONUTF8=1`. (3) Kernel had no internet until the account was phone verified. (4) We no longer pip install `ml` on Kaggle; the kernel puts `ml/src` on PYTHONPATH (torch, cv2, yaml are preinstalled). (5) DM-Count demo.py skips ImageNet normalisation; we normalise as in their test code.

## Repo facts (to save rechecking)
* `ml/src/sentinel_ml/density/` holds the DMCount module (step 2). `ml/configs/` and `ml/tests/` exist (the Farnebäck flow code is an example of the style).
* `eval/inspect_dataset.py`, `ml/configs/inspect_dataset.yaml` and `eval/targets.yaml` are **not** on this branch, though the team notes mention them.
* `data/*` is gitignored except `data/README.md` and `data/manifests/`. `*.pt/*.pth` are gitignored.
* This PC: AMD RX 560 4 GB, no CUDA. About 400 GB free on D:.
* Local venv: `.venv` at repo root (Python 3.13, torch 2.14 CPU, `pip install -e "ml[dev,eval]" kaggle`). Kaggle: torch 2.11 cu128, Python 3.13.
* Kaggle commands: in PowerShell set `$env:PATH="$PWD\.venv\Scripts;$env:PATH"; $env:PYTHONUTF8="1"; $env:KAGGLE_USERNAME="ahmar004"`, then `python eval/density_dmcount/kaggle/push.py src -m msg` and `... kernel dmcount_smoke`. Outputs: `kaggle kernels output ahmar004/<kernel> -p runs/kaggle/<name>`.
* `apps/backend` has only an empty Django skeleton (`manage.py`, Dockerfile, `.gitkeep` folders). The root `requirements.txt` lists FastAPI. The framework choice is an [ASK] for step 17.

## Next
* **Step 3: Dataset acquisition.** Start with the step 3 [ASK] questions (DroneCrowd vs own footage, how to get DroneCrowd onto Kaggle, inspect script). Then rerun the smoke test on one DroneCrowd frame.
