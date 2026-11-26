#!/usr/bin/env bash
# One check for the whole system. Run from the repo root on the PC host:
#   bash infra/pc/check_all.sh
# Prints PASS / FAIL / SKIP per stage and exits 1 if any stage FAILED. Small smoke tests only:
# one clip, five frames. Logs: /tmp/sentinel-check-$USER/. Outputs go to YOUR data/outputs/smoke.
set -uo pipefail
cd "$(dirname "$0")/../.."
export UID GID=$(id -g)
# Use the shared dataset unless this clone has its own copy. (Docker leaves an EMPTY data/dronecrowd
# folder in the clone as a mount point, so test for real content, not for the folder.)
[ -z "${SENTINEL_DATA_DIR:-}" ] && [ ! -d data/dronecrowd/val_data ] && [ -d /srv/sentinel/datasets/dronecrowd/val_data ] \
  && export SENTINEL_DATA_DIR=/srv/sentinel/datasets/dronecrowd
mkdir -p "$HOME/.cache/sentinel-container" data/outputs
LOGDIR="/tmp/sentinel-check-$USER"; mkdir -p "$LOGDIR"
DC=(docker compose -p "sentinel-$USER" -f infra/ml/compose.gpu.yaml)
ml() { "${DC[@]}" run --rm -T ml "$@"; }
RESULTS=(); FAILED=0

stage() {  # stage "<name>" <function>   (function returns 0 = PASS, 77 = SKIP, else FAIL)
  local name=$1 fn=$2 log="$LOGDIR/$(echo "$1" | tr -c 'A-Za-z0-9' _).log" rc
  echo; echo "=== $name"
  $fn >"$log" 2>&1; rc=$?
  tail -n 25 "$log"
  if [ $rc -eq 0 ]; then RESULTS+=("PASS  $name")
  elif [ $rc -eq 77 ]; then RESULTS+=("SKIP  $name  -- $(tail -n 1 "$log")")
  else RESULTS+=("FAIL  $name  (log: $log)"); FAILED=1; fi
}

s_gpu() {
  nvidia-smi || return 1
  echo "--- other GPU jobs (compute apps):"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  docker image inspect sentinel-ml-gpu >/dev/null 2>&1 || { echo "image sentinel-ml-gpu missing: build it (docs/gpu-pc-setup.md)"; return 1; }
  local line; line=$(ml python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tail -n 1)
  echo "$line"
  case "$line" in "2.1.0+cu121 True "*) ;; *) echo "unexpected torch/CUDA line (expected 2.1.0+cu121 True <GPU name>)"; return 1;; esac
}
s_ml_lint()  { ml bash -c "cd ml && ruff check ."; }   # same as CI (runs in ml/)
s_ml_tests() { ml pytest ml/tests -q -p no:cacheprovider; }
s_eval_tests() { ls eval/test_*.py eval/tests 2>/dev/null | grep -q . || { echo "eval/ has no tests (its scripts are covered by ml/tests)"; return 77; }
                 ml pytest eval -q -p no:cacheprovider; }
s_b_smoke() {
  [ -f ml/third_party/uav-dot/checkpoints/dot_pd_dronecrowd_51.00.ckpt ] || { echo "checkpoint missing: run infra/ml/get_checkpoint.sh"; return 1; }
  [ -d "${SENTINEL_DATA_DIR:-data/dronecrowd}/val_data" ] || { echo "dataset missing: set SENTINEL_DATA_DIR (see docs/gpu-pc-setup.md)"; return 1; }
  rm -rf data/outputs/smoke
  ml python -m sentinel_ml.localization.infer --config ml/configs/infer_b.yaml --split val \
     --max-clips 1 --max-frames 5 --out-dir data/outputs/smoke && ls -l data/outputs/smoke/val
}
s_tracking_smoke() {
  ls data/outputs/smoke/val/clip_*_points.json >/dev/null 2>&1 || { echo "no Model B smoke points (stage 3 failed or skipped)"; return 77; }
  ml python -m sentinel_ml.tracking.run_tracker --config ml/configs/tracker_selected_e2.yaml --split val \
     --points-dir data/outputs/smoke/val --out-dir data/outputs/smoke/tracks
}
s_module() {  # s_module <dir> : SKIP when the package has no code beyond __init__.py
  local n; n=$(find "ml/src/sentinel_ml/$1" -name '*.py' ! -name '__init__.py' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo "ml/src/sentinel_ml/$1 has no code yet"; return 77; }
  echo "$1 has $n code file(s); no separate smoke entry point is defined, covered by ml/tests"; return 77
}
s_backend() {
  [ -s apps/backend/manage.py ] && [ -f apps/backend/requirements.txt ] || {
    echo "backend not runnable yet: apps/backend/manage.py empty or requirements.txt missing"; return 77; }
  local net="sentinel-check-$USER" pg="sc-pg-$USER" rd="sc-rd-$USER" rc
  docker network create "$net" >/dev/null 2>&1
  docker run -d --rm --name "$pg" --network "$net" -e POSTGRES_USER=sentinel -e POSTGRES_PASSWORD=sentinel -e POSTGRES_DB=sentinel postgres:16 >/dev/null
  docker run -d --rm --name "$rd" --network "$net" redis:7 >/dev/null
  sleep 8
  docker run --rm --network "$net" -v "$PWD/apps/backend:/app" -w /app \
    -e DATABASE_URL="postgres://sentinel:sentinel@$pg:5432/sentinel" -e REDIS_URL="redis://$rd:6379/0" \
    -e DJANGO_SECRET_KEY=ci-only-secret -e DJANGO_DEBUG=0 -e PYTHONDONTWRITEBYTECODE=1 python:3.11-slim-bookworm \
    bash -c "pip install -q -r requirements.txt ruff pytest pytest-django && ruff check . && python manage.py makemigrations --check --dry-run && pytest -p no:cacheprovider"
  rc=$?; docker rm -f "$pg" "$rd" >/dev/null 2>&1; docker network rm "$net" >/dev/null 2>&1; return $rc
}
s_frontend() {
  [ -f apps/frontend/package.json ] || { echo "frontend has no package.json yet"; return 77; }
  docker run --rm --user "$UID:$GID" -e HOME=/tmp -v "$PWD/apps/frontend:/app" -w /app node:20 \
    bash -c "npm ci && npm run lint --if-present && npm test --if-present && npm run build"
}
s_pipeline() {
  find ml/src/sentinel_ml/pipeline -name '*.py' ! -name '__init__.py' 2>/dev/null | grep -q . \
    || { echo "ml/src/sentinel_ml/pipeline has no code yet"; return 77; }
  echo "pipeline code exists but no entry point is defined in this script yet"; return 77
}

stage "1 GPU + torch (host and container)"      s_gpu
stage "2a ML lint (ruff)"                       s_ml_lint
stage "2b ML tests (pytest ml/tests)"           s_ml_tests
stage "2c eval/ tests"                          s_eval_tests
stage "3 Model B smoke (GPU, 1 clip, 5 frames)" s_b_smoke
stage "4 Tracking smoke on those points"        s_tracking_smoke
for m in flow density geometry risk features; do stage "5 module: $m" "s_module $m"; done
stage "6 Backend (ruff, migrations, pytest)"    s_backend
stage "7 Frontend (npm ci, lint, test, build)"  s_frontend
stage "8 Pipeline end-to-end"                   s_pipeline

echo; echo "================ SUMMARY ================"; printf '%s\n' "${RESULTS[@]}"
[ $FAILED -eq 0 ] && echo "OVERALL: PASS (no failures; SKIPs are listed above)" || echo "OVERALL: FAIL"
exit $FAILED
