#!/usr/bin/env bash
# Start the backend and the viewer, sample memory every 10 s, run latency_headless.py, stop all.
# Every argument goes to latency_headless.py. Run from anywhere; needs the stored clip in
# data/samples (ingest_clip), node on PATH and `npm ci` done in apps/frontend.
#   bash eval/pipeline/run_latency.sh --streams 3 --runs 3 --clip live_demo48
#   PYTHON=.venv/bin/python OUT=data/outputs/latency bash eval/pipeline/run_latency.sh --raw
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}; case $PYTHON in /*) ;; *) PYTHON=$ROOT/$PYTHON ;; esac
OUT=${OUT:-data/outputs/latency}; case $OUT in /*) ;; *) OUT=$ROOT/$OUT ;; esac
mkdir -p "$OUT"
if ss -ltn | grep -qE ':(8000|5173) '; then echo "port 8000 or 5173 is in use"; exit 1; fi
(cd apps/backend && PIPELINE_RUNNER=local PIPELINE_DEVICE=${PIPELINE_DEVICE:-cuda} \
  exec "$PYTHON" manage.py runserver 127.0.0.1:8000 --noreload > "$OUT/backend.log" 2>&1) 9>&- & B=$!
(cd apps/frontend && exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 \
  --strictPort > "$OUT/frontend.log" 2>&1) 9>&- & F=$!
( echo "t,gpu_mb,worker_rss_mb,browser_rss_mb,backend_rss_mb,mem_available_mb"
  while true; do
    w=$(ps -eo rss,args | awk '/run_stream --worker/ && !/awk/ {s+=$1} END{print int(s/1024)}')
    c=$(ps -eo rss,args | awk '/chrome-headless-shell/ && !/awk/ {s+=$1} END{print int(s/1024)}')
    k=$(ps -o rss= -p $B | awk '{print int($1/1024)}')
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    m=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    echo "$(date +%s),$g,$w,$c,$k,$m"; sleep 10
  done ) 9>&- > "$OUT/memory_$(date +%Y%m%d_%H%M%S).csv" & M=$!
# 9>&- above: background jobs must not keep gpu-job.sh's lock (fd 9) after this script ends.
trap 'pkill -P $M 2>/dev/null; kill $B $F $M 2>/dev/null' EXIT
"$PYTHON" eval/pipeline/latency_headless.py --out "$OUT" "$@"
