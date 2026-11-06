#!/usr/bin/env bash
# Run one command while holding the team's GPU lock, so heavy jobs take turns.
#   infra/pc/gpu-job.sh [--wait] <command ...>
#   infra/pc/gpu-job.sh docker compose -p sentinel-$USER -f infra/ml/compose.gpu.yaml run --rm ml python -m ...
# Busy lock: prints who holds it and exits 1 (with --wait it waits instead).
# The lock is only a courtesy. If someone forgot to use it, you still see their job in nvidia-smi.
set -uo pipefail
LOCK="${SENTINEL_GPU_LOCK:-/srv/sentinel/locks/gpu.lock}"
WAIT=0; [ "${1:-}" = "--wait" ] && { WAIT=1; shift; }
[ $# -gt 0 ] || { sed -n '2,7p' "$0"; exit 2; }

echo "== nvidia-smi: processes on the GPU right now =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "(an empty list above means nobody is computing on the GPU)"

umask 002
exec 9>>"$LOCK" || { echo "cannot open $LOCK (are you in group 'sentinel'?)"; exit 1; }
if ! flock -n 9; then
  echo "GPU lock is HELD by: $(cat "$LOCK.info" 2>/dev/null || echo 'unknown (no info file)')"
  [ "$WAIT" = 1 ] || exit 1
  echo "waiting for the lock ..."; flock 9
fi
echo "$USER since $(date '+%F %T') running: $*" > "$LOCK.info"
trap 'rm -f "$LOCK.info"' EXIT
"$@"
