#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
export CARGO_TARGET_DIR="$PWD/target"
run=/workspace/alpha-zero-followups/runs/value-heads-20260906/pooling
test -f pooling-build-ready.json || exit 1
mkdir -p "$run"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/launcher-started-at.txt"
bash scripts/monitor_gpu.sh "$run/gpu.csv" 1000 &
gpu_monitor=$!
bash scripts/monitor_host.sh "$run/host.csv" 1000 $$ &
host_monitor=$!
cleanup() { kill "$gpu_monitor" "$host_monitor" 2>/dev/null || true; }
trap cleanup EXIT
python3 -u scripts/archive/run_pooling_followup.py \
  --predecessor /workspace/alpha-zero-followups/runs/value-heads-20260906 \
  --output-dir "$run" --validation-plan pooling-validation-plan.json
result=$?
printf '%s\n' "$result" > "$run/exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/finished-at.txt"
exit "$result"
