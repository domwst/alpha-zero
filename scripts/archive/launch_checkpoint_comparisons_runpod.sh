#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
run=/workspace/alpha-zero-followups/runs/value-heads-20260906/pooling-checkpoints
mkdir -p "$run"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/launcher-started-at.txt"
bash scripts/monitor_gpu.sh "$run/gpu.csv" 1000 &
gpu_monitor=$!
bash scripts/monitor_host.sh "$run/host.csv" 1000 $$ &
host_monitor=$!
cleanup() { kill "$gpu_monitor" "$host_monitor" 2>/dev/null || true; }
trap cleanup EXIT
python3 -u scripts/archive/run_checkpoint_comparisons.py \
  --predecessor /workspace/alpha-zero-followups/runs/value-heads-20260906 \
  --output-dir "$run" --binary "$PWD/target/release/alz" \
  --binary-sha256 27c4080dd6b8d9b59126f1485ce9fbf0aad78be66a6629d5ce8e19d1a816aca1
result=$?
printf '%s\n' "$result" > "$run/exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/finished-at.txt"
exit "$result"
