#!/usr/bin/env bash
# Deployment launcher for the September 6 experiment queue. Run inside tmux.
set -uo pipefail
cd /workspace/alpha-zero-followups || exit 1
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
run=/workspace/alpha-zero-followups/runs/value-heads-20260906
mkdir -p "$run"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/launcher-started-at.txt"
bash scripts/monitor_gpu.sh "$run/gpu.csv" 1000 &
gpu_monitor=$!
bash scripts/monitor_host.sh "$run/host.csv" 1000 $$ &
host_monitor=$!
cleanup() { kill "$gpu_monitor" "$host_monitor" 2>/dev/null || true; }
trap cleanup EXIT
python3 scripts/archive/run_replay_followups.py \
  --activation-dir /workspace/alpha-zero/runs/relu-vs-gelu-20260906 \
  --require-success-file /workspace/alpha-zero/runs/relu-vs-gelu-20260906/exit-code.txt \
  --latest-checkpoint /workspace/alpha-zero-followups/reference/00000069 \
  --output-dir "$run"
result=$?
printf '%s\n' "$result" > "$run/exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/finished-at.txt"
exit "$result"
