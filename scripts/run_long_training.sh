#!/usr/bin/env bash
set -euo pipefail

run_root=${1:?usage: run_long_training.sh RUN_ROOT}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
architecture=${ARCHITECTURE:-kata-v1}
epochs=${EPOCHS:-50}
inference_batch_size=${INFERENCE_BATCH_SIZE:-256}
inference_symmetry=${INFERENCE_SYMMETRY:-random}
games_parallelism=${GAMES_PARALLELISM:-500}
batch_timeout_us=${BATCH_TIMEOUT_US:-1000}
training_batch_size=${TRAINING_BATCH_SIZE:-256}
seed=${SEED:-20260901}
mkdir -p "$run_root"
run_root=$(cd -- "$run_root" && pwd)

segment_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
segment_dir="$run_root/segments/$segment_id"
mkdir -p "$segment_dir" "$run_root/checkpoints" "$run_root/games" "$run_root/stats"
printf '%s\n' "$segment_dir" >"$run_root/current-segment.txt"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKIO_WORKER_THREADS=${TOKIO_WORKER_THREADS:-16}
export GPU_MONITOR_INTERVAL_MS=1000
export ALZ_LOG_ANSI=${ALZ_LOG_ANSI:-always}

exec "$repo_dir/scripts/run_profiled.sh" "$segment_dir" \
  "$repo_dir/target/release/alz" train \
    --architecture "$architecture" \
    --device cuda \
    --epochs "$epochs" \
    --games-per-epoch 700 \
    --simulations 2000 \
    --inference-batch-size "$inference_batch_size" \
    --inference-symmetry "$inference_symmetry" \
    --games-parallelism "$games_parallelism" \
    --batch-timeout-us "$batch_timeout_us" \
    --training-batch-size "$training_batch_size" \
    --replay-games 1800 \
    --rendered-games 5 \
    --seed "$seed" \
    --progress-every-games 10 \
    --heartbeat-seconds 60 \
    --checkpoint-dir "$run_root/checkpoints" \
    --games-dir "$run_root/games" \
    --stats-dir "$run_root/stats"
