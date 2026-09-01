#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: repeat_self_play_finalists.sh OUTPUT_DIR}
architecture=${ARCHITECTURE:-kata-v1}
mkdir -p "$output_dir"

run_case() {
  local name=$1
  local cpu_list=$2
  local workers=$3
  local batch=$4
  local parallelism=$5
  local timeout_us=$6
  local repeat=$7
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS="$workers" \
    taskset --cpu-list "$cpu_list" \
      ./target/release/alz benchmark self-play --device cuda \
        --architecture "$architecture" \
        --games 384 \
        --simulations 32 \
        --inference-batch-size "$batch" \
        --games-parallelism "$parallelism" \
        --batch-timeout-us "$timeout_us" \
        --warmup-batches 5 \
        --seed "$((10000 + repeat))" \
        --output "$output_dir/${name}-r${repeat}.json"
}

for repeat in 1 2 3; do
  # Candidate A favors smaller, more consistently full batches.
  run_case b64-p384-t1000-w16 0-15 16 64 384 1000 "$repeat"
  # Candidate B favors the higher raw-inference throughput region.
  run_case b128-p384-t5000-w16 0-15 16 128 384 5000 "$repeat"
done

# Repeat the winner's likely configuration using one worker per physical core.
# On this host the sibling pairs are (0,1), (2,3), ..., (14,15).
for repeat in 1 2 3; do
  run_case b128-p384-t5000-w8 0,2,4,6,8,10,12,14 8 128 384 5000 "$repeat"
done
