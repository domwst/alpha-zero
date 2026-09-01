#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: sweep_self_play.sh OUTPUT_DIR}
architecture=${ARCHITECTURE:-kata-v1}
mkdir -p "$output_dir"

run_case() {
  local name=$1
  local batch=$2
  local parallelism=$3
  local timeout_us=$4
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ./target/release/alz benchmark self-play --device cuda \
      --architecture "$architecture" \
      --games 384 \
      --simulations 16 \
      --inference-batch-size "$batch" \
      --games-parallelism "$parallelism" \
      --batch-timeout-us "$timeout_us" \
      --warmup-batches 5 \
      --seed 17 \
      --output "$output_dir/${name}.json"
}

# Batch-size sweep at fixed concurrency and timeout.
run_case batch-b32 32 256 1000
run_case batch-b64 64 256 1000
run_case batch-b128 128 256 1000
run_case batch-b256 256 256 1000

# Concurrency sweep at the center inference candidate.
run_case parallel-p64 128 64 1000
run_case parallel-p128 128 128 1000
run_case parallel-p192 128 192 1000
run_case parallel-p384 128 384 1000

# Timeout sweep at batch 128 / concurrency 256.
run_case timeout-t0 128 256 0
run_case timeout-t100 128 256 100
run_case timeout-t500 128 256 500
run_case timeout-t2000 128 256 2000
run_case timeout-t5000 128 256 5000
run_case timeout-t100000 128 256 100000
