#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: sweep_self_play_interactions.sh OUTPUT_DIR}
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
      --seed 29 \
      --output "$output_dir/${name}.json"
}

# Cross the strongest levels from the one-factor sweep. This catches interactions
# that a batch/concurrency/timeout sweep around a single center point cannot show.
run_case b64-p256-t5000 64 256 5000
run_case b64-p256-t100000 64 256 100000
run_case b64-p384-t1000 64 384 1000
run_case b64-p384-t5000 64 384 5000
run_case b64-p384-t100000 64 384 100000
run_case b128-p384-t5000 128 384 5000
run_case b128-p384-t100000 128 384 100000
