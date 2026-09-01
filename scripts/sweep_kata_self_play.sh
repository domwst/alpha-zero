#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: sweep_kata_self_play.sh OUTPUT_DIR}
architecture=${ARCHITECTURE:-kata-v1}
mkdir -p "$output_dir"

run_case() {
  local name=$1
  local batch=$2
  local parallelism=$3
  local timeout_us=$4
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16 \
    ./target/release/alz benchmark self-play \
      --architecture "$architecture" \
      --device cuda \
      --games 500 \
      --simulations 16 \
      --inference-batch-size "$batch" \
      --games-parallelism "$parallelism" \
      --batch-timeout-us "$timeout_us" \
      --warmup-batches 5 \
      --seed 20260901 \
      --output "$output_dir/${name}.json"
}

run_case baseline-b128-p160-t100000 128 160 100000
run_case b128-p384-t1000 128 384 1000
run_case b192-p384-t1000 192 384 1000
run_case b256-p256-t1000 256 256 1000
run_case b256-p384-t1000 256 384 1000
run_case b256-p500-t1000 256 500 1000
run_case b256-p500-t100000 256 500 100000
