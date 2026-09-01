#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: repeat_self_play_baseline.sh OUTPUT_DIR}
architecture=${ARCHITECTURE:-kata-v1}
mkdir -p "$output_dir"

for repeat in 1 2 3; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16 \
    taskset --cpu-list 0-15 \
      ./target/release/alz benchmark self-play --device cuda \
        --architecture "$architecture" \
        --games 384 \
        --simulations 32 \
        --inference-batch-size 128 \
        --games-parallelism 160 \
        --batch-timeout-us 100000 \
        --warmup-batches 5 \
        --seed "$((10000 + repeat))" \
        --output "$output_dir/b128-p160-t100000-w16-r${repeat}.json"
done
