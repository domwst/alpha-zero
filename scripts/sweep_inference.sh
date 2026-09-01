#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: sweep_inference.sh OUTPUT_DIR}
architecture=${ARCHITECTURE:-kata-v1}
mkdir -p "$output_dir"

batches=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
for repeat in 1 2 3; do
  for batch in "${batches[@]}"; do
    iterations=200
    if ((batch >= 128)); then
      iterations=100
    fi
    if ((batch >= 1024)); then
      iterations=40
    fi
    ./target/release/alz benchmark inference --device cuda \
      --architecture "$architecture" \
      --batch-size "$batch" \
      --warmup-iterations 10 \
      --iterations "$iterations" \
      --seed 0 \
      --output "$output_dir/inference-b${batch}-r${repeat}.json"
  done
done
