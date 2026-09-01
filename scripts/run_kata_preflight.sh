#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: run_kata_preflight.sh OUTPUT_DIR}
mkdir -p "$output_dir/inference" "$output_dir/training" "$output_dir/selfplay"

for repeat in 1 2 3; do
  ./target/release/alz benchmark inference \
    --architecture kata-v1 \
    --device cuda \
    --batch-size 256 \
    --warmup-iterations 10 \
    --iterations 100 \
    --seed 0 \
    --output "$output_dir/inference/b256-r${repeat}.json"
done

for repeat in 1 2 3; do
  ./target/release/alz benchmark training \
    --architecture kata-v1 \
    --device cuda \
    --batch-size 256 \
    --warmup-iterations 3 \
    --iterations 10 \
    --seed 0 \
    --output "$output_dir/training/b256-r${repeat}.json"
done

for workers in 16 24; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=$workers \
    ./target/release/alz benchmark self-play \
      --architecture kata-v1 \
      --device cuda \
      --games 500 \
      --simulations 16 \
      --inference-batch-size 256 \
      --games-parallelism 500 \
      --batch-timeout-us 1000 \
      --warmup-batches 5 \
      --seed 20260901 \
      --output "$output_dir/selfplay/b256-p500-t1000-w${workers}.json"
done
