#!/usr/bin/env bash
set -u

output_dir=${1:?usage: sweep_training.sh OUTPUT_DIR}
architecture=${ARCHITECTURE:-kata-v1}
mkdir -p "$output_dir"

batches=(128 256 512 1024 2048 4096 8192)
status=0
for repeat in 1 2 3; do
  for batch in "${batches[@]}"; do
    if ! ./target/release/alz benchmark training --device cuda \
      --architecture "$architecture" \
      --batch-size "$batch" \
      --warmup-iterations 3 \
      --iterations 10 \
      --seed 0 \
      --output "$output_dir/training-b${batch}-r${repeat}.json"; then
      printf 'training benchmark failed: batch=%s repeat=%s\n' "$batch" "$repeat" >&2
      status=1
    fi
  done
done
exit "$status"
