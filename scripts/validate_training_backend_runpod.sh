#!/usr/bin/env bash
# Run at the queue boundary, with exclusive GPU access and prebuilt test binaries.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
export CARGO_TARGET_DIR="$repo/target"
fork_manifest=$1
./run.sh cargo test --release --locked --lib cuda_cache_and_pinned_prefetch_match_legacy_batches -- --ignored
./run.sh cargo test --release --offline --manifest-path "$fork_manifest" --test fused_adam_tests fused_adam_cuda -- --ignored
printf 'CUDA replay-cache, pinned-prefetch, fused Adam and optimizer checkpoint checks passed.\n'
