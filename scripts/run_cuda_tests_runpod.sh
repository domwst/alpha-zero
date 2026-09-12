#!/usr/bin/env bash
# Compile with Cargo, then run the test executable through the CUDA-loading wrapper.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
export CARGO_TARGET_DIR="$PWD/target"
run_test() {
  local test_filter=$1 test_binary
  shift
  test_binary=$(./run.sh cargo test --release --no-run --message-format=json "$@" | python3 -c '
import json,sys
artifacts=[json.loads(line) for line in sys.stdin]
executables={item["executable"] for item in artifacts if item.get("reason")=="compiler-artifact" and item.get("executable") and item.get("profile",{}).get("test")}
assert len(executables)==1, executables
print(executables.pop())')
  ./run.sh "$test_binary" "$test_filter" --ignored --list | python3 -c 'import sys; assert any(line.rstrip().endswith(": test") for line in sys.stdin), "No matching CUDA tests"'
  ./run.sh "$test_binary" "$test_filter" --ignored --nocapture
}
case "${1:-}" in
  capacity)
    # Compare CPU and CUDA at matching precision. This override is confined to
    # this validation process; actual replay preflights use training defaults.
    export NVIDIA_TF32_OVERRIDE=0
    run_test capacity_cuda_matches_cpu --locked --lib
    ;;
  architecture) run_test pooling_cuda_matches_cpu --locked --lib ;;
  cache) run_test cuda_cache_and_pinned_prefetch_match_legacy_batches --locked --lib ;;
  backend)
    run_test cuda_cache_and_pinned_prefetch_match_legacy_batches --locked --lib
    run_test fused_adam_cuda --offline --manifest-path "${2:?fork manifest required}" --test fused_adam_tests
    ;;
  *) exit 2 ;;
esac
