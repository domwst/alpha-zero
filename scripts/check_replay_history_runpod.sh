#!/usr/bin/env bash
# Validate ordered reconstruction, resume, and subsequent self-play on CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runpod_env.sh
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
check_dir=$(mktemp -d /tmp/alz-history-cuda.XXXXXX)
trap 'rm -rf "$check_dir"' EXIT
torch_root=$(./run.sh python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')
c++ -shared -fPIC -O2 -std=c++20 -I"$torch_root/include" \
    scripts/cuda_deterministic_test.cpp -L"$torch_root/lib" -ltorch_cpu -lc10 \
    -o "$check_dir/deterministic.so"
./run.sh cargo test --release --locked --bin alz --no-run --message-format=json > "$check_dir/build.json"
test_exe=$(python3 - "$check_dir/build.json" <<'PY'
import json, sys
items = [json.loads(line) for line in open(sys.argv[1])]
executables = {x['executable'] for x in items if x.get('reason') == 'compiler-artifact'
               and x.get('executable') and x.get('profile', {}).get('test')}
assert len(executables) == 1, executables
print(executables.pop())
PY
)
# run.sh first establishes Torch's runtime paths. Keep CUDA registration when
# adding the deterministic test preload to the actual test process.
./run.sh env LD_PRELOAD="$torch_root/lib/libtorch_cuda.so:$check_dir/deterministic.so" \
    "$test_exe" replay_history_cuda --ignored --test-threads=1
