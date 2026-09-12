#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
export CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target"
run=/workspace/alpha-zero-followups/runs/value-heads-20260906/training-recipes
mkdir -p "$run"
exec 9>"$run/.launcher-lock"
flock -n 9 || exit 1
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/launcher-started-at.txt"
bash scripts/monitor_gpu.sh "$run/gpu.csv" 1000 &
gpu_monitor=$!
bash scripts/monitor_host.sh "$run/host.csv" 1000 $$ &
host_monitor=$!
cleanup() { kill "$gpu_monitor" "$host_monitor" 2>/dev/null || true; }
trap cleanup EXIT
execute() {
  ./run.sh cargo build --release --locked --bin alz || return
  ./run.sh cargo test --release --locked --lib --bin alz || return
  mkdir -p validated || return
  cp target/release/alz validated/alz || return
  python3 -u scripts/archive/run_training_recipe_experiments.py \
    --output-dir "$run" --binary "$PWD/validated/alz" \
    --baseline-dir /workspace/alpha-zero-followups/runs/value-heads-20260906/kata-gelu-value64x2-v1
}
execute
result=$?
printf '%s\n' "$result" > "$run/exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/finished-at.txt"
if [ "$result" -ne 0 ]; then
  python3 - "$run" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from experiment_io import read,write
root=Path(sys.argv[1]); path=root/'status.json'
if not path.exists() or read(path).get('stage') != 'failed':
 write(path, {'stage':'failed', 'error':'Build or checks failed; see launcher.log'})
PY
fi
exit "$result"
