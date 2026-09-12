#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
export CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target"
run=/workspace/alpha-zero-followups/runs/value-heads-20260906/third-global
mkdir -p "$run"
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
  python3 - <<'PY' || return
import datetime,sys
from pathlib import Path
sys.path.insert(0,'scripts')
from experiment_io import digest,fixed,read,write
path=Path('global3-build-ready.json')
receipt={'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'binary_sha256':digest('validated/alz'),
 'source_sha256':{str(p):digest(p) for p in sorted(Path('src').rglob('*.rs'))},
 'cuda_checks':'Required by queue before training'}
if Path('/workspace/alpha-zero-followups/runs/value-heads-20260906/third-global/plan.json').exists():
 receipt['completed_at']=read(path)['completed_at']
 fixed(path,receipt)
else:
 write(path,receipt)
PY
  python3 -u scripts/archive/run_third_global_experiments.py \
    --predecessor /workspace/alpha-zero-followups/runs/value-heads-20260906/capacity \
    --output-dir "$run" --binary "$PWD/validated/alz"
}
execute
result=$?
printf '%s\n' "$result" > "$run/exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/finished-at.txt"
if [ "$result" -ne 0 ]; then
  python3 - "$run" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,'scripts')
from experiment_io import read,write
root=Path(sys.argv[1]);status=root/'status.json'
if not status.exists() or read(status).get('stage')!='failed':
 write(status,{'stage':'failed','failed_stage':'train-depth16-global3','error':'Build or tests failed; see launcher.log'})
PY
fi
exit "$result"
