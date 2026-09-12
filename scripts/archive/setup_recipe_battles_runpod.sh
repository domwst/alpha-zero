#!/usr/bin/env bash
set -euo pipefail
cd /workspace/alpha-zero-battles
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
export CARGO_BUILD_JOBS=4
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential ca-certificates curl git pkg-config tmux xz-utils
if [[ ! -x /workspace/.local/bin/uv ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/.local/bin sh
fi
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain nightly-2026-08-27
fi
uv python install 3.14
uv sync --frozen
./run.sh python -c 'import torch; assert torch.cuda.is_available(); assert torch.isfinite(torch.randn(64,64,device="cuda") @ torch.randn(64,64,device="cuda")).all(); torch.cuda.synchronize(); print(torch.__version__, torch.cuda.get_device_name())'
./run.sh cargo build --release --locked --example check_checkpoint_cuda
NVIDIA_TF32_OVERRIDE=0 ./run.sh target/release/examples/check_checkpoint_cuda --models-dir models --expected-models 6 > runs/recipe-battles/cuda-check.log 2>&1
./run.sh validated/alz benchmark inference --device cuda --architecture kata-gelu-value64x2-v1 --batch-size 64 --warmup-iterations 30 --iterations 200 --output runs/recipe-battles/inference-benchmark.json > runs/recipe-battles/inference-benchmark.log 2>&1
python3 - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from experiment_io import write, digest
from experiment_io import now
write('runs/recipe-battles/preflight.json', {'passed': True, 'completed_at': now(),
      'checks': ['six selected checkpoints CPU/CUDA outputs at batches 1, 4, 64', 'CUDA inference benchmark'],
      'binary_sha256': digest('validated/alz')})
PY
