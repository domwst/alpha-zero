#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s REPLAY_CHECKPOINT_OR_ROOT_DIR EXPERIMENT_DIR\n' "$0" >&2
  exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
replay_dir=$(realpath -- "$1")
experiment_dir=$(realpath -m -- "$2")
cd "$repo_dir"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TOKIO_WORKER_THREADS=${TOKIO_WORKER_THREADS:-16}
device=${DEVICE:-cuda}
epochs=${EPOCHS:-20}
seed=${SEED:-20260906}
alz_binary=${ALZ_BINARY:-$repo_dir/target/release/alz}
replay_snapshots=${REPLAY_SNAPSHOTS:-10}
[[ "$replay_snapshots" =~ ^[1-9][0-9]*$ ]] || { printf 'REPLAY_SNAPSHOTS must be positive.\n' >&2; exit 2; }
replay_args=()
if [[ -f "$replay_dir/metadata.json" ]]; then
  replay_args+=(--replay-checkpoint-dir "$replay_dir")
else
  sources=()
  for candidate in "$replay_dir"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]; do
    if [[ -f "$candidate/metadata.json" && -f "$candidate/replay.bin.zst" ]]; then
      sources+=("$candidate")
    fi
  done
  [[ ${#sources[@]} -gt 0 ]] || { printf 'No replay snapshots found.\n' >&2; exit 1; }
  start=$((${#sources[@]} > replay_snapshots ? ${#sources[@]} - replay_snapshots : 0))
  for candidate in "${sources[@]:start}"; do
    replay_args+=(--replay-checkpoint-dir "$candidate")
  done
fi

if [[ ! -x "$alz_binary" ]]; then
  printf 'Build the current binary with ./run.sh cargo build --release --locked first.\n' >&2
  exit 1
fi
mkdir -p "$experiment_dir"
# Do not overwrite an already completed match, or launch two writers into one experiment.
exec 9>"$experiment_dir/.lock"
flock -n 9 || { printf 'Experiment is already running.\n' >&2; exit 1; }
if [[ -e "$experiment_dir/battle.json" ]]; then
  printf 'Match already exists: %s/battle.json; use a fresh experiment directory.\n' "$experiment_dir" >&2
  exit 1
fi

binary_sha256=$(sha256sum "$alz_binary" | cut -d ' ' -f 1)
if [[ -f "$experiment_dir/binary.sha256" ]]; then
  [[ $(cat "$experiment_dir/binary.sha256") == "$binary_sha256" ]] || {
    printf 'Binary changed since this experiment started; use a fresh directory.\n' >&2
    exit 1
  }
else
  printf '%s\n' "$binary_sha256" >"$experiment_dir/binary.sha256"
fi

for architecture in kata-v1 kata-gelu-v1; do
  ./run.sh "$alz_binary" train-replay \
    "${replay_args[@]}" \
    --run-dir "$experiment_dir/$architecture" \
    --architecture "$architecture" \
    --device "$device" \
    --epochs "$epochs" \
    --training-batch-size "${TRAINING_BATCH_SIZE:-256}" \
    --validation-fraction 0.1 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --seed "$seed" \
    2>&1 | tee -a "$experiment_dir/$architecture.log"
done

# The trainer rejects resuming a run with incompatible data or training settings.
# Explicit epochs pin the match to equal training exposure, independent of later checkpoints.
checkpoint_epoch=$(printf '%08d' "$((epochs - 1))")
./run.sh "$alz_binary" battle \
  --first-checkpoint-dir "$experiment_dir/kata-v1/checkpoints/$checkpoint_epoch" \
  --second-checkpoint-dir "$experiment_dir/kata-gelu-v1/checkpoints/$checkpoint_epoch" \
  --device "$device" \
  --games "${BATTLE_GAMES:-300}" \
  --simulations "${SIMULATIONS:-4000}" \
  --temperature 0.7 \
  --games-parallelism "${GAMES_PARALLELISM:-128}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE:-64}" \
  --batch-timeout-us 1000 \
  --seed "$seed" \
  --heartbeat-seconds 60 \
  --no-move-logs \
  --output "$experiment_dir/battle.json" \
  2>&1 | tee -a "$experiment_dir/battle.log"
