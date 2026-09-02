#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?usage: run_checkpoint_tournament.sh RUN_DIR}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=runpod_env.sh
source "$repo_dir/scripts/runpod_env.sh"

kata_root=${KATA_CHECKPOINT_ROOT:-$repo_dir/runs/kata-v1-50e-20260901/checkpoints}
legacy_root=${LEGACY_CHECKPOINT_ROOT:-$repo_dir/runs/training-50e-20260829/checkpoints}
games=${GAMES:-200}
simulations=${SIMULATIONS:-2000}
temperature=${TEMPERATURE:-0.7}
games_parallelism=${GAMES_PARALLELISM:-200}
inference_batch_size=${INFERENCE_BATCH_SIZE:-128}
batch_timeout_us=${BATCH_TIMEOUT_US:-1000}
base_seed=${SEED:-20260901}
cpu_list=${RUNPOD_CPU_LIST:-64-95}
max_parallel_series=${MAX_PARALLEL_SERIES:-3}

if ((max_parallel_series < 1)); then
  printf 'MAX_PARALLEL_SERIES must be at least one\n' >&2
  exit 1
fi

mkdir -p "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKIO_WORKER_THREADS=${TOKIO_WORKER_THREADS:-16}
export GPU_MONITOR_INTERVAL_MS=${GPU_MONITOR_INTERVAL_MS:-1000}
export ALZ_LOG_ANSI=${ALZ_LOG_ANSI:-always}

checkpoint_path() {
  local root=$1
  local epoch=$2
  printf '%s/%08d\n' "$root" "$epoch"
}

require_checkpoint() {
  local path=$1
  if [[ ! -f "$path/metadata.json" || ! -f "$path/model.safetensors" ]]; then
    printf 'incomplete checkpoint: %s\n' "$path" >&2
    exit 1
  fi
}

run_battle() {
  local group=$1
  local first_name=$2
  local first_path=$3
  local second_name=$4
  local second_path=$5
  local match_index=$6
  local match_dir="$run_dir/$group/matches/$first_name-vs-$second_name"
  local result="$match_dir/result.json"

  require_checkpoint "$first_path"
  require_checkpoint "$second_path"
  if [[ -f "$result" ]] && jq -e --argjson games "$games" \
      '.schema_version == 1 and (.games | length) == $games' "$result" >/dev/null; then
    printf 'already complete: %s\n' "$first_name-vs-$second_name"
    return
  fi

  mkdir -p "$match_dir"
  printf 'starting %s vs %s (%s games)\n' "$first_name" "$second_name" "$games"
  local command=(
    "$repo_dir/target/release/alz" battle
    --first-checkpoint-dir "$first_path"
    --second-checkpoint-dir "$second_path"
    --device cuda
    --games "$games"
    --games-parallelism "$games_parallelism"
    --inference-batch-size "$inference_batch_size"
    --batch-timeout-us "$batch_timeout_us"
    --simulations "$simulations"
    --temperature "$temperature"
    --seed "$((base_seed + match_index * 1000003))"
    --heartbeat-seconds 60
    --no-move-logs
    --output "$result"
  )
  if [[ -n "$cpu_list" ]]; then
    "$repo_dir/scripts/run_profiled.sh" "$match_dir" \
      taskset --cpu-list "$cpu_list" "${command[@]}"
  else
    "$repo_dir/scripts/run_profiled.sh" "$match_dir" "${command[@]}"
  fi
}

battle_pids=()
start_battle() {
  run_battle "$@" &
  battle_pids+=("$!")
}

wait_for_battles() {
  local failed=0
  local pid
  for pid in "${battle_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  battle_pids=()
  if ((failed)); then
    printf 'one or more battle series failed\n' >&2
    return 1
  fi
}

wait_for_one_battle() {
  local completed_pid=
  local status=0
  local pid
  local remaining=()
  wait -n -p completed_pid "${battle_pids[@]}" || status=$?
  for pid in "${battle_pids[@]}"; do
    if [[ "$pid" != "$completed_pid" ]]; then
      remaining+=("$pid")
    fi
  done
  battle_pids=("${remaining[@]}")
  if ((status != 0)); then
    printf 'battle series process %s failed with status %s\n' "$completed_pid" "$status" >&2
    return "$status"
  fi
}

wait_if_full() {
  if ((${#battle_pids[@]} >= max_parallel_series)); then
    wait_for_one_battle
  fi
}

kata_epochs=(12 16 18 19)
match_index=0
for ((first_index = 0; first_index < ${#kata_epochs[@]}; first_index++)); do
  for ((second_index = first_index + 1; second_index < ${#kata_epochs[@]}; second_index++)); do
    first_epoch=${kata_epochs[$first_index]}
    second_epoch=${kata_epochs[$second_index]}
    start_battle \
      kata-round-robin \
      "kata-$first_epoch" "$(checkpoint_path "$kata_root" "$first_epoch")" \
      "kata-$second_epoch" "$(checkpoint_path "$kata_root" "$second_epoch")" \
      "$match_index"
    ((match_index += 1))
    wait_if_full
  done
done
wait_for_battles

"$repo_dir/scripts/analyze_battle_replays.py" \
  "$run_dir/kata-round-robin" \
  --json-output "$run_dir/kata-round-robin-summary.json" \
  --markdown-output "$run_dir/kata-round-robin-report.md"

mapfile -t top_kata_epochs < <(
  jq -r '.ranking[0:2][] | select(.architecture == "kata_v1") | .epoch' \
    "$run_dir/kata-round-robin-summary.json"
)
if [[ ${#top_kata_epochs[@]} -ne 2 ]]; then
  printf 'could not select exactly two Kata checkpoints from round-robin ranking\n' >&2
  exit 1
fi
printf 'top Kata checkpoints: %s %s\n' "${top_kata_epochs[0]}" "${top_kata_epochs[1]}"

legacy_epochs=(129 125 115 105 90 80 70)
for kata_epoch in "${top_kata_epochs[@]}"; do
  for legacy_epoch in "${legacy_epochs[@]}"; do
    start_battle \
      kata-vs-legacy \
      "kata-$kata_epoch" "$(checkpoint_path "$kata_root" "$kata_epoch")" \
      "legacy-$legacy_epoch" "$(checkpoint_path "$legacy_root" "$legacy_epoch")" \
      "$match_index"
    ((match_index += 1))
    wait_if_full
  done
done
wait_for_battles

"$repo_dir/scripts/analyze_battle_replays.py" \
  "$run_dir" \
  --json-output "$run_dir/tournament-summary.json" \
  --markdown-output "$run_dir/tournament-report.md"

printf 'tournament complete: %s\n' "$run_dir"
