#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runpod_env.sh
root="$PWD/runs/selfplay-nucleus"
# Refuse duplicate supervisors; the Python worker also holds an exclusive lock.
if tmux has-session -t alz-selfplay 2>/dev/null; then
  echo 'alz-selfplay already exists; attach with: tmux attach -t alz-selfplay'
  exit 1
fi
mkdir -p "$root/nucleus-p095" "$root/control-p100"
touch "$root/nucleus-p095/stdout.log" "$root/control-p100/stdout.log"
tmux new-session -d -s alz-selfplay -n supervisor -c "$PWD" \
  'source scripts/runpod_env.sh; set -o pipefail; python3 -u scripts/run_self_play_experiments.py --run-dir runs/selfplay-nucleus 2>&1 | tee -a runs/selfplay-nucleus/supervisor.log; code=${PIPESTATUS[0]}; printf "%s\n" "$code" > runs/selfplay-nucleus/supervisor-exit.txt; exec bash'
tmux new-window -t alz-selfplay -n nucleus -c "$PWD" \
  'tail -n 80 -F runs/selfplay-nucleus/nucleus-p095/stdout.log'
tmux new-window -t alz-selfplay -n control -c "$PWD" \
  'tail -n 80 -F runs/selfplay-nucleus/control-p100/stdout.log'
tmux new-window -t alz-selfplay -n resources -c "$PWD" \
  'watch -n 5 "cat runs/selfplay-nucleus/status.json; nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv"'
tmux select-window -t alz-selfplay:nucleus
printf 'Started. Attach with: tmux attach -t alz-selfplay\n'
