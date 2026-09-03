#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n ${RUNPOD_WORKSPACE:-} || $repo_dir == /workspace/* ]]; then
  # shellcheck source=runpod_env.sh
  source "$repo_dir/scripts/runpod_env.sh"
fi
cd "$repo_dir"

if [[ ! -d web/node_modules ]]; then
  npm --prefix web ci
fi
npm --prefix web run build
./run.sh cargo build --release --locked -p alz-demo-server

exec ./run.sh ./target/release/alz-demo-server \
  --assets "$repo_dir/web/dist" \
  "$@"
