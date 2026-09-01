#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=runpod_env.sh
source "$repo_dir/scripts/runpod_env.sh"

case "$repo_dir" in
  "$RUNPOD_WORKSPACE"/*) ;;
  *)
    printf 'repository must live below persistent workspace %s (got %s)\n' \
      "$RUNPOD_WORKSPACE" "$repo_dir" >&2
    exit 1
    ;;
esac

mkdir -p \
  "$CARGO_HOME" \
  "$RUSTUP_HOME" \
  "$RUNPOD_WORKSPACE/.cache/uv" \
  "$RUNPOD_WORKSPACE/.local/bin"

if [[ $EUID -eq 0 ]]; then
  apt=(apt-get)
else
  apt=(sudo apt-get)
fi

"${apt[@]}" update -qq
DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  jq \
  numactl \
  pkg-config \
  rsync \
  tmux \
  zstd

if [[ ! -x "$RUNPOD_WORKSPACE/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh |
    env UV_INSTALL_DIR="$RUNPOD_WORKSPACE/.local/bin" sh
fi

if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable
fi

rustup_toolchain=${RUNPOD_RUST_TOOLCHAIN:-nightly-2026-08-27}
if ! rustup run "$rustup_toolchain" rustc --version >/dev/null 2>&1; then
  rustup toolchain install "$rustup_toolchain" --profile minimal
fi
rustup default "$rustup_toolchain"

cd "$repo_dir"
uv python install 3.14
uv sync --frozen
./run.sh cargo build --release --locked
./run.sh python scripts/check_cuda.py
./run.sh ./target/release/alz --help >/dev/null

printf '\nRunPod setup complete. For an interactive shell, run:\n'
printf '  source %q\n' "$repo_dir/scripts/runpod_env.sh"
printf 'binary_sha256=%s\n' "$(sha256sum target/release/alz | awk '{print $1}')"
