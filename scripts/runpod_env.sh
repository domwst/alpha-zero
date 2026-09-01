#!/usr/bin/env bash

# Source this file to use the persistent RunPod toolchain from an interactive
# shell. Other RunPod scripts source it automatically.
export RUNPOD_WORKSPACE=${RUNPOD_WORKSPACE:-/workspace}
export CARGO_HOME=${CARGO_HOME:-$RUNPOD_WORKSPACE/.cargo}
export RUSTUP_HOME=${RUSTUP_HOME:-$RUNPOD_WORKSPACE/.rustup}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$RUNPOD_WORKSPACE/.cache/uv}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$RUNPOD_WORKSPACE/.cache}
export PATH="$RUNPOD_WORKSPACE/.local/bin:$CARGO_HOME/bin:$PATH"

