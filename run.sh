#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir"

venv_dir="$script_dir/.venv"
find_torch_lib() {
    for candidate in "$venv_dir"/lib/python*/site-packages/torch/lib; do
        if [ -d "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

torch_lib=$(find_torch_lib || true)
if [ -z "$torch_lib" ]; then
    uv sync --frozen
    torch_lib=$(find_torch_lib || true)
fi
if [ -z "$torch_lib" ]; then
    printf 'unable to locate torch/lib under %s after uv sync\n' "$venv_dir" >&2
    exit 1
fi

runtime_libs="$torch_lib"
nvidia_root=$(dirname -- "$(dirname -- "$torch_lib")")/nvidia
for nvidia_lib in "$nvidia_root"/*/lib; do
    if [ -d "$nvidia_lib" ]; then
        runtime_libs="$runtime_libs:$nvidia_lib"
    fi
done
cuda_driver=
system_zlib=
if command -v c++ >/dev/null 2>&1 && command -v nix-store >/dev/null 2>&1; then
    cxx_binary=$(readlink -f "$(command -v c++)" 2>/dev/null || true)
    case "$cxx_binary" in
        /nix/store/*)
            for dependency in $(nix-store -q --references "$cxx_binary" 2>/dev/null || true); do
                if [ -f "$dependency/lib/libstdc++.so.6" ]; then
                    runtime_libs="$runtime_libs:$dependency/lib"
                    break
                fi
            done
            if command -v ldconfig >/dev/null 2>&1; then
                cuda_driver=$(ldconfig -p 2>/dev/null | awk '$1 == "libcuda.so.1" { print $NF; exit }')
                system_zlib=$(ldconfig -p 2>/dev/null | awk '$1 == "libz.so.1" { print $NF; exit }')
            fi
            ;;
    esac
fi
export LIBTORCH_USE_PYTORCH=1
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C link-arg=-Wl,-rpath,$torch_lib"

preload_libs=
if [ "$(uname -s)" = Linux ] && [ "${1:-}" != cargo ] && [ -f "$torch_lib/libtorch_cuda.so" ]; then
    preload_libs="$torch_lib/libtorch_cuda.so"
    if [ -f "$torch_lib/libtorch_global_deps.so" ]; then
        preload_libs="$torch_lib/libtorch_global_deps.so:$preload_libs"
    fi
    if [ -f "$system_zlib" ]; then
        preload_libs="$system_zlib:$preload_libs"
    fi
    if [ -f "$cuda_driver" ]; then
        preload_libs="$cuda_driver:$preload_libs"
    fi
fi

exec uv run --frozen env \
    LIBTORCH_USE_PYTORCH="$LIBTORCH_USE_PYTORCH" \
    RUSTFLAGS="$RUSTFLAGS" \
    LD_LIBRARY_PATH="$runtime_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    DYLD_LIBRARY_PATH="$torch_lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
    LD_PRELOAD="$preload_libs${LD_PRELOAD:+:$LD_PRELOAD}" \
    "$@"
