#!/bin/sh

RUSTFLAGS="-C link-arg=-Wl,-rpath,.venv/lib/python3.14/site-packages/torch/lib" LIBTORCH_USE_PYTORCH=1 exec uv run $@
