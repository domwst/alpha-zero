# Training performance backends

Training now has independent opt-in switches for fused Adam, pre-encoded replay
data, and CPU batch prefetching. They apply to `train`, `train-replay`, and
`benchmark training`. Defaults remain `--adam-backend standard --replay-cache none
--prefetch-batches 0`.

The workspace, including the demo server, pins the published `tch-rs` revision
`7e7d436ac73523afedf1a192892dfc9346568ba6`. Build with
`./run.sh cargo build --release --locked --bin alz`; Cargo fetches the fork from
Git, so deployment does not require transferring a local `tch-rs` checkout.

## Usage

For an allocation with enough GPU memory, add these flags to a new training run:

```bash
--adam-backend fused --replay-cache device
```

For a dataset that fits RAM but leaves insufficient GPU memory for training:

```bash
--adam-backend fused --replay-cache cpu --prefetch-batches 2
```

`device` means the selected training device (CPU when `--device cpu`). CPU
prefetching requires `--replay-cache cpu`; queue depth is bounded to 16 batches.
On CUDA, the worker pins the gathered CPU batches and the training thread issues
nonblocking transfers on its own stream. This overlaps CPU preparation with
training; it does not introduce a separate CUDA copy stream.

The cache stores float32 states, policies and values for all eight symmetries:
`base_positions × 8 × 1084 × 4` bytes. The current pooled training split needs
about 11.4 GiB, plus about 1.3 GiB for validation. Model, optimizer, activations,
gathered batches and allocator overhead need additional space. Cache construction
uses bounded staging buffers and reports allocation failures without silently
switching modes. CPU prefetch adds at most the queued batches, a producer batch,
and a consumer batch to the full CPU cache.

Fixed replay training builds both caches once and reuses them across passes.
Self-play training rebuilds its cache after the replay buffer changes each epoch;
the one-pass cost can offset the benefit there. `batch-cache.json` records cache
bytes, mode and initialization time for fixed replay runs. Epoch throughput
excludes cache construction; account for that one-time cost in total job time.

Sample ordering, symmetry ordering, shuffle seeds, final partial batches, loss
reductions and float32 training precision are preserved. Cache changes can be
used when resuming a fixed replay run. Its Adam backend is recorded in
`replay-config.json` and cannot change within that run directory; old configs
without this field mean standard Adam. Fused arithmetic can change rounding, so
use the same backend for both arms of a fresh controlled experiment.

## Fork implementation and checkpoints

The new API in `tch-rs` is:

```rust
use tch::nn::{Adam, OptimizerConfig};
let optimizer = Adam::default().wd(1e-4).fused().build(&var_store, 1e-3)?;
```

`torch-sys/libtch/fused_adam.h` batches defined gradients by parameter group,
device and dtype, then calls libtorch's existing multi-tensor fused Adam kernel.
It preserves coupled Adam weight decay, AMSGrad, per-parameter step counts and
the standard C++ Adam checkpoint format. Device step tensors are derived state
and are rebuilt after loading. Standard and fused optimizers can load each
other's archives. Parameters must use float32, float64, float16 or bfloat16, with
dense non-overlapping storage and matching gradient strides. This API supports
CPU/CUDA; it does not add AMP, loss scaling, AdamW, or CUDA graph capture.

Replay-specific encoding and worker ownership live in `src/training_batches.rs`,
with CLI selection and training integration in `src/cli.rs` and `src/commands/`.

## Validation and GPU measurement

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./run.sh cargo test --workspace
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./run.sh cargo test \
  --manifest-path ../tch-rs/Cargo.toml --test fused_adam_tests
```

Tests cover sample/target parity against the previous batching order, bounded
worker shutdown, invalid configurations, Adam/AMSGrad numerical parity, parameter
groups, mixed dtypes, transposed tensors, missing gradients, late parameters and
cross-backend checkpoint loading.
CUDA tests are explicit and ignored by the ordinary CPU suite. On an idle GPU:

```bash
bash scripts/run_cuda_tests_runpod.sh cache
bash scripts/run_cuda_tests_runpod.sh backend /path/to/pinned/tch-rs/Cargo.toml
python3 scripts/benchmark_training_backends.py \
  --replay-checkpoint-dir /path/to/checkpoints/00000069 \
  --output-dir runs/training-backend-comparison --device cuda
```

The benchmark compares five configurations on identical shuffled replay batches,
using fresh seeded models, weight decay 0.0001, 30 warmup iterations and 100 timed
iterations. It runs three repetitions and saves per-run JSON/logs plus medians,
relative speedups, preparation time and final losses. The replay must contain
enough augmented positions for warmup and measurement without wrapping.
Run with exclusive GPU access; concurrent experiments invalidate throughput
comparisons. Timed steps include batch acquisition, transfers, forward/backward,
Adam and the same two scalar loss readbacks as production. Schema version 6
records these settings; historical synthetic benchmarks used one readback.

The implementation has been validated locally on CPU. The CUDA cache/prefetch
parity test also passed on the L40 after correcting the test launcher. Full
fused-Adam CUDA validation and comparative L40 throughput remain outstanding;
the earlier 1.35–1.75× estimate is not a result.

### Live queue audit, 7 September 2026

Both value-head runs completed using the approved fallback: the previous
executable, standard Adam and uncached batches. Validation initially failed
because executing tests through `run.sh cargo test` skipped CUDA preloading;
the test saw CUDA as unavailable. This was a launcher failure, not a numerical
parity failure. `scripts/run_cuda_tests_runpod.sh` now compiles the tests and
runs their executables through `run.sh` separately. Correcting the launcher did
not switch those value-head runs away from their already selected fallback.

Recent full-pass intervals were 356.6 s for the original GELU, 356.8 s for the
wide head and 356.8 s for the deep head. They therefore show no optimization
speedup. These intervals include validation and checkpoint writing; recorded
training alone was about 342 s in the final pass of each run.

The pending pooling comparison now reuses the selected existing checkpoint
and requests device caching for its one new model, after CUDA architecture
and cache checks. Adam stays matched to the baseline (standard for all current
candidates). Cache-check failure falls back to uncached batching. Fused Adam
would need both arms trained with that backend for an equally controlled
comparison; enabling it on just this new architecture would add a confound.

Local checks on 6 September 2026 passed: 80 workspace tests, four fork optimizer
tests, and a full self-play smoke epoch with fused Adam and CPU prefetching.
Saved replay smoke runs produced bit-for-bit identical weights between the old
executable, new standard-Adam default, CPU cache/prefetch and device cache on CPU.
Resuming an old standard checkpoint and resuming a fused checkpoint both matched
their uninterrupted runs exactly.

Standard versus fused Adam after two replay passes had a maximum parameter
difference of 0.00185, concentrated in the input convolution bias, with associated
BatchNorm running-mean drift. Training total loss differed by less than 8e-8;
validation total loss differed by about 0.00087. Strict full-model parameter
closeness did not hold. Tests with identical tiny gradients passed, consistent
with rounding being amplified through repeated network updates. Backend parity
is numerical, not bitwise. Artifacts are in `runs/training-optimization-20260906/`,
including `smoke-results.json`, checkpoints, logs and the CPU benchmark matrix.
