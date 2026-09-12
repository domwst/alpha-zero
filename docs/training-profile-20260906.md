# Training profile on the RunPod L40 — 6 September 2026

The training loop is limited substantially by CPU submission of many small GPU
operations. Unfused Adam is a concrete optimization target: the real training
step launches **702 optimizer kernels across 78 parameter tensors**, out of
**1,071 kernels per batch**. Serial replay preparation is also significant.
GPU memory transfers and explicit scalar-readback waits are much smaller costs.

No training implementation, checkpoint, hyperparameter, or queued experiment was
changed. Profiling used the existing native executable and disposable probe runs.

## Measurements

Three isolated, unprofiled synthetic GELU benchmarks, batch 256, 30 warmup and
100 measured iterations each:

| Repetition | Milliseconds per batch | Augmented positions/second |
| --- | ---: | ---: |
| 1 | 30.5253 | 8,386.5 |
| 2 | 30.5009 | 8,393.2 |
| 3 | 30.5269 | 8,386.0 |

Median: **30.5253 ms/batch**. The live replay trainer previously measured about
31.1 ms/batch, or 342 seconds for 11,004 batches in a full training pass.

The following are medians across **119 complete batches** in a five-second CUDA
trace of the actual `train-replay` command. They include profiler overhead and
are not unprofiled throughput estimates. Intervals can overlap and must not be
added as a breakdown of total time.

| Measurement in the replay trace | Median per batch |
| --- | ---: |
| Elapsed time between successive input batches | 41.363 ms |
| GPU kernel execution time, union of active intervals | 9.358 ms |
| Total kernels | 1,071 |
| Adam kernels | 702 |
| Elapsed span from first to last Adam kernel | 11.467 ms |
| GPU execution within that Adam span | 0.838 ms |
| Host gap after final loss readback, before next input transfer | 6.972 ms |
| CPU time inside kernel-launch APIs | 8.169 ms |
| GPU host-to-device copy time | 0.051 ms |
| GPU device-to-host copy time | 0.0026 ms |
| CPU time inside `cudaStreamSynchronize` | 0.071 ms |

The host gap includes cleanup, encoding, augmentation, stacking and conversion;
it is an **upper bound on encoding alone**, not a direct CPU stack profile.
The native loop does these operations serially before its next input transfer.

![GPU execution during one representative replay-training batch](../runs/training-profile-20260906/training-timeline.png)

The timeline uses the batch nearest the median total duration. Its individual
stage measurements therefore differ slightly from the table's stage medians.
An [SVG version](../runs/training-profile-20260906/training-timeline.svg) is also
available for zooming and export.

## Interpretation and priorities

1. **Reduce optimizer kernel launches.** The trace contains a repeating sequence
   of nine elementwise kernels per parameter tensor: coupled weight decay and
   Adam's moment/parameter updates. Their GPU execution takes under 1 ms, while
   the sequence spans over 11 ms under profiling. This identifies an optimizer
   implementation with substantial submission overhead. Benchmark a multi-tensor
   or fused Adam implementation while preserving coupled weight decay, epsilon,
   bias correction, gradient handling and checkpoint state. The current `tch`
   wrapper constructs C++ `torch::optim::Adam`; its exposed options do not include
   Python's `foreach` or `fused` switches. This requires implementation work, not
   merely adding a CLI flag. PyTorch documents its multi-tensor and fused
   implementations as generally faster than the per-tensor implementation.
   [PyTorch Adam documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html)

2. **Prepare batches ahead of GPU work.** Pre-encode or construct contiguous
   batches with fewer small tensor allocations, and prefetch the next batch.
   Preserve the exact sample order and augmentations. The approximately 7 ms
   host gap makes this worth measuring even though aggregate synthetic and live
   throughput happened to be similar. Those aggregate numbers alone were not
   sufficient to rule out preparation costs.

3. **Measure further reductions in network-operation submission overhead.** The
   remaining 369 kernels cover the network, loss and gradient computations.
   A CUDA graph or suitable fusion may help, but requires stable buffers and
   compatible optimizer state management. Measure after the first two changes;
   no speedup has yet been demonstrated for an optimized implementation.

The earlier suspicion about scalar readbacks was not borne out as a primary
bottleneck: explicit synchronization and copy timings are small in this capture.
Moving more data to the GPU alone will not remove the hundreds of optimizer
launches. Mixed precision and larger batches remain separate experiments because
they can change numerical or training behavior. Some convolution-gradient kernels
already use TF32 Tensor Core paths, as shown by their kernel names in the trace.

The synthetic trace provides a useful cross-check: all 121 complete batches have
993 kernels, including 624 Adam kernels. Its optimizer has zero weight decay,
so it performs eight kernels per tensor rather than the replay trainer's nine.
It also reads one loss scalar instead of two and skips sample encoding. It is
therefore a baseline, not an exact substitute for the production training loop.

## Capture conditions and limitations

- GPU: NVIDIA L40, 46,068 MiB; driver 580.95.05. The prior live observation was
  31% utilization, about 162 W, 61°C, with no active throttling reason reported.
- Native executable SHA-256:
  `d8324fc97ad5f8987b9310edf417f594869442cf5f08e37161bdbf9c458f6ab0`.
- Existing locked Torch 2.13.0+cu130 environment; `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`; no model or precision changes.
- Nsight Systems 2025.3.2.474, CUDA and OS runtime tracing. The NVIDIA package was
  extracted into a separate tools directory without changing the Torch runtime.
- CPU stack sampling is blocked by the container's `perf_event_open` permissions.
  Attribution uses CUDA API/kernel events, the repeating Adam sequence and
  inspection of the existing training code, not sampled C++ stacks.
- The replay probe used checkpoint 69's replay archive, a seeded 90/10 game
  split, 75,981 training positions, the same eight symmetries, GELU architecture,
  batch 256, Adam learning rate 0.001 and weight decay 0.0001. The five-second
  capture began after 15 seconds, during training, after initial validation.
  This profiles a fresh model on a representative subset of the live ten-archive
  pool; it does not resume or replace the experiment's model.
- Profiler overhead raised batch time from approximately 31 ms to approximately
  41 ms. Use unprofiled runs to assess any future speedup. The GPU-active interval
  is about 9.4 ms; substantial idle gaps also remain visible in the trace.
- The first replay capture produced Nsight import errors after forced target
  termination. It was discarded. The replacement report exported successfully,
  and every analyzed batch has the expected complete optimizer sequence.

### Effect on the live experiment

The live trainer was suspended in memory for 78.65 seconds for the first set of
measurements and 37.36 seconds for the replacement capture. Both suspensions had
an independent watchdog and a `finally` resume. Its original model, optimizer,
batch sequence and training budget were retained.

With `--kill=none`, Nsight exited after writing the replacement report while its
probe process continued. That probe overlapped with the resumed trainer for
56.58 seconds before being explicitly terminated. The accepted trace itself was
captured during exclusive GPU access. All profiling processes were subsequently
verified stopped, with only the original trainer remaining and GPU use back near
33%. **Live epoch 8 and epoch 9 wall-clock throughput is affected by these
interventions and should be excluded from performance comparisons.** Loss and
game-strength comparisons retain their original settings and training budget.

## Artifacts and reproduction

Local artifacts are in `runs/training-profile-20260906/`; pod artifacts are in
`/workspace/alz-training-profile-20260906/`. The verified retrieved archive has
SHA-256 `cdcf5d68772a2ef6b9f54e8e08e60e8df0cf4728862e5c9f26e89edf8f634c47`.

- `baseline-{1,2,3}.json`: unprofiled measurements.
- `synthetic-trace.nsys-rep`, `replay-complete-trace.nsys-rep`: successful traces.
- Corresponding `.sqlite` and `*_cuda_*.csv`: exported event tables and summaries.
- `synthetic-analysis.json`, `replay-analysis.json`: complete-batch statistics
  and representative raw kernel timelines.
- `capture-manifest.json`, `recapture-manifest.json`, `probe-cleanup.json`:
  executed commands, timing, executable identity and cleanup records.
- `plot.py`: Matplotlib code used for the standalone PNG/SVG timeline.

For a future isolated GPU allocation, reproduce the unprofiled benchmark with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./run.sh ./target/release/alz \
  benchmark training --architecture kata-gelu-v1 --device cuda \
  --batch-size 256 --warmup-iterations 30 --iterations 100 \
  --output runs/training-profile-baseline.json
```

Exact capture commands are in the manifests. If using `nsys --kill=none`, track
the target process separately: completion of the profiler does not guarantee
that the target has exited. Stop or await that disposable target before releasing
the GPU to another workload.

Recompute the accepted trace summary locally without GPU access:

```bash
python3 scripts/summarize_training_profile.py \
  runs/training-profile-20260906/replay-complete-trace.sqlite \
  --output runs/training-profile-20260906/replay-analysis.json
```

The analyzer opens SQLite read-only, checks that there is exactly one GPU process,
finds complete batches by their state-copy size, validates the repeating optimizer
tail, and computes the union of kernel execution intervals. It rejects an
unrecognized optimizer sequence rather than assigning misleading stage timings.
