# Live self-play profile, 15 September 2026

The main opportunity is faster inference during game generation, followed by
reducing tree memory and improving low-concurrency inference. Optimizing the
training loop has little effect on total epoch time in this run.

## Scope and limitations

- Job: `e14eed41-a865-4f43-b852-7c71473a72f4`, Sharp Temp / top-p 1.0.
- Live sample: 03:28:38–03:30:08 UTC, 90 seconds during epoch 24 (one-based).
- PID 187837, start ticks 1233242025, unchanged after profiling.
- Executable SHA-256: `2ad7a1ba6f0e1c1341d75428f77b5c1f211f1f032939717f83067c65517e9bb6`.
  This is the deployed September 13 binary, not the subsequently edited source.
- Settings: S3000/P400/B128 inference, 1,000 games/epoch, training B256,
  standard Adam, CPU replay cache, no prefetch, random inference symmetry.
- RTX 3090 24 GiB; cgroup CPU quota 27.2 cores; host memory limit 125 GB
  (116.4 GiB). The runtime has 16 Tokio workers.
- Both `perf_event_open` and `PTRACE_SEIZE` returned EPERM. No permissions were
  changed and no debugger remained attached. Consequently this is **live resource
  sampling plus existing instrumented phase timings**, not CPU stack sampling or
  a CUDA kernel timeline. Function-level causes below are hypotheses from source
  inspection; measured phase totals are identified separately.
- Completed-epoch timing evidence covers epochs 19–23. Tree diagnostics are a
  deterministic sample of 40 games (every 25th game ID) from epoch 23, comprising
  1,776 positions. This sample is not a census of peak simultaneous trees.

[Machine-readable summary](../reports/self-play-profile-20260915/summary.json).

## Live measurements

| Measurement | Observed |
| --- | ---: |
| Evaluations/s, heartbeat counter delta | 24,535 |
| Batch size, counter delta | 128.0 |
| Active games | 400 throughout |
| Completed games | 173 → 194 |
| GPU active time, NVIDIA samples | mean 49.4%, range 41–56% |
| GPU power | mean 291 W |
| GPU SM clock | 1,950 MHz throughout |
| VRAM | 1,614–1,875 MiB |
| Process CPU | 4.76 cores averaged over 90 seconds |
| CPU throttling | zero events/time |
| Process resident memory | 96.51 GiB |
| CPU runqueue delay, summed over all threads | 0.245 seconds over 90 seconds |

These readings do not suggest insufficient CPU quota or too few worker threads.
They do not exclude a **serial** CPU submission bottleneck: the inference task can
move between runtime threads. NVIDIA utilization is time with GPU activity, not
a percentage of theoretical FLOP throughput or an end-to-end speedup budget.

The batcher is already full during this part of generation. More games would
consume additional host RAM without necessarily improving this bottleneck.

## Where epoch time goes

| One-based epoch | Generation (s) | Training (s) | Mean inference batch | Last 10% of games: share of generation |
| --- | ---: | ---: | ---: | ---: |
| 19 | 7,279.9 | 52.4 | 83.9 | 23.4% |
| 20 | 6,176.3 | 54.6 | 96.0 | 21.4% |
| 21 | 5,766.7 | 57.1 | 105.0 | 18.6% |
| 22 | 5,689.6 | 60.2 | 89.1 | 25.2% |
| 23 | 5,269.3 | 61.8 | 95.2 | 26.4% |

Tail thresholds use the first heartbeat reporting at least that many games, so
there is approximately one heartbeat interval of quantization. Epoch 23's final
100 games took about 23.2 minutes; its final ten games took another 9.1 minutes
within that tail. Game duration depends on the checkpoint and sampled trajectories.

Training was 1.16% of epoch 23's 5,345.6 seconds. Even a hypothetical 2× training
speedup would save only 30.9 seconds (~0.58% of the epoch). Fused Adam and prefetch
remain available to benchmark, but are not the first optimization for this run.

## Inference-path timings

Epoch 23, 1,018,860 batches and 96,998,608 requests:

| Instrumented interval | Average per batch | Total elapsed in interval |
| --- | ---: | ---: |
| Input stack, conversion and device transfer | 0.295 ms | 300.3 s |
| `forward_t` submission | 2.292 ms | 2,334.8 s |
| Mask stacking | 0.110 ms | 111.6 s |
| Mask device-transfer submission | 0.027 ms | 27.3 s |
| Policy postprocessing submission | 0.070 ms | 71.2 s |
| Output copies / synchronization | 0.492 ms | 501.1 s |
| Entire executor service interval | 3.289 ms | 3,351.1 s |

The service interval covers 63.6% of generation wall time (63–65% over these five
epochs). Its remainder includes request/response handling outside the timer,
search work, collection and scheduling waits; it is **not** a measured MCTS-only
percentage. `forward_t` time includes host/libtorch/driver work and possible
synchronization. It is not GPU kernel execution time. GPU work overlaps other
intervals, and request-side timings overlap across games.

The oldest-request queue wait averages 10.76 ms. That is time already-submitted
requests spend waiting, not time spent deliberately filling a batch. Only 1.39%
of epoch 23's dispatches were deadline-triggered. Shortening the 1 ms deadline is
therefore not an attractive primary optimization.

Per request, position encoding took 8.74 µs, legal-mask construction 7.30 µs,
and policy decoding 20.37 µs. Summed across requests these intervals account for
847, 708 and 1,976 seconds respectively, **overlapping** other games/executor work.
They cannot be added to the service total to estimate a serial speedup.

## Recommended optimization order

1. **Benchmark CUDA-graph inference, especially for small tail batches.** The
   repeated eager forward is the largest measured executor interval. Capture a
   fixed-weight inference path with stable input/output buffers, starting with
   B128, then a small measured set of tail shapes. Rebuild captures when weights
   change. Keep eager execution as the reference and fallback. Compare complete
   batch latency and output parity, not GPU utilization alone. Graphs are a
   hypothesis here: without a CUDA timeline we have not isolated launch overhead.
   [PyTorch's CUDA graph documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
   describes reducing host submission overhead and the static-memory/shape
   requirements. Earlier batch-padding experiments did not test graph replay.

2. **Reduce per-request tensor construction and copying.** In
   `src/gomoku/codec.rs`, decoding makes a temporary host vector from a tensor and
   then a second vector in legal-move order; encoding/masks each build separate
   small tensors. The executor stacks them and splits outputs into tensor views
   per reply. A contiguous batch input/mask buffer, bulk CPU output transfer,
   and lightweight reply slices could remove allocations and FFI calls. Preserve
   symmetry reversal, move order, normalization checks and ownership across
   asynchronous requests. This targets measured costs, but its end-to-end benefit
   still needs an A/B benchmark because workers overlap.

3. **Compact the search-tree representation and measure move-boundary work.**
   In the sampled diagnostics, every allocated child occupies 48 bytes; there are
   318.7 allocated child entries per expanded node. Most child entries therefore
   carry storage for nodes that have never been expanded. Live controller counts
   imply approximately 57–61 GiB in child arrays alone at reported move boundaries.
   Those snapshots lag in-progress searches and do not include allocator overhead,
   replay buffers, or all process memory. Separate compact edge statistics from
   lazily allocated expanded-node state; an arena/index layout is worth testing.
   This is a well-supported memory opportunity, not a measured CPU speedup.

   `MonteCarloTree::advance` traverses every child of the retained subtree to
   rebuild telemetry, including unexpanded children. The sample's median retained
   count is 1.56 million nodes, maximum 9.87 million. Track expanded-child links
   or restructure bookkeeping so unexpanded siblings need not be visited one by
   one. Recursive destruction of discarded branches and PUCT child scans are
   other plausible CPU costs, but we have no stack samples to rank them.

4. **Cache the board-border contribution during fixed-weight inference.**
   `GomokuKataNet::forward_t` creates the same all-ones plane and convolves it on
   every batch. Its output depends on board size, device/dtype and the current
   weights, not on the positions. Compute it once per fixed-weight inference
   executor, invalidating after training/model replacement. This removes known
   redundant work, although it is only one small convolution and its individual
   cost was not measured. Keep the differentiable computation during training.

The long tail strengthens the case for reducing small-batch inference latency.
Overlapping an independent job only during the tail could improve overall GPU
throughput, but memory admission must use current measurements. This process now
uses ~96.5 GiB RSS; the earlier ~77 GiB overlap estimate is no longer representative.
Starting the next self-play epoch early would change the model/data schedule and
is not a transparent performance optimization.

Recent symmetry-counter changes have not been deployed in this executable. They
remove work, but a single counter increment per request is unlikely to outrank
these measured microsecond-scale tensor costs; no counter-specific speedup was
measured here.

## Evidence and reproduction

Raw evidence and collector: `runs/self-play-profile-20260915/`, mirrored on the pod
at `/workspace/alz-self-play-profile-20260915/`. `profile.tar.gz` SHA-256:
`c8454585096addb77fbb88ae0473b25dee5a9ddb783fafbb62e8f000c0e776eb`.
It contains 91 one-second process samples, 91 NVIDIA readings, five complete epoch
metric files, stage/progress records, and the deterministic tree sample.
`analyze.py` beside the local archive generates `summary.json` using only Python's
standard library. No training settings or active job state were changed.

A true CPU hot-function profile requires a pod permitting performance sampling
(or a separately launched representative process with an appropriate profiler).
A GPU timeline needs a CUDA profiler on a representative run. Neither was silently
substituted with estimates in the measurements above.
