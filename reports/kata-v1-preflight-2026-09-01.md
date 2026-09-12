# Kata v1 training preflight — 2026-09-01

## Decision

The Kata v1 implementation and L40S deployment are ready for a new, independent
50-epoch run. Use:

- architecture: `kata-v1`;
- inference batch size: 256;
- games parallelism: 500;
- partial-batch timeout: 1,000 us;
- training batch size: 256;
- games per epoch: 700;
- simulations per move: 2,000;
- Adam learning rate: 1e-3;
- Adam L2 weight decay: 1e-4.

The new run must not share a checkpoint directory with `legacy_resnet_v1`.

## Deployment identity and host

- Source commit: `60bce9dbea51c1c2f7bc7db4b7700538681d0c58`.
- Release binary SHA-256:
  `e3f7b9a73c2ee2b9163c8d5bf912b3099b3b6d51c826b2ef732a244c50ff7695`.
- GPU: NVIDIA L40S, 46,068 MiB, 350 W.
- Host: 16 logical CPUs, 62 GiB RAM, no swap.
- Disk at preflight: 188 GiB free.
- Uploaded legacy run: 118 format-v2 snapshots through epoch 117.

The remote Git metadata still describes the old checkout onto which releases
were deployed, so manifests now record the exact `alz` binary hash as the
authoritative executable identity.

## Correctness gates

- All 53 Rust unit/integration tests passed locally at the deployed commit.
- Kata CUDA forward inference passed.
- Kata CUDA backward and Adam update passed at training batch 256.
- A real four-game CUDA epoch saved model, optimizer and replay data with
  `architecture: kata_v1`; a new process restored it successfully.
- The saved model SHA-256 matched its format-v2 metadata.
- The same binary loaded the uploaded `legacy_resnet_v1` epoch-117 snapshot on
  CUDA and completed a policy-only game.
- Cross-architecture checkpoint protection remains enabled.

## Raw GPU batch sweeps

Each point was measured three times. Values below are medians.

| Inference batch | Examples/s | Iteration ms |
|---:|---:|---:|
| 64 | 41,957 | 1.525 |
| 128 | 79,476 | 1.611 |
| **256** | **122,684** | **2.087** |
| 512 | 116,687 | 4.388 |
| 1024 | 85,162 | 12.024 |

Batch 256 is 54.4% faster than batch 128 in isolated inference. Batch 512 is
already 4.9% slower than 256.

| Training batch | Samples/s | Iteration ms |
|---:|---:|---:|
| 128 | 13,312 | 9.615 |
| **256** | **26,549** | **9.642** |
| 512 | 26,171 | 19.563 |
| 1024 | 21,054 | 48.637 |

Batch 256 remains the training optimum. The complete training sweep reached a
maximum of 19,397 MiB GPU memory at batch 8192; the selected batch has ample
headroom.

## End-to-end self-play

The focused shallow interaction sweep used 500 complete games and 16
simulations per move.

| Inference / parallelism / timeout | Evals/s | Moves/s | Average batch | Request latency us |
|---|---:|---:|---:|---:|
| 128 / 160 / 100 ms | 30,588 | 1,925 | 101.1 | 4,161 |
| 192 / 384 / 1 ms | 34,296 | 2,158 | 131.0 | 7,712 |
| 256 / 256 / 1 ms | 35,063 | 2,206 | 159.7 | 5,427 |
| 256 / 384 / 1 ms | 36,382 | 2,289 | 169.9 | 7,499 |
| **256 / 500 / 1 ms** | **38,491** | **2,422** | **198.2** | **9,651** |

The selected candidate was 25.8% faster than the old baseline in this matched
interaction sweep. The separately measured 100 ms candidate was within noise
at full concurrency.

## Production-depth probe

Both configurations ran for four minutes with 500 requested games and 2,000
simulations per move. The final common 210-second heartbeat is used below.

| Configuration | Evals/s | Minimum available RAM | Average GPU utilization | Max GPU memory |
|---|---:|---:|---:|---:|
| 128 / 160 / 100 ms | 33,567 | 53.2 GiB | 29.1% | 877 MiB |
| **256 / 500 / 100 ms** | **39,204** | **38.2 GiB** | 28.2% | 955 MiB |

The candidate retained a 16.8% throughput gain at production search depth.
Host RAM remains the binding resource, but 38.2 GiB was still available after
four minutes. No checkpoint was published by either interrupted probe.

## Timeout tail test

With only 32 games, batch 256, and 256 simulations per move:

- 1 ms completed in 97.55 seconds at 6,525 eval/s;
- 100 ms did not finish within the 180-second safety limit.

The production default is therefore 1 ms. Full-concurrency throughput is
unchanged while the end-of-epoch tail avoids long partial-batch stalls.

## Launch guardrails

- Use a fresh run root such as `runs/kata-v1-50e-20260901`.
- Keep `ALZ_LOG_ANSI=always` in tmux and retain one-second GPU/host telemetry.
- Watch the first epoch closely because an untrained model produces games near
  100 moves, substantially longer than the mature legacy model.
- Stop and reassess if available host RAM falls below 12 GiB or throughput
  remains below 35,000 eval/s for two consecutive heartbeats.
- Verify the epoch-0 metadata says `kata_v1`, then perform a no-op restore before
  leaving the run unattended.

Raw results are archived under
`remote-results/kata-v1-preflight-20260901` locally and
`runs/kata-v1-preflight-20260901` on the VM.
