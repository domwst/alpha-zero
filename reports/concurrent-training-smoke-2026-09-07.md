# Concurrent training smoke test — 2026-09-07

Running two independent copies of the 16-block × 32-channel trainer on the
RunPod L40 produced **1.997× aggregate training throughput**, with negligible
per-worker slowdown. The disposable copy completed two epochs and exited
successfully. The production trainer continued throughout.

| Measurement | Solo trainer | Main trainer during full overlap | Disposable trainer |
| --- | ---: | ---: | ---: |
| Complete epochs used | 2 | 1 | 2 |
| Training seconds per epoch | 448.33 | 449.07 | 448.84 |
| Augmented positions/second | 6,283.04 | 6,272.74 | 6,275.92 |

Concurrent rates sum to **12,548.66 positions/second**. The solo epochs took
448.697 and 447.973 seconds; the fully overlapping main epoch took 449.071
seconds; the probe epochs took 451.471 and 446.216 seconds. Every epoch
processed 2,816,904 augmented training positions in 11,004 batches.

| GPU measurement during training intervals | Solo | Concurrent |
| --- | ---: | ---: |
| Mean utilization | 38.0% | 97.8% |
| Peak allocated memory | 14,817 MiB | 29,630 MiB |
| Mean power | 186.3 W | 281.6 W |
| Mean temperature | 65.8°C | 74.2°C |
| Peak temperature | 70°C | 78°C |

These GPU summaries use 897 one-second samples per condition, selected from
the original capacity telemetry within the measured training intervals.
After the probe exited, a spot check returned to 36% utilization and
14,817 MiB allocated, with the original trainer still running.

## End-to-end cross-check

The probe process ran for **998.34 seconds (16 minutes 38 seconds)**, including
replay loading, cache construction, initial validation, two training epochs,
validation, checkpoint writes, and up to five seconds of supervisor polling.
During that lifetime, both workers each started and completed two full
training epochs: **11,267,616 training positions** in total.

Counting only those completed epochs gives **11,286.39 positions/second**,
or **1.796× the solo training-only rate**, despite including startup and
non-training overhead in the concurrent denominator. This deliberately
discards the main worker's partially completed next epoch. It is a conservative
cross-check rather than an identical-overhead sequential comparison.

## Controls and interpretation

Both processes used the same validated executable, 16-block × 32-channel
architecture, two original global blocks, GELU, deep value head, replay
checkpoints 60–69, deduplicated dataset and split, seed 20260906, batch size
256, standard Adam, LR 0.001, weight decay 0.0001, and device replay cache.
Their saved replay configurations match exactly. MPS configuration and the
production queue were unchanged. Probe checkpoints are outside the
architecture-comparison queue.

The primary throughput estimate sums native training rates measured over
complete epochs. The main worker's first partly overlapping epoch is excluded;
only its following epoch, entirely inside the probe's training window, is
used. This is a short smoke test with two identical architectures, not a
repeated benchmark of the depth/width pair or proof of scaling to three
trainers. Losses remained finite; matching settings do not imply bitwise
identical GPU training trajectories.

The result supports running **two independent trainers concurrently** on
this pod, subject to the memory requirements of the chosen models. It does
not suggest that one individual trainer will finish twice as fast.

## Reproduction and audit

Protocol and runner: [protocol](../docs/concurrent-training-smoke.md) and
[script](../scripts/concurrent_training_smoke.py).
Detailed metrics, commands, hashes, timings, and checkpoint descriptors:
[JSON report](concurrent-training-smoke-2026-09-07.json).

Pod output:
`/workspace/alpha-zero-followups/runs/value-heads-20260906/capacity-concurrency-smoke-20260907`.

The probe launched at 16:53:54 UTC, began training at 16:55:04 UTC, and was
confirmed exited at 17:10:32 UTC. Its two standalone reference epochs preceded
the launch. No model or replay files were downloaded for this report.

