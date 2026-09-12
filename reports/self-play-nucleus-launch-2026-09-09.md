# Fresh self-play comparison — September 9, 2026

Two fresh, matched-seed self-play runs were launched on the existing RTX 3090
pod, `1amocr9mqhg10y`. The primary uses top-p 0.95; the control uses top-p 1.0.
Both target 100 epochs of 1,000 games, 3,000 simulations per move, game
parallelism 500, inference batch ceiling 256, and partial-batch timeout 1 ms.
The optimizer batch is independently set to 256. The network is the selected
GELU / 10-block, 32-channel trunk with current global pooling and a
64→64→64→1 value head. BatchNorm gamma starts at one. Initialization seed is
20260909 for both; trajectories and subsequent search RNG streams can diverge.

Temperature is 1.0 for moves 1–6, decreasing linearly in consecutive move pairs
to 0.7 at moves 19–20, then staying there. Nucleus filtering runs after this
transform, preserves all exact ties at its cutoff, and only changes move
sampling. Training policy targets retain normalized MCTS visits. Root Dirichlet
noise and randomized inference symmetries retain their existing settings.

Replay capacity starts at 62,500 stored positions and grows by 3,750 positions
per epoch after epoch 15. Whole games are retained/evicted after incoming games
are shuffled; actual occupancy can be slightly below capacity. LR is
`0.001 × (62,500 / capacity)^1.1`, reaching approximately 0.000137 at epoch 100.
Each cycle trains one pass over the retained buffer with all eight symmetries.
Both runs use CPU replay caches and two prefetched batches. At final capacity,
each encoded cache is approximately 12.3 GiB, so two full device caches would
not fit on the 24 GiB GPU.

## Capacity measurement

A bounded five-minute fresh-network probe ran one S3000/P500/B256 worker for
150 seconds, then overlapped a second for 150 seconds. Throughput uses heartbeat
deltas after excluding each phase's first 60 seconds and last five seconds.
This measures early search, not complete-epoch time or ultimate peak memory.

| Workers | Aggregate evaluations/s | Peak effective host memory | Peak GPU memory | Mean GPU utilization |
| --- | ---: | ---: | ---: | ---: |
| One | 23,843 | 30.3 GiB | 768 MiB | 40.1% |
| Two | 30,987 | 56.3 GiB | 1,547 MiB | 71.7% |

Aggregate gain was 30.0%; each run becomes slower individually. The effective
container limit is 124,999,999,488 bytes (116.4 GiB), with a CPU quota equivalent
to 27.2 cores. `/proc/meminfo` reflects the larger host and is not the container
budget. No fresh B64/B128 scheduler sweep was performed; B256 is a demonstrated
feasible setting, not an established optimum. The probe does not establish a
100-epoch ETA; early random-network games can be much longer than mature games.
The existing GPU compute quote is $0.50/hour for the shared pod, plus storage.

## Validation and optimizer fallback

- Local and remote native library/binary tests passed (86 passed, three ignored
  platform/GPU tests in the regular suite).
- Explicit CUDA cache/prefetch and fused-Adam equivalence/checkpoint tests passed.
- Actual fused training completed two epochs and a resumed third with finite loss.
- However, repeated training on identical empty boards produced non-finite loss
  with fused Adam while standard Adam remained finite at the same benchmark
  settings. The precise numerical cause remains unresolved.
- Production therefore uses **standard Adam for both runs**. Its own three-epoch
  real self-play/training smoke, including resume and replay-linked LR, passed.
  The real-replay B256 benchmark measured 11,822 augmented samples/s. This is
  optimizer throughput on a small smoke replay, not whole-run throughput.
- The production trainer rejects non-finite losses before applying the associated
  optimizer update. The supervisor stops on worker failure and records the error.

## Monitoring and resource handling

On the pod, attach with `tmux attach -t alz-selfplay`. Windows are `nucleus`,
`control`, `resources`, and `supervisor`; the first two follow the real log files.
The local read-only dashboard remains at `http://127.0.0.1:8765/experiments`.
It includes self-play progress and training curves; no validation curve is
invented for the on-policy training runs, which have no held-out split.

Run root: `/workspace/alpha-zero-battles/runs/selfplay-nucleus`. The supervisor
records the exact commands, phase status, timestamps, and resource telemetry.
Its preflight receipt is bound to the production binary SHA-256. If combined
working memory exceeds 82% of the container limit, or GPU use exceeds 22,500 MiB,
it interrupts only the control and resumes it from its last completed checkpoint
after the primary finishes. An interrupted partial epoch is recollected. At 93%
host-memory use with only one worker, it stops rather than risk cgroup OOM.
These are headroom guards, not a promise that every allocation spike is caught.

The pod's persistent volume was expanded from 20 to 100 GB before launch. The
completed recipe-comparison results were downloaded and their archive checked
before the resize. Code, binaries, checkpoints, and results reside in `/workspace`.
Local deployment receipts and downloaded preflight measurements are under
`runs/selfplay-nucleus-20260909-deploy`.

### Control resumed at P400 (2026-09-09 16:53 UTC)

The combined workers reached the 82% host-memory guard at 16:35 UTC. The control
was stopped safely with two complete epochs saved; no cgroup OOM was recorded.
At the user's request, the control resumed at **S3000/P400/B256**, restoring snapshot
`00000001` and restarting self-play for epoch 3. Its interrupted partial epoch was
not retained. The primary remains at P500, with its original process and in-flight
epoch preserved. Both runs retain the original training recipe and 100-epoch target.

The replacement supervisor adopted the live primary using its PID, Linux start
time and command; it monitors both workers with the original memory thresholds.
If the combined footprint reaches the guard again, the control returns to the
serial queue. The P400 override persists for future control launches. Dashboard
notes show the per-run concurrency; the existing tmux `control` window follows
resumed output. Handoff records and the preceding memory incident are saved in
`runs/selfplay-nucleus-20260909-deploy/p400-resume/`.

Validation: five local self-play supervisor/dashboard tests, four pod supervisor
tests, frontend type-check/build, complete-snapshot restore log, fresh heartbeats
from both workers, primary process identity preserved, dashboard API reporting
both runs active and P400 for the control, and zero cgroup OOM events after resume.

### Both runs resumed at P400 (2026-09-09 17:23 UTC)

The P500 primary plus P400 control reached the memory guard again at 16:56 UTC.
At the user's request, the active primary was interrupted and **both** runs were
restarted with **S3000/P400/B256**. The primary restored snapshot `00000002`
(three complete epochs, restarting epoch 4); the control restored `00000001`
(two complete epochs, restarting epoch 3). The primary's unfinished epoch was
discarded. The memory guard, training recipe and 100-epoch targets are unchanged.

Both the shared plan default and individual job settings now specify P400.
The original guard incident, commands, process identities and restart receipt are
archived in `runs/selfplay-nucleus-20260909-deploy/both-p400-resume/`.
The existing tmux log windows and read-only dashboard show both resumed runs.
