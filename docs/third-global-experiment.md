# Third global-pooling block — 2026-09-08

Train `kata-gelu-b16c32g3-value64x2-v1` from scratch, then compare it with two
existing validation-selected checkpoints. The trunk has 16 blocks and 32 channels;
original mean/max global-pooling blocks occupy zero-based indices 3, 7 and 11.
Block 11 replaces an ordinary residual block. GELU, the 64 → 64 → 64 → 1
value MLP, and both head widths match the previous capacity experiment.
Existing architecture identities and the production default retain their layouts.

The two reused opponents are:

| Opponent | Selected pass | Model SHA-256 |
| --- | ---: | --- |
| Current 10-block trunk, two global blocks | 18 | `6c9d758cfba6d1ed86a06d176efeb89f6584fb76425898f3f6a05a941dde2e8f` |
| 16-block trunk, two global blocks | 15 | `56f46d6687fa00d14b0e37f55f6531562738d9a5899f7719607fa3973b683e39` |

Training uses the same deduplicated checkpoint 60–69 replays, frozen split,
eight symmetries, seed 20260906, batch 256, standard Adam, LR 0.001, weight decay
0.0001, and 20 complete passes. The split has 15,750 training games and 1,750
validation games. Dataset SHA-256:
`2a391f1f93a588877e06dc550d2eaa8f953d1c198f8d6988053594a32a46f5bc`.
The new checkpoint minimizes validation policy loss + value MSE; ties favor
earlier passes. Selection must finish before either comparison starts.

Both matches play 1,000 games (500 per seat), 4,000 simulations per move,
temperature 0.7, concurrency 300, inference batch 64 and 1 ms batch timeout.
The current baseline is checkpoint A in the first match (seed 20260936);
the existing 16-block model is A in the second (seed 20260937).
The new three-pooling model is B in both. The second match starts once the first
has completed 600 games, or finished; at most two matches run together.

The queue pins code, binary, data and opponent identities. Architecture CUDA
checks must pass before training. A replay-cache correctness failure falls back
to uncached standard Adam. This preserves the previously authorized fallback;
an architecture failure stops the run instead of training an incorrect model.

Implementation: `scripts/archive/run_third_global_experiments.py`, launched by
`scripts/archive/launch_third_global_runpod.sh`. The dashboard displays training, both
matches, logs, results and network identities. HTTP remains read-only on loopback.

Deployment:

- Isolated source: `/workspace/alpha-zero-global3-20260908`.
- Run: `/workspace/alpha-zero-followups/runs/value-heads-20260906/third-global`.
- tmux session: `alz-global3-20260908`.
- Persistent executable: `validated/alz` inside the isolated source directory.
- Compilation cache: `/tmp/alz-global3-target`, linked as `target`; this uses
  container storage to preserve volume space for checkpoints. If that temporary
  cache disappears, rerunning the launcher rebuilds it. The executable and
  experiment data remain under `/workspace`.
- Local transfer receipt: `runs/third-global-20260908/bundle.json`.

Local validation passed 75 Rust tests (three CUDA tests deferred to the pod),
39 queue, scheduling, statistics and read-only dashboard tests, and the dashboard production build. The new
model participates in topology, finite-gradient, CPU/CUDA comparison,
serialization, and model/optimizer checkpoint-restore checks.

The pod's release tests and CPU/CUDA checks passed, including the new architecture.
Device-cache correctness passed without fallback. Training started at
2026-09-08 04:15:07 UTC. The 256-position training preflight measured about
6,528 examples/s over ten measured batches; this is a short preflight, not a
full-pass throughput estimate. The previous capacity binary remained unchanged.
