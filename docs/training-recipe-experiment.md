# Normalization initialization and learning-rate recipes

Approved on 2026-09-08: six fresh training runs, at most two concurrent trainers
on the existing L40. Finish training before migrating to a cheaper host for
comparisons. The runner does not start battles or change pod lifecycle.

| Recipe | BatchNorm gamma at initialization | LR by pass | Seeds |
| --- | --- | --- | --- |
| Control | Existing uniform [0, 1] | Constant 0.001 | 20260908, 20260909 |
| Cosine | Existing uniform [0, 1] | 0.001 → 0.0001 | Same pair |
| Gamma one | All ones | Constant 0.001 | Same pair |

All use `kata-gelu-value64x2-v1`: 10 blocks, 32 channels, two original global
pooling blocks and the 64→64→64→1 value head. Use replay checkpoints 60–69,
the same deduplicated frozen pool, split seed 20260906, validation fraction
0.1, batch 256, all eight symmetries, 20 passes, standard Adam, weight decay
0.0001 and device cache. Model initialization and batch order vary with the
replication seed. The trajectory split stays fixed across all six models.

For zero-based pass e, cosine LR is
`0.0001 + (0.001 - 0.0001) * (1 + cos(pi * e / 19)) / 2`.
LR is constant within a pass, includes both endpoints, is recorded in pass
metrics and is reapplied after optimizer restore. The cosine horizon is part
of the immutable run config; changing the target pass count on resume is
rejected. Old constant-LR configs remain readable.

Gamma is overwritten only in newly constructed models, after normal default
construction and before optimizer creation. This preserves random-number
consumption and every other tensor. Existing checkpoint gamma is never
modified: resume restores saved learned parameters. Initial tensor fingerprints
are checked on the GPU before training: control/cosine must be identical and
gamma-one must differ in exactly the BatchNorm gamma tensors.

Every run gets an independent directory, optimizer and checkpoints. The runner
validates replay hashes, exact recipe settings, split counts, CPU tests, CUDA
architecture/cache checks and frozen initialization manifests. A CUDA or pairing
failure blocks training. A failed trainer blocks subsequent starts while its
already active peer finishes. Completed checkpoints can resume with the same
recipe; the supervisor uses a lock to prevent a duplicate queue.

The selected checkpoint minimizes held-out policy cross-entropy plus value MSE
among all 20 passes, with ties preferring the earlier pass. Preserve selected
and final checkpoints and full metrics. Validation selects candidates; it does
not establish playing strength or superiority across training seeds.

The read-only local dashboard displays all six jobs, losses, timings and ETAs.
RunPod output: `/workspace/alpha-zero-followups/runs/value-heads-20260906/training-recipes`.
Runner: `scripts/archive/run_training_recipe_experiments.py`.
Launcher: `scripts/archive/launch_training_recipes_runpod.sh`.

Frozen-checkpoint BN running-statistics diagnostics remain separate work:
comparable train/validation inference losses, batch versus running statistics,
training-only recalibration and scale/saturation inspection. This queue does
not change normalization momentum, recalibrate these models, or combine gamma
one with cosine decay. Those changes require results from the separate tests.

## Selected-checkpoint comparisons (2026-09-08)

Run four primary matches: control versus cosine and control versus gamma one,
separately for each training seed. Each match plays 1,000 games, balanced between
seats, with 4,000 simulations, temperature 0.7, game concurrency 300, inference
batch 64 and a 1 ms batch timeout. Battle seeds are 20260940–20260943 in that
order (seed 1 cosine, seed 1 gamma, seed 2 cosine, seed 2 gamma). Start the next
match at 60% completion, with at most two active matches.

| Training run | Selected pass |
| --- | --- |
| Constant LR, seed 1 | 11 |
| Cosine LR, seed 1 | 15 |
| Gamma one, seed 1 | 11 |
| Constant LR, seed 2 | 10 |
| Cosine LR, seed 2 | 16 |
| Gamma one, seed 2 | 19 |

`scripts/archive/run_recipe_battles.py` checks selected model hashes, the validated
binary hash, the immutable plan and the GPU preflight receipt before running.
It validates completed reports, including checkpoint identities, game counts,
seat balance, seeds and search settings. A failed match prevents further starts
while an already active peer finishes. Interrupted matches restart in full.

`examples/check_checkpoint_cuda.rs` compares both network heads from all six selected
models on CPU and CUDA at batches 1, 4 and 64, with TF32 disabled for the
comparison only. The actual match uses the existing runtime defaults.
`scripts/archive/setup_recipe_battles_runpod.sh` also benchmarks CUDA inference.

Only model weights, metadata, source/runtime files and saved dashboard history
are deployed. The complete original replay/training archive remains local.
The new worker lives in `/workspace/alpha-zero-battles/runs/recipe-battles`.
The loopback dashboard merges its four live jobs with the 28 archived jobs;
its HTTP and SSH interfaces remain read-only.
