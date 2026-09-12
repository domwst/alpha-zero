# Replay training and value-head follow-ups

These experiments follow the [ReLU/GELU experiment](activation-experiment.md).
They use X = 64 and preserve the trunk, pooling, policy head, Adam update rule, and
loss definitions. Dense value-head parameters, including biases:

| Head | Parameters | ReLU architecture | GELU architecture |
| --- | ---: | --- | --- |
| 64 → 10 → 1 | 661 | `kata-v1` | `kata-gelu-v1` |
| 64 → 64 → 1 | 4,225 | `kata-value64-v1` | `kata-gelu-value64-v1` |
| 64 → 64 → 64 → 1 | 8,385 | `kata-value64x2-v1` | `kata-gelu-value64x2-v1` |

Each hidden dense layer has the architecture's activation. The output retains
`tanh`. Existing architecture identities, tensor registration order and shapes,
initialization, and forward semantics are preserved. New layouts have separate
versioned identities for loading, inference, and training resume.

## Ordered queue

1. Wait for the current activation experiment's successful exit and complete
   match report. Validate its model identities, equal training budgets, dataset,
   300 games, 4,000 simulations, temperature 0.7, and balanced seats.
2. Match the fresh replay-trained ReLU final checkpoint against archived Kata
   self-play checkpoint **69** (the newest snapshot in the replay source run).
   Its model SHA-256 is
   `91b74e5b63c6b92055331f9acbff9c68b15e96e0f9b3188aa92dc04673e2243a`.
3. Select the activation with the higher score in the original ReLU/GELU match,
   counting each draw as half a win. An exact tie selects ReLU. Record the score
   and confidence interval; selection alone does not establish superiority.
4. Run native inference and training preflights for each new head, then train
   the two new models from scratch. Both use the selected activation throughout
   the network. Reuse the selected original final checkpoint as the current-head
   baseline, so only **two additional models** need training.
5. Run three matches: current vs wide, current vs deep, and wide vs deep.
6. Run the separately queued [pooling comparison](pooling-experiment.md), using
   the highest-scoring checkpoint from the complete round robin as the baseline.
   Only its KataGo-pooling counterpart trains afresh, with matched replay budget
   and Adam optimizer and a validated device replay cache.

The completed activation and checkpoint-69 matches have 300 games and up to
128 concurrent games. The current/wide and current/deep value-head matches
retain **600 games**, 300 per seat. The remaining wide/deep and pooling matches
use **1,000 games**, 500 per seat, with **up to 300 concurrent games**. All comparisons use
4,000 simulations per move, temperature 0.7 and inference batches up to 64 per model.
Match seeds are the training seed plus 1, 2, 3, and 4 in the above match order.
No additional self-play training is scheduled.

The September 6 match-budget update is recorded in `match-plan.json`. It changes
only the four future value-head/pooling matches; completed results and the
original queue/training configuration retain their recorded settings.
`run_match_plan.py` wraps the original pinned worker and overrides game count
and concurrency only inside those battle calls. `handoff_match_plan.py` waits
for the active training command to finish before replacing its idle controller.
The approved standard-trainer fallback remains in effect for both value heads.
Follow the transition in `match-plan-handoff.json`.

The September 7 update uses match-plan schema 2: historical and active matches
keep their recorded counts, while unstarted matches and the displayed future
budget use 1,000 games. `handoff_match_budget.py` waits for the active match to
publish its result and the controller to pause with no active child. It then
replaces only the idle controller and resumes the queue automatically; progress
is recorded in `budget-handoff.json`. Completed training is validated and reused
without rerunning its command or rewriting its timestamps. The pooling worker
uses the mean score rate against each opponent to rank the mixed-budget round
robin, so each opponent contributes equally.

Training settings and replay source paths are read from the completed activation
run and verified against the recorded digests. For this deployment that means
the same deduplicated checkpoints 60–69, trajectory-group holdout, seed
20260906, all eight symmetries, batch 256, Adam learning rate 0.001, weight decay
0.0001, and **20 passes per new model**. Final checkpoints are used after equal
data exposure. Models train sequentially on the pod's GPU.

The seed and minibatch ordering are shared. Different head shapes consume
different amounts of initialization randomness, so the new architectures do
not have identical initial weights for every common tensor. In particular,
the policy head is initialized after the value head. This is a single-seed
architecture comparison; close results need replication with more seeds.

The checkpoint-69 match measures how well this fixed replay training protocol
recovers the archived model's strength. The archived model has a longer training
history and different data exposure. The match does not establish equivalence
between replay training and continued self-play learning.

## Running and tracking

The [read-only experiment dashboard](experiment-dashboard.md) provides live
queue status, training curves, results, and logs through GET/HEAD endpoints.
It cannot change the queue. The deployed worker continues to check its existing
`control.json` before each follow-up command.

Build the follow-up binary in a separate repository directory while the original
experiment runs. Do not replace the original job's binary or sources.

```bash
cd /workspace/alpha-zero-followups
source scripts/runpod_env.sh
python3 scripts/archive/run_replay_followups.py \
  --activation-dir /workspace/alpha-zero/runs/relu-vs-gelu-20260906 \
  --require-success-file /workspace/alpha-zero/runs/relu-vs-gelu-20260906/exit-code.txt \
  --latest-checkpoint /workspace/alpha-zero-followups/reference/00000069 \
  --output-dir /workspace/alpha-zero-followups/runs/value-heads-20260906
```

Run in tmux. The `--require-success-file` gate is written by the original job's
launcher only after it exits. Nonzero exit or invalid results stop the queue.
The queue writes `status.json` at every phase transition, `selection.json` after
the activation result is available, per-stage logs, and `summary.json` as matches
finish. Training directories include per-epoch metrics and full checkpoints.

The RunPod deployment uses tmux session `replay-followups`, a `progress` window,
and combined log `/workspace/alz-followups.log`:

```bash
tmux attach -t replay-followups
# Or, from an existing tmux session:
tmux select-window -t replay-followups:progress
tail -f /workspace/alz-followups.log
cat /workspace/alpha-zero-followups/runs/value-heads-20260906/status.json
```

Rerunning the exact command resumes completed training epochs and validates and
skips completed matches. An interrupted match restarts all its games. A lock
prevents duplicate queue writers. The binary, script, source checkpoint, and
configuration are pinned; changes normally require a fresh output directory. The queue
does not stop the pod when finished.

## Optimized training backend upgrade

The September 6 update uses `--adam-backend fused --replay-cache device` for the
two pending value-head training runs. The prepared executable lives in
`/workspace/alpha-zero-optimized-20260906/target/release/alz`. Inference and matches
keep `/workspace/alpha-zero-followups/target/release/alz`; the current checkpoint
comparison continues with its existing process. The original baseline checkpoint
was trained with standard Adam. Both new heads use fused Adam; backend rounding
is consequently an additional difference from that baseline. No extra model
training or changes to data, seeds, passes, batch size, or loss weights are added.

The queue now accepts `--training-binary`, `--adam-backend`, `--replay-cache` and
`--prefetch-batches` independently of its match `--binary`. Training preflights
use the selected backend and replay cache, and completed training configurations
are checked against those settings as well as the original data and budget.

`scripts/archive/handoff_replay_training.py` performs the deployment at a command boundary.
It gates upcoming commands, waits for the current comparison to finish and the
old worker to have no running child, then runs the prebuilt CUDA correctness
tests, with a five-minute timeout. The approved deployment policy automatically
resumes the original validated worker if checks fail or time out. Both new heads
then use standard Adam and uncached replay batching. This fallback verifies the
original executable, script and queue configuration and requires that value-head
work has not started. The validation process group is terminated on timeout.
After successful checks, the handoff replaces the idle worker, archives launcher
status and telemetry, and resumes the optimized queue.
The dashboard remains read-only; this gate is managed by the deployment process.

The one-time `--upgrade-from-config-sha256` operation preserves the old and new
configuration in `training-backend-upgrade.json`. It permits separate training
backends only after the checkpoint comparison finishes and before value-head
preflights or training begin. Match settings, match binary and source identities
must remain identical. The replacement configuration is then pinned normally.

Track the handoff in the existing run directory through `training-handoff.json`,
`training-backend-validation.json`, its companion log, and `queue-config.json`. A
`fallback_to_previous_trainer` handoff state means the original worker resumed
without changing its pinned configuration. The
prepared plan records executable hashes and exact worker/validation commands.
See [training performance](training-performance.md) for cache memory requirements
and numerical validation details.

For a small CPU smoke test, use an activation run produced with two games and
eight simulations, and add `--device cpu --games 2 --simulations 8 --parallelism 2
--inference-batch-size 2`. Training epochs are inherited from that activation run.

After the optimized worker is installed, an explicit restart uses the same
output directory and both pinned executables:

```bash
cd /workspace/alpha-zero-optimized-20260906
source scripts/runpod_env.sh
python3 scripts/archive/run_replay_followups.py \
  --activation-dir /workspace/alpha-zero/runs/relu-vs-gelu-20260906 \
  --require-success-file /workspace/alpha-zero/runs/relu-vs-gelu-20260906/exit-code.txt \
  --latest-checkpoint /workspace/alpha-zero-followups/reference/00000069 \
  --output-dir /workspace/alpha-zero-followups/runs/value-heads-20260906 \
  --binary /workspace/alpha-zero-followups/target/release/alz \
  --training-binary /workspace/alpha-zero-optimized-20260906/target/release/alz \
  --adam-backend fused --replay-cache device
```

Do not start this while the current worker or handoff process is running. The
queue lock prevents duplicate writers. The handoff runs in the `training-upgrade`
window of the existing `replay-followups` tmux session and writes its combined log
to `/workspace/alz-followups-optimized.log`.
