# Board-mask replay experiment

A fresh `kata-gelu-boardmask-value64x2-v1` model trains for 20 passes on the
original deduplicated checkpoint-60–69 replay pool. The comparison reuses
`recipe-gamma-one-s1`, selected pass 11. Both use BN gamma=1, seed 20260908,
split seed 20260906, batch 256, constant LR 0.001, weight decay 0.0001,
10% held-out validation, standard Adam and GPU replay caches. Shared initial
tensor hashes and dataset/split counts must match the archived baseline before
training proceeds. The new model is selected by minimum validation value MSE +
policy loss, with earlier passes breaking ties.

The new model adds 288 parameters. Input remains two stone planes; the model
constructs an all-ones mask internally and adds its zero-padded convolution to
the stone-input convolution. This is equivalent to a three-plane convolution,
and keeps redundant ones out of the GPU replay cache. Existing checkpoint
architectures are unchanged.

The auxiliary supervisor runs training first, then the strength comparison at
1,000 games, 4,000 simulations, temperature 0.7, P100/B64 and match seed 20260910.
The baseline is first checkpoint and board-mask model second, with balanced
seats. It runs alongside P500 self-play. Every second the supervisor checks
host working memory and GPU usage; above 88% of the host limit or 22,000 MiB
GPU memory it stops only its own process group and marks the experiment paused.
It does not lower P100, resume automatically, or stop primary self-play.

Remote root: `/workspace/alpha-zero-battles/runs/board-mask-20260910`.
Binary: `validated/alz-boardmask-20260910` (separate from production self-play).
Supervisor: `scripts/archive/run_board_mask_experiment.py`.
Logs: `tmux attach -t alz-boardmask`, with a live `logs` window.
The read-only dashboard lists both stages, losses, results, and logs using
`scripts/board_mask_dashboard.py`.

CPU and CUDA checks cover baseline initialization parity, zero-mask output
parity, finite optimizer gradients, save/reload parity, mask edge/corner counts,
and equivalence to a concatenated three-channel convolution. The watchdog tests
verify thresholds and cleanup of the owned subprocess group on memory pressure.
CUDA receipt and experiment plan pin the executable SHA-256.

## Comparison handoff after self-play epoch 20

The initial P100/B64 attempt stopped at the 88% host-memory guard after 3 games;
the kernel reported no OOM kill. Its log remains in the original experiment
root and its games are excluded from the replacement match.

Self-play was stopped only after epoch 20's checkpoint, rendering, statistics,
and completion log were saved. Both self-play jobs have automatic launch disabled.
`scripts/archive/run_board_mask_comparisons.py` owns the new queue under
`comparison-handoff/`, with these matches at P500/B128, S4000, temperature 0.7,
1,000 games each:

1. Replay baseline (BN gamma=1, seed 1, selected pass 11) versus board mask
   (selected pass 14).
2. The same replay baseline versus the self-play epoch-20 checkpoint, pinned
   by SHA-256 after the epoch-boundary receipt.

The second match starts after 600 games complete in the first. If overlap hits
the memory guard, only the second match is stopped and its retry waits for the
first to finish. Each attempt has a separate log and partial report. A single
match exhausting headroom is paused for intervention. Self-play remains paused
after both matches; this handoff does not automatically restart it.

The dashboard follows the handoff queue, uses the current attempt's log, and
keeps the old interrupted attempt's counts out of the new comparison. tmux
windows `comparisons` and `comparison-logs` show the supervisor and match logs.
