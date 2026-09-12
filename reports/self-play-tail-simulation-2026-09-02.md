# Self-play tail simulation

Date: 2026-09-02

## Conclusion

The recent production data supports the heavy-tail explanation. Increasing the
number of games in an epoch from 700 to 1,500 is predicted to improve generated
games per wall-clock hour by about 31%, whereas increasing concurrency from 500
to 700 without increasing the epoch size has no material scheduling benefit in
the fixed-throughput model.

The recommended next controlled experiment is therefore:

- 1,500 games per epoch;
- 500 games of parallelism;
- 4,000 replay games;
- unchanged inference batch 256, 1 ms timeout, training batch 256, and 2,000
  simulations per move.

Keeping parallelism at 500 isolates the epoch-size effect. A later P700 trial
can still test whether extra concurrency improves real inference batch fill, an
effect intentionally absent from this simulator.

## Inputs

The length sample is checkpoint 29's final 1,400 replay entries: all games from
epochs 28 and 29. The first 400 entries were excluded because they are only the
last-completed games from epoch 27. Since the replay deque is ordered by game
completion, that partial epoch is biased toward long games.

| Length statistic | Plies |
|---|---:|
| Games | 1,400 |
| Mean | 25.93 |
| Median | 23 |
| p80 | 35 |
| p90 | 51 |
| p95 | 64 |
| p99 | 91 |
| Maximum | 154 |

The longest 20% of games contain 42.4% of all plies. Eight of 1,400 games ran
past 100 plies. The two source epochs are consistent: their mean lengths were
26.44 and 25.41, both had p80 35 and p99 91, and their maxima were 154 and 132.

Observed completion curves came from production epochs 20 through 29, all run
as G700/P500. Their median makespan was 1,818.2 seconds. The median time to 80%
completion was 934.0 seconds, leaving 908.3 seconds, or 49.1% of the epoch, for
the final 20%.

Raw inputs are retained locally under
`remote-results/runpod-rtx3090-20260901/tail-simulation/`.

## Model

The simulator maintains up to the configured concurrency and immediately starts
a queued game when another finishes. All active games receive equal progress:

```text
per-game progress rate = min(1, 256 / active games)
```

Thus 10 or 256 active games each progress at rate 1, while 512 active games
each progress at rate 0.5. This is a processor-sharing model with global
capacity 256 and a per-game rate cap of 1.

Raw ply count does not validate as a direct compute estimate: it predicts a
60.2% median post-80% tail, versus 49.1% observed. Late moves are cheaper than
early moves in NN-evaluation work, so the final calibrated model uses
`plies^0.78` as effective game work. The exponent was selected by a small grid
search against the normalized completion curve. One wall-clock scale factor was
then fitted by matching the observed and simulated baseline median makespan.

This calibration reproduces the baseline curve sufficiently well for a
scheduling estimate:

| Milestone | Observed median | Simulated median | Error |
|---|---:|---:|---:|
| 50% complete | 685.8 s | 724.2 s | +5.6% |
| 80% complete | 934.0 s | 941.4 s | +0.8% |
| 90% complete | 1,096.9 s | 1,072.0 s | -2.3% |
| 95% complete | 1,225.2 s | 1,188.8 s | -3.0% |
| 99% complete | 1,541.8 s | 1,437.2 s | -6.8% |
| 100% complete | 1,818.2 s | 1,818.2 s | fitted |
| Post-80% tail fraction | 49.1% | 48.4% | -1.6% relative |

The simulated baseline p10–p90 makespan interval is 1,636–2,055 seconds,
compared with 1,665–2,101 seconds across the ten observed epochs.

## Scenario results

Each scenario used 10,000 bootstrap trials from the 1,400 empirical lengths.
Time ranges are p10–p90. Throughput speedup is relative to G700/P500.

| Scenario | Median self-play | p10–p90 | Post-80% tail | Median games/hour | Throughput change |
|---|---:|---:|---:|---:|---:|
| G700/P500 | 1,818 s (30.3 min) | 1,636–2,055 s | 48.4% | 1,386 | baseline |
| G700/P700 | 1,799 s (30.0 min) | 1,574–1,978 s | 48.7% | 1,401 | approximately 0% |
| G1500/P500 | 2,943 s (49.1 min) | 2,772–3,208 s | 34.6% | 1,835 | +30.7% |
| G1500/P700 | 2,934 s (48.9 min) | 2,746–3,178 s | 33.7% | 1,840 | +31.5% |

The P700 result is statistically indistinguishable from P500 in this model.
The paired trial speedup has median 0.992 and p10–p90 0.954–1.073 for G700,
and P700 adds less than one percentage point to the G1500 central estimate.
Starting more games earlier changes which tail games receive service first but
does not create more global work capacity.

By contrast, G1500 keeps the global engine busy for longer before it drains.
Predicted effective capacity utilization rises from a median 55.3% to 73.4% at
P500. The larger epoch is expected to finish self-play in about 49 minutes,
rather than the approximately 65 minutes implied by linear scaling from G700.

Using raw plies instead of the calibrated work proxy gives a conservative
sensitivity result of +43% throughput for G1500/P500. Because that model fails
the baseline tail validation (60% predicted versus 49% observed), 31% is the
better central estimate.

## Training and replay consequences

The current implementation trains one complete, eightfold-augmented pass over
the replay after each self-play epoch. The replay-to-new-game ratio is nearly
unchanged:

```text
current:  1800 / 700  = 2.57
proposed: 4000 / 1500 = 2.67
```

At the measured 25.93 positions per game, a 4,000-game replay contains about
103,700 positions or 830,000 augmented samples. Recent training throughput is
about 9,900 samples/s, so the optimizer pass should take approximately 84–90
seconds. The resulting complete epoch should take roughly 50.5 minutes.

Replay memory is small relative to active MCTS trees. Raising total games while
leaving parallelism at 500 does not materially raise peak tree memory. The main
algorithmic tradeoff is network freshness: self-play will use one checkpoint
for about 50 minutes and 1,500 games rather than about 31 minutes and 700 games.
Optimizer work per newly generated game and the replay horizon remain almost
unchanged, but updates arrive in larger, less frequent bursts.

## Reproduction

Export replay lengths:

```bash
./run.sh cargo run --example export_replay_lengths -- \
  CHECKPOINT/replay.bin.zst replay-lengths.json
```

Run the calibrated simulation:

```bash
python3 scripts/simulate_self_play_tail.py \
  --lengths replay-lengths.json \
  --last-games 1400 \
  --observed-log stdout.log \
  --observed-epochs 20-29 \
  --length-exponent 0.78 \
  --trials 10000 \
  --output tail-simulation.json
```

The complete generated result is
`reports/self-play-tail-simulation-2026-09-02.json`.

## Limitations

- The effective-work exponent is empirical and was fitted on the same ten
  epochs used for validation. It is a scheduling proxy, not a claim about MCTS
  complexity.
- Only epochs 28 and 29 supply the bootstrap length distribution, although
  their distributions are very similar and the ten observed mean lengths span
  only 24.54–28.09.
- The model holds global throughput fixed above 256 active games. It does not
  predict possible batch-fill, CPU, memory-bandwidth, or GPU-efficiency changes
  from P500 to P700.
- Future checkpoints may change the game-length distribution. The simulation
  should be rerun periodically from the most recent two complete epochs.
