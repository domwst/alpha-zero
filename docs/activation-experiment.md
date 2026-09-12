# ReLU versus GELU on fixed Gomoku replays

The experiment trains fresh `kata-v1` and `kata-gelu-v1` networks on the same
data, then plays 300 games with 4,000 simulations per move and temperature 0.7.
Seats alternate, giving each checkpoint 150 games as the first player.

`kata-gelu-v1` replaces every hidden ReLU in Kata v1 with exact GELU
(`Tensor::gelu("none")`). Tensor registration, widths, global blocks, BatchNorm
initialization, optimizer, loss, and the final value `tanh` stay identical.
The existing `kata_v1` checkpoint meaning is preserved.

## Dataset and training protocol

Use the latest ten archived Kata snapshots, epochs 60–69, from:

```text
deployments/runpod-rtx3090-final-20260903/repository/runs/kata-v1-50e-20260901/checkpoints
```

Local validation found 40,000 stored game entries, of which 22,500 are repeated
across buffers. The resulting dataset has **17,500 unique games and 391,212
positions**. With the default seed, training has 15,750 games / 352,113 positions
and validation has 1,750 games / 39,099 positions. The combined dataset SHA-256 is
`2a391f1f93a588877e06dc550d2eaa8f953d1c198f8d6988053594a32a46f5bc`.

Each snapshot contains a rolling replay buffer. The loader validates replay
schemas, counts, legal policy targets, and value targets. It removes duplicate
games by hashing their complete contents, including targets. It sorts unique
games by content, so changing source paths, source ordering, or including a
buffer twice cannot change the dataset or split. Source file hashes and a
combined dataset hash are recorded in `dataset.json` and `replay-config.json`.

Copies of the same sequence of board states with different policy/value
targets remain distinct training examples but belong to the same split.
Ten percent of these trajectory groups are held out using seed `20260906`.
This prevents a repeated game from leaking across the split; common opening
positions and transpositions can still occur in different games.

Both networks start from the same seeded parameter initialization and use:

- 20 passes through the training partition;
- all eight board symmetries, with the same minibatch shuffle each pass;
- batch size 256, Adam learning rate 0.001, L2 weight decay 0.0001;
- policy cross-entropy plus value MSE;
- validation after every pass, with inference-mode BatchNorm and no gradients.

The match uses the final checkpoints after equal training exposure. Validation
curves diagnose overfitting; they do not select a different training budget for
each activation. This is an initial fixed-data, single-seed experiment. It does
not establish which activation learns best in a complete self-play loop.
Around an even result, 300 games give roughly a ±5.6 percentage-point 95% interval
under a simple independent-game model. Small differences remain inconclusive.

## Running on a GPU pod

Use one NVIDIA GPU. The repository's earlier RTX 3090 benchmark demonstrates
that a 24 GB card is sufficient for this network. A practical starting host is
16 vCPUs, 64 GB system RAM, and 100 GB persistent disk. The script starts the
match with 128 concurrent games and inference batches of up to 64 per model;
measure memory and batching on the actual host before increasing concurrency.

The locked Torch environment uses CUDA 13. Choose a host with an R580 or newer
driver, then verify CUDA with `scripts/check_cuda.py`. See
[NVIDIA's driver compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
For data that must survive pod termination, use a network volume or retain an
external copy; a pod volume has a different lifetime. See
[RunPod's storage documentation](https://docs.runpod.io/pods/storage/types).

Place the repository below `/workspace` and transfer only `metadata.json` and
`replay.bin.zst` from each selected numeric snapshot directory. Source model and
optimizer files are not required for replay training. For example, keep the
ten source directories below `/workspace/alpha-zero/replays`.

```bash
cd /workspace/alpha-zero
bash scripts/setup_runpod.sh
source scripts/runpod_env.sh
scripts/archive/run_activation_experiment.sh \
  /workspace/alpha-zero/replays \
  /workspace/alpha-zero/runs/relu-vs-gelu-20260906
```

Run this in tmux. The experiment script trains the two models sequentially and
then invokes `battle` with the requested settings. `REPLAY_SNAPSHOTS` defaults
to 10 when the first argument is a checkpoint root; passing an individual
snapshot uses only that buffer. Defaults can be overridden through `EPOCHS`,
`SEED`, `TRAINING_BATCH_SIZE`, `GAMES_PARALLELISM`, and `INFERENCE_BATCH_SIZE`.
`DEVICE`, `BATTLE_GAMES`, and `SIMULATIONS` exist for small smoke tests; leave
them at their defaults for the actual experiment.

Each model has an independent output directory with configuration, dataset
provenance, per-epoch training/validation losses, and complete checkpoints.
`battle.json` contains the final outcomes, confidence interval, per-seat
results, model identities, timings, and move records. Logs are retained beside
the model directories. The script refuses to overwrite a completed match and
records the binary hash to prevent an accidental executable change on resume.

Interrupted training resumes from the latest complete epoch when rerunning
the same script. Changing data, activation, seed, optimizer settings, batch
size, or split requires a new run directory. The dataset source must remain
available. An interrupted match restarts its 300 games; it has no partial-game
resume facility.

## Individual commands

`train-replay` accepts repeated source arguments and supports inspection without
creating outputs or initializing a model:

```bash
./run.sh ./target/release/alz train-replay \
  --replay-checkpoint-dir replays/00000068 \
  --replay-checkpoint-dir replays/00000069 \
  --run-dir runs/inspect-only \
  --architecture kata-gelu-v1 --inspect-only
```

Remove `--inspect-only`, add `--device cuda`, and set the desired `--epochs` to
train. The standalone command uses the same defaults as the experiment script.
