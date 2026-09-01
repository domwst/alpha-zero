# AlphaZero performance experiment methodology

The objective is end-to-end self-play throughput, measured as completed network evaluations and games per second. GPU utilization is diagnostic rather than the optimization target: this network is small enough for CPU-side MCTS and batch formation to dominate.

## Reproducibility controls

- Build and run through `./run.sh`, which uses the locked Python/PyTorch environment and embeds the matching Torch library path.
- Use a release binary with line-level debug information for `perf` and Nsight Systems.
- Give every run a unique directory and execute it through `scripts/run_profiled.sh`.
- Set an explicit seed. Training derives independent deterministic streams for each game, epoch shuffle, and rendered-game selection.
- Keep batch size, concurrent games, and batch timeout fixed within a measured run.
- Warm CUDA and convolution selection before recording a benchmark.
- Repeat finalists at least three times and compare medians, not the single best observation.

## Benchmark order

1. `benchmark inference`: find the batch-throughput knee without MCTS.
2. `benchmark self-play`: tune batch size, concurrent games, and timeout together using a fixed game/simulation workload.
3. Compare 8 physical cores with all 16 logical CPUs for the best scheduler candidates.
4. `benchmark training`: tune optimizer batch size independently.
5. Validate host-memory scaling at the production simulation count. Shallow MCTS
   sweeps do not expose the memory retained by expanded search trees.
6. Confirm the selected settings with a complete training epoch before beginning a long run.

Example:

```bash
./run.sh cargo build --release
scripts/run_profiled.sh runs/inference-b128 \
  ./target/release/alz benchmark inference --device cuda \
  --batch-size 128 --warmup-iterations 20 --iterations 100 \
  --output runs/inference-b128/result.json
```

For timeline profiling, profile only representative baseline and finalist runs because profiler overhead invalidates throughput comparison:

```bash
nsys profile --trace=cuda,osrt --sample=cpu --cpuctxsw=process-tree \
  --output runs/nsys-final/profile --force-overwrite=true \
  ./target/release/alz benchmark self-play --device cuda \
  --games 384 --simulations 32 --inference-batch-size 128 \
  --games-parallelism 384 --batch-timeout-us 5000
```

Epoch metrics are written both as individual JSON documents and as `stats/epochs.jsonl`. The latter is the primary health-monitoring stream for long runs.
