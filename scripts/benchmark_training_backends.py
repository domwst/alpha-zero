#!/usr/bin/env python3
"""Compare Adam and replay batching backends with identical real replay batches."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--architecture", default="kata-gelu-v1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-iterations", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.iterations, args.repetitions) <= 0 or args.warmup_iterations < 0:
        parser.error("batch size, iterations and repetitions must be positive; warmup must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    replay = args.replay_checkpoint_dir.resolve()
    modes = [
        ("standard-none", "standard", "none", 0),
        ("fused-none", "fused", "none", 0),
        ("standard-device", "standard", "device", 0),
        ("fused-cpu-prefetch", "fused", "cpu", 2),
        ("fused-device", "fused", "device", 0),
    ]
    results = {name: [] for name, *_ in modes}
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for repeat in range(args.repetitions):
        # Rotate execution order between repetitions to reduce ordering bias.
        offset = repeat % len(modes)
        for name, adam, cache, prefetch in modes[offset:] + modes[:offset]:
            path = output / f"{name}-{repeat}.json"
            command = [str(root / "run.sh"), str(root / "target/release/alz"),
                       "benchmark", "training", "--device", args.device,
                       "--cuda-index", str(args.cuda_index), "--architecture", args.architecture,
                       "--replay-checkpoint-dir", str(replay), "--seed", "20260906",
                       "--weight-decay", "0.0001", "--batch-size", str(args.batch_size),
                       "--warmup-iterations", str(args.warmup_iterations),
                       "--iterations", str(args.iterations), "--adam-backend", adam,
                       "--replay-cache", cache, "--prefetch-batches", str(prefetch),
                       "--output", str(path)]
            print(f"{repeat + 1}/{args.repetitions}: {name}", flush=True)
            with path.with_suffix(".log").open("w") as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=log, check=True)
            results[name].append(json.loads(path.read_text()))
    baseline = statistics.median(r["milliseconds_per_iteration"] for r in results["standard-none"])
    summary = {}
    for name, runs in results.items():
        median = statistics.median(r["milliseconds_per_iteration"] for r in runs)
        summary[name] = {
            "median_ms_per_batch": median,
            "speedup_vs_standard_none": baseline / median,
            "median_preparation_seconds": statistics.median(r["preparation_seconds"] for r in runs),
            "final_losses": [r["final_loss"] for r in runs],
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
