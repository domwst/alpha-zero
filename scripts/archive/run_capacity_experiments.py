#!/usr/bin/env python3
"""Train deeper/wider trunks on frozen replays and compare validation-selected models."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from contextlib import ExitStack
import fcntl
from pathlib import Path

from archive.run_checkpoint_comparisons import select_checkpoints, matched_config
from archive.run_pooling_followup import PoolingQueue
from experiment_io import digest, fixed, read, require, write


MODELS = [
    {"id": "capacity-depth16", "architecture": "kata-gelu-b16c32-value64x2-v1",
     "title": "Train deeper trunk · 16 blocks × 32 channels", "blocks": 16, "channels": 32},
    {"id": "capacity-width48", "architecture": "kata-gelu-b10c48-value64x2-v1",
     "title": "Train wider trunk · 10 blocks × 48 channels", "blocks": 10, "channels": 48},
]
MATCHES = [
    {"id": "capacity-baseline-vs-depth16", "title": "Current trunk vs 16-block trunk", "first": "baseline", "second": "capacity-depth16"},
    {"id": "capacity-baseline-vs-width48", "title": "Current trunk vs 48-channel trunk", "first": "baseline", "second": "capacity-width48"},
    {"id": "capacity-depth16-vs-width48", "title": "16-block trunk vs 48-channel trunk", "first": "capacity-depth16", "second": "capacity-width48"},
]


def schedule():
    return {"schema_version": 1, "models": MODELS, "matches": MATCHES, "epochs": 20,
            "settings": {"games": 1000, "simulations": 4000, "parallelism": 300,
                         "inference_batch_size": 64, "temperature": 0.7},
            "note": "Same frozen replays and 20-pass training budget; minimum combined validation loss selects every model. Two original global blocks; fixed head widths."}


class CapacityQueue(PoolingQueue):
    def prepare(self):
        args = self.args
        fixed(self.root / "schedule.json", schedule())
        for key, value in schedule()["settings"].items():
            if key != "temperature":
                setattr(args, key, value)
        for directory in (args.predecessor, args.predecessor / "pooling"):
            require(read(directory / "status.json")["stage"] == "complete", "Predecessor incomplete")
        stopped = read(args.predecessor / "pooling-checkpoints" / "cancellation.json")
        require(set(stopped["jobs"]) == {"current-pooling-selected-vs-final", "katago-pooling-selected-vs-final"},
                "Checkpoint comparison cancellation changed")
        original = read(args.predecessor / "pooling" / "selection.json")
        baseline = select_checkpoints(Path(original["selected_checkpoint"]["path"]).parents[1], "kata_gelu_value64x2_v1")
        require(baseline["selected"]["epoch"] == 17, "Baseline validation selection changed")
        common = matched_config(baseline)
        require(common["adam_backend"] == "standard", "Baseline optimizer changed")
        dataset_path = Path(baseline["directory"]) / "dataset.json"
        dataset = read(dataset_path)
        require(dataset["dataset_sha256"] == common["dataset_sha256"], "Dataset mismatch")
        replays = []
        for source in dataset["sources"]:
            path = Path(source["checkpoint"]["path"])
            require(digest(path / "replay.bin.zst") == source["replay_sha256"], "Replay source changed")
            replays.extend(["--replay-checkpoint-dir", str(path)])
        ready = read(self.repo / "capacity-build-ready.json")
        require(digest(self.binary) == ready["binary_sha256"], "Validated binary changed")
        plan = {"schedule": schedule(), "baseline": baseline, "dataset_sha256": common["dataset_sha256"],
                "dataset_file_sha256": digest(dataset_path), "common_training": common,
                "binary_sha256": ready["binary_sha256"], "build": ready,
                "code_sha256": {name: digest(self.repo / name) for name in (
                    "scripts/archive/run_capacity_experiments.py", "scripts/archive/run_checkpoint_comparisons.py",
                    "scripts/archive/run_pooling_followup.py", "scripts/archive/run_replay_followups.py",
                    "scripts/experiment_control.py", "scripts/run_cuda_tests_runpod.sh", "run.sh")}}
        fixed(self.root / "plan.json", plan)
        return baseline, common, replays

    def run(self):
        baseline, common, replays = self.prepare()
        self.status("validating_capacity_cuda")
        self.configure_backend({
            "architecture_command": ["bash", str(self.repo / "scripts/run_cuda_tests_runpod.sh"), "capacity"],
            "cache_command": ["bash", str(self.repo / "scripts/run_cuda_tests_runpod.sh"), "cache"],
        }, {"adam_backend": "standard", "replay_cache": "device", "prefetch_batches": 0})
        selections = {"baseline": baseline}
        # Record baseline throughput with the same benchmark protocol as the candidates.
        self.preflight("capacity-baseline", "kata-gelu-value64x2-v1", common, replays)
        for model in MODELS:
            identity, architecture = model["id"], model["architecture"]
            directory = self.root / identity
            self.preflight(identity, architecture, common, replays)
            if not (directory / "result.json").exists():
                self.command("train-" + identity, ["train-replay", *replays,
                    "--run-dir", directory, "--architecture", architecture, "--device", self.args.device,
                    "--epochs", 20, "--training-batch-size", common["batch_size"],
                    "--learning-rate", common["learning_rate"], "--weight-decay", common["weight_decay"],
                    "--validation-fraction", common["validation_fraction"], "--seed", common["seed"]])
            selected = select_checkpoints(directory, architecture.replace("-", "_"))
            require(matched_config(selected) == common, "New model used different training settings")
            selections[identity] = selected
            write(self.root / "selection.json", selections)
        summary = {"plan_sha256": digest(self.root / "plan.json"), "matches": {}}
        for index, match in enumerate(MATCHES):
            result = self.battle(match["id"], selections[match["first"]]["selected"],
                                 selections[match["second"]]["selected"], common["seed"] + 20 + index)
            config = read(self.root / (match["id"] + ".json"))["config"]
            require(config["games_parallelism"] == 300 and config["inference_batch_size"] == 64,
                    "Match concurrency changed")
            summary["matches"][match["id"]] = result
            write(self.root / "summary.json", summary)
        self.status("complete", summary=str(self.root / "summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for directory in (args.predecessor, args.predecessor / "pooling",
                          args.predecessor / "pooling-checkpoints", args.output_dir):
            lock = stack.enter_context((directory / ".lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = CapacityQueue(args)
        try:
            if args.prepare_only: queue.prepare()
            else: queue.run()
        except Exception as error:
            queue.status("failed", failed_stage=queue.stage, error=str(error))
            raise

if __name__ == "__main__":
    main()
