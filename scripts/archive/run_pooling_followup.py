#!/usr/bin/env python3
"""Reuse the winning value-head checkpoint and train its KataGo pooling counterpart."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import fcntl
import math
from fractions import Fraction
from pathlib import Path
import time

from archive.handoff_replay_training import validate_backend
from match_plan import match_settings, read_match_plan
from experiment_io import canonical_descriptor, checkpoint, digest, fixed, read, require, validate_battle, write
from archive.run_replay_followups import Queue


MATCHES = [("current-vs-wide64", "current", "wide64"),
           ("current-vs-deep64", "current", "deep64"),
           ("wide64-vs-deep64", "wide64", "deep64")]
HEADS = {"current": [10], "wide64": [64], "deep64": [64, 64]}
STANDARD = {"adam_backend": "standard", "replay_cache": "none", "prefetch_batches": 0}
BASELINE_POLICY = "reuse-selected-checkpoint"


def select_head(reports, models, games, simulations):
    scores = {name: 0.0 for name in HEADS}
    rates = {name: Fraction(0) for name in HEADS}
    played = {name: 0 for name in HEADS}
    budgets = {name: games if isinstance(games, int) else games[name] for name, _, _ in MATCHES}
    for name, first, second in MATCHES:
        report = reports[name]
        count = budgets[name]
        validate_battle(report, models[first], models[second], count, simulations, 0.7)
        scores[first] += report["first_checkpoint_result"]["score"]
        scores[second] += report["second_checkpoint_result"]["score"]
        for label, side in [(first, "first"), (second, "second")]:
            rates[label] += Fraction(str(report[side + "_checkpoint_result"]["score"])) / count / 2
            played[label] += count
    # Insertion order makes exact ties deterministic: current, wide, deep.
    selected = max(rates, key=rates.get)
    return {"head": selected, "hidden_dims": HEADS[selected], "scores": scores,
            "score_rates": {name: float(rate) for name, rate in rates.items()},
            "games_per_match": budgets, "games_played": played,
            "rule": "highest mean score rate across two opponents, equally weighted; draws count half; exact ties prefer smaller head",
            "selection_establishes_statistical_superiority": False}


def pooling_architecture(current):
    require(current in ("kata-v1", "kata-value64-v1", "kata-value64x2-v1",
                        "kata-gelu-v1", "kata-gelu-value64-v1", "kata-gelu-value64x2-v1"),
            "Unknown current pooling architecture")
    prefix = "kata-gelu" if current.startswith("kata-gelu") else "kata"
    return prefix + "-pool" + current[len(prefix):]


def predecessor_ready(root):
    status = read(root / "status.json") if (root / "status.json").exists() else {}
    require(status.get("stage") != "failed", "Value-head experiments failed; pooling work stopped")
    # Ignore an old launcher receipt while the trainer handoff is in progress.
    if status.get("stage") != "complete":
        return False
    success = root / "exit-code.txt"
    if not success.exists():
        return False
    require(success.read_text().strip() == "0", "Value-head launcher did not exit successfully")
    return True


class PoolingQueue(Queue):
    def run(self):
        args = self.args
        self.match_plan = read_match_plan(args.predecessor)
        validation_plan = read(args.validation_plan) if args.validation_plan else None
        require(args.device != "cuda" or validation_plan is not None,
                "CUDA deployment requires architecture and optimizer validation commands")
        fixed(self.root / "queue-config.json", {
            "schema_version": 2, "predecessor": str(args.predecessor),
            "binary_sha256": digest(self.binary), "device": args.device,
            "validation_plan": validation_plan,
            "match_plan": self.match_plan,
            "code_sha256": {name: digest(self.repo / name) for name in [
                "scripts/archive/run_pooling_followup.py", "scripts/archive/run_replay_followups.py",
                "scripts/archive/handoff_replay_training.py", "scripts/experiment_control.py",
                "scripts/match_plan.py", "scripts/run_cuda_tests_runpod.sh", "run.sh"]},
            "baseline_policy": BASELINE_POLICY,
            "training": "reuse selected checkpoint; train KataGo pooling with matched Adam and validated device cache",
            "selection": "mean opponent score rate; exact ties prefer smaller value head",
        })
        self.status("waiting_for_value_heads")
        # Retain the predecessor lock after completion to exclude accidental restarts
        # competing for the GPU. Never touch the running worker or its control file.
        with (args.predecessor / ".lock").open("a") as lock:
            while True:
                if predecessor_ready(args.predecessor):
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        require(predecessor_ready(args.predecessor), "Predecessor changed at queue boundary")
                        break
                time.sleep(args.poll_seconds)
            self.run_after_predecessor(validation_plan)

    def run_after_predecessor(self, validation_plan):
        args = self.args
        source = args.predecessor
        source_config = read(source / "queue-config.json")
        require(source_config["device"] == args.device, "Predecessor used a different device")
        for key in ("games", "simulations", "parallelism", "inference_batch_size"):
            setattr(args, key, source_config[key])
        round_games = {name: match_settings(self.match_plan, name, args.games, args.parallelism)['games']
                       for name, _, _ in MATCHES}
        pooling_settings = match_settings(self.match_plan, 'current-vs-katago-pooling', args.games, args.parallelism)
        args.games, args.parallelism = pooling_settings['games'], pooling_settings['parallelism']
        require(args.games > 0 and args.games % 2 == 0 and source_config["temperature"] == 0.7,
                "Invalid predecessor match settings")
        selection = read(source / "selection.json")
        common = selection["training_config"]
        epochs = selection["epochs"]
        prefix = "kata-gelu" if selection["activation"] == "kata-gelu-v1" else "kata"
        model_roots = {"current": Path(source_config["activation_dir"]) / selection["activation"],
                       "wide64": source / (prefix + "-value64-v1"),
                       "deep64": source / (prefix + "-value64x2-v1")}
        models = {"current": self.trained_checkpoint(model_roots["current"], selection["activation"],
                                                     common, epochs, backend=None)}
        require(models["current"] == canonical_descriptor(selection["baseline"]),
                "Selected baseline checkpoint changed")
        for label, suffix in [("wide64", "-value64-v1"), ("deep64", "-value64x2-v1")]:
            models[label] = self.trained_checkpoint(model_roots[label], prefix + suffix,
                                                    common, epochs, backend=None)
        reports = {name: read(source / (name + ".json")) for name, _, _ in MATCHES}
        selected = select_head(reports, models, round_games, args.simulations)
        for offset, (name, _, _) in enumerate(MATCHES, 2):
            require(reports[name]["config"]["seed"] == common["seed"] + offset,
                    "Value-head match seed changed")
        selected.update(activation=selection["activation"],
                        selected_checkpoint=models[selected["head"]],
                        matches_sha256={name: digest(source / (name + ".json")) for name, _, _ in MATCHES},
                        predecessor_config_sha256=digest(source / "queue-config.json"),
                        predecessor_match_plan=self.match_plan,
                        epochs=epochs, training_config=common)
        current = models[selected["head"]]["model"]["architecture"].replace("_", "-")
        variants = {"current-pooling": current, "katago-pooling": pooling_architecture(current)}
        selected["architectures"] = variants
        fixed(self.root / "selection.json", selected)
        baseline_root = model_roots[selected["head"]]
        baseline_backend = read(baseline_root / "replay-config.json").get("adam_backend", "standard")
        require(baseline_backend in ("standard", "fused"), "Unknown baseline Adam backend")
        baseline = {"checkpoint": models[selected["head"]], "training_directory": str(baseline_root),
                    "adam_backend": baseline_backend, "epochs": epochs,
                    "dataset_sha256": common["dataset_sha256"],
                    "source_sha256": {name: digest(baseline_root / name)
                                      for name in ("replay-config.json", "result.json")}}
        fixed(self.root / "reused-baseline.json", baseline)
        dataset = read(Path(source_config["activation_dir"]) / selection["activation"] / "dataset.json")
        require(dataset["dataset_sha256"] == common["dataset_sha256"], "Source dataset changed")
        replay_args = []
        for replay in dataset["sources"]:
            path = Path(replay["checkpoint"]["path"])
            require(digest(path / "replay.bin.zst") == replay["replay_sha256"], "Replay source changed")
            replay_args.extend(["--replay-checkpoint-dir", str(path)])
        requested = {"adam_backend": baseline_backend, "replay_cache": "device", "prefetch_batches": 0}
        self.configure_backend(validation_plan, requested)
        require(self.performance["adam_backend"] == baseline_backend, "Pooling optimizer must match reused baseline")
        fixed(self.root / "experiment.json", {
            "selection": selected, "performance": self.performance,
            "games": args.games, "simulations": args.simulations, "temperature": 0.7,
            "parallelism": args.parallelism, "inference_batch_size": args.inference_batch_size,
            "match_seed": common["seed"] + 5, "fresh_models": False,
            "baseline_policy": BASELINE_POLICY, "reused_baseline": baseline,
        })
        trained = {"current-pooling": models[selected["head"]]}
        label, architecture = "katago-pooling", variants["katago-pooling"]
        if not (self.root / label / "result.json").exists():
            self.preflight(label, architecture, common, replay_args)
            self.command("train-" + label, ["train-replay", *replay_args,
                "--run-dir", self.root / label, "--architecture", architecture,
                "--device", args.device, "--epochs", epochs,
                "--training-batch-size", common["batch_size"],
                "--validation-fraction", common["validation_fraction"],
                "--learning-rate", common["learning_rate"],
                "--weight-decay", common["weight_decay"], "--seed", common["seed"]])
        trained[label] = self.trained_checkpoint(self.root / label, architecture, common,
                                                 epochs, self.performance["adam_backend"])
        result = self.battle("current-vs-katago-pooling", trained["current-pooling"],
                             trained["katago-pooling"], common["seed"] + 5)
        write(self.root / "summary.json", {"selection": selected, "performance": self.performance,
                                          "reused_baseline": baseline,
                                          "matches": {"current-vs-katago-pooling": result}})
        self.status("complete", summary=str(self.root / "summary.json"))

    def configure_backend(self, plan, requested):
        receipt = self.root / "backend.json"
        if receipt.exists():
            recorded = read(receipt)
            require(recorded["requested"] == requested, "Requested pooling backend changed")
            self.performance = recorded["performance"]
            require(self.performance["adam_backend"] == requested["adam_backend"], "Baseline optimizer changed")
            return
        result = {"requested": requested, "performance": requested}
        if self.args.device == "cuda":
            self.status("validating_pooling_cuda")
            architecture = validate_backend(plan["architecture_command"], self.repo,
                                            self.root / "architecture-validation.log", 300)
            write(self.root / "architecture-validation.json", architecture)
            require(architecture["passed"], "Pooling architecture CUDA checks failed")
            if requested != STANDARD:
                self.status("validating_pooling_backend")
                command = plan["cache_command"] if requested["adam_backend"] == "standard" else plan["backend_command"]
                backend = validate_backend(command, self.repo,
                                           self.root / "backend-validation.log", 300)
                result["validation"] = backend
                if not backend["passed"]:
                    require(requested["adam_backend"] == "standard",
                            "Inherited fused Adam failed validation; cannot change reused baseline optimizer")
                    result["performance"] = STANDARD
                    result["fallback"] = "standard Adam matching reused baseline, with uncached batching"
        fixed(receipt, result)
        self.performance = result["performance"]

    def preflight(self, label, architecture, common, replay_args):
        for mode, batch, field in [("inference", self.args.inference_batch_size, "checksum"),
                                   ("training", common["batch_size"], "final_loss")]:
            name = f"preflight-{label}-{mode}"
            output = self.root / (name + ".json")
            if not output.exists():
                extra = []
                warmup, iterations = 3, 10
                if mode == "training":
                    extra = ["--weight-decay", common["weight_decay"]]
                    if self.performance["replay_cache"] != "none":
                        extra += replay_args[:2]
                        batches = read(Path(replay_args[1]) / "metadata.json")["replay_positions"] * 8 // batch
                        require(batches > 0, "No complete preflight replay batch")
                        warmup = min(warmup, batches - 1)
                        iterations = min(iterations, batches - warmup)
                self.command(name, ["benchmark", mode, "--architecture", architecture,
                    "--device", self.args.device, "--batch-size", batch, "--seed", common["seed"],
                    "--warmup-iterations", warmup, "--iterations", iterations, "--output", output, *extra])
            bench = read(output)
            require(isinstance(bench[field], (int, float)) and math.isfinite(bench[field]), "Non-finite preflight")
            require(bench["device"] == ("Cuda(0)" if self.args.device == "cuda" else "Cpu"), "Wrong preflight device")
            if mode == "training":
                require(all(bench["config"].get(k) == v for k, v in self.performance.items()),
                        "Preflight used different training backends")

    @staticmethod
    def trained_checkpoint(root, architecture, common, epochs, backend):
        config = read(root / "replay-config.json")
        result = read(root / "result.json")
        require(config.pop("model") == {"architecture": architecture.replace("-", "_")},
                "Training architecture changed")
        actual_backend = config.pop("adam_backend", "standard")
        require(backend is None or actual_backend == backend, "Training backend changed")
        require(config == {k: v for k, v in common.items() if k != "adam_backend"},
                "Training budget or data changed")
        model = checkpoint(result["checkpoint"]["path"])
        require(model == canonical_descriptor(result["checkpoint"]) and
                model["model"] == result["model"] == {"architecture": architecture.replace("-", "_")},
                "Training result identity changed")
        require(model["epoch"] + 1 == result["completed_epochs"] == epochs and
                result["dataset_sha256"] == common["dataset_sha256"], "Wrong final training budget or data")
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--validation-plan", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    require(args.poll_seconds > 0, "Polling interval must be positive")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    require(args.output_dir != args.predecessor, "Pooling must use a separate output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = PoolingQueue(args)
        try:
            queue.run()
        except Exception as error:
            queue.status("failed", failed_stage=queue.stage, error=str(error))
            (args.output_dir / "exit-code.txt").write_text("1\n")
            raise
        (args.output_dir / "exit-code.txt").write_text("0\n")


if __name__ == "__main__":
    main()
