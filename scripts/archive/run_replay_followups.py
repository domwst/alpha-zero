#!/usr/bin/env python3
"""Wait for an activation experiment, then run the fixed-replay follow-ups.

Run inside tmux. Completed stages are validated and reused on restart; training
resumes completed epochs. Interrupted matches restart in full. Python stdlib only.
"""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

from experiment_control import read_control
from experiment_io import canonical_descriptor, checkpoint, digest, fixed, read, require, validate_battle, write












def upgrade_queue_config(root, value, expected_digest=None):
    """Audit an explicitly requested backend upgrade before future training starts."""
    path = root / "queue-config.json"
    if not path.exists() or read(path) == value or expected_digest is None:
        return fixed(path, value)
    require(digest(path) == expected_digest, "Queue configuration changed before upgrade")
    previous = read(path)
    require(previous["schema_version"] == 1 and value["schema_version"] == 2,
            "Only a version-1 queue can adopt separate training backends")
    allowed = {"schema_version", "script_sha256", "training_binary_sha256", "training_performance"}
    require({k: v for k, v in previous.items() if k not in allowed} ==
            {k: v for k, v in value.items() if k not in allowed},
            "Upgrade changed match settings, match binary, or source inputs")
    require(not any(root.glob("preflight-*")) and not any(root.glob("kata*")),
            "Cannot upgrade after value-head preflight or training has started")
    require((root / "replay-relu-vs-selfplay.json").is_file(),
            "Checkpoint comparison must finish before upgrading the worker")
    fixed(root / "training-backend-upgrade.json", {
        "previous": previous, "replacement": value, "previous_config_sha256": expected_digest,
        "reason": "User requested optimized backends for future training; completed baseline retained",
    })
    write(path, value)








def select_activation(report):
    # This is a scheduling rule, not a claim of statistical significance.
    return ("kata-gelu-v1" if report["second_checkpoint_result"]["score"] >
            report["first_checkpoint_result"]["score"] else "kata-v1")


class Queue:
    def __init__(self, args):
        self.args = args
        self.root = args.output_dir
        self.repo = Path(__file__).resolve().parents[2]
        self.binary = args.binary or self.repo / "target/release/alz"
        self.training_binary = getattr(args, "training_binary", None) or self.binary
        self.performance = {"adam_backend": getattr(args, "adam_backend", "standard"),
                            "replay_cache": getattr(args, "replay_cache", "none"),
                            "prefetch_batches": getattr(args, "prefetch_batches", 0)}
        self.stage = "initializing"
        self.env = dict(os.environ)
        for key, value in {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                           "TOKIO_WORKER_THREADS": "16"}.items():
            self.env.setdefault(key, value)

    def status(self, stage, **details):
        self.stage = stage
        value = {"stage": stage, "updated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), "pid": os.getpid(), **details}
        write(self.root / "status.json", value)
        print(json.dumps(value), flush=True)

    def command_line(self, arguments):
        training = arguments[0] == "train-replay" or arguments[:2] == ["benchmark", "training"]
        binary = self.training_binary if training else self.binary
        command = [str(self.repo / "run.sh"), str(binary), *map(str, arguments)]
        if training:
            for key, value in self.performance.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
        return command

    def command(self, name, arguments):
        self.wait_until_resumed(name)
        command = self.command_line(arguments)
        self.status(name, log=str(self.root / (name + ".log")))
        print(shlex.join(command), flush=True)
        with (self.root / (name + ".log")).open("a") as log:
            with subprocess.Popen(command, cwd=self.repo, env=self.env, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if "BATTLE_RESULT" not in line:
                        print(line, end="", flush=True)
                require(process.wait() == 0, f"{name} failed; see {log.name}")

    def wait_until_resumed(self, next_stage):
        announced = False
        while read_control(self.root)["paused"]:
            if not announced:
                self.status("paused", next_stage=next_stage)
                announced = True
            time.sleep(1)

    def battle(self, name, first, second, seed):
        output = self.root / (name + ".json")
        if not output.exists():
            # Publish only a complete, validated report. An interrupted partial
            # report can be overwritten when rerunning this same stage.
            temporary = output.with_suffix(".partial.json")
            self.command(name, ["battle", "--first-checkpoint-dir", first["path"],
                "--second-checkpoint-dir", second["path"], "--device", self.args.device,
                "--games", self.args.games, "--simulations", self.args.simulations,
                "--temperature", "0.7", "--games-parallelism", self.args.parallelism,
                "--inference-batch-size", self.args.inference_batch_size,
                "--batch-timeout-us", "1000", "--seed", seed, "--heartbeat-seconds", "60",
                "--no-move-logs", "--output", temporary])
            validate_battle(read(temporary), first, second, self.args.games,
                            self.args.simulations, 0.7)
            temporary.replace(output)
        report = read(output)
        validate_battle(report, first, second, self.args.games, self.args.simulations, 0.7)
        require(report["config"]["seed"] == seed, "Battle seed changed")
        return {key: report[key] for key in ("first_checkpoint", "second_checkpoint",
                "first_checkpoint_result", "second_checkpoint_result", "duration_seconds")}

    def run(self):
        args = self.args
        source = checkpoint(args.latest_checkpoint)
        require(source["model"]["architecture"] == "kata_v1", "Expected a ReLU Kata checkpoint")
        upgrade_queue_config(self.root, {
            "schema_version": 2, "binary_sha256": digest(self.binary),
            "training_binary_sha256": digest(self.training_binary),
            "training_performance": self.performance,
            "script_sha256": digest(__file__), "latest_checkpoint": source,
            "activation_dir": str(args.activation_dir), "device": args.device,
            "games": args.games, "simulations": args.simulations, "temperature": 0.7,
            "parallelism": args.parallelism, "inference_batch_size": args.inference_batch_size,
            "require_success_file": str(args.require_success_file) if args.require_success_file else None,
            "selection": "highest activation-match score; exact tie selects ReLU",
        }, getattr(args, "upgrade_from_config_sha256", None))
        self.status("waiting_for_activation_experiment", activation_dir=str(args.activation_dir))
        while True:
            if args.require_success_file:
                if not args.require_success_file.exists():
                    time.sleep(args.poll_seconds)
                    continue
                require(args.require_success_file.read_text().strip() == "0",
                        "Activation experiment failed; follow-ups stopped")
            if (args.activation_dir / "battle.json").exists():
                break
            require(not args.require_success_file, "Successful activation job has no match report")
            time.sleep(args.poll_seconds)

        models = {}
        common_config = None
        epochs = None
        for name in ("kata-v1", "kata-gelu-v1"):
            directory = args.activation_dir / name
            result = read(directory / "result.json")
            config = read(directory / "replay-config.json")
            require(config.pop("model")["architecture"] == name.replace("-", "_"),
                    "Activation model has the wrong architecture")
            model = checkpoint(result["checkpoint"]["path"])
            require(model == canonical_descriptor(result["checkpoint"]), "Training result differs from checkpoint")
            require(model["model"] == result["model"] == {"architecture": name.replace("-", "_")},
                    "Training result has the wrong architecture")
            require(result["completed_epochs"] == model["epoch"] + 1, "Wrong training budget")
            require(result["dataset_sha256"] == config["dataset_sha256"], "Training dataset mismatch")
            if common_config is None:
                common_config, epochs = config, result["completed_epochs"]
            require(config == common_config and result["completed_epochs"] == epochs,
                    "Activation runs do not have matched training settings")
            models[name] = model
        activation_report = read(args.activation_dir / "battle.json")
        validate_battle(activation_report, models["kata-v1"], models["kata-gelu-v1"],
                        args.games, args.simulations, 0.7)
        selected = select_activation(activation_report)
        stats = activation_report[("first" if selected == "kata-v1" else "second") + "_checkpoint_result"]
        fixed(self.root / "selection.json", {
            "activation": selected, "baseline": models[selected], "score": stats,
            "interval_includes_half": stats["score_rate_95_percent_low"] <= 0.5 <=
                                      stats["score_rate_95_percent_high"],
            "rule": "highest score, draw = half win; exact tie selects ReLU",
            "activation_battle_sha256": digest(args.activation_dir / "battle.json"),
            "training_config": common_config, "epochs": epochs,
        })
        dataset = read(args.activation_dir / selected / "dataset.json")
        require(dataset["dataset_sha256"] == common_config["dataset_sha256"], "Dataset mismatch")
        replay_args = []
        for replay in dataset["sources"]:
            path = Path(replay["checkpoint"]["path"])
            require(digest(path / "replay.bin.zst") == replay["replay_sha256"],
                    f"Replay source changed: {path}")
            replay_args.extend(["--replay-checkpoint-dir", str(path)])

        seed = common_config["seed"]
        summary = {"selection": read(self.root / "selection.json"), "matches": {},
                   "training_performance": self.performance,
                   "baseline_adam_backend": common_config.get("adam_backend", "standard")}
        summary["matches"]["replay-relu-vs-selfplay"] = self.battle(
            "replay-relu-vs-selfplay", models["kata-v1"], source, seed + 1)
        write(self.root / "summary.json", summary)

        prefix = "kata-gelu" if selected == "kata-gelu-v1" else "kata"
        variants = [prefix + "-value64-v1", prefix + "-value64x2-v1"]
        for name in variants:
            # Run actual native inference and optimizer steps before full training.
            for mode, batch_size, field in [("inference", args.inference_batch_size, "checksum"),
                                           ("training", common_config["batch_size"], "final_loss")]:
                output = self.root / f"preflight-{name}-{mode}.json"
                if not output.exists():
                    training_args = []
                    warmup, iterations = 3, 10
                    if mode == "training":
                        training_args = ["--weight-decay", common_config["weight_decay"]]
                        if self.performance["replay_cache"] != "none":
                            training_args += replay_args[:2]
                            positions = read(Path(replay_args[1]) / "metadata.json")["replay_positions"]
                            full_batches = positions * 8 // batch_size
                            require(full_batches > 0, "Replay has no full preflight batch")
                            warmup = min(warmup, full_batches - 1)
                            iterations = min(iterations, full_batches - warmup)
                    self.command(f"preflight-{name}-{mode}", ["benchmark", mode,
                        "--architecture", name, "--device", args.device, "--batch-size", batch_size,
                        "--warmup-iterations", warmup, "--iterations", iterations, "--seed", seed,
                        "--output", output, *training_args])
                bench = read(output)
                require(isinstance(bench[field], (int, float)) and math.isfinite(bench[field]),
                        f"Non-finite {name} {mode} preflight")
                expected_device = "Cuda(0)" if args.device == "cuda" else "Cpu"
                require(bench["device"] == expected_device, "Preflight used the wrong device")
                if mode == "training":
                    require(all(bench["config"].get(k) == v for k, v in self.performance.items()),
                            "Preflight used different training backends")
            self.command("train-" + name, ["train-replay", *replay_args,
                "--run-dir", self.root / name, "--architecture", name, "--device", args.device,
                "--epochs", epochs, "--training-batch-size", common_config["batch_size"],
                "--validation-fraction", common_config["validation_fraction"],
                "--learning-rate", common_config["learning_rate"],
                "--weight-decay", common_config["weight_decay"], "--seed", seed])
            trained = read(self.root / name / "result.json")
            config = read(self.root / name / "replay-config.json")
            require(config.pop("model") == {"architecture": name.replace("-", "_")},
                    "Wrong trained architecture")
            require(config.pop("adam_backend", "standard") == self.performance["adam_backend"],
                    "Follow-up used a different Adam backend")
            expected = {k: v for k, v in common_config.items() if k != "adam_backend"}
            require(config == expected and trained["completed_epochs"] == epochs,
                    "Follow-up training budget or dataset changed")
            models[name] = checkpoint(trained["checkpoint"]["path"])
            require(models[name] == canonical_descriptor(trained["checkpoint"]), "Follow-up checkpoint mismatch")

        for offset, (name, first, second) in enumerate([
                ("current-vs-wide64", selected, variants[0]),
                ("current-vs-deep64", selected, variants[1]),
                ("wide64-vs-deep64", variants[0], variants[1])], start=2):
            summary["matches"][name] = self.battle(name, models[first], models[second], seed + offset)
            write(self.root / "summary.json", summary)
        self.status("complete", summary=str(self.root / "summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--latest-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-success-file", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--training-binary", type=Path, help="Separate executable for future training")
    parser.add_argument("--adam-backend", choices=("standard", "fused"), default="standard")
    parser.add_argument("--replay-cache", choices=("none", "cpu", "device"), default="none")
    parser.add_argument("--prefetch-batches", type=int, default=0)
    parser.add_argument("--upgrade-from-config-sha256", help="Explicit audited upgrade of a version-1 queue")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--simulations", type=int, default=4000)
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    require(args.games > 0 and args.games % 2 == 0, "Games must be positive and even")
    require(min(args.simulations, args.parallelism, args.inference_batch_size, args.poll_seconds) > 0,
            "Budgets and polling interval must be positive")
    require(0 <= args.prefetch_batches <= 16 and
            (args.prefetch_batches == 0 or args.replay_cache == "cpu"), "Invalid prefetch configuration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = Queue(args)
        try:
            queue.run()
        except Exception as error:
            queue.status("failed", failed_stage=queue.stage, error=str(error))
            raise


if __name__ == "__main__":
    main()
