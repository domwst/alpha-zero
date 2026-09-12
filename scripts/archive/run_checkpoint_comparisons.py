#!/usr/bin/env python3
"""Compare validation-selected pooling checkpoints and their final checkpoints."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from contextlib import ExitStack
import fcntl
import math
from pathlib import Path

from experiment_io import checkpoint, digest, fixed, read, require, write
from archive.run_replay_followups import Queue


def select_checkpoints(directory, architecture, epochs=20):
    directory = Path(directory)
    config = read(directory / "replay-config.json")
    result = read(directory / "result.json")
    require(config["model"] == {"architecture": architecture}, "Wrong source architecture")
    require(result["completed_epochs"] == epochs and
            result["dataset_sha256"] == config["dataset_sha256"], "Wrong source training budget or data")
    metrics, hashes = [], {}
    for epoch in range(epochs):
        path = directory / "epochs" / f"{epoch:08}.json"
        metric = read(path)
        require(metric["epoch"] == epoch and
                (directory / "checkpoints" / f"{epoch:08}" / "metadata.json").is_file(),
                "Missing completed source checkpoint")
        losses = metric["validation"]
        require(all(math.isfinite(losses[k]) for k in ("value_loss", "policy_loss", "total_loss"))
                and math.isclose(losses["total_loss"], losses["value_loss"] + losses["policy_loss"],
                                 rel_tol=1e-9, abs_tol=1e-9), "Invalid validation loss")
        metrics.append(metric)
        hashes[path.name] = digest(path)
    best = min(metrics, key=lambda metric: (metric["validation"]["total_loss"], metric["epoch"]))
    early = checkpoint(directory / "checkpoints" / f'{best["epoch"]:08}')
    final = checkpoint(directory / "checkpoints" / f"{epochs - 1:08}")
    require(early["model"] == final["model"] == config["model"], "Checkpoint architecture mismatch")
    require(final == result["checkpoint"], "Final checkpoint differs from training result")
    return {"directory": str(directory), "config": config, "selected": early, "final": final,
            "selected_validation": best["validation"], "final_validation": metrics[-1]["validation"],
            "metrics_sha256": hashes, "config_sha256": digest(directory / "replay-config.json"),
            "result_sha256": digest(directory / "result.json")}


def matched_config(source):
    config = dict(source["config"])
    config.pop("model")
    config.setdefault("adam_backend", "standard")
    return config


class CheckpointQueue(Queue):
    def prepare(self):
        source = self.args.predecessor
        for directory in (source, source / "pooling"):
            require(read(directory / "status.json")["stage"] == "complete" and
                    (directory / "exit-code.txt").read_text().strip() == "0",
                    "Preceding experiments must have completed successfully")
        selected = read(source / "pooling" / "selection.json")
        current = select_checkpoints(Path(selected["selected_checkpoint"]["path"]).parents[1],
                                     "kata_gelu_value64x2_v1")
        katago = select_checkpoints(source / "pooling" / "katago-pooling",
                                    "kata_gelu_pool_value64x2_v1")
        require(current["final"] == selected["selected_checkpoint"], "Previous baseline identity changed")
        require(current["selected"]["epoch"] == 17 and katago["selected"]["epoch"] == 14,
                "Validation selection differs from the proposed passes 18 and 15")
        require(matched_config(current) == matched_config(katago), "Source training settings differ")
        require(digest(self.binary) == self.args.binary_sha256, "Validated match binary changed")
        matches = []
        for name, title, first, second, note, offset in [
            ("selected-current-vs-katago-pooling", "Selected checkpoints · current pass 18 vs KataGo pass 15",
             current["selected"], katago["selected"],
             "Each checkpoint minimizes validation policy loss + value MSE within its 20-pass run.", 6),
            ("current-pooling-selected-vs-final", "Current pooling · selected pass 18 vs final pass 20",
             current["selected"], current["final"],
             "Checks whether validation selection improves the current-pooling model's playing strength.", 7),
            ("katago-pooling-selected-vs-final", "KataGo pooling · selected pass 15 vs final pass 20",
             katago["selected"], katago["final"],
             "Checks whether validation selection improves the KataGo-style model's playing strength.", 8),
        ]:
            matches.append({"id": name, "title": title, "first": first, "second": second,
                            "note": note, "seed": current["config"]["seed"] + offset})
        plan = {"schema_version": 1, "predecessor": str(source), "device": self.args.device,
                "binary": str(self.binary), "binary_sha256": digest(self.binary),
                "code_sha256": {name: digest(self.repo / name) for name in (
                    "scripts/archive/run_checkpoint_comparisons.py", "scripts/archive/run_replay_followups.py",
                    "scripts/experiment_control.py", "run.sh")},
                "selection_rule": "minimum validation policy_loss + value_loss; ties prefer earlier pass",
                "sources": {"current": current, "katago": katago}, "matches": matches,
                "settings": {"games": self.args.games, "simulations": self.args.simulations,
                             "parallelism": self.args.parallelism,
                             "inference_batch_size": self.args.inference_batch_size, "temperature": 0.7}}
        fixed(self.root / "plan.json", plan)
        return plan

    def run(self):
        plan = self.prepare()
        summary = {"plan_sha256": digest(self.root / "plan.json"), "matches": {}}
        for match in plan["matches"]:
            result = self.battle(match["id"], match["first"], match["second"], match["seed"])
            config = read(self.root / (match["id"] + ".json"))["config"]
            require(config["games_parallelism"] == self.args.parallelism and
                    config["inference_batch_size"] == self.args.inference_batch_size,
                    "Match concurrency or inference batch size changed")
            summary["matches"][match["id"]] = result
            write(self.root / "summary.json", summary)
        self.status("complete", summary=str(self.root / "summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--simulations", type=int, default=4000)
    parser.add_argument("--parallelism", type=int, default=300)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    require(args.games > 0 and args.games % 2 == 0 and
            min(args.simulations, args.parallelism, args.inference_batch_size) > 0,
            "Invalid match settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for directory in (args.predecessor, args.predecessor / "pooling", args.output_dir):
            lock = stack.enter_context((directory / ".lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = CheckpointQueue(args)
        try:
            if args.prepare_only:
                queue.prepare()
            else:
                queue.run()
        except Exception as error:
            if not args.prepare_only:
                queue.status("failed", failed_stage=queue.stage, error=str(error))
            raise


if __name__ == "__main__":
    main()
