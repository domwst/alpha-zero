"""Typed job inputs compiled to argv. No shell, arbitrary executable, or output path."""

from pathlib import Path
import re

# Checkpoints/replays are artifact IDs resolved by the service, never client paths.
ARCHITECTURES = (
    "legacy-resnet-v1",
    "kata-v1",
    "kata-gelu-v1",
    "kata-value64-v1",
    "kata-value64x2-v1",
    "kata-gelu-value64-v1",
    "kata-gelu-value64x2-v1",
    "kata-gelu-boardmask-value64x2-v1",
    "kata-gelu-b16c32-value64x2-v1",
    "kata-gelu-b16c32g3-value64x2-v1",
    "kata-gelu-b10c48-value64x2-v1",
    "kata-pool-v1",
    "kata-gelu-pool-v1",
    "kata-pool-value64-v1",
    "kata-pool-value64x2-v1",
    "kata-gelu-pool-value64-v1",
    "kata-gelu-pool-value64x2-v1",
)
COMMON = {
    "device": ("cpu", "cuda", "mps"),
    "cuda-index": int,
    "seed": int,
    "inference-batch-grid": "grid",
}
TRAINING = {
    "architecture": ARCHITECTURES,
    "epochs": int,
    "training-batch-size": int,
    "learning-rate": float,
    "weight-decay": float,
    "bn-gamma-one": bool,
    "adam-backend": ("standard", "fused"),
    "replay-cache": ("none", "cpu", "device"),
    "prefetch-batches": int,
}
KINDS = {
    "self_play": TRAINING
    | {
        "games-per-epoch": int,
        "simulations": int,
        "games-parallelism": int,
        "inference-batch-size": int,
        "top-p": float,
        "temperature-schedule": ("paired", "sharp"),
        "replay-positions": int,
        "replay-position-growth": int,
        "replay-growth-start-epoch": int,
        "replay-lr-exponent": float,
        "inference-symmetry": ("none", "random"),
        "batch-timeout-us": int,
        "c-puct": float,
        "rendered-games": int,
        "heartbeat-seconds": int,
    },
    "replay_train": TRAINING
    | {
        "validation-fraction": float,
        "lr-schedule": ("constant", "cosine"),
        "final-learning-rate": float,
        "split-seed": int,
    },
    "comparison": {
        "games": int,
        "simulations": int,
        "games-parallelism": int,
        "inference-batch-size": int,
        "batch-timeout-us": int,
        "temperature": float,
        "first-temperature": float,
        "second-temperature": float,
        "c-puct": float,
        "heartbeat-seconds": int,
        "no-move-logs": bool,
    },
}
KINDS["benchmark_executor"] = {
    "producers": int,
    "requests": int,
    "batch-size": int,
    "timeout-us": int,
    "grid": "grid",
    "tail": bool,
    "reacquire": bool,
    "profile-allocator": bool,
    "task-count-baseline": bool,
    "disable-tf32": bool,
}
KINDS["benchmark_self_play"] = {
    "architecture": ARCHITECTURES,
    "games": int,
    "top-p": float,
    "simulations": int,
    "games-parallelism": int,
    "inference-batch-size": int,
    "batch-timeout-us": int,
    "warmup-batches": int,
    "c-puct": float,
    "inference-symmetry": ("none", "random"),
}
KINDS["reconstruction"] = KINDS["self_play"] | {"replay-history-epochs": int}
COMMANDS = {
    "reconstruction": "train",
    "self_play": "train",
    "replay_train": "train-replay",
    "comparison": "battle",
    "benchmark_executor": "benchmark",
    "benchmark_self_play": "benchmark",
}


def input_artifact_kinds(name):
    if name == "history":
        return {"history"}
    if name == "replay" or name.startswith("replay_"):
        return {"replay", "checkpoint"}
    return {"checkpoint"}


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) - {
        "kind",
        "options",
        "inputs",
        "resources",
        "binary_artifact",
    }:
        raise ValueError("Unknown job specification fields")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError("Unknown job kind")
    options = spec.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    rules = COMMON | KINDS[kind]
    for key, value in options.items():
        rule = rules.get(key)
        if rule is None:
            raise ValueError(f"Unsupported option: {key}")
        if rule == "grid":
            if not isinstance(value, str):
                raise ValueError("Batch grid must be comma-separated integers")
            try:
                grid = [int(n) for n in value.split(",")]
            except ValueError:
                raise ValueError("Invalid batch grid")
            if (
                not grid
                or any(n <= 0 or n > 4096 for n in grid)
                or any(a >= b for a, b in zip(grid, grid[1:]))
            ):
                raise ValueError("Batch grid must be positive and increasing")
        elif isinstance(rule, tuple):
            if value not in rule:
                raise ValueError(f"Invalid {key}")
        elif rule is bool:
            if not isinstance(value, bool):
                raise ValueError(f"Invalid Boolean {key}")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (rule is int and not isinstance(value, int))
        ):
            raise ValueError(f"Invalid numeric {key}")
        elif not 0 <= value <= 10**9:
            raise ValueError(f"{key} is out of range")
    if options.get("device") not in ("cpu", "cuda", "mps"):
        raise ValueError("Choose an explicit device so its resources can be reserved")
    for key in (
        "epochs",
        "producers",
        "requests",
        "batch-size",
        "games-per-epoch",
        "games",
        "simulations",
        "games-parallelism",
        "inference-batch-size",
        "training-batch-size",
        "replay-positions",
    ):
        if key in options and options[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if "top-p" in options and not 0 < options["top-p"] <= 1:
        raise ValueError("top-p must be in (0,1]")
    if "validation-fraction" in options and not 0 < options["validation-fraction"] < 1:
        raise ValueError("validation-fraction must be in (0,1)")
    if "learning-rate" in options and options["learning-rate"] <= 0:
        raise ValueError("learning-rate must be positive")
    inputs = spec.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be an object")
    allowed = (
        {"checkpoint"}
        if kind in ("self_play", "benchmark_executor", "benchmark_self_play")
        else (
            {"history"}
            if kind == "reconstruction"
            else {"replay"}
            if kind == "replay_train"
            else {"first", "second"}
        )
    )
    if kind == "replay_train":
        if (
            "replay" not in inputs
            or len(inputs) > 100
            or any(not re.fullmatch(r"replay(_[1-9][0-9]*)?", key) for key in inputs)
        ):
            raise ValueError(
                "Replay inputs must be named replay, replay_2, ... (at most 100)"
            )
    elif set(inputs) - allowed or (kind != "self_play" and set(inputs) != allowed):
        raise ValueError("Invalid input artifacts")
    if kind == "reconstruction" and (
        not options.get("replay-history-epochs")
        or not isinstance(inputs.get("history"), str)
    ):
        raise ValueError(
            "Reconstruction requires a registered history and a positive replay-history-epochs count"
        )
    for value in inputs.values():
        if isinstance(value, str) and value:
            continue
        if (
            not isinstance(value, dict)
            or set(value) != {"job_id", "selection"}
            or not isinstance(value["job_id"], str)
            or value["selection"] not in ("latest", "best_value_validation")
        ):
            raise ValueError(
                "Inputs must be artifact IDs or checkpoint selections from a job"
            )
    resources = spec.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("resources must be an object")
    if set(resources) != {"slots", "host_memory_mb", "gpu_memory_mb"}:
        raise ValueError(
            "Explicit slots, host_memory_mb and gpu_memory_mb reservations are required"
        )
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0
            for v in resources.values()
        )
        or resources["slots"] < 1
        or resources["host_memory_mb"] < 1
    ):
        raise ValueError("Invalid resource reservation")
    if options["device"] != "cpu" and resources["gpu_memory_mb"] <= 0:
        raise ValueError("Accelerator jobs require a positive GPU memory reservation")
    normalized = {
        "kind": kind,
        "options": options,
        "inputs": inputs,
        "resources": resources,
    }
    if "binary_artifact" in spec:
        if not isinstance(spec["binary_artifact"], str) or not spec["binary_artifact"]:
            raise ValueError("binary_artifact must name a registered executable")
        normalized["binary_artifact"] = spec["binary_artifact"]
    return normalized


def compile_argv(spec, binary, output, resolve_artifact):
    validate_spec(spec)
    args = [str(binary), COMMANDS[spec["kind"]]]
    if spec["kind"].startswith("benchmark_"):
        args.append("executor" if spec["kind"] == "benchmark_executor" else "self-play")
    for key, value in sorted(spec["options"].items()):
        if key == "disable-tf32":
            continue  # Applied by the attempt owner before loading the native process.
        if isinstance(value, bool):
            if value:
                args.append("--" + key)
        else:
            args += ["--" + key, str(value)]
    if (
        spec["kind"] in ("self_play", "reconstruction", "comparison")
        and "heartbeat-seconds" not in spec["options"]
    ):
        args += ["--heartbeat-seconds", "5"]
    # Native command-specific paths are server-owned.
    if spec["kind"].startswith("benchmark_"):
        args += [
            "--checkpoint"
            if spec["kind"] == "benchmark_executor"
            else "--checkpoint-dir",
            str(resolve_artifact(spec["inputs"]["checkpoint"])),
            "--output",
            str(Path(output) / "result.json"),
        ]
    elif spec["kind"] == "comparison":
        args += [
            "--first-checkpoint-dir",
            str(resolve_artifact(spec["inputs"]["first"])),
            "--second-checkpoint-dir",
            str(resolve_artifact(spec["inputs"]["second"])),
            "--output",
            str(Path(output) / "result.json"),
        ]
    elif spec["kind"] == "replay_train":
        args += ["--run-dir", str(output)]
        for name in sorted(spec["inputs"]):
            args += [
                "--replay-checkpoint-dir",
                str(resolve_artifact(spec["inputs"][name])),
            ]
    else:
        args += [
            "--checkpoint-dir",
            str(Path(output) / "checkpoints"),
            "--games-dir",
            str(Path(output) / "games"),
            "--stats-dir",
            str(Path(output) / "stats"),
        ]
    if spec["kind"] == "reconstruction":
        args += [
            "--replay-history-dir",
            str(resolve_artifact(spec["inputs"]["history"])),
        ]
    return args


def environment_overrides(spec):
    validate_spec(spec)
    return {"NVIDIA_TF32_OVERRIDE": "0"} if spec["options"].get("disable-tf32") else {}
