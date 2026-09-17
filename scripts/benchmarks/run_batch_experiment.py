"""Submit BATCH-001 as typed, sequential jobs and retain a reproducible trial manifest.

Run on the service host. This tool does not execute workers; the service owns them.
Rerunning with the same experiment ID reuses the acknowledged job commands.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from scripts.job_service.store import Store
from scripts.job_service.worker import atomic_json

GRIDS = {
    "unpadded": None,
    "coarse": "1,2,4,8,16,32,64,128,256",
    "dense": "1,2,4,8,16,32,48,64,80,96,112,128,160,192,224,256",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Registered exact checkpoint artifact ID"
    )
    parser.add_argument(
        "--baseline-binary",
        required=True,
        help="Trusted registered old executable artifact ID",
    )
    parser.add_argument("--experiment-id", default="BATCH-001-20260912")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Wait for an existing, explicitly amended trial manifest",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    store = Store(root / "jobs.sqlite3")
    directory = root / "experiments" / args.experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    if args.resume:
        manifest = json.loads((directory / "trials.json").read_text())
        if (
            manifest["checkpoint"] != args.checkpoint
            or manifest["baseline_binary"] != args.baseline_binary
        ):
            raise ValueError("Cannot resume a different benchmark input")
        wait_trials(store, directory, manifest["trials"])
        return
    trials = []
    previous = None

    def add(name, kind, options, arm, repeat, workload, legacy=False, profile=False):
        nonlocal previous
        if kind == "benchmark_executor":
            options = {**options, "disable-tf32": True}
        spec = {
            "kind": kind,
            "options": {"device": "cuda", **options},
            "inputs": {"first": args.checkpoint, "second": args.checkpoint}
            if kind == "comparison"
            else {"checkpoint": args.checkpoint},
            "resources": {
                "slots": 1,
                "host_memory_mb": 40000 if kind == "comparison" else 8192,
                "gpu_memory_mb": 8192,
            },
        }
        if legacy:
            spec["binary_artifact"] = args.baseline_binary
        request = {
            "action": "create",
            "title": name,
            "experiment_id": args.experiment_id,
            "experiment_title": "BATCH-001 · inference batching",
            "spec": spec,
            "dependencies": [previous] if previous else [],
        }
        identifier = store.command(
            args.experiment_id + ":" + str(len(trials)), request
        )["job_id"]
        previous = identifier
        trials.append(
            {
                "job_id": identifier,
                "arm": arm,
                "repeat": repeat,
                "workload": workload,
                "profile": profile,
                "spec": spec,
            }
        )

    add(
        "BATCH-001 · CUDA padding preflight",
        "benchmark_executor",
        {
            "producers": 3,
            "batch-size": 4,
            "requests": 16,
            "grid": "1,2,4",
            "tail": True,
            "reacquire": True,
        },
        "coarse",
        0,
        "preflight",
    )
    for repeat in range(3):
        arms = ["task_count", *GRIDS]
        arms = arms[repeat:] + arms[:repeat]
        for workload in ("steady", "tail", "demo"):
            for arm in arms:
                options = {
                    "producers": 2 if workload == "demo" else 400,
                    "batch-size": 256,
                    "requests": 512 if workload == "demo" else 3000,
                    "timeout-us": 1000,
                    "tail": workload != "steady",
                    "reacquire": workload == "demo",
                    "seed": 0,
                }
                if arm == "task_count":
                    options["task-count-baseline"] = True
                elif GRIDS[arm]:
                    options["grid"] = GRIDS[arm]
                add(
                    f"BATCH-001 · {workload} · {arm} · repeat {repeat + 1}",
                    "benchmark_executor",
                    options,
                    arm,
                    repeat,
                    workload,
                )
    # Closed-loop, pinned weights; compares old task counting with the new controller,
    # and exercises separate per-checkpoint executors. Same seed within each repeat.
    for repeat in range(3):
        arms = ["legacy", "unpadded", "coarse", "dense"]
        arms = arms[repeat:] + arms[:repeat]
        for arm in arms:
            options = {
                "games": 96,
                "simulations": 256,
                "games-parallelism": 100,
                "inference-batch-size": 64,
                "batch-timeout-us": 1000,
                "temperature": 0.7,
                "seed": 12600 + repeat,
                "heartbeat-seconds": 10,
                "no-move-logs": True,
            }
            if arm in GRIDS and GRIDS[arm]:
                options["inference-batch-grid"] = ",".join(
                    n for n in GRIDS[arm].split(",") if int(n) <= 64
                )
            add(
                f"BATCH-001 · MCTS comparison · {arm} · repeat {repeat + 1}",
                "comparison",
                options,
                arm,
                repeat,
                "comparison",
                legacy=arm == "legacy",
            )
    # Separate allocator runs: no probe is loaded in primary timing runs.
    for arm, grid in GRIDS.items():
        options = {
            "producers": 400,
            "batch-size": 256,
            "requests": 3000,
            "tail": True,
            "timeout-us": 1000,
            "profile-allocator": True,
            "seed": 0,
        }
        if grid:
            options["grid"] = grid
        add(
            f"BATCH-001 · allocator profile · {arm}",
            "benchmark_executor",
            options,
            arm,
            0,
            "tail",
            profile=True,
        )
    atomic_json(
        directory / "trials.json",
        {
            "schema_version": 1,
            "checkpoint": args.checkpoint,
            "baseline_binary": args.baseline_binary,
            "trials": trials,
        },
    )
    wait_trials(store, directory, trials)


def wait_trials(store, directory, trials):
    for trial in trials:
        print("Waiting:", trial["job_id"], trial["workload"], trial["arm"], flush=True)
        while True:
            job = store.snapshot(job_id=trial["job_id"])["jobs"][0]
            if job["state"] == "succeeded":
                break
            if job["state"] in ("failed", "paused", "cancelled"):
                atomic_json(
                    directory / "status.json",
                    {"state": "needs_attention", "trial": trial, "job": job},
                )
                raise RuntimeError(f"Trial stopped: {trial['job_id']} ({job['state']})")
            atomic_json(
                directory / "status.json",
                {"state": "running", "trial": trial, "job_state": job["state"]},
            )
            time.sleep(5)
        print("Completed:", trial["job_id"], flush=True)
    atomic_json(
        directory / "status.json", {"state": "completed", "trials": len(trials)}
    )
    print("All trials completed", flush=True)


if __name__ == "__main__":
    main()
