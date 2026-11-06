"""Summarize completed BATCH-001 trials without counting padding as useful work."""

from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from scripts.job_service.store import Store
from scripts.job_service.worker import atomic_json


def summarize(root, experiment_id):
    root = Path(root).resolve()
    directory = root / "experiments" / experiment_id
    manifest = json.loads((directory / "trials.json").read_text())
    store = Store(root / "jobs.sqlite3")
    groups = defaultdict(list)
    trials = []
    for trial in manifest["trials"]:
        job = store.snapshot(job_id=trial["job_id"])["jobs"][0]
        if job["state"] != "succeeded":
            raise ValueError(f"Incomplete trial: {trial['job_id']} ({job['state']})")
        output = root / "jobs" / trial["job_id"]
        result = json.loads((output / "result.json").read_text())
        row = {
            key: trial[key]
            for key in ("job_id", "arm", "repeat", "workload", "profile")
        }
        row["precision"] = (
            "ieee_fp32"
            if trial["spec"]["options"].get("disable-tf32")
            else "production_default"
        )
        row["seconds"] = result["duration_seconds"]
        if trial["workload"] == "comparison":
            row["games_per_hour"] = result["games_per_second"] * 3600
            row["evaluations_per_second"] = sum(
                result[f"{side}_checkpoint_inference"]["evaluations_per_second"]
                for side in ("first", "second")
            )
        else:
            if not result["parity"]["passed"]:
                raise ValueError("Numerical parity failed")
            network = result["network"]
            row.update(
                evaluations_per_second=result["requests_per_second"],
                p50_ms=result["latency_us"]["p50"] / 1000,
                p95_ms=result["latency_us"]["p95"] / 1000,
                p99_ms=result["latency_us"]["p99"] / 1000,
                mean_real_batch=network["requests"] / network["invocations"],
                padding_percent=100
                * network["padded_requests"]
                / (network["requests"] + network["padded_requests"]),
                shape_count=len(network["physical_batch_size_histogram"]),
            )
        if trial["profile"]:
            row["allocator"] = json.loads(
                (output / "result.allocator.json").read_text()
            )
        else:
            groups[(trial["workload"], trial["arm"])].append(row)
        trials.append(row)
    aggregates = []
    for (workload, arm), rows in groups.items():
        values = {}
        for field in (
            "seconds",
            "evaluations_per_second",
            "games_per_hour",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "mean_real_batch",
            "padding_percent",
            "shape_count",
        ):
            samples = [r[field] for r in rows if field in r]
            if samples:
                values[field] = {
                    "mean": statistics.mean(samples),
                    "median": statistics.median(samples),
                    "min": min(samples),
                    "max": max(samples),
                }
        aggregates.append(
            {"workload": workload, "arm": arm, "repeats": len(rows), "metrics": values}
        )
    result = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "checkpoint": manifest["checkpoint"],
        "baseline_binary": manifest["baseline_binary"],
        "aggregates": aggregates,
        "trials": trials,
    }
    atomic_json(directory / "summary.json", result)
    store.artifact(None, None, "benchmark_summary", directory / "summary.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-id", default="BATCH-001-20260912")
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.experiment_id), indent=2))


if __name__ == "__main__":
    main()
