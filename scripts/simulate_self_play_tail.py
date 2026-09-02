#!/usr/bin/env python3
"""Bootstrap self-play makespans from an empirical game-length distribution."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
PROGRESS = re.compile(
    r'self-play progress .*update="completion" .*games_completed=(\d+) '
    r".*games_total=(\d+) .*elapsed_seconds=([0-9.]+)"
)
COMPLETE = re.compile(
    r"self-play complete epoch=(\d+) .*self_play_seconds=([0-9.]+)"
)
COMPLETION_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


@dataclass(frozen=True)
class Scenario:
    name: str
    games: int
    concurrency: int


def parse_scenario(value: str) -> Scenario:
    try:
        name, games, concurrency = value.split(":", 2)
        scenario = Scenario(name, int(games), int(concurrency))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME:GAMES:CONCURRENCY") from error
    if not name or scenario.games <= 0 or scenario.concurrency <= 0:
        raise argparse.ArgumentTypeError("scenario values must be non-empty and positive")
    return scenario


def parse_epoch_range(value: str) -> set[int]:
    try:
        first, last = (int(part) for part in value.split("-", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected FIRST-LAST") from error
    if first < 0 or last < first:
        raise argparse.ArgumentTypeError("invalid epoch range")
    return set(range(first, last + 1))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p10": percentile(values, 0.1),
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
    }


def completion_key(fraction: float) -> str:
    return f"t{round(fraction * 100):02d}"


def completion_time(completions: list[float], fraction: float) -> float:
    index = max(0, math.ceil(len(completions) * fraction) - 1)
    return completions[index]


def simulate(
    game_lengths: list[float], concurrency: int, global_capacity: float
) -> dict[str, float]:
    """Simulate equal processor sharing with a per-game progress-rate cap of one."""
    if not game_lengths or any(length <= 0 for length in game_lengths):
        raise ValueError("game lengths must be positive")
    if concurrency <= 0 or global_capacity <= 0:
        raise ValueError("concurrency and global capacity must be positive")

    concurrency = min(concurrency, len(game_lengths))
    virtual_progress = 0.0
    elapsed = 0.0
    next_game = concurrency
    finish_thresholds = [float(length) for length in game_lengths[:concurrency]]
    heapq.heapify(finish_thresholds)
    completions: list[float] = []

    while finish_thresholds:
        active_games = len(finish_thresholds)
        next_threshold = finish_thresholds[0]
        progress = next_threshold - virtual_progress
        per_game_rate = min(1.0, global_capacity / active_games)
        elapsed += progress / per_game_rate
        virtual_progress = next_threshold

        finished = 0
        while finish_thresholds and finish_thresholds[0] == next_threshold:
            heapq.heappop(finish_thresholds)
            finished += 1
        completions.extend([elapsed] * finished)

        for _ in range(finished):
            if next_game == len(game_lengths):
                break
            heapq.heappush(
                finish_thresholds,
                virtual_progress + game_lengths[next_game],
            )
            next_game += 1

    makespan = completions[-1]
    total_work = float(sum(game_lengths))
    metrics = {
        "makespan": makespan,
        "tail_after_80": makespan - completion_time(completions, 0.8),
        "tail_after_80_fraction": (
            makespan - completion_time(completions, 0.8)
        )
        / makespan,
        "capacity_utilization": total_work / (global_capacity * makespan),
        "total_work": total_work,
    }
    for fraction in COMPLETION_FRACTIONS:
        time = completion_time(completions, fraction)
        key = completion_key(fraction)
        metrics[key] = time
        metrics[f"{key}_fraction"] = time / makespan
    return metrics


def interpolate_milestone(
    milestones: dict[int, float], total_games: int, fraction: float
) -> float:
    target = total_games * fraction
    points = [(0, 0.0), *sorted(milestones.items())]
    if target <= 0:
        return 0.0
    for (lower_games, lower_time), (upper_games, upper_time) in zip(points, points[1:]):
        if target <= upper_games:
            if upper_games == lower_games:
                return upper_time
            weight = (target - lower_games) / (upper_games - lower_games)
            return lower_time + weight * (upper_time - lower_time)
    raise ValueError(f"completion log stops before target {target:g}/{total_games}")


def load_observed_curves(path: Path, epochs: set[int]) -> list[dict[str, float]]:
    current: dict[int, float] = {}
    curves = []
    with path.open(errors="replace") as stream:
        for raw_line in stream:
            line = ANSI_ESCAPE.sub("", raw_line)
            if match := PROGRESS.search(line):
                completed, total, elapsed = match.groups()
                current[int(completed)] = float(elapsed)
                current[-1] = float(total)
                continue
            if not (match := COMPLETE.search(line)):
                continue
            epoch = int(match.group(1))
            makespan = float(match.group(2))
            total_games = int(current.get(-1, 0))
            if epoch in epochs:
                if not total_games:
                    raise ValueError(f"missing progress milestones for epoch {epoch}")
                metrics: dict[str, float] = {
                    "epoch": float(epoch),
                    "makespan": makespan,
                }
                for fraction in COMPLETION_FRACTIONS:
                    key = completion_key(fraction)
                    time = interpolate_milestone(current, total_games, fraction)
                    metrics[key] = time
                    metrics[f"{key}_fraction"] = time / makespan
                metrics["tail_after_80"] = makespan - metrics["t80"]
                metrics["tail_after_80_fraction"] = (
                    metrics["tail_after_80"] / makespan
                )
                curves.append(metrics)
            current = {}
    missing = epochs - {int(curve["epoch"]) for curve in curves}
    if missing:
        raise ValueError(f"observed log is missing epochs: {sorted(missing)}")
    return curves


def summarize_trials(
    metrics: list[dict[str, float]], games: int, seconds_per_unit: float
) -> dict[str, Any]:
    time_keys = [
        "makespan",
        "tail_after_80",
        *(completion_key(fraction) for fraction in COMPLETION_FRACTIONS),
    ]
    result: dict[str, Any] = {
        f"{key}_seconds": summary(
            [trial[key] * seconds_per_unit for trial in metrics]
        )
        for key in time_keys
    }
    result.update(
        {
            "tail_after_80_fraction": summary(
                [trial["tail_after_80_fraction"] for trial in metrics]
            ),
            "capacity_utilization": summary(
                [trial["capacity_utilization"] for trial in metrics]
            ),
            "games_per_hour": summary(
                [games * 3600.0 / (trial["makespan"] * seconds_per_unit) for trial in metrics]
            ),
        }
    )
    return result


def summarize_observed(curves: list[dict[str, float]], games: int) -> dict[str, Any]:
    time_keys = [
        "makespan",
        "tail_after_80",
        *(completion_key(fraction) for fraction in COMPLETION_FRACTIONS),
    ]
    result: dict[str, Any] = {
        "epochs": [int(curve["epoch"]) for curve in curves],
        **{
            f"{key}_seconds": summary([curve[key] for curve in curves])
            for key in time_keys
        },
        "tail_after_80_fraction": summary(
            [curve["tail_after_80_fraction"] for curve in curves]
        ),
        "games_per_hour": summary(
            [games * 3600.0 / curve["makespan"] for curve in curves]
        ),
    }
    return result


def length_distribution(lengths: list[int]) -> dict[str, Any]:
    ordered = sorted(lengths)
    total = sum(ordered)
    top_20_start = math.floor(len(ordered) * 0.8)
    return {
        "games": len(ordered),
        "positions": total,
        "mean": statistics.fmean(ordered),
        "minimum": ordered[0],
        "p50": percentile([float(value) for value in ordered], 0.5),
        "p80": percentile([float(value) for value in ordered], 0.8),
        "p90": percentile([float(value) for value in ordered], 0.9),
        "p95": percentile([float(value) for value in ordered], 0.95),
        "p99": percentile([float(value) for value in ordered], 0.99),
        "maximum": ordered[-1],
        "over_50": sum(length > 50 for length in ordered),
        "over_100": sum(length > 100 for length in ordered),
        "top_20_percent_work_fraction": sum(ordered[top_20_start:]) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", required=True, type=Path)
    parser.add_argument(
        "--last-games",
        type=int,
        help="Use only this many games from the end of the replay deque.",
    )
    parser.add_argument("--observed-log", required=True, type=Path)
    parser.add_argument("--observed-epochs", required=True, type=parse_epoch_range)
    parser.add_argument("--baseline", default="g700-p500")
    parser.add_argument("--global-capacity", type=float, default=256.0)
    parser.add_argument(
        "--length-exponent",
        type=float,
        default=1.0,
        help="Convert plies to effective work as plies raised to this exponent.",
    )
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--scenario", action="append", type=parse_scenario)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    scenarios = args.scenario or [
        Scenario("g700-p500", 700, 500),
        Scenario("g700-p700", 700, 700),
        Scenario("g1500-p500", 1500, 500),
        Scenario("g1500-p700", 1500, 700),
    ]
    if args.trials <= 0 or args.global_capacity <= 0 or args.length_exponent <= 0:
        parser.error("trials, global capacity, and length exponent must be positive")
    if len({scenario.name for scenario in scenarios}) != len(scenarios):
        parser.error("scenario names must be unique")
    scenario_by_name = {scenario.name: scenario for scenario in scenarios}
    if args.baseline not in scenario_by_name:
        parser.error("baseline must name one of the scenarios")

    with args.lengths.open() as stream:
        replay = json.load(stream)
    lengths = [int(value) for value in replay["game_lengths"]]
    if args.last_games is not None:
        if args.last_games <= 0 or args.last_games > len(lengths):
            parser.error("last-games must fit within the replay")
        lengths = lengths[-args.last_games :]
    if not lengths or any(length <= 0 for length in lengths):
        parser.error("length input contains no usable games")

    observed_curves = load_observed_curves(args.observed_log, args.observed_epochs)
    baseline = scenario_by_name[args.baseline]
    if any(int(curve["epoch"]) < 0 for curve in observed_curves):
        parser.error("invalid observed epoch")

    rng = random.Random(args.seed)
    max_games = max(scenario.games for scenario in scenarios)
    trials: dict[str, list[dict[str, float]]] = {
        scenario.name: [] for scenario in scenarios
    }
    for _ in range(args.trials):
        sampled = [
            float(length) ** args.length_exponent
            for length in rng.choices(lengths, k=max_games)
        ]
        for scenario in scenarios:
            trials[scenario.name].append(
                simulate(
                    sampled[: scenario.games],
                    scenario.concurrency,
                    args.global_capacity,
                )
            )

    observed = summarize_observed(observed_curves, baseline.games)
    baseline_unit_makespan = summary(
        [trial["makespan"] for trial in trials[baseline.name]]
    )["median"]
    seconds_per_unit = (
        observed["makespan_seconds"]["median"] / baseline_unit_makespan
    )

    scenario_results = {}
    baseline_trials = trials[baseline.name]
    for scenario in scenarios:
        result = summarize_trials(
            trials[scenario.name], scenario.games, seconds_per_unit
        )
        result["games"] = scenario.games
        result["concurrency"] = scenario.concurrency
        result["throughput_speedup_vs_baseline"] = summary(
            [
                (scenario.games / trial["makespan"])
                / (baseline.games / baseline_trial["makespan"])
                for trial, baseline_trial in zip(
                    trials[scenario.name], baseline_trials
                )
            ]
        )
        scenario_results[scenario.name] = result

    simulated_baseline = scenario_results[baseline.name]
    validation = {}
    for key in (
        "t50_seconds",
        "t80_seconds",
        "t90_seconds",
        "t95_seconds",
        "t99_seconds",
        "makespan_seconds",
        "tail_after_80_seconds",
        "tail_after_80_fraction",
    ):
        observed_median = observed[key]["median"]
        simulated_median = simulated_baseline[key]["median"]
        validation[key] = {
            "observed_median": observed_median,
            "simulated_median": simulated_median,
            "relative_error": (simulated_median - observed_median) / observed_median,
        }

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "lengths": str(args.lengths),
            "last_games": args.last_games,
            "observed_log": str(args.observed_log),
            "observed_epochs": sorted(args.observed_epochs),
        },
        "game_length_distribution": length_distribution(lengths),
        "model": {
            "description": (
                "All active games receive equal progress. Per-game speed is "
                "min(1, global_capacity / active_games), and completed slots "
                "are immediately refilled while queued games remain."
            ),
            "global_capacity": args.global_capacity,
            "effective_work": f"plies ** {args.length_exponent:g}",
            "length_exponent": args.length_exponent,
            "trials": args.trials,
            "seed": args.seed,
        },
        "calibration": {
            "baseline": baseline.name,
            "seconds_per_work_unit": seconds_per_unit,
            "method": "match simulated and observed median baseline makespan",
        },
        "observed_baseline": observed,
        "validation": validation,
        "scenarios": scenario_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")

    print("scenario       games  parallel  median s  tail80 %  games/h  speedup")
    for scenario in scenarios:
        scenario_result = scenario_results[scenario.name]
        print(
            f"{scenario.name:13} {scenario.games:5d} {scenario.concurrency:9d} "
            f"{scenario_result['makespan_seconds']['median']:9.1f} "
            f"{100 * scenario_result['tail_after_80_fraction']['median']:8.1f} "
            f"{scenario_result['games_per_hour']['median']:8.1f} "
            f"{scenario_result['throughput_speedup_vs_baseline']['median']:7.3f}"
        )


if __name__ == "__main__":
    main()
