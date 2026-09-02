#!/usr/bin/env python3
"""Aggregate `alz battle` reports and summarize their saved game replays."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def checkpoint_name(checkpoint: dict[str, Any]) -> str:
    architecture = checkpoint["model"]["architecture"]
    prefix = "kata" if architecture == "kata_v1" else "legacy"
    return f"{prefix}-{checkpoint['epoch']}"


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def wilson(score_rate: float, games: int) -> tuple[float, float]:
    z = 1.959_963_984_540_054
    z_squared = z * z
    denominator = 1.0 + z_squared / games
    center = (score_rate + z_squared / (2.0 * games)) / denominator
    margin = (
        z
        * math.sqrt(
            (score_rate * (1.0 - score_rate) + z_squared / (4.0 * games))
            / games
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def elo(score_rate: float) -> float | None:
    if not 0.0 < score_rate < 1.0:
        return None
    return 400.0 * math.log10(score_rate / (1.0 - score_rate))


def opening_key(game: dict[str, Any], plies: int) -> tuple[tuple[int, int], ...]:
    return tuple((move["row"], move["column"]) for move in game["moves"][:plies])


def load_reports(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    reports = []
    for path in sorted(root.rglob("result.json")):
        with path.open() as stream:
            report = json.load(stream)
        if report.get("schema_version") != 1 or not isinstance(report.get("games"), list):
            continue
        reports.append((path, report))
    if not reports:
        raise SystemExit(f"no battle result.json files found below {root}")
    return reports


def analyze(root: Path) -> dict[str, Any]:
    reports = load_reports(root)
    competitors: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"series": 0, "games": 0, "wins": 0, "losses": 0, "draws": 0}
    )
    series = []
    all_plies: list[int] = []
    all_games: list[dict[str, Any]] = []

    for path, report in reports:
        first = checkpoint_name(report["first_checkpoint"])
        second = checkpoint_name(report["second_checkpoint"])
        games = report["games"]
        plies = [game["plies"] for game in games]
        all_plies.extend(plies)
        all_games.extend(games)

        first_result = report["first_checkpoint_result"]
        second_result = report["second_checkpoint_result"]
        for name, checkpoint, result in (
            (first, report["first_checkpoint"], first_result),
            (second, report["second_checkpoint"], second_result),
        ):
            aggregate = competitors[name]
            aggregate["model"] = name
            aggregate["architecture"] = checkpoint["model"]["architecture"]
            aggregate["epoch"] = checkpoint["epoch"]
            aggregate["series"] += 1
            aggregate["games"] += len(games)
            aggregate["wins"] += result["wins"]
            aggregate["losses"] += result["losses"]
            aggregate["draws"] += result["draws"]

        first_seat_wins = sum(
            game["winner"] is not None and game["winner"] == game["first_seat"]
            for game in games
        )
        second_seat_wins = sum(
            game["winner"] is not None and game["winner"] == game["second_seat"]
            for game in games
        )
        draws = len(games) - first_seat_wins - second_seat_wins
        value_estimates = [
            move["value_estimate"]
            for game in games
            for move in game["moves"]
            if move["value_estimate"] is not None
        ]
        combined_evaluations_per_second = (
            report["first_checkpoint_inference"]["evaluations_per_second"]
            + report["second_checkpoint_inference"]["evaluations_per_second"]
        )
        series.append(
            {
                "path": str(path),
                "first": first,
                "second": second,
                "games": len(games),
                "first_wins": first_result["wins"],
                "draws": first_result["draws"],
                "first_losses": first_result["losses"],
                "first_score": first_result["score"],
                "first_score_rate": first_result["score_rate"],
                "first_elo_difference": first_result["elo_difference"],
                "duration_seconds": report["duration_seconds"],
                "evaluations_per_second": combined_evaluations_per_second,
                "plies": {
                    "minimum": min(plies),
                    "mean": statistics.fmean(plies),
                    "median": statistics.median(plies),
                    "p90": percentile(plies, 0.90),
                    "maximum": max(plies),
                },
                "seat_outcomes": {
                    "first_wins": first_seat_wins,
                    "second_wins": second_seat_wins,
                    "draws": draws,
                },
                "unique_openings": {
                    "four_plies": len({opening_key(game, 4) for game in games}),
                    "eight_plies": len({opening_key(game, 8) for game in games}),
                },
                "value_estimates": {
                    "count": len(value_estimates),
                    "mean": statistics.fmean(value_estimates),
                    "mean_absolute": statistics.fmean(map(abs, value_estimates)),
                },
            }
        )

    ranking = []
    for aggregate in competitors.values():
        score = aggregate["wins"] + aggregate["draws"] * 0.5
        score_rate = score / aggregate["games"]
        low, high = wilson(score_rate, aggregate["games"])
        aggregate.update(
            score=score,
            score_rate=score_rate,
            score_rate_95_percent_low=low,
            score_rate_95_percent_high=high,
            elo_difference=elo(score_rate),
        )
        ranking.append(aggregate)
    ranking.sort(key=lambda row: (row["score_rate"], row["wins"], row["epoch"]), reverse=True)

    first_seat_wins = sum(
        game["winner"] is not None and game["winner"] == game["first_seat"]
        for game in all_games
    )
    second_seat_wins = sum(
        game["winner"] is not None and game["winner"] == game["second_seat"]
        for game in all_games
    )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "report_files": len(reports),
        "games": len(all_games),
        "ranking": ranking,
        "series": series,
        "replays": {
            "plies": {
                "minimum": min(all_plies),
                "mean": statistics.fmean(all_plies),
                "median": statistics.median(all_plies),
                "p90": percentile(all_plies, 0.90),
                "maximum": max(all_plies),
            },
            "seat_outcomes": {
                "first_wins": first_seat_wins,
                "second_wins": second_seat_wins,
                "draws": len(all_games) - first_seat_wins - second_seat_wins,
            },
            "unique_openings": {
                "four_plies": len({opening_key(game, 4) for game in all_games}),
                "eight_plies": len({opening_key(game, 8) for game in all_games}),
            },
        },
    }


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint battle and replay report",
        "",
        f"Analyzed {summary['games']:,} games from {summary['report_files']} series.",
        "",
        "## Aggregate checkpoint results",
        "",
        "| Rank | Checkpoint | Series | Games | W-D-L | Score | Score rate | 95% CI | Aggregate Elo |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(summary["ranking"], 1):
        elo_value = row["elo_difference"]
        elo_text = "n/a" if elo_value is None else f"{elo_value:+.1f}"
        lines.append(
            f"| {rank} | {row['model']} | {row['series']} | {row['games']} | "
            f"{row['wins']}-{row['draws']}-{row['losses']} | {row['score']:.1f} | "
            f"{percentage(row['score_rate'])} | "
            f"{percentage(row['score_rate_95_percent_low'])}–{percentage(row['score_rate_95_percent_high'])} | "
            f"{elo_text} |"
        )

    lines.extend(
        [
            "",
            "## Series",
            "",
            "The W-D-L and score columns are from the first checkpoint's perspective.",
            "",
            "| Match | W-D-L | Score | Elo | Seconds | eval/s | Plies mean / p50 / p90 / max | Openings unique at 4 / 8 plies |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["series"]:
        elo_value = row["first_elo_difference"]
        elo_text = "n/a" if elo_value is None else f"{elo_value:+.1f}"
        plies = row["plies"]
        openings = row["unique_openings"]
        lines.append(
            f"| {row['first']} vs {row['second']} | "
            f"{row['first_wins']}-{row['draws']}-{row['first_losses']} | "
            f"{row['first_score']:.1f}/{row['games']} ({percentage(row['first_score_rate'])}) | "
            f"{elo_text} | {row['duration_seconds']:.1f} | {row['evaluations_per_second']:.0f} | "
            f"{plies['mean']:.1f} / {plies['median']:.1f} / {plies['p90']} / {plies['maximum']} | "
            f"{openings['four_plies']} / {openings['eight_plies']} |"
        )

    replay = summary["replays"]
    seat = replay["seat_outcomes"]
    openings = replay["unique_openings"]
    plies = replay["plies"]
    lines.extend(
        [
            "",
            "## Replay statistics",
            "",
            f"- Game length: mean {plies['mean']:.1f}, median {plies['median']:.1f}, "
            f"p90 {plies['p90']}, range {plies['minimum']}–{plies['maximum']} plies.",
            f"- Seat outcomes: first player {seat['first_wins']} wins, second player "
            f"{seat['second_wins']} wins, {seat['draws']} draws.",
            f"- Opening diversity: {openings['four_plies']} unique four-ply sequences and "
            f"{openings['eight_plies']} unique eight-ply sequences.",
            "- The per-series `result.json` files contain every move and network value estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    summary = analyze(args.root.resolve())
    rendered_json = json.dumps(summary, indent=2) + "\n"
    rendered_markdown = markdown(summary)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered_json)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered_markdown)
    if not args.json_output and not args.markdown_output:
        print(rendered_json, end="")


if __name__ == "__main__":
    main()
