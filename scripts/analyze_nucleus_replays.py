#!/usr/bin/env python3
"""Audit top-p sampling on exported replay distributions; never alter training data."""
import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path


def nucleus(policy, threshold):
    """Retain the smallest ranked prefix reaching p, extended to include exact ties."""
    total = math.fsum(p for _, p in policy)
    if total <= 0 or not 0 < threshold <= 1:
        raise ValueError("expected a positive distribution and 0 < top-p <= 1")
    ranked = sorted(((a, p / total) for a, p in policy if p > 0), key=lambda item: -item[1])
    if threshold == 1:
        return ranked, {a for a, _ in ranked}, 0.0, [p for _, p in ranked]
    cumulative, cutoff = 0.0, 0.0
    for _, weight in ranked:
        cumulative += weight
        cutoff = weight
        if cumulative >= threshold or math.isclose(cumulative, threshold, abs_tol=1e-12, rel_tol=0):
            break
    kept = {a for a, weight in ranked if weight >= cutoff}
    removed = math.fsum(weight for a, weight in ranked if a not in kept)
    filtered = [weight / (1 - removed) for a, weight in ranked if a in kept]
    return ranked, kept, removed, filtered


def describe(values):
    values = sorted(values)
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "p90": values[max(0, math.ceil(0.9 * len(values)) - 1)], "max": values[-1]}


class Aggregate:
    def __init__(self):
        self.removed, self.before, self.after, self.entropy_before, self.entropy_after = [], [], [], [], []
        self.actions = self.excluded = self.expected_excluded = 0
        self.games, self.excluded_games = set(), set()

    def add(self, row, ranked, kept, removed, filtered):
        self.removed.append(removed)
        self.before.append(len(ranked))
        self.after.append(len(kept))
        self.entropy_before.append(-math.fsum(p * math.log(p) for _, p in ranked))
        self.entropy_after.append(-math.fsum(p * math.log(p) for p in filtered))
        self.games.add(row["game"])
        if row["chosen"] is not None:
            self.actions += 1
            self.expected_excluded += removed
            if row["chosen"] not in kept:
                self.excluded += 1
                self.excluded_games.add(row["game"])

    def result(self):
        return {"positions": len(self.before), "games": len(self.games),
                "removed_mass": describe(self.removed), "support_before": describe(self.before),
                "support_after": describe(self.after), "entropy_before_nats": describe(self.entropy_before),
                "entropy_after_nats": describe(self.entropy_after), "observed_actions": self.actions,
                "excluded_actions": self.excluded,
                "excluded_action_fraction": self.excluded / self.actions if self.actions else None,
                "expected_excluded_actions_at_recorded_positions": self.expected_excluded,
                "games_with_excluded_observed_action": len(self.excluded_games),
                "games_with_excluded_observed_action_fraction": len(self.excluded_games) / len(self.games)}


def stage(ply):
    return "01-06" if ply < 6 else "07-12" if ply < 12 else "13-20" if ply < 20 else "21+"


def exported_rows(path, follow):
    # A live exporter flushes complete JSON lines and writes metadata after closing
    # its data stream. Never consume a partially written line.
    with path.open() as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if line.endswith("\n"):
                yield json.loads(line)
            elif follow and not path.with_suffix(".metadata.json").exists():
                stream.seek(offset)
                time.sleep(1)
            elif line:
                raise ValueError("incomplete final JSON line")
            else:
                return


def analyze(path, follow=False):
    thresholds = [0.90, 0.95, 0.98, 0.99, 1.0]
    aggregates = {p: Aggregate() for p in thresholds}
    stages, seats, outcomes = defaultdict(Aggregate), defaultdict(Aggregate), defaultdict(Aggregate)
    examples = []
    for row in exported_rows(path, follow):
        for threshold, aggregate in aggregates.items():
            ranked, kept, removed, filtered = nucleus(row["policy"], threshold)
            aggregate.add(row, ranked, kept, removed, filtered)
            if threshold == 0.95:
                for group, key in [(stages, stage(row["ply"])), (seats, "first" if row["ply"] % 2 == 0 else "second"),
                                   (outcomes, str(row["value"]))]:
                    group[key].add(row, ranked, kept, removed, filtered)
                if row["chosen"] is not None and row["chosen"] not in kept and len(examples) < 12:
                    examples.append({**row, "policy": ranked, "retained_actions": sorted(kept),
                                     "removed_mass": removed, "chosen_probability": dict(ranked)[row["chosen"]]})
    return {"schema_version": 1, "input": str(path), "policy": "stored distribution, normalized for float rounding only",
            "ties": "include every action exactly tied with the boundary action",
            "limitations": ["No counterfactual outcomes or strength estimates; positions follow original trajectories.",
                            "Actual-action counts exclude each game's final action.",
                            "Archived policies already include sampling temperature; no new temperature applied.",
                            "Positive support counts exclude legal actions that had zero visits.",
                            "Examples are the first 12 excluded actions in input order, not a quality-ranked sample."],
            "thresholds": {str(p): a.result() for p, a in aggregates.items()},
            "top_p_0.95_by_stage": {key: value.result() for key, value in sorted(stages.items())},
            "top_p_0.95_by_seat": {key: value.result() for key, value in sorted(seats.items())},
            "top_p_0.95_by_actor_outcome": {key: value.result() for key, value in sorted(outcomes.items())},
            "examples": examples}


def report(result):
    rows = ["# Nucleus sampling replay audit", "", "Stored, already temperature-adjusted policies; whole games deduplicated across checkpoints.", "",
            "| Top-p | Mean mass removed | Mean supported moves before → after | Recorded actions excluded | Games with an excluded recorded action |",
            "|---|---:|---:|---:|---:|"]
    for p, a in result["thresholds"].items():
        rows.append(f'| {p} | {a["removed_mass"]["mean"]:.2%} | {a["support_before"]["mean"]:.1f} → {a["support_after"]["mean"]:.1f} | '
                    f'{a["excluded_actions"]:,}/{a["observed_actions"]:,} ({a["excluded_action_fraction"]:.2%}) | '
                    f'{a["games_with_excluded_observed_action"]:,}/{a["games"]:,} ({a["games_with_excluded_observed_action_fraction"]:.2%}) |')
    rows += ["", "## Top-p 0.95 by move stage", "", "| Moves | Positions | Mean mass removed | Recorded actions excluded | Median retained moves |",
             "|---|---:|---:|---:|---:|"]
    for stage_name, a in result["top_p_0.95_by_stage"].items():
        rows.append(f'| {stage_name} | {a["positions"]:,} | {a["removed_mass"]["mean"]:.2%} | {a["excluded_action_fraction"]:.2%} | {a["support_after"]["median"]:g} |')
    rows += ["", "## Top-p 0.95 by seat", "", "| Seat | Recorded actions excluded | Exclusion rate |", "|---|---:|---:|"]
    for seat, a in result["top_p_0.95_by_seat"].items():
        rows.append(f'| {seat} | {a["excluded_actions"]:,}/{a["observed_actions"]:,} | {a["excluded_action_fraction"]:.2%} |')
    rows += ["", "## Interpretation", "", *[f"- {note}" for note in result["limitations"]], "",
             "Excluded actions are not established blunders. This measures which historical choices the filter would forbid, not which alternative would be sampled or how the game would end.", "",
             "The JSON report includes seat/outcome breakdowns, entropy and support quantiles, and 12 inspectable examples (row-major action IDs; stones 1=current actor, 2=opponent).", ""]
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--follow-export", action="store_true", help="wait for a running exporter to finish")
    args = parser.parse_args()
    result = analyze(args.input, args.follow_export)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(report(result))
