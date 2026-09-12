#!/usr/bin/env python3
"""Build position-weighted histograms without retaining individual replay rows."""
import argparse
import bisect
import json
import math
from pathlib import Path
from analyze_nucleus_replays import nucleus, stage

THRESHOLDS = [0.9, 0.95, 0.98, 0.99, 1.0]


def cuts(policy):
    ranked = sorted(policy, key=lambda item: -item[1])
    total = math.fsum(p for _, p in ranked)
    weights = [p / total for _, p in ranked]
    cumulative = []
    acc = 0.0
    for p in weights:
        acc += p
        cumulative.append(acc)
    negative = [-p for p in weights]
    result = []
    for threshold in THRESHOLDS:
        if threshold == 1:
            kept, mass = len(weights), 0.0
        else:
            boundary = min(bisect.bisect_left(cumulative, threshold - 1e-12), len(weights) - 1)
            kept = bisect.bisect_right(negative, negative[boundary])
            mass = max(0.0, 1.0 - cumulative[kept - 1])
        result.append((kept, mass))
    return ranked, result


def empty():
    return {"positions": 0, "support_before": [0] * 362, "support_after": [0] * 362,
            "removed_count": [0] * 362, "removed_fraction": [0] * 1001,
            "removed_mass": [0] * 1001, "sum_before": 0, "sum_removed": 0,
            "sum_fraction": 0.0, "sum_mass": 0.0, "observed": 0, "excluded": 0,
            "support_joint": {}}


def build(path):
    groups = {str(p): {f"{s}/{seat}": empty() for s in ["01-06", "07-12", "13-20", "21+"]
                       for seat in ["first", "second"]} for p in THRESHOLDS}
    count = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            ranked, results = cuts(row["policy"])
            before = len(ranked)
            chosen_rank = next((i for i, (a, _) in enumerate(ranked) if a == row["chosen"]), None)
            key = f'{stage(row["ply"])}/{"first" if row["ply"] % 2 == 0 else "second"}'
            for threshold, (kept, mass) in zip(THRESHOLDS, results):
                g = groups[str(threshold)][key]
                removed = before - kept
                fraction = removed / before
                g["positions"] += 1
                g["support_before"][before] += 1
                g["support_after"][kept] += 1
                pair = before * 362 + kept
                g["support_joint"][pair] = g["support_joint"].get(pair, 0) + 1
                g["removed_count"][removed] += 1
                g["removed_fraction"][min(1000, int(fraction * 1000))] += 1
                g["removed_mass"][min(1000, int(mass * 10000))] += 1
                g["sum_before"] += before
                g["sum_removed"] += removed
                g["sum_fraction"] += fraction
                g["sum_mass"] += mass
                if chosen_rank is not None:
                    g["observed"] += 1
                    g["excluded"] += chosen_rank >= kept
            # Cross-check the optimized cuts against the original audited filter.
            if count < 100 or count % 10000 == 0:
                for threshold, (kept, mass) in zip(THRESHOLDS, results):
                    _, reference_kept, reference_mass, _ = nucleus(row["policy"], threshold)
                    assert len(reference_kept) == kept
                    assert abs(reference_mass - mass) < 1e-10
            count += 1
            if count % 50000 == 0:
                print(f"Processed {count:,} positions", flush=True)
    for cells in groups.values():
        for g in cells.values():
            g["support_joint"] = [[key // 362, key % 362, n] for key, n in sorted(g["support_joint"].items())]
    return {"schema_version": 2, "positions": count, "thresholds": groups,
            "joint_definition": "support_joint triples are [nonzero moves before, nonzero moves retained, position count]; exact integer pairs, not inferred from marginal histograms",
            "bins": {"support_before": "exact integer count", "support_after": "exact integer count",
                     "removed_count": "exact integer count", "removed_fraction": "floor(fraction * 1000), width 0.1 percentage point",
                     "removed_mass": "floor(mass * 10000), width 0.01 percentage point"},
            "weighting": "Each stored position counts once, before symmetry augmentation; stage/seat groups are disjoint."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input)
    old = json.loads(args.summary.read_text())
    for p, groups in result["thresholds"].items():
        reference = old["thresholds"][p]
        total = sum(g["positions"] for g in groups.values())
        assert total == reference["positions"]
        assert sum(g["excluded"] for g in groups.values()) == reference["excluded_actions"]
        assert sum(g["observed"] for g in groups.values()) == reference["observed_actions"]
        assert abs(sum(g["sum_mass"] for g in groups.values()) / total - reference["removed_mass"]["mean"]) < 1e-10
        assert abs(sum(g["sum_before"] for g in groups.values()) / total - reference["support_before"]["mean"]) < 1e-10
        for g in groups.values():
            for field in result["bins"]:
                assert sum(g[field]) == g["positions"]
            before, after, removed = [0] * 362, [0] * 362, [0] * 362
            for b, a, n in g["support_joint"]:
                assert 1 <= a <= b <= 361 and n > 0
                before[b] += n
                after[a] += n
                removed[b-a] += n
                if p == "1.0":
                    assert a == b
            assert before == g["support_before"]
            assert after == g["support_after"]
            assert removed == g["removed_count"]
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
