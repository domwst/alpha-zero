import unittest
import json
import tempfile
from pathlib import Path
from analyze_nucleus_replays import analyze, nucleus


class NucleusTests(unittest.TestCase):
    def test_known_tail_and_boundary(self):
        _, kept, removed, filtered = nucleus([(0, 0.6), (1, 0.35), (2, 0.03), (3, 0.02)], 0.95)
        self.assertEqual(kept, {0, 1})
        self.assertAlmostEqual(removed, 0.05)
        self.assertAlmostEqual(sum(filtered), 1)

    def test_ties_do_not_choose_arbitrary_coordinates(self):
        policy = [(i, 0.1) for i in range(10)]
        self.assertEqual(nucleus(policy, 0.95)[1], set(range(10)))
        self.assertEqual(nucleus(list(reversed(policy)), 0.95)[1], set(range(10)))

    def test_disabled_filter_preserves_all_positive_support(self):
        policy = [(0, 0.9999), (1, 0.0001), (2, 0)]
        self.assertEqual(nucleus(policy, 1.0)[1], {0, 1})
        self.assertEqual(nucleus(policy, 1.0)[2], 0)

    def test_audit_counts_observed_actions_separately_from_final_positions(self):
        rows = [{"game": "a", "ply": 0, "value": 1, "chosen": 1, "policy": [[0, 0.96], [1, 0.04]]},
                {"game": "a", "ply": 1, "value": -1, "chosen": None, "policy": [[0, 1.0]]},
                {"game": "b", "ply": 0, "value": 1, "chosen": 0, "policy": [[0, 1.0]]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = analyze(path)["thresholds"]["0.95"]
        self.assertEqual(result["positions"], 3)
        self.assertEqual(result["observed_actions"], 2)
        self.assertEqual(result["excluded_actions"], 1)
        self.assertEqual(result["games_with_excluded_observed_action"], 1)
        self.assertAlmostEqual(result["expected_excluded_actions_at_recorded_positions"], 0.04)


if __name__ == "__main__":
    unittest.main()
