"""Summary and event-history feeds don't require the browser to replay diagnostic logs."""

from pathlib import Path
import tempfile
import unittest
from scripts.job_service.store import Store
from scripts.job_service.summaries import summarize


class SummaryTests(unittest.TestCase):
    def test_resume_deduplicates_games_and_preserves_original_finish_times(self):
        def game(n, plies, time):
            return {
                "time": time,
                "kind": "comparison_game",
                "payload": {
                    "game": n,
                    "plies": plies,
                    "first_seat": "first_checkpoint",
                    "winner": "second_checkpoint",
                    "duration_seconds": time - 100,
                },
            }

        events = [
            {"time": 100, "kind": "worker_started", "payload": {}},
            game(1, 26, 102),
            game(2, 28, 104),
            game(3, 40, 106),
            game(4, 44, 108),
            game(1, 26, 110),
            {"time": 111, "kind": "comparison_completed", "payload": {}},
        ]
        stats = summarize(events)["comparison"]
        self.assertEqual(stats["games"], 4)
        self.assertEqual(stats["lengths"]["all"]["median"], 34)
        self.assertEqual(stats["outcomes"]["second_player_wins"], 4)
        self.assertEqual(stats["completion"]["last_finish_seconds"], 8)
        self.assertEqual(stats["completion"]["points"][0]["seconds"], 2)
        self.assertEqual(stats["durations"]["count"], 4)
        self.assertEqual(
            stats["seat_results"]["second_checkpoint"]["second"]["wins"], 4
        )

    def test_saved_aggregates_are_not_replaced_by_a_partial_import(self):
        saved = {"games": 300, "completion": None}
        events = [
            {"kind": "comparison_completed", "payload": {"game_statistics": saved}},
            {
                "kind": "comparison_game",
                "time": 2,
                "payload": {
                    "game": 1,
                    "plies": 9,
                    "first_seat": "first_checkpoint",
                    "winner": None,
                },
            },
        ]
        self.assertEqual(summarize(events)["comparison"], saved)

    def test_summary_ignores_large_diagnostic_history_and_pages_recent_events(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "jobs.sqlite3")
            job = store.command(
                "create",
                {
                    "action": "create",
                    "spec": {
                        "kind": "self_play",
                        "options": {"device": "cpu"},
                        "inputs": {},
                        "resources": {
                            "slots": 1,
                            "host_memory_mb": 64,
                            "gpu_memory_mb": 0,
                        },
                    },
                },
            )["job_id"]
            with store.transaction() as db:
                for _ in range(1200):
                    store.event(db, job, None, "heartbeat", {})
                store.event(
                    db,
                    job,
                    None,
                    "comparison_game",
                    {
                        "game": 1,
                        "plies": 9,
                        "first_seat": "first_checkpoint",
                        "winner": None,
                    },
                )
            summary = store.job_summary(job)
            self.assertEqual(summary["comparison"]["games"], 1)
            self.assertEqual(summary["comparison"]["outcomes"]["draws"], 1)
            page = store.event_page(job)
            self.assertEqual(len(page["events"]), 100)
            self.assertEqual(page["events"][0]["kind"], "comparison_game")
            older = store.event_page(job, page["events"][-1]["id"])
            self.assertTrue(page["has_more"])
            self.assertLess(older["events"][0]["id"], page["events"][-1]["id"])
