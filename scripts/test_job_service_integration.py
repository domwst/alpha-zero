"""Opt-in CPU exercise of real jobs, recovery, cataloging, and checkpoint selection.

ALZ_TEST_BINARY=target/debug/alz OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./run.sh python -m unittest scripts.test_job_service_integration
"""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid

from scripts.job_service.scheduler import Scheduler
from scripts.job_service.store import Store


@unittest.skipUnless(
    os.environ.get("ALZ_TEST_BINARY"), "set ALZ_TEST_BINARY for native CPU integration"
)
class NativeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="alz-service-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = Store(self.root / "jobs.sqlite3")
        self.device = os.environ.get("ALZ_TEST_DEVICE", "cpu")
        self.resources = {
            "slots": 1,
            "host_memory_mb": 4096,
            "gpu_memory_mb": 4096 if self.device == "cuda" else 0,
        }
        self.scheduler = Scheduler(
            self.store,
            self.root,
            Path(os.environ["ALZ_TEST_BINARY"]),
            self.resources,
            Path.cwd(),
        )
        self.addCleanup(self.stop_workers)

    def command(self, request):
        return self.store.command(str(uuid.uuid4()), request)

    def create(self, kind, options, inputs=None):
        return self.command(
            {
                "action": "create",
                "spec": {
                    "kind": kind,
                    "options": {"device": self.device, **options},
                    "inputs": inputs or {},
                    "resources": self.resources,
                },
            }
        )["job_id"]

    def wait_for(self, job_id, state):
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            self.scheduler.tick()
            job = self.store.snapshot(job_id=job_id)["jobs"][0]
            if job["state"] == state:
                return job
            self.assertNotIn(job["state"], ("failed", "cancelled"), job)
            time.sleep(0.2)
        self.fail(f"Job did not reach {state}")

    def stop_workers(self):
        for attempt in self.store.active_attempts():
            self.command(
                {"action": "cancel", "job_id": attempt["job_id"], "mode": "immediate"}
            )
        for _ in range(50):
            if not self.store.active_attempts():
                break
            self.scheduler.tick()
            time.sleep(0.2)
        for child in self.scheduler.children:
            child.wait(timeout=10)

    def test_real_training_and_comparison_recovery(self):
        job = self.create(
            "self_play",
            {
                "epochs": 2,
                "games-per-epoch": 2,
                "simulations": 2,
                "games-parallelism": 2,
                "inference-batch-size": 2,
                "training-batch-size": 64,
                "rendered-games": 0,
                "heartbeat-seconds": 1,
                "seed": 241,
                "top-p": 0.95,
                "inference-batch-grid": "1,2,4",
            },
        )
        self.wait_for(job, "running")
        self.command({"action": "pause", "job_id": job, "mode": "boundary"})
        self.wait_for(job, "paused")
        output = self.root / "jobs" / job
        archive = output / "games" / "archive" / "00000000"
        original = {p.name: p.read_bytes() for p in archive.glob("[0-9]*.json")}
        self.assertEqual(len(original), 2)
        receipt = json.loads((archive / "receipt.json").read_text())
        self.command({"action": "resume", "job_id": job})
        finished = self.wait_for(job, "succeeded")
        self.assertEqual(len(finished["attempts"]), 2)
        self.assertTrue(
            all(
                (archive / name).read_bytes() == data for name, data in original.items()
            )
        )
        stats = json.loads((output / "stats" / "00000000.json").read_text())
        self.assertEqual(
            stats["network"]["requests"], receipt["batch_stats"]["requests"]
        )
        self.assertAlmostEqual(
            stats["self_play_seconds"],
            receipt["duration"]["secs"] + receipt["duration"]["nanos"] / 1e9,
        )

        # Replay training consumes the source buffers through cataloged references.
        reference = {"job_id": job, "selection": "latest"}
        replay = self.create(
            "replay_train",
            {"epochs": 1, "training-batch-size": 64, "seed": 244},
            {"replay": reference},
        )
        self.wait_for(replay, "succeeded")
        battle = self.create(
            "comparison",
            {
                "games": 4,
                "simulations": 2,
                "games-parallelism": 2,
                "inference-batch-size": 2,
                "temperature": 0.7,
                "seed": 242,
            },
            {
                "first": reference,
                "second": {"job_id": replay, "selection": "best_value_validation"},
            },
        )
        self.wait_for(battle, "running")
        self.command({"action": "pause", "job_id": battle, "mode": "boundary"})
        self.wait_for(battle, "paused")
        battle_output = self.root / "jobs" / battle
        saved = {
            p: p.read_bytes()
            for p in (battle_output / "games" / "archive" / "00000000").glob(
                "[0-9]*.json"
            )
        }
        self.assertTrue(saved)
        self.command({"action": "resume", "job_id": battle})
        self.wait_for(battle, "succeeded")
        result = json.loads((battle_output / "result.json").read_text())
        self.assertEqual(len(result["games"]), 4)
        self.assertGreaterEqual(result["recovered_games"], len(saved))
        self.assertTrue(all(p.read_bytes() == data for p, data in saved.items()))
        self.assertTrue(all(g["duration_seconds"] > 0 for g in result["games"]))

        broken = self.create(
            "comparison",
            {
                "games": 100,
                "simulations": 1000,
                "games-parallelism": 2,
                "inference-batch-size": 2,
                "seed": 245,
            },
            {"first": reference, "second": reference},
        )
        broken_output = self.root / "jobs" / broken
        archive = broken_output / "games/archive/00000000"
        archive.mkdir(parents=True)
        (archive / "00000000.json").write_text("invalid saved game")
        self.wait_for(broken, "failed")
        events = self.store.events(job_id=broken)
        self.assertTrue(any(e["kind"] == "native_failed" for e in events))
        self.assertFalse(
            any(
                e["payload"].get("games_completed", 0) > 0
                for e in events
                if e["kind"] == "comparison_progress"
            )
        )
        self.assertFalse((broken_output / "result.json").exists())
