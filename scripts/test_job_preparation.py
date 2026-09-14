"""Preparation must remain recoverable without blocking admission/control/ingest."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts.job_service.scheduler import Scheduler
from scripts.job_service.store import Store

CAPACITY = {"slots": 1, "host_memory_mb": 100, "gpu_memory_mb": 0}
SPEC = {
    "kind": "self_play",
    "options": {"device": "cpu"},
    "inputs": {},
    "resources": CAPACITY,
}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / "jobs.db")
        self.scheduler = Scheduler(
            self.store, self.root, "/bin/true", CAPACITY, Path.cwd()
        )

    def create(self, key="job", spec=SPEC):
        return self.store.command(key, {"action": "create", "spec": spec})["job_id"]

    def job(self, identifier):
        return self.store.snapshot(job_id=identifier)["jobs"][0]

    def test_preparation_does_not_block_commands_ingestion_or_reconciliation(self):
        job = self.create()
        running = self.store.claim(CAPACITY, lambda j, a, s: {"resources": CAPACITY})
        self.store.ingest(
            running["id"], [{"time": time.time(), "kind": "worker_started"}], 1
        )
        preparing = self.create("preparing")
        self.scheduler.capacity = {k: v * 2 for k, v in CAPACITY.items()}
        entered, release = threading.Event(), threading.Event()

        def effective(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release preparation")
            return {"resources": CAPACITY}

        with (
            patch.object(self.scheduler, "effective", side_effect=effective),
            patch.object(self.scheduler, "reconcile") as reconcile,
            patch("scripts.job_service.scheduler.memory_pressure", return_value=None),
        ):
            try:
                self.scheduler.tick()
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.job(preparing)["state"], "preparing")
                # Would hit SQLite's 30-second timeout if effective held its writer lock.
                self.store.ingest(
                    running["id"], [{"time": time.time(), "kind": "heartbeat"}], 2
                )
                self.store.command("cancel", {"action": "cancel", "job_id": preparing})
                self.scheduler.tick()
                self.assertGreaterEqual(reconcile.call_count, 2)
                self.assertIsNone(self.store.claim(self.scheduler.capacity))
            finally:
                release.set()
                for thread in self.scheduler.preparations.values():
                    thread.join(5)
        self.assertEqual(self.job(preparing)["state"], "cancelled")
        self.assertEqual(self.job(job)["state"], "running")
        self.assertFalse(
            (
                self.root
                / "attempts"
                / self.job(preparing)["attempts"][0]["id"]
                / "request.json"
            ).exists()
        )

    def test_restart_recovers_reservation_and_only_one_scheduler_prepares_it(self):
        job = self.create()
        attempt = self.store.claim(CAPACITY)
        entered, release = threading.Event(), threading.Event()

        def effective(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release preparation")
            return {"resources": CAPACITY}

        restarted = Scheduler(
            Store(self.store.path), self.root, "/bin/true", CAPACITY, Path.cwd()
        )
        with (
            patch.object(self.scheduler, "effective", side_effect=effective) as first,
            patch.object(
                restarted,
                "effective",
                side_effect=AssertionError("duplicate preparation"),
            ) as duplicate,
            patch("scripts.job_service.scheduler.memory_pressure", return_value=None),
        ):
            try:
                self.scheduler.tick()
                self.assertTrue(entered.wait(2))
                restarted.tick()
                for thread in restarted.preparations.values():
                    thread.join(2)
                duplicate.assert_not_called()
            finally:
                release.set()
                for thread in self.scheduler.preparations.values():
                    thread.join(5)
            first.assert_called_once()
        self.assertEqual(self.job(job)["state"], "starting")
        self.assertEqual(len(self.job(job)["attempts"]), 1)
        self.assertFalse(
            self.store.finish_preparation(attempt["id"], error="stale preparation")
        )
        self.assertEqual(self.job(job)["state"], "starting")

    def test_failed_and_paused_preparations_release_resources(self):
        for action in ("fail", "pause"):
            job = self.create(action)
            attempt = self.store.claim(CAPACITY)
            if action == "pause":
                self.store.command("pause-command", {"action": "pause", "job_id": job})
            self.store.finish_preparation(attempt["id"], error="missing file")
            self.assertEqual(
                self.job(job)["state"], "paused" if action == "pause" else "failed"
            )
            self.assertEqual(self.store.active_attempts(), [])
            self.assertIsNotNone(self.job(job)["attempts"][0]["ended"])

    def test_capacity_diagnostic_is_visible_deduplicated_and_rechecked(self):
        job = self.create(spec=SPEC | {"resources": CAPACITY | {"host_memory_mb": 200}})
        self.assertIsNone(self.store.claim(CAPACITY))
        self.assertIn("Exceeds host capacity", self.job(job)["reason"])
        events = len(self.store.events())
        self.store.claim(CAPACITY)
        self.assertEqual(len(self.store.events()), events)
        self.assertIsNotNone(self.store.claim(CAPACITY | {"host_memory_mb": 200}))
        self.assertIsNone(self.job(job)["reason"])
        waiting = self.create("waiting")
        self.store.claim(CAPACITY | {"host_memory_mb": 200})
        self.assertIn("Waiting for resources", self.job(waiting)["reason"])

    def test_input_roles_and_checkpoint_producers_are_validated(self):
        file = self.root / "replay.bin.zst"
        file.write_bytes(b"replay")
        replay = self.store.artifact(None, None, "replay", file)["id"]
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.create("bad", SPEC | {"inputs": {"checkpoint": replay}})
        model = self.root / "model.safetensors"
        model.write_bytes(b"model")
        checkpoint = self.store.artifact(None, None, "checkpoint", model)["id"]
        replay_spec = SPEC | {"kind": "replay_train", "inputs": {"replay": checkpoint}}
        with self.assertRaisesRegex(ValueError, "no registered replay"):
            self.create("missing-replay", replay_spec)
        self.store.artifact(
            None, None, "checkpoint", model, {"files": {"replay.bin.zst": "digest"}}
        )
        producer = self.create("replay-train", replay_spec)
        with self.assertRaisesRegex(ValueError, "does not produce self-play replays"):
            self.create(
                "bad-producer",
                replay_spec
                | {"inputs": {"replay": {"job_id": producer, "selection": "latest"}}},
            )
        comparison = SPEC | {
            "kind": "comparison",
            "inputs": {"first": checkpoint, "second": checkpoint},
        }
        battle = self.create("battle", comparison)
        with self.assertRaisesRegex(ValueError, "does not produce checkpoints"):
            self.create(
                "bad-selection",
                SPEC
                | {"inputs": {"checkpoint": {"job_id": battle, "selection": "latest"}}},
            )

    def test_imported_jobs_with_registered_checkpoints_remain_selectable(self):
        historical = self.create("historical")
        with self.store.transaction() as db:
            db.execute(
                "UPDATE jobs SET spec=? WHERE id=?",
                (json.dumps(SPEC | {"kind": "historical_training"}), historical),
            )
        reference = {"job_id": historical, "selection": "latest"}
        spec = SPEC | {"inputs": {"checkpoint": reference}}
        with self.assertRaisesRegex(ValueError, "does not produce checkpoints"):
            self.create("no-checkpoint", spec)
        file = self.root / "model.safetensors"
        file.write_bytes(b"historical model")
        self.store.artifact(historical, None, "checkpoint", file, {"epoch": 0})
        self.create("with-checkpoint", spec)

    def test_effective_resolves_each_input_once_and_detects_subsequent_changes(self):
        from scripts.job_service.catalog import register_checkpoint

        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"model")
        import hashlib

        (checkpoint / "metadata.json").write_text(
            json.dumps(
                {"epoch": 0, "model_sha256": hashlib.sha256(b"model").hexdigest()}
            )
        )
        reference = register_checkpoint(self.store, checkpoint)["id"]
        spec = SPEC | {"inputs": {"checkpoint": reference}}
        job = self.create(spec=spec)
        with patch.object(
            self.scheduler, "resolve_artifact", wraps=self.scheduler.resolve_artifact
        ) as resolve:
            effective = self.scheduler.effective(job, "first", spec)
            resolve.assert_called_once_with(reference)
        self.assertEqual(
            effective["resolved_inputs"]["checkpoint"]["path"], str(checkpoint)
        )
        (checkpoint / "model.safetensors").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "content changed"):
            self.scheduler.effective(job, "second", spec)

    def test_native_request_errors_preserve_worker_but_analysis_failures_reset_it(self):
        from scripts.job_service.analysis import AnalysisRequestError, AnalysisWorker

        (self.root / "metadata.json").write_text(
            json.dumps({"model_sha256": "fixture"})
        )
        binary = self.root / "fake-analysis"
        binary.write_text("""#!/usr/bin/env python3
import sys,json,os
for line in sys.stdin:
 body=json.loads(line)
 if body.get('failure'):
  print(json.dumps({'error':'test error','error_kind':body['failure']}),flush=True)
 else:
  print(json.dumps({'checkpoint':{'model_sha256':'fixture'},'result':{'complete':True,'pid':os.getpid(),'searched_simulations':body['simulations'],'carried_visits':0}}),flush=True)
""")
        binary.chmod(0o755)
        worker = AnalysisWorker(binary, self.root)
        self.addCleanup(worker.close)
        first = worker.request({"simulations": 1})["result"]["pid"]
        with self.assertRaises(AnalysisRequestError):
            worker.request({"simulations": 1, "failure": "invalid_request"})
        self.assertEqual(worker.request({"simulations": 1})["result"]["pid"], first)
        with self.assertRaises(ValueError):
            worker.request({"simulations": 1, "failure": "analysis_failed"})
        self.assertIsNone(worker.process)
        self.assertNotEqual(worker.request({"simulations": 1})["result"]["pid"], first)
