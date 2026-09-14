"""Finalization is durable, isolated, and never owns the scheduler or writer lock during I/O."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.job_service.catalog import register_checkpoint, register_history
from scripts.job_service.journal import ingest_native
from scripts.job_service.scheduler import Scheduler
from scripts.job_service.store import Store
from scripts.job_service.worker import atomic_json, emit, process_identity

RESOURCES = {"slots": 1, "host_memory_mb": 64, "gpu_memory_mb": 0}
SPEC = {
    "kind": "self_play",
    "options": {"device": "cpu"},
    "inputs": {},
    "resources": RESOURCES,
}


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = Store(self.root / "jobs.sqlite3")
        self.scheduler = Scheduler(
            self.store,
            self.root,
            "/bin/true",
            RESOURCES | {"slots": 2, "host_memory_mb": 128},
            Path.cwd(),
        )
        self.addCleanup(self.join)

    def join(self):
        for thread in [
            *self.scheduler.finalizations.values(),
            *self.scheduler.preparations.values(),
            *(entry[0] for entry in self.scheduler.checkpoint_publications.values()),
        ]:
            thread.join(5)
            self.assertFalse(thread.is_alive())

    def create(self, key):
        return self.store.command(key, {"action": "create", "spec": SPEC})["job_id"]

    def running(self, key, finished=True):
        job = self.create(key)
        attempt = self.store.claim(
            self.scheduler.capacity,
            lambda j, a, s: {
                "spec": s,
                "resources": RESOURCES,
                "output": str(self.root / "jobs" / j),
            },
        )
        directory = self.root / "attempts" / attempt["id"]
        directory.mkdir(parents=True)
        owner = {"pid": 999999999} if finished else process_identity(os.getpid())
        atomic_json(directory / "started.json", {"owner": owner})
        emit(directory, "worker_started", {})
        if finished:
            atomic_json(
                directory / "result.json",
                {"returncode": 0, "stopped": False, "reason": None},
            )
        return job, attempt, directory

    def state(self, job):
        return self.store.snapshot(job_id=job)["jobs"][0]["state"]

    def checkpoint(self, job, directory, epoch):
        path = self.root / "jobs" / job / "checkpoints" / directory
        path.mkdir(parents=True)
        model = f"model {epoch}".encode()
        (path / "model.safetensors").write_bytes(model)
        (path / "replay.bin.zst").write_bytes(b"replays")
        atomic_json(
            path / "metadata.json",
            {"epoch": epoch, "model_sha256": hashlib.sha256(model).hexdigest()},
        )
        return path

    def test_running_checkpoints_are_published_and_restart_backfills(self):
        job, attempt, directory = self.running("live", finished=False)
        attempt = self.store.active_attempts()[0]
        self.checkpoint(job, "00000000", 0)
        pending = self.checkpoint(job, ".00000001.pending", 1)
        self.scheduler.reconcile(attempt, directory)
        self.join()
        with self.store.transaction(write=False) as db:
            rows = db.execute(
                "SELECT metadata FROM artifacts WHERE job_id=?", (job,)
            ).fetchall()
        self.assertEqual([json.loads(r[0])["epoch"] for r in rows], [0])
        self.assertEqual(self.state(job), "running")
        pending.rename(pending.parent / "00000001")
        # Restarting discovers epochs even if their original event was already ingested.
        other = Scheduler(self.store, self.root, "/bin/true", RESOURCES, Path.cwd())
        other.publish_checkpoints(attempt, directory)
        other.checkpoint_publications[attempt["id"]][0].join(5)
        self.assertFalse(other.checkpoint_publications[attempt["id"]][0].is_alive())
        with self.store.transaction(write=False) as db:
            rows = db.execute(
                "SELECT metadata FROM artifacts WHERE job_id=?", (job,)
            ).fetchall()
        self.assertEqual(sorted(json.loads(r[0])["epoch"] for r in rows), [0, 1])

    def test_live_hashing_does_not_block_controls_and_is_not_restarted_each_tick(self):
        job, attempt, directory = self.running("live", finished=False)
        attempt = self.store.active_attempts()[0]
        self.checkpoint(job, "00000000", 0)
        entered, release = threading.Event(), threading.Event()

        def register(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("gate timed out")
            return register_checkpoint(*args)

        with patch(
            "scripts.job_service.catalog.register_checkpoint", side_effect=register
        ) as mocked:
            try:
                self.scheduler.reconcile(attempt, directory)
                self.assertTrue(entered.wait(2))
                self.store.command("pause-live", {"action": "pause", "job_id": job})
                self.scheduler.reconcile(attempt, directory)
                self.assertTrue((directory / "control.json").exists())
                self.assertEqual(mocked.call_count, 1)
            finally:
                release.set()
                self.join()
            self.scheduler.publish_checkpoints(attempt, directory)
            self.assertEqual(mocked.call_count, 1)
        self.assertEqual(self.state(job), "stopping")

    def test_slow_catalog_does_not_block_control_or_release_reservation_early(self):
        first, _, _ = self.running("first")
        second, _, directory = self.running("second", finished=False)
        entered, release = threading.Event(), threading.Event()

        def collect(_):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("gate timed out")

        with (
            patch.object(self.scheduler, "collect_artifacts", side_effect=collect),
            patch("scripts.job_service.scheduler.memory_pressure", return_value=None),
        ):
            try:
                self.scheduler.tick()
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.state(first), "finalizing")
                queued = self.create("queued")
                self.store.command("pause", {"action": "pause", "job_id": second})
                self.scheduler.tick()
                self.assertTrue((directory / "control.json").exists())
                self.assertEqual(self.state(queued), "queued")
            finally:
                release.set()
                self.join()
        self.assertEqual(self.state(first), "succeeded")

    def test_bad_checkpoint_fails_only_its_job_and_frees_admission(self):
        job, _, _ = self.running("bad")
        checkpoint = self.root / "jobs" / job / "checkpoints/00000000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"incomplete checkpoint")
        self.scheduler.tick()
        self.join()
        self.assertEqual(self.state(job), "failed")
        self.assertTrue(
            any(e["kind"] == "finalization_failed" for e in self.store.events(job))
        )
        other = self.create("other")
        with (
            patch.object(self.scheduler, "prepare"),
            patch("scripts.job_service.scheduler.memory_pressure", return_value=None),
        ):
            self.scheduler.tick()
        self.assertEqual(self.state(other), "preparing")

    def test_restart_and_duplicate_finalizers_collect_once(self):
        job, attempt, directory = self.running("restart")
        self.store.begin_finalization(attempt["id"])
        other = Scheduler(
            Store(self.store.path),
            self.root,
            "/bin/true",
            self.scheduler.capacity,
            Path.cwd(),
        )
        entered, release = threading.Event(), threading.Event()
        calls = []

        def collect(_):
            calls.append(1)
            entered.set()
            if not release.wait(5):
                raise RuntimeError("gate timed out")

        with (
            patch.object(self.scheduler, "collect_artifacts", side_effect=collect),
            patch.object(other, "collect_artifacts", side_effect=collect),
        ):
            try:
                self.scheduler.finalize(attempt, directory)
                self.assertTrue(entered.wait(2))
                other.finalize(attempt, directory)
                for thread in other.finalizations.values():
                    thread.join(2)
                self.assertEqual(len(calls), 1)
            finally:
                release.set()
                self.join()
        self.assertEqual(self.state(job), "succeeded")
        self.assertFalse(self.store.begin_finalization(attempt["id"]))

    def test_native_backlog_is_bounded_and_idempotent(self):
        job, attempt, directory = self.running("journal")
        path = directory / "events.jsonl"
        path.write_text(
            "".join(
                json.dumps({"kind": "heartbeat", "time": i, "payload": {"n": i}}) + "\n"
                for i in range(1201)
            )
        )
        self.assertFalse(ingest_native(self.store, attempt, path))
        with self.store.transaction(write=False) as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM events WHERE kind='heartbeat'"
                ).fetchone()[0],
                500,
            )
        self.store.command("pause", {"action": "pause", "job_id": job})
        while not ingest_native(self.store, attempt, path):
            pass
        self.assertTrue(ingest_native(self.store, attempt, path))
        with self.store.transaction(write=False) as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM events WHERE kind='heartbeat'"
                ).fetchone()[0],
                1201,
            )

    def test_published_checkpoint_hashes_are_reused_but_changes_are_rechecked(self):
        directory = self.root / "history/checkpoints/00000000"
        directory.mkdir(parents=True)
        (directory / "model.safetensors").write_bytes(b"model")
        (directory / "replay.bin.zst").write_bytes(b"replay")
        (directory / "metadata.json").write_text(
            json.dumps(
                {"epoch": 0, "model_sha256": hashlib.sha256(b"model").hexdigest()}
            )
        )
        stats = self.root / "history/stats"
        stats.mkdir()
        (stats / "00000000.json").write_text("{}")
        first = register_checkpoint(self.store, directory)
        with patch(
            "scripts.job_service.catalog.digest",
            side_effect=AssertionError("unexpected rehash"),
        ):
            self.assertEqual(
                register_checkpoint(self.store, directory)["id"], first["id"]
            )
        from scripts.job_service.catalog import digest

        with patch("scripts.job_service.catalog.digest", wraps=digest) as observed:
            register_history(self.store, self.root, self.root / "history")
            self.assertEqual(
                [call.args[0].name for call in observed.call_args_list],
                ["00000000.json"],
            )
        (directory / "replay.bin.zst").write_bytes(b"edited")
        updated = register_checkpoint(self.store, directory)
        self.assertNotEqual(
            json.loads(first["metadata"])["files"],
            json.loads(updated["metadata"])["files"],
        )
