import http.client
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

from scripts.job_service.auth import Auth
from scripts.job_service.server import Application, Handler
from scripts.job_service.store import Conflict, Store
from scripts.job_service.spec import compile_argv
from scripts.job_service.scheduler import Scheduler


RESOURCES = {"slots": 1, "host_memory_mb": 100, "gpu_memory_mb": 0}
SPEC = {
    "kind": "self_play",
    "options": {"device": "cpu", "epochs": 2},
    "inputs": {},
    "resources": RESOURCES,
}


class AuthPersistenceTests(unittest.TestCase):
    def test_session_and_csrf_survive_restart_but_logout_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            password = Auth.hash_password("long test password")
            first = Auth(password, sessions_path=path)
            token, session = first.login("long test password")
            first.close()
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(token.encode(), path.read_bytes())
            second = Auth(password, sessions_path=path)
            self.assertEqual(second.require(token, session["csrf"]), session)
            with self.assertRaises(PermissionError):
                second.require(token, "wrong csrf")
            second.logout(token)
            second.close()
            third = Auth(password, sessions_path=path)
            self.assertIsNone(third.get(token))
            third.close()

    def test_sessions_survive_years_but_password_changes_revoke_them(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            password = Auth.hash_password("long test password")
            with patch("scripts.job_service.auth.time.time", return_value=100):
                first = Auth(password, sessions_path=path)
                token, session = first.login("long test password")
                first.close()
            with patch(
                "scripts.job_service.auth.time.time",
                return_value=100 + 10 * 365 * 86400,
            ):
                second = Auth(password, sessions_path=path)
                self.assertEqual(second.require(token, session["csrf"]), session)
                second.close()
                third = Auth(
                    Auth.hash_password("new test password"), sessions_path=path
                )
                self.assertIsNone(third.get(token))
                third.close()
                restored = Auth(password, sessions_path=path)
                self.assertIsNone(restored.get(token))
                restored.close()

    def test_legacy_migration_keeps_active_sessions_without_reviving_expired_ones(self):
        import sqlite3
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            password = Auth.hash_password("long test password")
            credential = Auth.token_hash(password)
            with sqlite3.connect(path) as db:
                db.execute(
                    "CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, csrf TEXT NOT NULL, expires REAL NOT NULL, credential TEXT NOT NULL)"
                )
                db.executemany(
                    "INSERT INTO sessions VALUES (?,?,?,?)",
                    [
                        (Auth.token_hash("active"), "active-csrf", 200, credential),
                        (Auth.token_hash("expired"), "expired-csrf", 99, credential),
                        (
                            Auth.token_hash("different-password"),
                            "other-csrf",
                            200,
                            "other",
                        ),
                    ],
                )
            db.close()
            with patch("scripts.job_service.auth.time.time", return_value=100):
                migrated = Auth(password, sessions_path=path)
                self.assertEqual(
                    migrated.require("active", "active-csrf"), {"csrf": "active-csrf"}
                )
                self.assertIsNone(migrated.get("expired"))
                self.assertIsNone(migrated.get("different-password"))
                self.assertNotIn(
                    "expires",
                    {
                        row[1]
                        for row in migrated.sessions.execute(
                            "PRAGMA table_info(sessions)"
                        )
                    },
                )
                migrated.close()
            with patch("scripts.job_service.auth.time.time", return_value=10**10):
                reopened = Auth(password, sessions_path=path)
                self.assertIsNotNone(reopened.get("active"))
                reopened.logout("active")
                self.assertIsNone(reopened.get("active"))
                reopened.close()


class StoreTests(unittest.TestCase):
    def test_exact_checkpoint_continuation_inherits_epoch_history_once(self):
        source = self.create("source")
        with self.store.transaction() as db:
            for epoch in range(3):
                self.store.event(
                    db,
                    source,
                    None,
                    "epoch_completed",
                    {"epoch": epoch, "training": {"value_loss": epoch / 10}},
                    100 + epoch,
                )
        model = self.root / "model.safetensors"
        model.write_bytes(b"checkpoint")
        artifact = self.store.artifact(
            source, None, "checkpoint", model, {"epoch": 1}
        )
        spec = SPEC | {"inputs": {"checkpoint": artifact["id"]}}
        continuation = self.create("continuation", spec=spec)
        with self.store.transaction() as db:
            self.assertEqual(
                self.store.inherit_epoch_history(db, continuation, artifact["id"]), 0
            )
        events = [
            event
            for event in self.store.events(continuation)
            if event["kind"] == "epoch_completed"
        ]
        self.assertEqual([event["payload"]["epoch"] for event in events], [0, 1])
        self.assertEqual([event["time"] for event in events], [100, 101])
        self.assertTrue(
            all(
                event["payload"]["inherited_from"]
                == {
                    "job_id": source,
                    "checkpoint_artifact": artifact["id"],
                    "checkpoint_epoch": 1,
                }
                for event in events
            )
        )

    def test_benchmark_precision_is_applied_before_native_startup(self):
        from scripts.job_service.spec import environment_overrides, validate_spec

        spec = {
            "kind": "benchmark_executor",
            "options": {"device": "cpu", "disable-tf32": True},
            "inputs": {"checkpoint": "registered-checkpoint"},
            "resources": RESOURCES,
        }
        self.assertEqual(environment_overrides(spec), {"NVIDIA_TF32_OVERRIDE": "0"})
        self.assertNotIn(
            "--disable-tf32",
            compile_argv(spec, "/binary", "/output", lambda _: "/checkpoint"),
        )
        with self.assertRaises(ValueError):
            validate_spec(SPEC | {"options": spec["options"]})

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "jobs.db")

    def create(self, key="first", **fields):
        return self.store.command(key, {"action": "create", "spec": SPEC, **fields})[
            "job_id"
        ]

    def test_read_transactions_are_query_only(self):
        with self.store.transaction(write=False) as db:
            self.assertEqual(db.execute("PRAGMA query_only").fetchone()[0], 1)
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            with self.assertRaises(sqlite3.OperationalError):
                db.execute("INSERT INTO settings VALUES ('read-test','1')")

    def test_command_retry_and_conflict_are_atomic(self):
        first = self.create()
        self.assertEqual(self.create(), first)
        self.assertEqual(self.store.snapshot()["total"], 1)
        with self.assertRaises(Conflict):
            self.create(title="different")
        with self.assertRaises(ValueError):
            self.create("invalid", dependencies=["missing"])
        self.assertEqual(self.store.snapshot()["total"], 1)
        self.assertEqual(len(self.store.events()), 1)

    def test_dependency_and_resource_reservations_survive_reopen(self):
        first = self.create()
        self.create("second", dependencies=[first])
        effective = lambda j, a, s: {"resources": s["resources"]}
        claim = self.store.claim(RESOURCES, effective)
        self.assertEqual(claim["job_id"], first)
        reopened = Store(self.store.path)
        self.assertIsNone(reopened.claim(RESOURCES, effective))
        reopened.ingest(
            claim["id"],
            [
                {
                    "time": time.time(),
                    "kind": "worker_finished",
                    "payload": {"returncode": 0},
                }
            ],
            100,
        )
        self.assertIsNotNone(reopened.claim(RESOURCES, effective))

    def test_acknowledged_stop_requires_worker_receipt(self):
        job = self.create()
        claim = self.store.claim(RESOURCES, lambda j, a, s: {"resources": RESOURCES})
        response = self.store.command(
            "pause", {"action": "pause", "job_id": job, "mode": "epoch"}
        )
        self.assertEqual(response["state"], "stopping")
        self.store.ingest(
            claim["id"],
            [
                {
                    "time": time.time(),
                    "kind": "worker_finished",
                    "payload": {"returncode": 0, "stopped": True},
                }
            ],
            50,
        )
        self.assertEqual(self.store.snapshot()["jobs"][0]["state"], "paused")
        self.store.command("resume", {"action": "resume", "job_id": job})
        attempt = self.store.claim(RESOURCES, lambda j, a, s: {"resources": RESOURCES})
        self.assertNotEqual(attempt["id"], claim["id"])

    def test_admission_keeps_clean_cache_separate_from_memory_pressure(self):
        from unittest.mock import patch

        scheduler = Scheduler(self.store, self.root, "/bin/true", RESOURCES, Path.cwd())
        for pressure, should_admit in ((50, True), (95, False)):
            with (
                patch(
                    "scripts.job_service.scheduler.memory_pressure",
                    return_value={
                        "used_bytes": 98,
                        "pressure_bytes": pressure,
                        "limit_bytes": 100,
                    },
                ),
                patch.object(self.store, "claim", return_value=None) as claim,
            ):
                scheduler.tick()
                self.assertEqual(claim.called, should_admit)

    def test_worker_outlives_scheduler_and_receipt_is_ingested_once(self):
        binary = self.root / "fake-alz"
        binary.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(.2)\n")
        binary.chmod(0o755)
        self.create()
        scheduler = Scheduler(self.store, self.root, binary, RESOURCES, Path.cwd())
        scheduler.tick()
        restarted = Scheduler(
            Store(self.store.path), self.root, binary, RESOURCES, Path.cwd()
        )
        for _ in range(50):
            restarted.tick()
            if self.store.snapshot()["jobs"][0]["state"] == "succeeded":
                break
            time.sleep(0.05)
        self.assertEqual(self.store.snapshot()["jobs"][0]["state"], "succeeded")
        self.assertEqual(
            len([e for e in self.store.events() if e["kind"] == "worker_finished"]), 1
        )
        for child in scheduler.children + restarted.children:
            child.wait(timeout=5)

    def test_only_registered_binary_artifacts_override_the_executor(self):
        source = self.root / "trusted-binary"
        source.write_bytes(b"known executable")
        source.chmod(0o755)
        wrong = self.store.artifact(None, None, "checkpoint", source)
        with self.assertRaises(ValueError):
            self.store.command(
                "wrong-binary",
                {"action": "create", "spec": SPEC | {"binary_artifact": wrong["id"]}},
            )
        other = self.root / "baseline"
        other.write_bytes(b"validated baseline executable")
        other.chmod(0o755)
        registered = self.store.artifact(None, None, "binary", other)
        spec = SPEC | {"binary_artifact": registered["id"]}
        job = self.store.command("baseline", {"action": "create", "spec": spec})[
            "job_id"
        ]
        scheduler = Scheduler(self.store, self.root, source, RESOURCES, Path.cwd())
        effective = scheduler.effective(job, "test-attempt", spec)
        self.assertEqual(Path(effective["argv"][0]).read_bytes(), other.read_bytes())
        other.write_bytes(b"changed executable")
        with self.assertRaises(ValueError):
            scheduler.effective(job, "another-attempt", spec)

    def test_argv_is_typed_and_has_server_owned_paths(self):
        argv = compile_argv(SPEC, "/binary", "/output", lambda _: None)
        self.assertIn("--games-dir", argv)
        self.assertIn("/output/checkpoints", argv)
        replay = SPEC | {
            "kind": "replay_train",
            "inputs": {"replay": "a", "replay_2": "b"},
        }
        pooled = compile_argv(
            replay, "/binary", "/output", lambda identifier: "/" + identifier
        )
        self.assertEqual(pooled.count("--replay-checkpoint-dir"), 2)
        with self.assertRaises(ValueError):
            self.store.command(
                "evil",
                {
                    "action": "create",
                    "spec": SPEC | {"options": {"checkpoint-dir": "/etc"}},
                },
            )

    def test_overlap_threshold_is_an_explicit_dependency_condition(self):
        # A typed comparison fixture with no executable is sufficient to test admission.
        parent = self.create()
        with self.store.transaction() as db:
            spec = SPEC | {"kind": "comparison"}
            db.execute("UPDATE jobs SET spec=? WHERE id=?", (json.dumps(spec), parent))
        child = self.create(
            "overlap", dependencies=[{"job_id": parent, "fraction": 0.6}]
        )
        capacity = RESOURCES | {"slots": 2, "host_memory_mb": 200}
        claim = self.store.claim(capacity, lambda j, a, s: {"resources": RESOURCES})
        self.store.ingest(
            claim["id"],
            [{"time": time.time(), "kind": "worker_started", "payload": {}}],
            10,
        )
        self.assertIsNone(
            self.store.claim(capacity, lambda j, a, s: {"resources": RESOURCES})
        )
        with self.store.transaction() as db:
            self.store.event(
                db,
                parent,
                claim["id"],
                "comparison_progress",
                {"games_completed": 60, "games_total": 100},
            )
        self.assertEqual(
            self.store.claim(capacity, lambda j, a, s: {"resources": RESOURCES})[
                "job_id"
            ],
            child,
        )

    def test_import_preserves_missing_evidence_and_is_idempotent(self):
        from scripts.job_service.importer import import_snapshot, converted_game

        source = self.root / "snapshot.json"
        source.write_text(
            json.dumps(
                {
                    "phases": [
                        {
                            "id": "old",
                            "title": "Old run",
                            "kind": "training",
                            "state": "completed",
                            "metrics": [
                                {
                                    "epoch": 1,
                                    "training": {"value_loss": 0.3, "policy_loss": 2},
                                    "validation": None,
                                }
                            ],
                        }
                    ]
                }
            )
        )
        original = source.read_bytes()
        first = import_snapshot(self.store, self.root, source)
        self.assertEqual(first["jobs"], 1)
        self.assertTrue(
            import_snapshot(self.store, self.root, source)["already_imported"]
        )
        self.assertEqual(source.read_bytes(), original)
        job = self.store.snapshot()["jobs"][0]
        self.assertEqual(job["state"], "archived")
        self.assertIsNone(job["attempts"][0]["started"])
        rows = [
            {
                "game": "a" * 64,
                "source_epoch": 69,
                "ply": 0,
                "stones": [],
                "policy": [[0, 1]],
                "value": 1,
            },
            {
                "game": "a" * 64,
                "source_epoch": 69,
                "ply": 1,
                "stones": [[0, 2]],
                "policy": [[1, 1]],
                "value": -1,
            },
        ]
        game = converted_game(rows, "temperature_adjusted_policy", "f" * 64)
        self.assertEqual(game["record"]["plies"][0]["action"], {"x": 0, "y": 0})
        self.assertIsNone(game["record"]["plies"][1]["action"])
        self.assertIsNone(
            game["record"]["plies"][0]["decision"]["diagnostics"]["network_prior"]
        )
        self.assertIsNone(game["generation_epoch"])
        self.assertEqual(game["policy_semantics"], "temperature_adjusted_policy")

    def test_checkpoint_selector_uses_validation_and_records_exact_source(self):
        job = self.create()
        with self.store.transaction() as db:
            db.execute('UPDATE jobs SET state="succeeded" WHERE id=?', (job,))
            for epoch, loss in [(0, 0.1), (1, 0.3)]:
                self.store.event(
                    db,
                    job,
                    None,
                    "epoch_completed",
                    {"epoch": epoch, "validation": {"value_loss": loss}},
                )
        for epoch in range(2):
            directory = self.root / "models" / str(epoch)
            directory.mkdir(parents=True)
            model = directory / "model.safetensors"
            model.write_bytes(bytes([epoch]))
            metadata = {
                "epoch": epoch,
                "model_sha256": __import__("hashlib")
                .sha256(model.read_bytes())
                .hexdigest(),
            }
            (directory / "metadata.json").write_text(json.dumps(metadata))
            self.store.artifact(job, None, "checkpoint", model, metadata)
        scheduler = Scheduler(self.store, self.root, "/bin/true", RESOURCES, Path.cwd())
        self.assertEqual(
            scheduler.resolve_artifact(
                {"job_id": job, "selection": "best_value_validation"}
            ).name,
            "0",
        )
        self.assertEqual(
            scheduler.resolve_artifact({"job_id": job, "selection": "latest"}).name, "1"
        )

    def test_catalog_reregistration_fills_metadata_without_changing_identity(self):
        path = self.root / "model.safetensors"
        path.write_bytes(b"model")
        job = self.create()
        first = self.store.artifact(job, None, "checkpoint", path)
        second = self.store.artifact(job, None, "checkpoint", path, {"epoch": 7})
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(json.loads(second["metadata"])["epoch"], 7)

    def test_execution_limits_change_only_while_inactive(self):
        job = self.create()
        self.store.command(
            "resize",
            {"action": "configure", "job_id": job, "options": {"games-parallelism": 4}},
        )
        self.assertEqual(
            self.store.snapshot(job_id=job)["jobs"][0]["spec"]["options"][
                "games-parallelism"
            ],
            4,
        )
        with self.assertRaises(ValueError):
            self.store.command(
                "lr",
                {
                    "action": "configure",
                    "job_id": job,
                    "options": {"learning-rate": 0.1},
                },
            )
        self.store.claim(RESOURCES, lambda j, a, s: {"resources": RESOURCES})
        with self.assertRaises(Conflict):
            self.store.command(
                "resize-running",
                {
                    "action": "configure",
                    "job_id": job,
                    "options": {"games-parallelism": 2},
                },
            )

    def test_history_registration_detects_modified_replay(self):
        from scripts.job_service.catalog import register_history

        history = self.root / "history"
        checkpoint = history / "checkpoints" / "00000000"
        checkpoint.mkdir(parents=True)
        (history / "stats").mkdir()
        (checkpoint / "replay.bin.zst").write_bytes(b"original")
        (checkpoint / "metadata.json").write_text('{"epoch":0}')
        (history / "stats" / "00000000.json").write_text("{}")
        artifact = register_history(self.store, self.root, history)
        scheduler = Scheduler(self.store, self.root, "/bin/true", RESOURCES, Path.cwd())
        self.assertEqual(scheduler.resolve_artifact(artifact["id"]), history)
        spec = SPEC | {
            "kind": "reconstruction",
            "options": {"device": "cpu", "replay-history-epochs": 1},
            "inputs": {"history": artifact["id"]},
        }
        argv = compile_argv(spec, "/binary", "/output", scheduler.resolve_artifact)
        self.assertEqual(argv[argv.index("--replay-history-dir") + 1], str(history))
        (checkpoint / "replay.bin.zst").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "History input changed"):
            scheduler.resolve_artifact(artifact["id"])

    def test_metrics_after_checkpoint_are_recovered_once(self):
        job = self.create()
        output = self.root / "jobs" / job
        (output / "stats").mkdir(parents=True)
        (output / "checkpoints" / "00000000").mkdir(parents=True)
        (output / "checkpoints" / "00000000" / "metadata.json").write_text("{}")
        (output / "stats" / "00000000.json").write_text(
            '{"epoch":0,"training":{"value_loss":0.2}}'
        )
        scheduler = Scheduler(self.store, self.root, "/bin/true", RESOURCES, Path.cwd())
        attempt = {"job_id": job, "id": None, "effective": {"output": str(output)}}
        scheduler.recover_metrics(attempt)
        scheduler.recover_metrics(attempt)
        self.assertEqual(
            len([e for e in self.store.events(job) if e["kind"] == "epoch_completed"]),
            1,
        )

    def test_owner_journal_recovers_a_torn_final_record(self):
        from scripts.job_service.worker import emit

        path = self.root / "worker-events.jsonl"
        path.write_bytes(b'{"kind":"worker_started"}\n{"kind":"worker_fin')
        emit(self.root, "worker_finished", {"returncode": 0})
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(
            [r["kind"] for r in records], ["worker_started", "worker_finished"]
        )


class ArchiveStartupTests(unittest.TestCase):
    def test_archive_requires_existing_data_and_rejects_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "must-not-create"
            for extra, message in (
                ([], "existing jobs.sqlite3"),
                (["--binary", "alz"], "cannot be combined"),
                (["--analysis-checkpoint", "checkpoint"], "cannot be combined"),
                (["--analysis-binary", "alz"], "cannot be combined"),
                (["--set-password"], "cannot be combined"),
            ):
                result = subprocess.run(
                    [sys.executable, "-m", "scripts.job_service.server",
                     "--root", str(root), "--read-only", *extra],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertFalse(root.exists())


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = Store(root / "jobs.db")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.server.app = Application(
            self.store,
            root,
            root,
            Auth(Auth.hash_password("long test password")),
            self.origin,
        )
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.server.app.auth.close()
        self.tmp.cleanup()

    def request(self, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        conn.request(
            "GET" if body is None else "POST",
            path,
            None if body is None else json.dumps(body),
            {
                "Origin": self.origin,
                "Content-Type": "application/json",
                **(headers or {}),
            },
        )
        result = conn.getresponse()
        status, data, response_headers = (
            result.status,
            json.loads(result.read()),
            dict(result.getheaders()),
        )
        conn.close()
        return status, data, response_headers

    def test_read_only_archive_rejects_even_authenticated_mutations(self):
        token, session = self.server.app.auth.login("long test password")
        self.server.app.read_only = True
        status, body, _ = self.request(
            "/api/v1/session", headers={"Cookie": f"alz_session={token}"}
        )
        self.assertEqual(status, 200)
        self.assertFalse(body["authenticated"])
        self.assertFalse(body["login_enabled"])
        for path, body in (
            ("/api/v1/login", {"password": "long test password"}),
            (
                "/api/v1/commands",
                {"command_id": "blocked", "request": {"action": "create", "spec": SPEC}},
            ),
        ):
            status, data, _ = self.request(
                path,
                body,
                {"Cookie": f"alz_session={token}", "X-CSRF-Token": session["csrf"]},
            )
            self.assertEqual(status, 403)
            self.assertIn("read-only", data["error"])
        status, data, _ = self.request("/api/v1/jobs")
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 0)

    def test_artifact_choices_include_series_title_without_exposing_paths(self):
        job = self.store.command(
            "named-series",
            {
                "action": "create",
                "title": "Board mask selected training pass",
                "spec": SPEC,
            },
        )["job_id"]
        path = self.server.app.root / "model.safetensors"
        path.write_bytes(b"fixture")
        self.store.artifact(job, None, "checkpoint", path, {"epoch": 13})
        status, data, _ = self.request("/api/v1/artifacts")
        self.assertEqual(status, 200)
        self.assertEqual(
            data["artifacts"][0]["job_title"], "Board mask selected training pass"
        )
        self.assertNotIn("path", data["artifacts"][0])

    def test_epoch_summaries_skip_event_backlog_and_support_incremental_reads(self):
        job = self.store.command("epochs", {"action": "create", "spec": SPEC})["job_id"]
        other = self.store.command("other-epochs", {"action": "create", "spec": SPEC})[
            "job_id"
        ]
        with self.store.transaction() as db:
            for epoch in range(510):
                self.store.event(db, job, None, "game_completed", {"game": epoch})
                self.store.event(db, job, None, "self_play_progress", {})
                self.store.event(
                    db,
                    job,
                    None,
                    "epoch_completed",
                    {"epoch": epoch, "training": {"value_loss": 0.1}},
                )
            self.store.event(db, other, None, "epoch_completed", {"epoch": 999})
        status, result, _ = self.request(f"/api/v1/epoch-summaries?job_id={job}")
        self.assertEqual(status, 200)
        summaries = result["events"]
        self.assertEqual([e["payload"]["epoch"] for e in summaries], list(range(510)))
        cursor = summaries[-1]["id"]
        with self.store.transaction() as db:
            self.store.event(db, job, None, "epoch_completed", {"epoch": 510})
        _, result, _ = self.request(
            f"/api/v1/epoch-summaries?job_id={job}&after={cursor}"
        )
        self.assertEqual([e["payload"]["epoch"] for e in result["events"]], [510])
        _, result, _ = self.request(
            f"/api/v1/epoch-summaries?job_id={job}&after={result['events'][0]['id']}"
        )
        self.assertEqual(result["events"], [])

    def test_job_filters_narrow_the_page_and_its_total(self):
        for title, state in (("alpha run", "queued"), ("beta run", "queued")):
            job = self.store.command(
                title,
                {
                    "action": "create",
                    "spec": SPEC,
                    "title": title,
                    "experiment_title": f"{title} series",
                },
            )["job_id"]
            if state != "queued":
                self.store.command(f"{title}-cancel", {"action": "cancel", "job_id": job})
        _, page, _ = self.request("/api/v1/jobs?q=BETA")
        self.assertEqual([j["title"] for j in page["jobs"]], ["beta run"])
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["jobs"][0]["experiment_title"], "beta run series")
        _, page, _ = self.request("/api/v1/jobs?kind=comparison")
        self.assertEqual(page["jobs"], [])
        self.assertEqual(page["total"], 0)
        _, page, _ = self.request("/api/v1/jobs?kind=self_play&state=queued")
        self.assertEqual(page["total"], 2)
        experiment = page["jobs"][0]["experiment_id"]
        _, page, _ = self.request(f"/api/v1/jobs?experiment_id={experiment}")
        self.assertEqual(page["total"], 1)
        _, session, _ = self.request("/api/v1/session")
        self.assertFalse(session["read_only"])
        self.assertIsNone(session["demo_url"])

    def test_overview_resolves_comparison_checkpoints_and_run_epochs(self):
        run = self.store.command("run", {"action": "create", "spec": SPEC, "title": "Run"})[
            "job_id"
        ]
        baseline = self.store.command(
            "baseline", {"action": "create", "spec": SPEC, "title": "Baseline"}
        )["job_id"]
        run_checkpoint = self.archive_checkpoint(run, 4)
        base_checkpoint = self.archive_checkpoint(baseline, 1)
        with self.store.transaction() as db:
            for epoch in range(3):
                self.store.event(
                    db,
                    run,
                    None,
                    "epoch_completed",
                    {
                        "epoch": epoch,
                        "training": {"value_loss": 0.5 - epoch / 10, "policy_loss": 2.0},
                        "self_play": {"games": 10, "first_player_win_rate": 0.6},
                    },
                )
        comparison = self.store.command(
            "match",
            {
                "action": "create",
                "title": "Match",
                "spec": SPEC
                | {
                    "kind": "comparison",
                    "options": {"device": "cpu", "games": 2, "simulations": 4},
                    "inputs": {"first": run_checkpoint, "second": base_checkpoint},
                },
            },
        )["job_id"]
        shas = {}
        with self.store.transaction() as db:
            for row in db.execute("SELECT id,sha256 FROM artifacts"):
                shas[row["id"]] = row["sha256"]
            self.store.event(
                db,
                comparison,
                None,
                "imported",
                {
                    "participants": {
                        "first": {"label": "Run · epoch 5", "model_sha256": shas[run_checkpoint]},
                        "second": {
                            "label": "Baseline · epoch 2",
                            "model_sha256": shas[base_checkpoint],
                        },
                    }
                },
            )
            self.store.event(
                db,
                comparison,
                None,
                "comparison_completed",
                {
                    "game_statistics": {
                        "games": 4,
                        "seat_results": {
                            "first_checkpoint": {
                                "first": {"games": 2, "wins": 2, "draws": 0, "losses": 0},
                                "second": {"games": 2, "wins": 1, "draws": 0, "losses": 1},
                            },
                            "second_checkpoint": {
                                "first": {"games": 2, "wins": 1, "draws": 0, "losses": 1},
                                "second": {"games": 2, "wins": 0, "draws": 0, "losses": 2},
                            },
                        },
                        "outcomes": {"first_player_wins": 3, "second_player_wins": 1, "draws": 0},
                        "lengths": {},
                        "completion": None,
                    }
                },
            )
            db.execute("UPDATE jobs SET state='succeeded' WHERE id=?", (comparison,))
        status, overview, _ = self.request("/api/v1/overview")
        self.assertEqual(status, 200)
        self.assertEqual(overview["jobs"], 3)
        self.assertEqual(overview["by_kind"], {"self_play": 2, "comparison": 1})
        self.assertEqual(len(overview["comparisons"]), 1)
        match = overview["comparisons"][0]
        self.assertEqual((match["wins"], match["draws"], match["losses"]), (3, 0, 1))
        self.assertEqual(match["first"]["job_id"], run)
        self.assertEqual(match["first"]["epoch"], 4)
        self.assertEqual(match["first"]["job_title"], "Run")
        self.assertEqual(match["second"]["job_id"], baseline)
        self.assertEqual(match["second"]["label"], "Baseline · epoch 2")
        self.assertEqual(match["settings"], {})
        runs = {r["id"]: r for r in overview["runs"]}
        self.assertEqual(
            [e["value_loss"] for e in runs[run]["epochs"]], [0.5, 0.4, 0.3]
        )
        self.assertEqual(runs[run]["epochs"][0]["first_player_win_rate"], 0.6)
        # The cache is keyed on the job table, so a new job refreshes it.
        self.store.command("later", {"action": "create", "spec": SPEC, "title": "Later"})
        _, refreshed, _ = self.request("/api/v1/overview")
        self.assertEqual(refreshed["jobs"], 4)

    def test_games_sort_and_filter_within_one_epoch(self):
        job = self.store.command("sorted", {"action": "create", "spec": SPEC})["job_id"]
        directory = self.server.app.archive(job) / "00000000"
        directory.mkdir(parents=True)
        for index, (plies, actor, value) in enumerate(
            ((9, "Second", -1), (25, "First", -1), (12, "Second", 0))
        ):
            (directory / f"{index:08}.json").write_text(
                json.dumps(
                    {
                        "record": {
                            "plies": [{"actor": "First"}] * plies,
                            "terminal_actor": actor,
                            "terminal_value": value,
                        }
                    }
                )
            )
        _, page, _ = self.request(
            f"/api/v1/games?job_id={job}&epoch=00000000&sort=longest"
        )
        self.assertEqual([g["plies"] for g in page["games"]], [25, 12, 9])
        _, page, _ = self.request(
            f"/api/v1/games?job_id={job}&epoch=00000000&winner=first&sort=shortest"
        )
        self.assertEqual([(g["plies"], g["winner"]) for g in page["games"]], [(9, "First")])
        self.assertEqual(page["total"], 1)
        status, error, _ = self.request(f"/api/v1/games?job_id={job}&sort=longest")
        self.assertEqual(status, 400)
        self.assertIn("single epoch", error["error"])
        status, _, _ = self.request(f"/api/v1/games?job_id={job}&epoch=00000000&sort=bogus")
        self.assertEqual(status, 400)

    def test_unknown_page_serves_the_app_with_a_404_status(self):
        (self.server.app.assets / "index.html").write_text(
            "<head></head><body>app</body>"
        )
        for path, expected in (
            ("/analyze", 200),
            ("/play", 200),
            ("/missing-page", 404),
        ):
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.server.server_port
            )
            connection.request("GET", path)
            response = connection.getresponse()
            self.assertEqual(response.status, expected)
            self.assertIn(b"alz-service", response.read())
            connection.close()

    def test_archive_discovery_recent_order_and_outcome_summaries(self):
        def create(key):
            return self.store.command(
                key, {"action": "create", "spec": SPEC, "title": key}
            )["job_id"]

        old, empty, recent = create("old"), create("empty"), create("recent")
        for job, epoch, actor, value in (
            (old, 0, "Second", -1),
            (recent, 1, "First", -1),
            (recent, 2, "Second", 0),
        ):
            directory = self.server.app.archive(job) / f"{epoch:08}"
            directory.mkdir(parents=True)
            (directory / "00000000.json").write_text(
                json.dumps(
                    {
                        "record": {
                            "plies": [{"actor": "First"}] * 9,
                            "terminal_actor": actor,
                            "terminal_value": value,
                        }
                    }
                )
            )
        _, catalog, _ = self.request("/api/v1/game-jobs")
        self.assertEqual([j["id"] for j in catalog["jobs"]], [recent, old])
        _, page, _ = self.request(f"/api/v1/games?job_id={recent}&limit=1")
        self.assertEqual(page["epochs"], ["00000002", "00000001"])
        self.assertEqual(page["games"][0]["winner"], "Draw")
        self.assertEqual(page["games"][0]["plies"], 9)
        _, second, _ = self.request(f"/api/v1/games?job_id={recent}&limit=1&offset=1")
        self.assertEqual(second["games"][0]["winner"], "Second")
        _, detail, _ = self.request(f"/api/v1/jobs/{empty}")
        self.assertFalse(detail["has_games"])
        _, detail, _ = self.request(f"/api/v1/jobs/{old}")
        self.assertTrue(detail["has_games"])

    def archive_job(self, name, checkpoint=None):
        spec = SPEC | {"inputs": {"checkpoint": checkpoint} if checkpoint else {}}
        return self.store.command(name, {"action": "create", "spec": spec})["job_id"]

    def archive_checkpoint(self, job, epoch):
        model = self.server.app.root / f"{job}-{epoch}.safetensors"
        # Distinct bytes per checkpoint so hashes identify one job and epoch.
        model.write_bytes(f"checkpoint {job} {epoch}".encode())
        return self.store.artifact(job, None, "checkpoint", model, {"epoch": epoch})[
            "id"
        ]

    def archive_epoch(self, job, epoch, label):
        directory = self.server.app.archive(job) / f"{epoch:08}"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"label": label}))
        path = directory / "00000000.json"
        path.write_text(
            json.dumps(
                {
                    "label": label,
                    "record": {"plies": [{"actor": "First"}], "terminal_value": 0},
                }
            )
        )
        return path

    def test_continuation_replays_are_zero_copy_bounded_and_available_over_http(self):
        parent = self.archive_job("parent")
        originals = [self.archive_epoch(parent, e, f"Parent {e}") for e in range(4)]
        child = self.archive_job("child", self.archive_checkpoint(parent, 1))
        status, page, _ = self.request(f"/api/v1/games?job_id={child}&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(page["total"], 2)
        self.assertEqual(page["epochs"], ["00000001", "00000000"])
        self.assertEqual(page["epoch_labels"]["00000000"], "Parent 0")
        self.assertEqual(page["games"][0]["epoch"], "00000001")
        _, next_page, _ = self.request(f"/api/v1/games?job_id={child}&limit=1&offset=1")
        self.assertEqual(next_page["games"][0]["epoch"], "00000000")
        _, filtered, _ = self.request(f"/api/v1/games?job_id={child}&epoch=00000000")
        self.assertEqual(filtered["total"], 1)
        _, game, _ = self.request(
            f"/api/v1/game?job_id={child}&epoch=00000000&id=00000000"
        )
        self.assertEqual(game["label"], "Parent 0")
        status, _, _ = self.request(
            f"/api/v1/game?job_id={child}&epoch=00000002&id=00000000"
        )
        self.assertEqual(status, 400)
        _, detail, _ = self.request(f"/api/v1/jobs/{child}")
        self.assertTrue(detail["has_games"])
        _, catalog, _ = self.request("/api/v1/game-jobs")
        self.assertIn(child, [job["id"] for job in catalog["jobs"]])
        self.assertEqual(
            self.server.app.game_paths(child), [originals[1], originals[0]]
        )
        self.assertFalse(self.server.app.archive(child).exists())

    def test_mixed_format_inherited_replays(self):
        from scripts.job_service import replays

        parent = self.archive_job("mixed-parent")
        old = self.archive_epoch(parent, 0, "old")
        newer = self.archive_epoch(parent, 1, "new")
        game = json.loads(newer.read_text()) | {
            "schema_version": 1,
            "game_id": "00000000",
        }
        replays.save(newer.parent, game)
        child = self.archive_job("mixed-child", self.archive_checkpoint(parent, 1))
        status, page, _ = self.request(f"/api/v1/games?job_id={child}")
        self.assertEqual(status, 200)
        self.assertEqual(page["total"], 2)
        self.assertEqual([g["id"] for g in page["games"]], ["00000000"] * 2)
        newer.unlink()
        for epoch, label in ((0, "old"), (1, "new")):
            status, restored, _ = self.request(
                f"/api/v1/game?job_id={child}&epoch={epoch:08}&id=00000000"
            )
            self.assertEqual(status, 200)
            self.assertEqual(restored["label"], label)
        self.assertTrue(old.exists())
        self.assertFalse(self.server.app.archive(child).exists())

    def test_replay_ancestry_chains_local_precedence_and_cycles(self):
        parent = self.archive_job("parent")
        for epoch in range(4):
            self.archive_epoch(parent, epoch, f"parent-{epoch}")
        child = self.archive_job("child", self.archive_checkpoint(parent, 1))
        local = self.archive_epoch(child, 1, "local")
        self.archive_epoch(child, 2, "child-2")
        grandchild = self.archive_job("grandchild", self.archive_checkpoint(child, 2))
        self.archive_epoch(grandchild, 3, "grandchild-3")
        paths = self.server.app.game_paths(grandchild)
        self.assertEqual(
            [p.parent.name for p in paths], [f"{e:08}" for e in (3, 2, 1, 0)]
        )
        self.assertEqual(paths[2], local)
        early = self.archive_job("early", self.archive_checkpoint(child, 0))
        self.assertEqual(
            [p.parent.name for p in self.server.app.game_paths(early)], ["00000000"]
        )
        # Old migrated archives may already contain symlinks; do not duplicate them.
        link = self.server.app.archive(child) / "00000000"
        link.symlink_to(
            self.server.app.archive(parent) / "00000000", target_is_directory=True
        )
        self.assertEqual(len(self.server.app.game_paths(grandchild)), 4)
        cycle = self.archive_checkpoint(child, 1)
        with self.store.transaction() as db:
            db.execute(
                "UPDATE jobs SET spec=? WHERE id=?",
                (
                    json.dumps(SPEC | {"inputs": {"checkpoint": cycle}}),
                    parent,
                ),
            )
        self.assertEqual(len(self.server.app.game_paths(grandchild)), 4)
        # A different job kind with checkpoint inputs does not imply continuation.
        with self.store.transaction() as db:
            db.execute(
                "UPDATE jobs SET spec=? WHERE id=?",
                (
                    json.dumps(
                        SPEC
                        | {
                            "kind": "historical_training",
                            "inputs": {"checkpoint": cycle},
                        }
                    ),
                    early,
                ),
            )
        self.assertEqual(self.server.app.game_paths(early), [])

    def test_dynamic_replay_parent_uses_pinned_epoch_not_current_latest(self):
        parent = self.archive_job("parent")
        for epoch in range(3):
            self.archive_epoch(parent, epoch, f"epoch-{epoch}")
        self.archive_checkpoint(parent, 2)
        child = self.archive_job("child", {"job_id": parent, "selection": "latest"})
        self.assertEqual(self.server.app.game_paths(child), [])
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO attempts(id,job_id,number,state,created,effective) VALUES (?,?,?,?,?,?)",
                (
                    "resolved",
                    child,
                    1,
                    "succeeded",
                    0,
                    json.dumps({"resolved_inputs": {"checkpoint": {"epoch": 1}}}),
                ),
            )
        self.assertEqual(
            [p.parent.name for p in self.server.app.game_paths(child)],
            ["00000001", "00000000"],
        )

    def test_live_progress_does_not_depend_on_historical_event_pages(self):
        job = self.store.command("live", {"action": "create", "spec": SPEC})["job_id"]
        with self.store.transaction() as db:
            for epoch in range(3):
                self.store.event(
                    db, job, None, "stage", {"stage": "self_play", "epoch": epoch}
                )
                for games in range(300):
                    self.store.event(
                        db,
                        job,
                        None,
                        "self_play_progress",
                        {"games_completed": games, "epoch": epoch},
                    )
            self.store.event(
                db,
                job,
                None,
                "game_completed",
                {"games_completed": 300, "games_total": 300},
            )
        _, history, _ = self.request(f"/api/v1/events?job_id={job}&after=0")
        self.assertEqual(len(history["events"]), 500)
        _, detail, _ = self.request(f"/api/v1/jobs/{job}")
        self.assertEqual(
            detail["runtime"]["self_play_progress"]["payload"],
            {"games_completed": 300, "games_total": 300, "active_games": 0, "epoch": 2},
        )
        with self.store.transaction() as db:
            self.store.event(db, job, None, "stage", {"stage": "self_play", "epoch": 3})
        _, detail, _ = self.request(f"/api/v1/jobs/{job}")
        self.assertNotIn("self_play_progress", detail["runtime"])

    def test_collection_and_comparison_telemetry_remain_stage_aware(self):
        job = self.store.command("stages", {"action": "create", "spec": SPEC})["job_id"]
        with self.store.transaction() as db:
            self.store.event(db, job, None, "stage", {"stage": "self_play", "epoch": 1})
            self.store.event(
                db,
                job,
                None,
                "self_play_progress",
                {
                    "games_completed": 10,
                    "games_total": 10,
                    "inference": {"invocations": 12},
                },
            )
            self.store.event(db, job, None, "stage", {"stage": "training", "epoch": 1})
        _, detail, _ = self.request(f"/api/v1/jobs/{job}")
        self.assertNotIn("self_play_progress", detail["runtime"])
        self.assertEqual(
            detail["runtime"]["last_collection"]["payload"]["games_completed"], 10
        )
        with self.store.transaction() as db:
            self.store.event(
                db,
                job,
                None,
                "comparison_progress",
                {
                    "games_completed": 2,
                    "first_inference": {"invocations": 20},
                    "second_inference": {"invocations": 30},
                },
            )
            self.store.event(
                db, job, None, "comparison_progress", {"games_completed": 3}
            )
        _, detail, _ = self.request(f"/api/v1/jobs/{job}")
        payload = detail["runtime"]["comparison_progress"]["payload"]
        self.assertEqual(payload["games_completed"], 3)
        self.assertEqual(payload["first_inference"]["invocations"], 20)
        self.assertEqual(payload["second_inference"]["invocations"], 30)

    def test_analysis_cancellation_events_correlate_without_credentials(self):
        class FakeAnalysis:
            digest = "fixture"

            def request(self, body, on_update=None, owner=None):
                return {"result": {"cancelled": True}}

            def cancel(self, request_id, owner):
                return {"cancelled": True}

        self.server.app.analysis = FakeAnalysis()
        _, session, headers = self.request(
            "/api/v1/login", {"password": "long test password"}
        )
        credentials = {
            "Cookie": headers["Set-Cookie"].split(";")[0],
            "X-CSRF-Token": session["csrf"],
        }
        body = {"request_id": "client-search", "simulations": 5}
        self.assertEqual(self.request("/api/v1/analyze", body, credentials)[0], 200)
        self.assertEqual(
            self.request("/api/v1/analysis/cancel", body, credentials)[0], 200
        )
        events = [e for e in self.store.events() if e["kind"].startswith("analysis_")]
        self.assertEqual(
            [e["kind"] for e in events],
            [
                "analysis_requested",
                "analysis_cancelled",
                "analysis_cancel_requested",
                "analysis_cancel_acknowledged",
            ],
        )
        self.assertTrue(
            all(e["payload"]["client_request_id"] == "client-search" for e in events)
        )
        self.assertEqual(
            events[0]["payload"]["request_id"], events[1]["payload"]["request_id"]
        )
        self.assertNotIn(session["csrf"], json.dumps(events))
        self.assertNotIn(credentials["Cookie"], json.dumps(events))

    def test_summary_and_recent_history_endpoints(self):
        job = self.store.command("summary-create", {"action": "create", "spec": SPEC})[
            "job_id"
        ]
        with self.store.transaction() as db:
            for _ in range(110):
                self.store.event(db, job, None, "heartbeat", {})
            self.store.event(
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
        status, summary, _ = self.request(f"/api/v1/job-summary?job_id={job}")
        self.assertEqual(status, 200)
        self.assertEqual(summary["comparison"]["games"], 1)
        status, page, _ = self.request(f"/api/v1/event-history?job_id={job}")
        self.assertEqual(status, 200)
        self.assertEqual(page["events"][0]["kind"], "comparison_game")
        self.assertEqual(len(page["events"]), 100)
        self.assertTrue(page["has_more"])
        status, older, _ = self.request(
            f"/api/v1/event-history?job_id={job}&before={page['events'][-1]['id']}"
        )
        self.assertEqual(status, 200)
        self.assertFalse(older["has_more"])
        self.assertLess(older["events"][0]["id"], page["events"][-1]["id"])

    def test_analysis_stream_requires_auth_and_delivers_progress_then_completion(self):
        class FakeAnalysis:
            digest = "test-checkpoint"

            def request(self, body, on_update=None, owner=None):
                for complete in (False, True):
                    result = {
                        "checkpoint": {"model_sha256": self.digest},
                        "result": {"complete": complete},
                    }
                    on_update(result)
                return result

        self.server.app.analysis = FakeAnalysis()
        self.assertEqual(self.request("/api/v1/analyze/stream", {})[0], 403)
        self.assertEqual(
            self.request("/api/v1/analysis/cancel", {"request_id": "search"})[0], 403
        )
        _, session, headers = self.request(
            "/api/v1/login", {"password": "long test password"}
        )
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        conn.request(
            "POST",
            "/api/v1/analyze/stream",
            "{}",
            {
                "Origin": self.origin,
                "Content-Type": "application/json",
                "Cookie": headers["Set-Cookie"].split(";")[0],
                "X-CSRF-Token": session["csrf"],
            },
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("X-Accel-Buffering"), "no")
        self.assertEqual(
            [
                json.loads(line)["result"]["complete"]
                for line in response.read().splitlines()
            ],
            [False, True],
        )
        conn.close()

    def test_public_cannot_mutate_and_login_requires_csrf_and_origin(self):
        self.assertEqual(self.request("/api/v1/jobs")[0], 200)
        command = {"command_id": "new", "request": {"action": "create", "spec": SPEC}}
        self.assertEqual(self.request("/api/v1/commands", command)[0], 403)
        status, session, headers = self.request(
            "/api/v1/login", {"password": "long test password"}
        )
        self.assertEqual(status, 200)
        cookie = headers["Set-Cookie"].split(";")[0]
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("Max-Age=31536000", headers["Set-Cookie"])
        self.assertNotIn("expires", session)
        _, restored, refreshed_headers = self.request(
            "/api/v1/session", headers={"Cookie": cookie}
        )
        self.assertTrue(restored["authenticated"])
        self.assertEqual(restored["csrf"], session["csrf"])
        self.assertEqual(refreshed_headers["Set-Cookie"], headers["Set-Cookie"])
        self.assertEqual(
            self.request("/api/v1/commands", command, {"Cookie": cookie})[0], 403
        )
        credentials = {"Cookie": cookie, "X-CSRF-Token": session["csrf"]}
        self.assertEqual(
            self.request(
                "/api/v1/commands",
                command,
                credentials | {"Origin": "https://evil.invalid"},
            )[0],
            403,
        )
        self.assertEqual(self.request("/api/v1/commands", command, credentials)[0], 200)
        self.request("/api/v1/logout", {}, credentials)
        self.assertEqual(self.request("/api/v1/commands", command, credentials)[0], 403)


class AnalysisLifetimeTests(unittest.TestCase):
    def test_cancel_yields_without_killing_tree_and_is_scoped_to_request_owner(self):
        from scripts.job_service.analysis import AnalysisWorker

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({"model_sha256": "fixture"}))
            binary = root / "fake-analysis"
            binary.write_text("""#!/usr/bin/env python3
import sys,json,os,time
completed=0
for line in sys.stdin:
 body=json.loads(line)
 assert 'request_id' not in body
 carried=completed
 time.sleep(.02)
 completed=carried+body['simulations']
 print(json.dumps({'checkpoint':{'model_sha256':'fixture'},'result':{'complete':True,'pid':os.getpid(),'searched_simulations':completed,'carried_visits':carried}}),flush=True)
""")
            binary.chmod(0o755)
            worker = AnalysisWorker(binary, root)
            updates, errors = [], []
            ready = threading.Event()

            def update(result):
                updates.append(result["result"])
                ready.set()

            def run():
                try:
                    worker.request(
                        {"simulations": 4000, "request_id": "search"},
                        update,
                        owner="one",
                    )
                except Exception as error:
                    errors.append(error)

            try:
                thread = threading.Thread(target=run)
                thread.start()
                self.assertTrue(ready.wait(5))
                worker.cancel("search", "other")
                self.assertFalse(worker.active[2].is_set())
                worker.cancel("different", "one")
                self.assertFalse(worker.active[2].is_set())
                worker.cancel("search", "one")
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                last = updates[-1]
                self.assertTrue(last["complete"])
                self.assertTrue(last["cancelled"])
                self.assertLess(last["searched_simulations"], 4000)
                self.assertTrue(all(r["target_simulations"] == 4000 for r in updates))
                self.assertTrue(all(not r["complete"] for r in updates[:-1]))
                continued = worker.request(
                    {"simulations": last["searched_simulations"] + 64}, owner="one"
                )["result"]
                self.assertEqual(continued["pid"], last["pid"])
                self.assertEqual(
                    continued["carried_visits"], last["searched_simulations"]
                )
                self.assertFalse(continued["cancelled"])
                worker.cancel("early", "one")
                with self.assertRaisesRegex(Conflict, "cancelled"):
                    worker.request(
                        {"simulations": 4000, "request_id": "early"}, owner="one"
                    )
                self.assertIsNone(worker.process.poll())
            finally:
                worker.close()

    def test_initial_snapshot_contains_retained_tree_and_selected_activations(self):
        from scripts.job_service.analysis import AnalysisWorker

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({"model_sha256": "fixture"}))
            binary = root / "fake-analysis"
            binary.write_text("""#!/usr/bin/env python3
import sys,json
completed=990
for line in sys.stdin:
 body=json.loads(line)
 carried=completed
 completed=carried+body['simulations']
 activations=[{'name':body['layers'][0],'values':[1]}] if body['inspect'] else []
 print(json.dumps({'checkpoint':{'model_sha256':'fixture'},'result':{'complete':True,'searched_simulations':completed,'carried_visits':carried,'activations':activations}}),flush=True)
""")
            binary.chmod(0o755)
            worker = AnalysisWorker(binary, root)
            try:
                self.assertEqual(worker.maximum, 20000)
                self.assertEqual(worker.argv[-1], "20000")
                self.assertEqual(worker.timeout, 900)
                with self.assertRaisesRegex(ValueError, "Invalid simulation budget"):
                    worker.request({"simulations": 20001})
                frames = []
                result = worker.request(
                    {"simulations": 20000, "inspect": True, "layers": ["chosen-layer"]},
                    frames.append,
                )
                initial = frames[0]["result"]
                self.assertFalse(initial["complete"])
                self.assertEqual(initial["searched_simulations"], 990)
                self.assertEqual(initial["carried_visits"], 990)
                self.assertEqual(initial["activations"][0]["name"], "chosen-layer")
                # 20000 new simulations on top of the 990 retained visits.
                self.assertEqual(result["result"]["searched_simulations"], 20990)
                self.assertEqual(result["result"]["target_simulations"], 20990)
                self.assertTrue(result["result"]["complete"])
                self.assertEqual(
                    result["result"]["activations"], initial["activations"]
                )
            finally:
                worker.close()

    def test_native_process_survives_request_thread_exit(self):
        from scripts.job_service.analysis import AnalysisWorker

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({"model_sha256": "fixture"}))
            binary = root / "fake-analysis"
            binary.write_text(
                "#!/usr/bin/env python3\nimport sys,json,os\nfor line in sys.stdin:\n print(json.dumps({'checkpoint':{'model_sha256':'fixture'},'result':{'complete':True,'pid':os.getpid()}}),flush=True)\n"
            )
            binary.chmod(0o755)
            worker = AnalysisWorker(binary, root)
            try:
                results = []

                def call():
                    results.append(worker.request({"simulations": 1})["result"]["pid"])

                first = threading.Thread(target=call)
                first.start()
                first.join()
                time.sleep(0.1)
                second = threading.Thread(target=call)
                second.start()
                second.join()
                self.assertEqual(len(results), 2)
                self.assertEqual(results[0], results[1])
                self.assertIsNone(worker.process.poll())
            finally:
                worker.close()


class HistoricalMigrationTests(unittest.TestCase):
    def test_retained_source_rejects_a_partial_previous_copy(self):
        from scripts.job_service.importer import retain_source

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "replay.jsonl"
            source.write_bytes(b"complete source evidence\n")
            expected, retained = retain_source(root, source)
            self.assertEqual(retain_source(root, source), (expected, retained))
            retained.write_bytes(b"complete")
            with self.assertRaisesRegex(ValueError, "Retained source checksum"):
                retain_source(root, source)
            self.assertEqual(source.read_bytes(), b"complete source evidence\n")

    def test_verified_copy_refuses_changed_source_and_preserves_comparison_moves(self):
        from scripts.job_service.migrate import verified_copy
        from scripts.job_service.importer import converted_battle_game

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.write_bytes(b"original")
            first = verified_copy(source, target)
            self.assertEqual(verified_copy(source, target), first)
            replay = root / "replay.bin.zst"
            replay.write_bytes(b"original")
            resumed = root / "resumed" / "replay.bin.zst"
            resumed.parent.mkdir()
            partial = resumed.with_name(resumed.name + ".copying")
            partial.write_bytes(b"torn")
            self.assertEqual(verified_copy(replay, resumed, {first: target}), first)
            self.assertEqual(resumed.read_bytes(), b"original")
            self.assertFalse(partial.exists())
            source.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verified_copy(source, target)
            self.assertEqual(target.read_bytes(), b"original")
        report = {
            "first_checkpoint": {"model_sha256": "a" * 64},
            "second_checkpoint": {"model_sha256": "b" * 64},
        }
        game = {
            "game": 1,
            "plies": 2,
            "first_seat": "first_checkpoint",
            "second_seat": "second_checkpoint",
            "winner": "second_checkpoint",
            "moves": [
                {"ply": 1, "row": 4, "column": 7, "checkpoint": "first_checkpoint"},
                {"ply": 2, "row": 6, "column": 8, "checkpoint": "second_checkpoint"},
            ],
        }
        converted = converted_battle_game(game, report, "c" * 64)
        self.assertEqual(converted["models_by_seat"]["First"], "a" * 64)
        self.assertEqual(converted["record"]["plies"][0]["action"], {"x": 3, "y": 6})
        self.assertEqual(converted["record"]["plies"][1]["actor"], "Second")
        self.assertIsNone(
            converted["record"]["plies"][0]["decision"]["training_policy"]
        )
        self.assertEqual(converted["record"]["plies"][0]["value_actual"], -1)
        game["moves"][1].update(row=4, column=7)
        with self.assertRaises(ValueError):
            converted_battle_game(game, report, "c" * 64)


class MemoryAccountingTests(unittest.TestCase):
    def test_clean_active_cache_does_not_look_like_anonymous_pressure(self):
        from scripts.job_service.resources import cgroup_memory

        value = cgroup_memory(
            124,
            125,
            {
                "total_cache": 21,
                "total_inactive_file": 9,
                "total_mapped_file": 1,
                "cache": 1,
            },
        )
        self.assertEqual(value["used_bytes"], 115)
        self.assertEqual(value["pressure_bytes"], 104)
        self.assertEqual(value["raw_used_bytes"], 124)

    def test_shared_mapped_and_dirty_memory_remain_charged(self):
        from scripts.job_service.resources import cgroup_memory

        value = cgroup_memory(
            100,
            125,
            {
                "file": 40,
                "shmem": 10,
                "file_mapped": 5,
                "file_dirty": 7,
                "file_writeback": 2,
                "unevictable": 1,
            },
        )
        self.assertEqual(value["pressure_bytes"], 85)
        self.assertEqual(cgroup_memory(100, 125, {})["pressure_bytes"], 100)
        self.assertEqual(
            cgroup_memory(100, 125, {"file": 10, "shmem": 20})["pressure_bytes"], 100
        )


class BackfillTests(unittest.TestCase):
    def test_reuse_completes_an_interrupted_archive_copy(self):
        from scripts.job_service.backfill import backfill
        from scripts.job_service.catalog import digest
        from scripts.job_service.importer import identity, import_snapshot
        from scripts.job_service.store import encode
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new"
            previous = Path(temporary) / "previous"
            store = Store(root / "jobs.sqlite3")
            snapshot = root / "snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "phases": [
                            {"id": "run", "kind": "training", "state": "completed"}
                        ]
                    }
                )
            )
            import_snapshot(store, root, snapshot)
            job = identity("legacy", "run")
            replay = root / "jobs" / job / "checkpoints/00000000/replay.bin.zst"
            replay.parent.mkdir(parents=True)
            replay.write_bytes(b"immutable replay")
            policy = "normalized_root_visits"
            key = hashlib.sha256(
                encode(
                    {"replays": [digest(replay)], "policy_semantics": policy}
                ).encode()
            ).hexdigest()
            archive = previous / "jobs" / job / "games/archive/00000000"
            archive.mkdir(parents=True)
            for name in ("a.json", "b.json"):
                (archive / name).write_text(name)
            partial = root / "jobs" / job / "games/archive/00000000"
            partial.mkdir(parents=True)
            (partial / "a.json").write_text("a.json")
            rows = previous / "rows.jsonl"
            rows.write_text("retained export")
            receipt = previous / "backfill" / key / "receipt.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(
                json.dumps(
                    {
                        "dataset_key": key,
                        "retained_source": str(rows),
                        "source_sha256": digest(rows),
                        "imported_games": 2,
                        "total_games": 2,
                    }
                )
            )
            result = backfill(root, root / "must-not-export", {job: policy}, previous)
            self.assertEqual(result[key]["games"], 2)
            self.assertEqual((partial / "b.json").read_text(), "b.json")
            self.assertEqual((partial / "a.json").read_text(), "a.json")

    def test_identical_replay_bytes_cannot_have_conflicting_policy_labels(self):
        from scripts.job_service.backfill import backfill
        from scripts.job_service.importer import identity, import_snapshot

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store(root / "jobs.sqlite3")
            snapshot = root / "snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "phases": [
                            {"id": name, "kind": "training", "state": "completed"}
                            for name in ("first", "second")
                        ]
                    }
                )
            )
            import_snapshot(store, root, snapshot)
            semantics = {}
            for name, policy in zip(
                ("first", "second"),
                ("normalized_root_visits", "temperature_adjusted_policy"),
            ):
                job = identity("legacy", name)
                replay = root / "jobs" / job / "checkpoints/00000000/replay.bin.zst"
                replay.parent.mkdir(parents=True)
                replay.write_bytes(b"same immutable replay")
                semantics[job] = policy
            with self.assertRaisesRegex(ValueError, "conflicting policy semantics"):
                backfill(root, root / "exporter-must-not-run", semantics)


if __name__ == "__main__":
    unittest.main()
