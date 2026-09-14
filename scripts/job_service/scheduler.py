"""Admission, durable reconciliation, and artifact collection; no experiment identities."""

from __future__ import annotations

import fcntl
import logging
import threading
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .spec import compile_argv, environment_overrides
from .resources import memory_pressure
from .worker import atomic_json, emit, process_identity
from .store import encode
from .journal import ingest_native, read_batch


class Scheduler:
    def __init__(self, store, root, binary, capacity, repository):
        self.store, self.root = store, Path(root).resolve()
        self.binary, self.capacity = Path(binary).resolve(), capacity
        self.repository = Path(repository).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.children = []
        self.preparations = {}
        self.finalizations = {}
        self.checkpoint_publications = {}

    def resolve_artifact(self, identifier):
        with self.store.transaction(write=False) as db:
            if isinstance(identifier, dict):
                choices = db.execute(
                    "SELECT * FROM artifacts WHERE job_id=? AND kind='checkpoint'",
                    (identifier["job_id"],),
                ).fetchall()
                if not choices:
                    raise ValueError("Dependency produced no checkpoints")
                if any(
                    not isinstance(json.loads(a["metadata"]).get("epoch"), int)
                    for a in choices
                ):
                    raise ValueError(
                        "Checkpoint catalog is missing epoch metadata; re-register its input"
                    )
                if identifier["selection"] == "latest":
                    artifact = max(
                        choices, key=lambda a: json.loads(a["metadata"])["epoch"]
                    )
                else:
                    scores = {}
                    for event in db.execute(
                        "SELECT payload FROM events WHERE job_id=? AND kind='epoch_completed'",
                        (identifier["job_id"],),
                    ):
                        metric = json.loads(event[0])
                        loss = (metric.get("validation") or {}).get("value_loss")
                        if isinstance(loss, (int, float)):
                            scores[metric["epoch"]] = loss
                    available = [
                        a
                        for a in choices
                        if json.loads(a["metadata"])["epoch"] in scores
                    ]
                    if not available:
                        raise ValueError(
                            "No recorded validation losses for checkpoint selection"
                        )
                    artifact = min(
                        available,
                        key=lambda a: (
                            scores[json.loads(a["metadata"])["epoch"]],
                            json.loads(a["metadata"])["epoch"],
                        ),
                    )
                identifier = artifact["id"]
            artifact = db.execute(
                "SELECT * FROM artifacts WHERE id=?", (identifier,)
            ).fetchone()
            if artifact is None:
                raise ValueError("Unknown artifact")
        path = Path(artifact["path"])
        with path.open("rb") as f:
            primary_digest = hashlib.file_digest(f, "sha256").hexdigest()
            if primary_digest != artifact["sha256"]:
                raise ValueError("Artifact content changed")
        if artifact["kind"] in ("checkpoint", "replay"):
            for name, expected in (
                json.loads(artifact["metadata"]).get("files", {}).items()
            ):
                if path.parent / name == path:
                    actual = primary_digest
                else:
                    with (path.parent / name).open("rb") as file:
                        actual = hashlib.file_digest(file, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError(f"Checkpoint input changed: {name}")
        if artifact["kind"] == "history":
            manifest = json.loads(path.read_text())
            directory = Path(manifest["directory"])
            for relative, expected in manifest["files"].items():
                with (directory / relative).open("rb") as file:
                    if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                        raise ValueError("History input changed")
            return directory
        return path.parent if artifact["kind"] in ("checkpoint", "replay") else path

    def effective(self, job_id, attempt_id, spec):
        with self.store.transaction(write=False) as db:
            self.store.validate_inputs(db, spec)
        paths = {}

        def resolve(reference):
            key = encode(reference)
            if key not in paths:
                paths[key] = self.resolve_artifact(reference)
            return paths[key]

        output = self.root / "jobs" / job_id
        output.mkdir(parents=True, exist_ok=True)
        initial = spec["inputs"].get("checkpoint")
        if initial and spec["kind"] == "self_play":
            source = resolve(initial)
            metadata = json.loads((source / "metadata.json").read_text())
            target = output / "checkpoints" / f"{metadata['epoch']:08}"
            if not (output / "initialization.json").exists():
                if target.exists():
                    with (target / "model.safetensors").open("rb") as f:
                        if (
                            hashlib.file_digest(f, "sha256").hexdigest()
                            != metadata["model_sha256"]
                        ):
                            raise ValueError(
                                "Existing initialization differs from requested checkpoint"
                            )
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".initializing")
                    if temporary.exists():
                        shutil.rmtree(temporary)
                    shutil.copytree(source, temporary)
                    os.replace(temporary, target)
                atomic_json(
                    output / "initialization.json",
                    {
                        "checkpoint_artifact": initial,
                        "model_sha256": metadata["model_sha256"],
                    },
                )
        binary = self.binary
        if spec.get("binary_artifact"):
            binary = resolve(spec["binary_artifact"])
        with binary.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        pinned = self.root / "binaries" / digest
        pinned.parent.mkdir(parents=True, exist_ok=True)
        if not pinned.exists():
            temporary = pinned.with_suffix(f".{attempt_id}.tmp")
            shutil.copy2(binary, temporary)
            with temporary.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                    raise ValueError("Binary changed while being pinned")
            os.replace(temporary, pinned)
        resolved = {}
        participants = {}
        for name, reference in spec["inputs"].items():
            path = resolve(reference)
            if name == "history":
                resolved[name] = {"path": str(path), "artifact_id": reference}
                continue
            descriptor = json.loads((path / "metadata.json").read_text())
            resolved[name] = {
                "path": str(path),
                "model_sha256": descriptor["model_sha256"],
                "epoch": descriptor["epoch"],
            }
            with self.store.transaction(write=False) as db:
                if isinstance(reference, dict):
                    source_job = reference["job_id"]
                else:
                    source_job = db.execute(
                        "SELECT job_id FROM artifacts WHERE id=?", (reference,)
                    ).fetchone()[0]
                source = db.execute(
                    "SELECT title FROM jobs WHERE id=?", (source_job,)
                ).fetchone()
            if name in ("first", "second"):
                participants[name] = {
                    "label": f"{source['title'] if source else 'Checkpoint'} · epoch {descriptor['epoch'] + 1}",
                    "model_sha256": descriptor["model_sha256"],
                }
        return {
            "job_id": job_id,
            "attempt_id": attempt_id,
            "resources": spec["resources"],
            "spec": spec,
            "environment_overrides": environment_overrides(spec),
            "argv": compile_argv(spec, pinned, output, resolve),
            "binary_sha256": digest,
            "resolved_inputs": resolved,
            "participants": participants,
            "cwd": str(self.repository),
            "output": str(output),
        }

    def tick(self):
        self.children = [p for p in self.children if p.poll() is None]
        self.preparations = {
            key: thread
            for key, thread in self.preparations.items()
            if thread.is_alive()
        }
        self.finalizations = {
            key: thread
            for key, thread in self.finalizations.items()
            if thread.is_alive()
        }
        attempts = self.store.active_attempts()
        active_ids = {a["id"] for a in attempts}
        self.checkpoint_publications = {
            key: entry
            for key, entry in self.checkpoint_publications.items()
            if key in active_ids or entry[0].is_alive()
        }
        for attempt in attempts:
            directory = self.root / "attempts" / attempt["id"]
            directory.mkdir(parents=True, exist_ok=True)
            if attempt["state"] == "preparing":
                self.prepare(attempt, directory)
                continue
            if not (directory / "request.json").exists():
                atomic_json(directory / "request.json", attempt["effective"])
            try:
                self.reconcile(attempt, directory)
            except Exception:
                # One damaged journal must not starve controls/admission for other jobs.
                logging.exception("Reconciliation failed for %s", attempt["job_id"])
        memory = memory_pressure()
        if (
            memory
            and memory.get("pressure_bytes", memory["used_bytes"])
            > memory["limit_bytes"] * 0.9
        ):
            return None
        claimed = self.store.claim(self.capacity)
        if claimed:
            directory = self.root / "attempts" / claimed["id"]
            directory.mkdir(parents=True, exist_ok=True)
            self.prepare(claimed, directory)
        return claimed

    def prepare(self, attempt, directory):
        """Recoverable preparation outside the scheduler loop and database transactions."""
        if attempt["id"] in self.preparations:
            return

        def run():
            # Independent schedulers/restarts may see the same reservation. Only one
            # may copy inputs; the durable state is checked again after locking.
            with (directory / "preparation.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
                active = next(
                    (
                        a
                        for a in self.store.active_attempts()
                        if a["id"] == attempt["id"] and a["state"] == "preparing"
                    ),
                    None,
                )
                if active is None:
                    return
                try:
                    if self.store.pending_control(attempt["job_id"]):
                        self.store.finish_preparation(attempt["id"])
                        return
                    effective = self.effective(
                        attempt["job_id"], attempt["id"], active["effective"]["spec"]
                    )
                    self.store.finish_preparation(attempt["id"], effective)
                except Exception as error:
                    logging.exception(
                        "Job preparation failed for %s", attempt["job_id"]
                    )
                    self.store.finish_preparation(attempt["id"], error=str(error))

        thread = threading.Thread(
            target=run, name=f"prepare-{attempt['id']}", daemon=True
        )
        self.preparations[attempt["id"]] = thread
        thread.start()

    def launch(self, directory):
        log = (directory / "owner.log").open("ab")
        try:
            self.children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "scripts.job_service.worker",
                        str(directory),
                    ],
                    cwd=self.repository,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        finally:
            log.close()

    def reconcile(self, attempt, directory):
        if attempt["state"] == "finalizing":
            self.finalize(attempt, directory)
            return
        control = self.store.pending_control(attempt["job_id"])
        if control:
            atomic_json(directory / "control.json", control)
        started = directory / "started.json"
        if not started.exists():
            # Re-launching is safe: the worker's exclusive lock and started marker
            # close the scheduler-crash window between reservation and spawning.
            self.launch(directory)
            return
        owner = json.loads(started.read_text())["owner"]
        result = directory / "result.json"
        if not result.exists() and process_identity(owner["pid"]) != owner:
            process_path = directory / "process.json"
            process = (
                json.loads(process_path.read_text()) if process_path.exists() else {}
            )
            child = process.get("child")
            if child and process_identity(child["pid"]) == child:
                # Native work still owns these resources. Never start a duplicate.
                return
            payload = {
                "returncode": -1,
                "stopped": False,
                "reason": "Worker disappeared without a completion receipt",
            }
            atomic_json(result, payload)
            emit(directory, "worker_finished", payload)
        if result.exists() or attempt["state"] == "finalizing":
            self.finalize(attempt, directory)
            return
        ingest_native(self.store, attempt, directory / "events.jsonl")
        records, cursor, _ = read_batch(
            directory / "worker-events.jsonl", attempt["cursor"]
        )
        if any(r["kind"] == "worker_finished" for r in records):
            self.finalize(attempt, directory)
            return
        if records:
            self.store.ingest(attempt["id"], records, cursor)
        self.publish_checkpoints(attempt, directory)

    def publish_checkpoints(self, attempt, directory):
        """Expose immutable epoch snapshots without waiting for the run to stop."""
        if attempt["effective"]["spec"]["kind"] not in (
            "self_play",
            "replay_train",
            "reconstruction",
        ):
            return
        previous = self.checkpoint_publications.get(attempt["id"])
        now = time.monotonic()
        if previous and (previous[0].is_alive() or now < previous[1]):
            return

        def run():
            try:
                with (directory / "checkpoints.lock").open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return
                    self.collect_checkpoints(attempt, live=True)
            except Exception:
                logging.exception(
                    "Checkpoint publication failed for %s", attempt["job_id"]
                )

        thread = threading.Thread(
            target=run, name=f"checkpoints-{attempt['id']}", daemon=True
        )
        self.checkpoint_publications[attempt["id"]] = (thread, now + 30)
        thread.start()

    def finalize(self, attempt, directory):
        if attempt["id"] in self.finalizations:
            return
        if not self.store.begin_finalization(attempt["id"]):
            return

        def run():
            with (directory / "finalization.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
                if not self.store.begin_finalization(attempt["id"]):
                    return
                try:
                    while not ingest_native(
                        self.store, attempt, directory / "events.jsonl"
                    ):
                        pass
                    with (directory / "checkpoints.lock").open("a") as checkpoint_lock:
                        fcntl.flock(checkpoint_lock, fcntl.LOCK_EX)
                        self.collect_artifacts(attempt)
                    self.recover_metrics(attempt)
                    # The receipt precedes the owner's final journal append. Wait
                    # for that append, or repair it once the owner is gone.
                    while True:
                        with self.store.transaction(write=False) as db:
                            row = db.execute(
                                "SELECT cursor,state FROM attempts WHERE id=?",
                                (attempt["id"],),
                            ).fetchone()
                        if row["state"] != "finalizing":
                            return
                        records, cursor, caught_up = read_batch(
                            directory / "worker-events.jsonl", row["cursor"]
                        )
                        if records:
                            self.store.ingest(attempt["id"], records, cursor)
                            if any(r["kind"] == "worker_finished" for r in records):
                                return
                        if not caught_up:
                            continue
                        owner = json.loads((directory / "started.json").read_text())[
                            "owner"
                        ]
                        if process_identity(owner["pid"]) != owner:
                            emit(
                                directory,
                                "worker_finished",
                                json.loads((directory / "result.json").read_text()),
                            )
                        else:
                            time.sleep(0.05)
                except Exception as error:
                    logging.exception("Finalization failed for %s", attempt["job_id"])
                    self.store.fail_finalization(attempt["id"], str(error))

        thread = threading.Thread(
            target=run, name=f"finalize-{attempt['id']}", daemon=True
        )
        self.finalizations[attempt["id"]] = thread
        thread.start()

    def recover_metrics(self, attempt):
        """A crash after checkpoint publication must not lose its durable metrics."""
        output = Path(attempt["effective"]["output"])
        metrics = []
        for directory in (output / "stats", output / "epochs"):
            for path in directory.glob("[0-9]*.json"):
                metric = json.loads(path.read_text())
                epoch = metric.get("epoch")
                if (
                    epoch is not None
                    and (
                        output / "checkpoints" / f"{epoch:08}" / "metadata.json"
                    ).exists()
                ):
                    metrics.append(metric)
        with self.store.transaction() as db:
            recorded = {
                json.loads(row[0]).get("epoch")
                for row in db.execute(
                    "SELECT payload FROM events WHERE job_id=? AND kind='epoch_completed'",
                    (attempt["job_id"],),
                )
            }
            for metric in metrics:
                epoch = metric["epoch"]
                if epoch not in recorded:
                    self.store.event(
                        db,
                        attempt["job_id"],
                        attempt["id"],
                        "epoch_completed",
                        metric
                        | {
                            "metrics_recovery": "published metric file; event receipt was missing"
                        },
                    )
                    recorded.add(epoch)

    def collect_checkpoints(self, attempt, live=False):
        output = Path(attempt["effective"]["output"])
        from .catalog import register_checkpoint

        for path in sorted(output.glob("checkpoints/*/model.safetensors")):
            # Native writers atomically rename a pending directory to an epoch
            # number only after all checkpoint files have been written and synced.
            if not path.parent.name.isdecimal():
                continue
            try:
                register_checkpoint(
                    self.store, path.parent, attempt["job_id"], attempt["id"]
                )
            except Exception:
                if not live:
                    raise
                logging.exception(
                    "Cannot publish checkpoint %s; will retry", path.parent
                )

    def collect_artifacts(self, attempt):
        output = Path(attempt["effective"]["output"])
        self.collect_checkpoints(attempt)
        if attempt["effective"]["spec"]["kind"] in (
            "self_play",
            "reconstruction",
        ) and any(output.glob("checkpoints/*/replay.bin.zst")):
            from .catalog import register_history

            register_history(
                self.store, self.root, output, attempt["job_id"], attempt["id"]
            )
        for path in output.glob("result.allocator.json"):
            self.store.artifact(
                attempt["job_id"], attempt["id"], "allocator_profile", path
            )
        for path in output.glob("result.json"):
            self.store.artifact(attempt["job_id"], attempt["id"], "result", path)
            if attempt["effective"]["spec"]["kind"] == "comparison":
                report = json.loads(path.read_text())
                with self.store.transaction() as db:
                    completed = db.execute(
                        "SELECT 1 FROM events WHERE job_id=? AND kind='comparison_completed'",
                        (attempt["job_id"],),
                    ).fetchone()
                    if not completed:
                        recorded = {
                            json.loads(row[0]).get("game")
                            for row in db.execute(
                                "SELECT payload FROM events WHERE job_id=? AND kind='comparison_game'",
                                (attempt["job_id"],),
                            )
                        }
                        for game in report["games"]:
                            if game["game"] not in recorded:
                                self.store.event(
                                    db,
                                    attempt["job_id"],
                                    attempt["id"],
                                    "comparison_game",
                                    {k: v for k, v in game.items() if k != "moves"}
                                    | {"finish_time_unavailable": True},
                                )
                        self.store.event(
                            db,
                            attempt["job_id"],
                            attempt["id"],
                            "comparison_completed",
                            {k: v for k, v in report.items() if k != "games"},
                        )
