"""Transactional source of truth. Each operation uses its own SQLite connection."""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, metadata TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    title TEXT NOT NULL, spec TEXT NOT NULL, state TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL, reason TEXT, priority INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS dependencies (
                    job_id TEXT NOT NULL REFERENCES jobs(id), dependency_id TEXT NOT NULL REFERENCES jobs(id),
                    PRIMARY KEY(job_id, dependency_id));
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), number INTEGER NOT NULL,
                    state TEXT NOT NULL, created REAL NOT NULL, started REAL, ended REAL,
                    effective TEXT NOT NULL, cursor INTEGER NOT NULL DEFAULT 0, result TEXT,
                    UNIQUE(job_id,number));
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT REFERENCES jobs(id),
                    attempt_id TEXT REFERENCES attempts(id), time REAL NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS commands (
                    id TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, job_id TEXT REFERENCES jobs(id), attempt_id TEXT REFERENCES attempts(id),
                    kind TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, metadata TEXT NOT NULL,
                    UNIQUE(path,sha256));
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS event_job ON events(job_id,id);
                CREATE INDEX IF NOT EXISTS event_job_kind ON events(job_id,kind,id);
            """)
            if "fraction" not in {
                r[1] for r in db.execute("PRAGMA table_info(dependencies)")
            }:
                db.execute("ALTER TABLE dependencies ADD COLUMN fraction REAL")
            db.execute(
                "INSERT OR IGNORE INTO settings VALUES (?,?)", ("schema_version", "1")
            )
            if (
                db.execute(
                    'SELECT value FROM settings WHERE key="schema_version"'
                ).fetchone()[0]
                != "1"
            ):
                raise ValueError("Unsupported database schema")

    @contextlib.contextmanager
    def transaction(self, write=True):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def event(db, job_id, attempt_id, kind, payload, timestamp=None):
        db.execute(
            "INSERT INTO events(job_id,attempt_id,time,kind,payload) VALUES (?,?,?,?,?)",
            (
                job_id,
                attempt_id,
                time.time() if timestamp is None else timestamp,
                kind,
                encode(payload),
            ),
        )

    @staticmethod
    def validate_inputs(db, spec):
        from .spec import input_artifact_kinds

        for name, reference in spec["inputs"].items():
            allowed = input_artifact_kinds(name)
            if isinstance(reference, dict):
                parent = db.execute(
                    "SELECT spec FROM jobs WHERE id=?", (reference["job_id"],)
                ).fetchone()
                if not parent:
                    raise ValueError("Unknown checkpoint-producing job")
                producer = json.loads(parent["spec"])["kind"]
                historical_checkpoint = (
                    producer.startswith("historical_")
                    and db.execute(
                        "SELECT 1 FROM artifacts WHERE job_id=? AND kind='checkpoint' LIMIT 1",
                        (reference["job_id"],),
                    ).fetchone()
                )
                if (
                    producer not in ("self_play", "replay_train", "reconstruction")
                    and not historical_checkpoint
                ):
                    raise ValueError(
                        f"Input {name}: selected job does not produce checkpoints"
                    )
                if name.startswith("replay") and producer == "replay_train":
                    raise ValueError(
                        f"Input {name}: replay training does not produce self-play replays"
                    )
                continue
            artifact = db.execute(
                "SELECT kind,metadata FROM artifacts WHERE id=?", (reference,)
            ).fetchone()
            if not artifact or artifact["kind"] not in allowed:
                raise ValueError(
                    f"Input {name} requires a registered {' or '.join(sorted(allowed))} artifact"
                )
            if (
                name.startswith("replay")
                and artifact["kind"] == "checkpoint"
                and "replay.bin.zst"
                not in json.loads(artifact["metadata"]).get("files", {})
            ):
                raise ValueError(
                    f"Input {name}: checkpoint has no registered replay buffer"
                )

    def command(self, command_id, request):
        """Idempotency covers effects AND acknowledgement, in the same transaction."""
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 128:
            raise ValueError("A command id of 1–128 characters is required")
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        with self.transaction() as db:
            old = db.execute(
                "SELECT * FROM commands WHERE id=?", (command_id,)
            ).fetchone()
            if old:
                if old["request"] != encode(request):
                    raise Conflict(
                        "Command id was already used for a different request"
                    )
                return json.loads(old["response"])
            action = request.get("action")
            now = time.time()
            if action == "create":
                from .spec import validate_spec

                spec = validate_spec(request["spec"])
                if spec.get("binary_artifact"):
                    registered = db.execute(
                        "SELECT kind FROM artifacts WHERE id=?",
                        (spec["binary_artifact"],),
                    ).fetchone()
                    if not registered or registered["kind"] != "binary":
                        raise ValueError("Choose a trusted registered binary")
                self.validate_inputs(db, spec)
                experiment = request.get("experiment_id") or str(uuid.uuid4())
                title = str(request.get("title", spec["kind"]))[:200]
                db.execute(
                    "INSERT OR IGNORE INTO experiments VALUES (?,?,?,?)",
                    (experiment, request.get("experiment_title", title), "{}", now),
                )
                job_id = str(uuid.uuid4())
                db.execute(
                    "INSERT INTO jobs(id,experiment_id,title,spec,state,created,updated) VALUES (?,?,?,?,?,?,?)",
                    (job_id, experiment, title, encode(spec), "queued", now, now),
                )
                for dependency in request.get("dependencies", []):
                    fraction = (
                        dependency.get("fraction")
                        if isinstance(dependency, dict)
                        else None
                    )
                    dependency = (
                        dependency["job_id"]
                        if isinstance(dependency, dict)
                        else dependency
                    )
                    parent = db.execute(
                        "SELECT spec FROM jobs WHERE id=?", (dependency,)
                    ).fetchone()
                    if not parent:
                        raise ValueError("Unknown dependency")
                    if fraction is not None and (
                        isinstance(fraction, bool)
                        or not isinstance(fraction, (int, float))
                        or not 0 < fraction <= 1
                        or json.loads(parent["spec"])["kind"] != "comparison"
                    ):
                        raise ValueError(
                            "Progress thresholds require a comparison and a fraction in (0,1]"
                        )
                    db.execute(
                        "INSERT INTO dependencies(job_id,dependency_id,fraction) VALUES (?,?,?)",
                        (job_id, dependency, fraction),
                    )
                for ref in spec["inputs"].values():
                    if isinstance(ref, dict):
                        db.execute(
                            "INSERT OR IGNORE INTO dependencies(job_id,dependency_id,fraction) VALUES (?,?,NULL)",
                            (job_id, ref["job_id"]),
                        )
                response = {
                    "job_id": job_id,
                    "state": "queued",
                    "command_id": command_id,
                }
            elif action in ("pause", "resume", "cancel"):
                job_id = request["job_id"]
                row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                if row is None:
                    raise ValueError("Unknown job")
                state = row["state"]
                if action == "resume":
                    if state not in ("paused", "failed"):
                        raise Conflict("Only paused or failed jobs can be resumed")
                    target = "queued"
                else:
                    if state not in (
                        "queued",
                        "preparing",
                        "starting",
                        "running",
                        "stopping",
                        "paused",
                    ):
                        raise Conflict("Job is already terminal")
                    mode = request.get("mode", "epoch")
                    if mode not in ("epoch", "boundary", "immediate"):
                        raise ValueError("Unknown pause mode")
                    target = (
                        ("cancelled" if action == "cancel" else "paused")
                        if state in ("queued", "paused")
                        else "stopping"
                    )
                db.execute(
                    "UPDATE jobs SET state=?,updated=?,reason=? WHERE id=?",
                    (target, now, action, job_id),
                )
                response = {"job_id": job_id, "state": target, "command_id": command_id}
            elif action == "configure":
                from .spec import validate_spec

                job_id = request["job_id"]
                row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                if row is None or row["state"] not in ("queued", "paused", "failed"):
                    raise Conflict(
                        "Execution limits can only change while a job is inactive"
                    )
                options = request.get("options", {})
                allowed = {
                    "games-parallelism",
                    "inference-batch-size",
                    "batch-timeout-us",
                    "heartbeat-seconds",
                }
                if json.loads(row["spec"])["kind"] == "benchmark_executor":
                    allowed.add("disable-tf32")
                if not isinstance(options, dict) or set(options) - allowed:
                    raise ValueError(
                        "Only execution limits can change within a logical job"
                    )
                before = json.loads(row["spec"])
                spec = validate_spec(
                    before
                    | {
                        "options": before["options"] | options,
                        "resources": request.get("resources", before["resources"]),
                    }
                )
                db.execute(
                    "UPDATE jobs SET spec=?,updated=? WHERE id=?",
                    (encode(spec), now, job_id),
                )
                response = {
                    "job_id": job_id,
                    "state": row["state"],
                    "command_id": command_id,
                }
            elif action == "scheduler":
                if not isinstance(request.get("paused"), bool):
                    raise ValueError("paused must be Boolean")
                db.execute(
                    "INSERT OR REPLACE INTO settings VALUES (?,?)",
                    ("scheduler_paused", encode(request["paused"])),
                )
                job_id = None
                response = {"paused": request["paused"], "command_id": command_id}
            else:
                raise ValueError("Unknown command")
            self.event(
                db, job_id, None, "command", request | {"command_id": command_id}
            )
            db.execute(
                "INSERT INTO commands VALUES (?,?,?,?)",
                (command_id, encode(request), encode(response), now),
            )
            return response

    def snapshot(self, limit=25, offset=0, job_id=None):
        with self.transaction(write=False) as db:
            jobs = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM jobs WHERE (? IS NULL OR id=?) ORDER BY created DESC,id DESC LIMIT ? OFFSET ?",
                    (job_id, job_id, limit, offset),
                )
            ]
            for job in jobs:
                job["spec"] = json.loads(job["spec"])
                job["attempts"] = [
                    self.decode_attempt(r)
                    for r in db.execute(
                        "SELECT * FROM attempts WHERE job_id=? ORDER BY number",
                        (job["id"],),
                    )
                ]
                job["heartbeat"] = None
                if job["attempts"]:
                    heartbeat = (
                        self.path.parent
                        / "attempts"
                        / job["attempts"][-1]["id"]
                        / "heartbeat.json"
                    )
                    try:
                        job["heartbeat"] = json.loads(heartbeat.read_text())
                    except FileNotFoundError:
                        pass
                job["dependencies"] = [
                    r[0]
                    for r in db.execute(
                        "SELECT dependency_id FROM dependencies WHERE job_id=?",
                        (job["id"],),
                    )
                ]
            paused = db.execute(
                'SELECT value FROM settings WHERE key="scheduler_paused"'
            ).fetchone()
            return {
                "schema_version": 1,
                "jobs": jobs,
                "total": db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
                "scheduler_paused": bool(paused and json.loads(paused[0])),
                "server_time": time.time(),
            }

    @staticmethod
    def decode_attempt(row):
        result = dict(row)
        result["effective"] = json.loads(result["effective"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        return result

    def events(self, job_id=None, after=0, limit=500):
        with self.transaction(write=False) as db:
            rows = db.execute(
                "SELECT * FROM events WHERE id>? AND (? IS NULL OR job_id=?) ORDER BY id LIMIT ?",
                (after, job_id, job_id, limit),
            )
            return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    def event_page(self, job_id, before=None, limit=100):
        with self.transaction(write=False) as db:
            rows = db.execute(
                "SELECT * FROM events WHERE job_id=? AND id<? ORDER BY id DESC LIMIT ?",
                (
                    job_id,
                    before if before is not None else 9223372036854775807,
                    limit + 1,
                ),
            ).fetchall()
        return {
            "events": [
                dict(row) | {"payload": json.loads(row["payload"])}
                for row in rows[:limit]
            ],
            "has_more": len(rows) > limit,
        }

    def job_summary(self, job_id):
        from .summaries import summarize

        with self.transaction(write=False) as db:
            if not db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                raise ValueError("Unknown job")
            rows = db.execute(
                "SELECT time,kind,payload FROM events WHERE job_id=? AND kind IN "
                "('worker_started','comparison_game','comparison_completed','benchmark_completed','imported') ORDER BY id",
                (job_id,),
            ).fetchall()
        return summarize(
            [dict(row) | {"payload": json.loads(row["payload"])} for row in rows]
        )

    def epoch_summaries(self, job_id, after=0):
        """Chart data without paging through game and heartbeat events."""
        with self.transaction(write=False) as db:
            rows = db.execute(
                "SELECT * FROM events WHERE job_id=? AND kind='epoch_completed' "
                "AND id>? ORDER BY id",
                (job_id, after),
            )
            return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    def reserve(self, capacity):
        """Reserve resources before launching. Only one scheduler can claim a job."""
        with self.transaction() as db:
            paused = db.execute(
                'SELECT value FROM settings WHERE key="scheduler_paused"'
            ).fetchone()
            if paused and json.loads(paused[0]):
                return None
            active = [
                json.loads(r[0])["resources"]
                for r in db.execute(
                    "SELECT effective FROM attempts WHERE state IN ('preparing','starting','running','stopping','finalizing')"
                )
            ]
            used = {k: sum(r[k] for r in active) for k in capacity}
            for row in db.execute(
                "SELECT * FROM jobs WHERE state='queued' ORDER BY priority DESC,created,id"
            ).fetchall():
                spec = json.loads(row["spec"])
                oversized = [
                    f"{k}: requested {spec['resources'][k]}, capacity {capacity[k]}"
                    for k in capacity
                    if spec["resources"][k] > capacity[k]
                ]
                unavailable = [
                    f"{k}: requested {spec['resources'][k]}, available {max(0, capacity[k] - used[k])}"
                    for k in capacity
                    if used[k] + spec["resources"][k] > capacity[k]
                ]
                reason = (
                    "Exceeds host capacity: " + "; ".join(oversized)
                    if oversized
                    else "Waiting for resources: " + "; ".join(unavailable)
                    if unavailable
                    else None
                )
                if row["reason"] != reason:
                    db.execute(
                        "UPDATE jobs SET reason=?,updated=? WHERE id=?",
                        (reason, time.time(), row["id"]),
                    )
                    self.event(
                        db, row["id"], None, "admission_status", {"reason": reason}
                    )
                if reason:
                    continue
                ready = True
                for dep in db.execute(
                    "SELECT d.*,j.state,j.spec FROM dependencies d JOIN jobs j ON j.id=d.dependency_id WHERE d.job_id=?",
                    (row["id"],),
                ).fetchall():
                    if dep["state"] == "succeeded":
                        continue
                    if dep["fraction"] is None or dep["state"] != "running":
                        ready = False
                        break
                    progress = db.execute(
                        "SELECT payload FROM events WHERE job_id=? AND kind='comparison_progress' ORDER BY id DESC LIMIT 1",
                        (dep["dependency_id"],),
                    ).fetchone()
                    payload = json.loads(progress[0]) if progress else {}
                    if payload.get("games_completed", 0) < dep[
                        "fraction"
                    ] * payload.get("games_total", float("inf")):
                        ready = False
                        break
                if not ready:
                    continue
                number = db.execute(
                    "SELECT COALESCE(MAX(number),0)+1 FROM attempts WHERE job_id=?",
                    (row["id"],),
                ).fetchone()[0]
                attempt_id = str(uuid.uuid4())
                effective = {
                    "job_id": row["id"],
                    "attempt_id": attempt_id,
                    "resources": spec["resources"],
                    "spec": spec,
                }
                now = time.time()
                db.execute(
                    "INSERT INTO attempts(id,job_id,number,state,created,effective) VALUES (?,?,?,?,?,?)",
                    (
                        attempt_id,
                        row["id"],
                        number,
                        "preparing",
                        now,
                        encode(effective),
                    ),
                )
                db.execute(
                    'UPDATE jobs SET state="preparing",updated=?,reason=NULL WHERE id=?',
                    (now, row["id"]),
                )
                self.event(db, row["id"], attempt_id, "preparing", effective)
                return {"id": attempt_id, "job_id": row["id"], "effective": effective}
            return None

    def claim(self, capacity, effective_factory=None):
        """Reserve atomically; optional synchronous preparation never holds the write lock."""
        attempt = self.reserve(capacity)
        if attempt is None or effective_factory is None:
            return attempt
        try:
            effective = effective_factory(
                attempt["job_id"], attempt["id"], attempt["effective"]["spec"]
            )
        except Exception as error:
            self.finish_preparation(attempt["id"], error=str(error))
            return None
        if self.finish_preparation(attempt["id"], effective):
            return attempt | {"effective": effective}
        return None

    def finish_preparation(self, attempt_id, effective=None, error=None):
        """Commit preparation once, respecting a pause/cancel acknowledged during I/O."""
        with self.transaction() as db:
            attempt = db.execute(
                "SELECT * FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if not attempt or attempt["state"] != "preparing":
                return False
            job = db.execute(
                "SELECT * FROM jobs WHERE id=?", (attempt["job_id"],)
            ).fetchone()
            now = time.time()
            if job["state"] == "stopping":
                state = "cancelled" if job["reason"] == "cancel" else "paused"
                reason = job["reason"]
            elif error is not None:
                state, reason = "failed", error
            else:
                state, reason = "starting", None
            if state == "starting":
                db.execute(
                    "UPDATE attempts SET state=?,effective=? WHERE id=?",
                    (state, encode(effective), attempt_id),
                )
                payload = effective
            else:
                payload = {
                    "reason": reason,
                    "stopped": state in ("paused", "cancelled"),
                }
                db.execute(
                    "UPDATE attempts SET state=?,ended=?,result=? WHERE id=?",
                    (state, now, encode(payload), attempt_id),
                )
            db.execute(
                "UPDATE jobs SET state=?,updated=?,reason=? WHERE id=?",
                (state, now, reason, job["id"]),
            )
            self.event(
                db,
                job["id"],
                attempt_id,
                "configuration_failed" if state == "failed" else state,
                payload,
            )
            return state == "starting"

    def begin_finalization(self, attempt_id):
        """Retain the reservation until journals and published artifacts are committed."""
        with self.transaction() as db:
            attempt = db.execute(
                "SELECT * FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if not attempt or attempt["state"] not in (
                "starting",
                "running",
                "stopping",
                "finalizing",
            ):
                return False
            if attempt["state"] != "finalizing":
                db.execute(
                    "UPDATE attempts SET state='finalizing' WHERE id=?", (attempt_id,)
                )
                db.execute(
                    "UPDATE jobs SET state='finalizing',updated=? WHERE id=? AND state<>'stopping'",
                    (time.time(), attempt["job_id"]),
                )
                self.event(db, attempt["job_id"], attempt_id, "finalizing", {})
            return True

    def fail_finalization(self, attempt_id, error):
        with self.transaction() as db:
            attempt = db.execute(
                "SELECT * FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if not attempt or attempt["state"] != "finalizing":
                return
            now = time.time()
            payload = {
                "returncode": -1,
                "stopped": False,
                "reason": f"Finalization failed: {error}",
            }
            db.execute(
                "UPDATE attempts SET state='failed',ended=?,result=? WHERE id=?",
                (now, encode(payload), attempt_id),
            )
            db.execute(
                "UPDATE jobs SET state='failed',updated=?,reason=? WHERE id=?",
                (now, payload["reason"], attempt["job_id"]),
            )
            self.event(
                db, attempt["job_id"], attempt_id, "finalization_failed", payload
            )

    def active_attempts(self):
        with self.transaction(write=False) as db:
            return [
                self.decode_attempt(r)
                for r in db.execute(
                    "SELECT * FROM attempts WHERE state IN ('preparing','starting','running','stopping','finalizing')"
                )
            ]

    def ingest(self, attempt_id, records, cursor):
        with self.transaction() as db:
            attempt = db.execute(
                "SELECT * FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if cursor <= attempt["cursor"]:
                return
            for record in records:
                self.event(
                    db,
                    attempt["job_id"],
                    attempt_id,
                    record["kind"],
                    record.get("payload", {}),
                    timestamp=record.get("time"),
                )
                if record["kind"] == "worker_started":
                    db.execute(
                        'UPDATE attempts SET state=CASE WHEN state="finalizing" THEN state ELSE "running" END,started=? WHERE id=?',
                        (record["time"], attempt_id),
                    )
                    db.execute(
                        'UPDATE jobs SET state="running",updated=? WHERE id=? AND state="starting"',
                        (record["time"], attempt["job_id"]),
                    )
                if record["kind"] == "worker_finished":
                    payload = record["payload"]
                    job = db.execute(
                        "SELECT * FROM jobs WHERE id=?", (attempt["job_id"],)
                    ).fetchone()
                    stopped = payload.get("stopped", False)
                    state = (
                        ("cancelled" if job["reason"] == "cancel" else "paused")
                        if stopped
                        else ("succeeded" if payload["returncode"] == 0 else "failed")
                    )
                    db.execute(
                        "UPDATE attempts SET state=?,ended=?,result=? WHERE id=?",
                        (state, record["time"], encode(payload), attempt_id),
                    )
                    db.execute(
                        "UPDATE jobs SET state=?,updated=?,reason=? WHERE id=?",
                        (
                            state,
                            record["time"],
                            payload.get("reason"),
                            attempt["job_id"],
                        ),
                    )
            db.execute("UPDATE attempts SET cursor=? WHERE id=?", (cursor, attempt_id))

    def pending_control(self, job_id):
        with self.transaction(write=False) as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row["state"] != "stopping":
                return None
            event = db.execute(
                "SELECT payload FROM events WHERE job_id=? AND kind='command' ORDER BY id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            return json.loads(event[0])

    def artifact(self, job_id, attempt_id, kind, path, metadata=None):
        path = Path(path).resolve()
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        with self.transaction() as db:
            identifier = str(uuid.uuid4())
            db.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?,?,?)",
                (
                    identifier,
                    job_id,
                    attempt_id,
                    kind,
                    str(path),
                    digest,
                    encode(metadata or {}),
                ),
            )
            if metadata is not None:
                db.execute(
                    "UPDATE artifacts SET metadata=?, job_id=COALESCE(job_id,?), attempt_id=COALESCE(attempt_id,?) WHERE path=? AND sha256=? AND kind=?",
                    (encode(metadata), job_id, attempt_id, str(path), digest, kind),
                )
            return dict(
                db.execute(
                    "SELECT * FROM artifacts WHERE path=? AND sha256=?",
                    (str(path), digest),
                ).fetchone()
            )
