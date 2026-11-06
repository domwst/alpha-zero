"""Same-origin application gateway. Public reads; authenticated typed mutations."""

from __future__ import annotations

import argparse
from functools import lru_cache
import getpass
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
from pathlib import Path
import threading
import time
import uuid
import re
from urllib.parse import parse_qs, urlsplit

from .auth import Auth
from .scheduler import Scheduler
from .store import Conflict, Store, encode
from . import replays


class Application:
    def __init__(
        self,
        store,
        root,
        assets,
        auth,
        origin,
        scheduler=None,
        analysis=None,
        read_only=False,
        demo_url=None,
    ):
        self.store, self.root, self.assets = (
            store,
            Path(root).resolve(),
            Path(assets).resolve(),
        )
        self.archive_availability = {}
        self.game_index_cache = {}
        self.game_index_locks = {}
        self.game_index_guard = threading.Lock()
        self.overview_cache = None
        self.read_only = read_only
        self.demo_url = demo_url
        self.auth, self.origin, self.scheduler, self.analysis = (
            auth,
            origin.rstrip("/"),
            scheduler,
            analysis,
        )

    def archive(self, job_id):
        # Resolve through job ownership, never a client-supplied file path.
        with self.store.transaction(write=False) as db:
            if not db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                raise ValueError("Unknown job")
        return self.root / "jobs" / job_id / "games" / "archive"

    def archive_directories(self, job_id):
        """A zero-copy view of local epochs and the checkpoint's bounded ancestry."""
        lineage = []
        visited = set()
        limit = None
        with self.store.transaction(write=False) as db:
            while job_id not in visited:
                row = db.execute(
                    "SELECT spec FROM jobs WHERE id=?", (job_id,)
                ).fetchone()
                if row is None:
                    if not lineage:
                        raise ValueError("Unknown job")
                    break
                visited.add(job_id)
                lineage.append((job_id, limit))
                spec = json.loads(row["spec"])
                if spec["kind"] != "self_play":
                    break
                reference = spec.get("inputs", {}).get("checkpoint")
                if isinstance(reference, str):
                    artifact = db.execute(
                        "SELECT job_id,metadata FROM artifacts WHERE id=? AND kind='checkpoint'",
                        (reference,),
                    ).fetchone()
                    if not artifact:
                        break
                    parent = artifact["job_id"]
                    epoch = json.loads(artifact["metadata"]).get("epoch")
                elif isinstance(reference, dict):
                    # A latest/best selector must use its recorded resolution,
                    # never the parent's current (possibly newer) checkpoint.
                    parent = reference.get("job_id")
                    epoch = None
                    for attempt in db.execute(
                        "SELECT effective FROM attempts WHERE job_id=? ORDER BY number",
                        (job_id,),
                    ):
                        resolved = json.loads(attempt["effective"]).get(
                            "resolved_inputs", {}
                        )
                        checkpoint = resolved.get("checkpoint", {})
                        if type(checkpoint.get("epoch")) is int:
                            epoch = checkpoint["epoch"]
                            break
                else:
                    break
                if not parent or type(epoch) is not int or epoch < 0:
                    break
                limit = epoch if limit is None else min(limit, epoch)
                job_id = parent
        directories = {}
        for owner, limit in lineage:
            archive = self.root / "jobs" / owner / "games" / "archive"
            for path in archive.glob("*"):
                if (
                    re.fullmatch(r"[0-9]{8}", path.name)
                    and path.is_dir()
                    and (limit is None or int(path.name) <= limit)
                ):
                    # A local epoch wins as a whole: never mix divergent games
                    # with identical IDs from different branches of training.
                    directories.setdefault(path.name, path)
        return directories

    def game_paths(self, job_id, epoch=None, *, directories=None):
        if epoch is not None and not re.fullmatch(r"[0-9]{8}", epoch):
            raise ValueError("Invalid epoch")
        if directories is None:
            directories = self.archive_directories(job_id)
        selected = directories.values()
        if epoch is not None:
            selected = [directories[epoch]] if epoch in directories else []
        return sorted(
            (path for directory in selected for path in replays.game_paths(directory)),
            key=lambda p: (-int(p.parent.name), replays.game_id(p)),
        )

    GAME_SORTS = ("newest", "longest", "shortest")
    GAME_WINNERS = {"first": "First", "second": "Second", "draw": "Draw"}

    def game_index(self, directory):
        """Winner and length of every game in one epoch directory, cached
        until the directory changes. Only used for sorted or filtered pages,
        which are limited to a single epoch."""
        key = str(directory)
        with self.game_index_guard:
            lock = self.game_index_locks.setdefault(key, threading.Lock())
        # Concurrent requests for the same epoch wait for one build instead of
        # each decoding every replay.
        with lock:
            stamp = directory.stat().st_mtime_ns
            cached = self.game_index_cache.get(key)
            if cached and cached[0] == stamp:
                return cached[1]
            index = {}
            for path in replays.game_paths(directory):
                identifier = replays.game_id(path)
                index[identifier] = self.game_summary(
                    path, replays.find_game(directory, identifier).stat().st_mtime_ns
                )
            self.game_index_cache[key] = (stamp, index)
            return index

    def game_page(self, job_id, epoch, offset, limit, winner=None, sort="newest"):
        if sort not in self.GAME_SORTS:
            raise ValueError("Unknown game sort")
        if winner is not None and winner not in self.GAME_WINNERS:
            raise ValueError("Unknown winner filter")
        directory_map = self.archive_directories(job_id)
        directories = [directory_map[key] for key in sorted(directory_map, reverse=True)]
        paths = self.game_paths(job_id, epoch, directories=directory_map)
        narrowed = winner is not None or sort != "newest"
        if narrowed and epoch is None:
            raise ValueError("Sorting or filtering games requires a single epoch")
        if narrowed and paths:
            index = self.game_index(paths[0].parent)
            if winner is not None:
                wanted = self.GAME_WINNERS[winner]
                paths = [p for p in paths if index[replays.game_id(p)]["winner"] == wanted]
            if sort != "newest":
                paths.sort(
                    key=lambda p: (
                        -index[replays.game_id(p)]["plies"]
                        if sort == "longest"
                        else index[replays.game_id(p)]["plies"],
                        replays.game_id(p),
                    )
                )
        return {
            "total": len(paths),
            "epochs": [p.name for p in directories],
            "epoch_labels": {
                p.name: json.loads((p / "manifest.json").read_text()).get(
                    "label", f"Epoch {int(p.name) + 1}"
                )
                for p in directories
                if (p / "manifest.json").exists()
            },
            "games": [
                {
                    "id": replays.game_id(p),
                    "epoch": p.parent.name,
                    "job_id": job_id,
                    **self.game_summary(
                        p,
                        replays.find_game(p.parent, replays.game_id(p))
                        .stat()
                        .st_mtime_ns,
                    ),
                }
                for p in paths[offset : offset + limit]
            ],
        }

    RUN_KINDS = ("self_play", "replay_train", "reconstruction", "historical_training")

    def overview(self):
        """Cross-job aggregates for the landing page: headline counts, every
        completed comparison with both checkpoints resolved to their source
        job and epoch, and the per-epoch losses of each training run."""
        with self.store.transaction(write=False) as db:
            # Epoch events and published checkpoints change the overview
            # without touching the job rows, so they are part of the key.
            key = (
                tuple(db.execute("SELECT COUNT(*), MAX(updated) FROM jobs").fetchone())
                + tuple(db.execute("SELECT MAX(id) FROM events").fetchone())
                + tuple(db.execute("SELECT COUNT(*) FROM artifacts").fetchone())
            )
            if self.overview_cache and self.overview_cache[0] == key:
                return self.overview_cache[1]
            jobs = [
                dict(r)
                for r in db.execute(
                    "SELECT j.id,j.title,j.state,j.created,j.updated,j.spec,j.experiment_id,"
                    "e.title AS experiment_title FROM jobs j LEFT JOIN experiments e "
                    "ON e.id=j.experiment_id ORDER BY j.created DESC,j.id DESC"
                )
            ]
            attempts = {}
            for row in db.execute(
                "SELECT job_id,number,state,started,ended,effective FROM attempts ORDER BY job_id,number"
            ):
                attempts.setdefault(row["job_id"], []).append(dict(row))
            checkpoints = {}
            artifacts = {}
            for row in db.execute(
                "SELECT id,job_id,sha256,metadata FROM artifacts WHERE kind='checkpoint'"
            ):
                epoch = json.loads(row["metadata"]).get("epoch")
                checkpoints[row["sha256"]] = (row["job_id"], epoch)
                artifacts[row["id"]] = (row["job_id"], epoch)
        titles = {j["id"]: j["title"] for j in jobs}
        by_kind, by_state = {}, {}
        attempt_count = attempt_seconds = 0
        last = 0.0
        for job in jobs:
            job["spec"] = json.loads(job["spec"])
            by_kind[job["spec"]["kind"]] = by_kind.get(job["spec"]["kind"], 0) + 1
            by_state[job["state"]] = by_state.get(job["state"], 0) + 1
            last = max(last, job["updated"] or 0)
            for attempt in attempts.get(job["id"], []):
                attempt_count += 1
                if attempt["started"] and attempt["ended"]:
                    attempt_seconds += attempt["ended"] - attempt["started"]
                    last = max(last, attempt["ended"])

        def checkpoint_side(job, name, participants, resolved):
            info = dict(participants.get(name) or {})
            reference = resolved.get(name) or {}
            sha = info.get("model_sha256") or reference.get("model_sha256")
            # The comparison's own input reference names the producing job;
            # the hash alone is ambiguous because continuations register
            # copies of their source checkpoints under their own job.
            owner = epoch = None
            spec_input = job["spec"].get("inputs", {}).get(name)
            if isinstance(spec_input, str) and spec_input in artifacts:
                owner, epoch = artifacts[spec_input]
            elif isinstance(spec_input, dict) and spec_input.get("job_id"):
                owner = spec_input["job_id"]
            if owner is None:
                owner, epoch = checkpoints.get(sha, (None, None))
            if type(reference.get("epoch")) is int:
                epoch = reference["epoch"]
            return {
                "label": info.get("label"),
                "model_sha256": sha,
                "job_id": owner,
                "job_title": titles.get(owner),
                "epoch": epoch,
            }

        comparisons = []
        for job in jobs:
            if job["spec"]["kind"] != "comparison" or job["state"] not in (
                "succeeded",
                "archived",
            ):
                continue
            summary = self.store.job_summary(job["id"])
            comparison = summary.get("comparison")
            if not comparison:
                continue
            latest = attempts.get(job["id"], [])
            effective = json.loads(latest[-1]["effective"]) if latest else {}
            participants = (
                summary.get("participants") or effective.get("participants") or {}
            )
            resolved = effective.get("resolved_inputs") or {}
            config = effective.get("resolved_config") or {}
            seats = comparison["seat_results"]["first_checkpoint"]
            comparisons.append(
                {
                    "id": job["id"],
                    "title": job["title"],
                    "created": job["created"],
                    "state": job["state"],
                    "games": comparison["games"],
                    # Search settings that change playing strength; the UI
                    # only pools or connects matches with equal settings.
                    "settings": {
                        key: config[key]
                        for key in (
                            "simulations",
                            "temperature",
                            "first_temperature",
                            "second_temperature",
                            "c_puct",
                        )
                        if config.get(key) is not None
                    },
                    "first": checkpoint_side(job, "first", participants, resolved),
                    "second": checkpoint_side(job, "second", participants, resolved),
                    **{
                        k: seats["first"][k] + seats["second"][k]
                        for k in ("wins", "draws", "losses")
                    },
                }
            )

        runs = []
        recorded_games = 0
        for job in jobs:
            archive = self.root / "jobs" / job["id"] / "games" / "archive"
            if archive.is_dir():
                for directory in archive.glob("[0-9]*"):
                    if directory.is_dir():
                        recorded_games += len(replays.game_paths(directory))
            if job["spec"]["kind"] not in self.RUN_KINDS:
                continue
            epochs = []
            for event in self.store.epoch_summaries(job["id"]):
                payload = event["payload"]
                training = payload.get("training") or {}
                self_play = payload.get("self_play") or {}
                epochs.append(
                    {
                        "epoch": payload.get("epoch"),
                        "value_loss": training.get("value_loss"),
                        "policy_loss": training.get("policy_loss"),
                        "first_player_win_rate": self_play.get("first_player_win_rate"),
                        "games": self_play.get("games", payload.get("games")),
                    }
                )
            source = None
            reference = job["spec"].get("inputs", {}).get("checkpoint")
            latest = attempts.get(job["id"], [])
            resolved = (
                json.loads(latest[-1]["effective"]).get("resolved_inputs", {})
                if latest
                else {}
            )
            if isinstance(reference, str) and reference in artifacts:
                owner, epoch = artifacts[reference]
                source = {"job_id": owner, "job_title": titles.get(owner), "epoch": epoch}
            elif isinstance(reference, dict) and reference.get("job_id"):
                owner = reference["job_id"]
                checkpoint = resolved.get("checkpoint") or {}
                source = {
                    "job_id": owner,
                    "job_title": titles.get(owner),
                    "epoch": checkpoint.get("epoch")
                    if type(checkpoint.get("epoch")) is int
                    else None,
                }
            runs.append(
                {
                    "id": job["id"],
                    "title": job["title"],
                    "kind": job["spec"]["kind"],
                    "state": job["state"],
                    "created": job["created"],
                    "experiment_id": job["experiment_id"],
                    "experiment_title": job["experiment_title"],
                    "source": source,
                    "epochs": epochs,
                }
            )
        result = {
            "jobs": len(jobs),
            "experiments": len({j["experiment_id"] for j in jobs}),
            "by_kind": by_kind,
            "by_state": by_state,
            "attempts": attempt_count,
            "attempt_seconds": attempt_seconds,
            "first_activity": min((j["created"] for j in jobs), default=None),
            "last_activity": last or None,
            "recorded_games": recorded_games,
            "comparisons": comparisons,
            "runs": runs,
        }
        self.overview_cache = (key, result)
        return result

    def has_games(self, job_id):
        cached = self.archive_availability.get(job_id)
        if cached and (cached[1] or time.monotonic() - cached[0] < 10):
            return cached[1]
        found = any(
            replays.has_games(directory)
            for directory in self.archive_directories(job_id).values()
        )
        self.archive_availability[job_id] = (time.monotonic(), found)
        return found

    def jobs(self, **kwargs):
        page = self.store.snapshot(**kwargs)
        for job in page["jobs"]:
            job["has_games"] = self.has_games(job["id"])
            if kwargs.get("job_id"):
                # Live status must not be reconstructed while the browser pages
                # through historical events from older epochs.
                with self.store.transaction(write=False) as db:
                    runtime = {}
                    for kind in (
                        "stage",
                        "self_play_progress",
                        "comparison_progress",
                        "training_progress",
                        "game_completed",
                    ):
                        row = db.execute(
                            "SELECT id,time,kind,payload,attempt_id FROM events WHERE job_id=? AND kind=? ORDER BY id DESC LIMIT 1",
                            (job["id"], kind),
                        ).fetchone()
                        if row:
                            runtime[kind] = dict(row) | {
                                "payload": json.loads(row["payload"])
                            }
                    comparison = runtime.get("comparison_progress")
                    if comparison:
                        # Small completion updates omit per-executor telemetry.
                        heartbeat = db.execute(
                            "SELECT payload FROM events WHERE job_id=? AND kind='comparison_progress' AND attempt_id IS ? AND id<=? AND json_type(payload,'$.first_inference')='object' ORDER BY id DESC LIMIT 1",
                            (job["id"], comparison["attempt_id"], comparison["id"]),
                        ).fetchone()
                        if heartbeat:
                            prior = json.loads(heartbeat[0])
                            comparison["payload"] = {
                                k: prior[k]
                                for k in ("first_inference", "second_inference")
                                if k in prior
                            } | comparison["payload"]
                stage = runtime.get("stage")
                if stage and stage["payload"].get("stage") == "self_play":
                    for key in (
                        "self_play_progress",
                        "training_progress",
                        "game_completed",
                    ):
                        if runtime.get(key, {}).get("id", 0) < stage["id"]:
                            runtime.pop(key, None)
                completion = runtime.pop("game_completed", None)
                progress = runtime.get("self_play_progress")
                if progress and completion and completion["id"] > progress["id"]:
                    # Completion events have only a subset of heartbeat fields.
                    # Merge their counts instead of replacing the live snapshot.
                    payload, done = progress["payload"], completion["payload"]
                    payload.update(
                        {
                            key: done[key]
                            for key in ("games_completed", "games_total")
                            if key in done
                        }
                    )
                    if payload.get("games_completed") == payload.get("games_total"):
                        payload["active_games"] = 0
                        if payload.get("inference"):
                            payload["inference"].update(
                                active_producers=0,
                                outstanding_requests=0,
                                in_flight_requests=0,
                            )
                if (
                    stage
                    and stage["payload"].get("stage") != "self_play"
                    and "self_play_progress" in runtime
                ):
                    runtime["last_collection"] = runtime.pop("self_play_progress")
                job["runtime"] = runtime
        return page

    @staticmethod
    @lru_cache(maxsize=512)
    def game_summary(path, mtime_ns):
        game = replays.load(path)
        plies = game["record"]["plies"]
        # MatchRecord terminal_value is from the player to move AFTER the last ply.
        value = game["record"].get("terminal_value")
        winner = None
        if value is not None and plies:
            actor = game["record"].get("terminal_actor") or (
                "Second" if plies[-1]["actor"] == "First" else "First"
            )
            winner = (
                (actor if value > 0 else ("Second" if actor == "First" else "First"))
                if value
                else "Draw"
            )
        return {
            "plies": len(plies),
            "winner": winner,
            "duration_seconds": game.get("duration_seconds"),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "AlphaZero"

    @property
    def app(self):
        return self.server.app

    def stream_reply(self, value):
        if not getattr(self, "streaming", False):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.streaming = True
        self.wfile.write((encode(value) + "\n").encode())
        self.wfile.flush()

    def reply(self, status, value, *, headers=None):
        if getattr(self, "streaming", False):
            self.stream_reply(value)
            return
        data = encode(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        for key, val in (headers or {}).items():
            self.send_header(key, val)
        self.end_headers()
        self.wfile.write(data)

    def token(self):
        jar = cookies.SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie", ""))
        except cookies.CookieError:
            return ""
        return jar["alz_session"].value if "alz_session" in jar else ""

    def body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if (
            not 0 < length <= 65536
            or self.headers.get_content_type() != "application/json"
        ):
            raise ValueError("A JSON body of at most 64 KiB is required")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def session_cookie(self, token):
        secure = "; Secure" if self.app.origin.startswith("https://") else ""
        # Browsers limit persistent cookie lifetimes. Renew on each page load;
        # the server-side session itself has no expiry.
        return f"alz_session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=31536000{secure}"

    def do_POST(self):
        try:
            if self.app.read_only:
                raise PermissionError("Historical archive is read-only")
            if self.headers.get("Origin") != self.app.origin:
                raise PermissionError("Request origin does not match this application")
            body = self.body()
            path = urlsplit(self.path).path
            secure = "; Secure" if self.app.origin.startswith("https://") else ""
            if path == "/api/v1/login":
                token, session = self.app.auth.login(body.get("password"))
                self.reply(
                    200,
                    {"authenticated": True, **session},
                    headers={"Set-Cookie": self.session_cookie(token)},
                )
                return
            self.app.auth.require(self.token(), self.headers.get("X-CSRF-Token"))
            if path == "/api/v1/logout":
                self.app.auth.logout(self.token())
                self.reply(
                    200,
                    {"authenticated": False},
                    headers={
                        "Set-Cookie": f"alz_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0{secure}"
                    },
                )
            elif path == "/api/v1/commands":
                self.reply(
                    200, self.app.store.command(body["command_id"], body["request"])
                )
            elif path == "/api/v1/analysis/cancel" and self.app.analysis:
                with self.app.store.transaction() as db:
                    self.app.store.event(
                        db,
                        None,
                        None,
                        "analysis_cancel_requested",
                        {"client_request_id": body.get("request_id")},
                    )
                try:
                    result = self.app.analysis.cancel(
                        body.get("request_id"), self.token()
                    )
                except Exception as error:
                    with self.app.store.transaction() as db:
                        self.app.store.event(
                            db,
                            None,
                            None,
                            "analysis_cancel_failed",
                            {
                                "client_request_id": body.get("request_id"),
                                "error": str(error),
                            },
                        )
                    raise
                with self.app.store.transaction() as db:
                    self.app.store.event(
                        db,
                        None,
                        None,
                        "analysis_cancel_acknowledged",
                        {"client_request_id": body.get("request_id"), **result},
                    )
                self.reply(200, result)
            elif (
                path in ("/api/v1/analyze", "/api/v1/analyze/stream")
                and self.app.analysis
            ):
                request_id, started = str(uuid.uuid4()), time.monotonic()
                with self.app.store.transaction() as db:
                    self.app.store.event(
                        db,
                        None,
                        None,
                        "analysis_requested",
                        {
                            "request_id": request_id,
                            "client_request_id": body.get("request_id"),
                            "checkpoint_sha256": self.app.analysis.digest,
                            "simulations": body.get("simulations", 0),
                            "inspect": body.get("inspect", False),
                        },
                    )
                try:
                    result = self.app.analysis.request(
                        body,
                        self.stream_reply if path.endswith("/stream") else None,
                        owner=self.token(),
                    )
                except Exception as error:
                    with self.app.store.transaction() as db:
                        self.app.store.event(
                            db,
                            None,
                            None,
                            "analysis_failed",
                            {
                                "request_id": request_id,
                                "client_request_id": body.get("request_id"),
                                "error": str(error),
                                "seconds": time.monotonic() - started,
                            },
                        )
                    raise
                with self.app.store.transaction() as db:
                    self.app.store.event(
                        db,
                        None,
                        None,
                        "analysis_cancelled"
                        if result.get("result", {}).get("cancelled")
                        else "analysis_completed",
                        {
                            "request_id": request_id,
                            "client_request_id": body.get("request_id"),
                            "seconds": time.monotonic() - started,
                        },
                    )
                if not path.endswith("/stream"):
                    self.reply(200, result)
            else:
                self.reply(404, {"error": "Unknown endpoint"})
        except PermissionError as error:
            self.reply(403, {"error": str(error)})
        except Conflict as error:
            self.reply(409, {"error": str(error)})
        except (ValueError, KeyError, TypeError) as error:
            self.reply(400, {"error": str(error)})
        except Exception:
            logging.exception("Request failed")
            self.reply(500, {"error": "Internal request failure; see service log"})

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            if url.path == "/api/v1/session":
                session = None if self.app.read_only else self.app.auth.get(self.token())
                self.reply(
                    200,
                    {
                        "authenticated": bool(session),
                        **(session or {}),
                        "login_enabled": bool(self.app.auth.password_hash)
                        and not self.app.read_only,
                        "read_only": self.app.read_only,
                        "demo_url": self.app.demo_url,
                    },
                    headers={"Set-Cookie": self.session_cookie(self.token())}
                    if session
                    else None,
                )
            elif url.path == "/api/v1/analysis":
                worker = self.app.analysis
                self.reply(
                    200,
                    {
                        "available": bool(worker),
                        "checkpoint": worker.checkpoint if worker else None,
                        "device": worker.device if worker else None,
                        "max_simulations": worker.maximum if worker else 0,
                    },
                )
            elif url.path == "/api/v1/jobs":
                self.reply(
                    200,
                    self.app.jobs(
                        limit=min(100, max(1, int(query.get("limit", ["25"])[0]))),
                        offset=max(0, int(query.get("offset", ["0"])[0])),
                        kind=query.get("kind", [None])[0] or None,
                        state=query.get("state", [None])[0] or None,
                        search=(query.get("q", [""])[0] or "").strip()[:200] or None,
                        experiment_id=query.get("experiment_id", [None])[0] or None,
                    ),
                )
            elif url.path == "/api/v1/overview":
                self.reply(200, self.app.overview())
            elif url.path.startswith("/api/v1/jobs/"):
                jobs = self.app.jobs(job_id=url.path.rsplit("/", 1)[1])["jobs"]
                self.reply(200, jobs[0]) if jobs else self.reply(
                    404, {"error": "Job not found"}
                )
            elif url.path == "/api/v1/epoch-summaries":
                self.reply(
                    200,
                    {
                        "events": self.app.store.epoch_summaries(
                            query["job_id"][0],
                            max(0, int(query.get("after", ["0"])[0])),
                        )
                    },
                )
            elif url.path == "/api/v1/event-history":
                before = int(query["before"][0]) if "before" in query else None
                self.reply(200, self.app.store.event_page(query["job_id"][0], before))
            elif url.path == "/api/v1/job-summary":
                self.reply(200, self.app.store.job_summary(query["job_id"][0]))
            elif url.path == "/api/v1/events":
                self.reply(
                    200,
                    {
                        "events": self.app.store.events(
                            query.get("job_id", [None])[0],
                            max(0, int(query.get("after", ["0"])[0])),
                        )
                    },
                )
            elif url.path == "/api/v1/artifacts":
                with self.app.store.transaction(write=False) as db:
                    rows = db.execute(
                        "SELECT a.id,a.job_id,a.attempt_id,a.kind,a.sha256,a.metadata,"
                        "j.title AS job_title FROM artifacts a LEFT JOIN jobs j ON j.id=a.job_id "
                        "ORDER BY a.rowid DESC LIMIT 1000"
                    )
                    self.reply(200, {"artifacts": [dict(r) for r in rows]})
            elif url.path == "/api/v1/game-jobs":
                with self.app.store.transaction(write=False) as db:
                    candidates = [
                        dict(r)
                        for r in db.execute(
                            "SELECT id,title FROM jobs ORDER BY created DESC,id DESC"
                        )
                    ]
                self.reply(
                    200,
                    {"jobs": [j for j in candidates if self.app.has_games(j["id"])]},
                )
            elif url.path == "/api/v1/games":
                self.reply(
                    200,
                    self.app.game_page(
                        query["job_id"][0],
                        query.get("epoch", [None])[0],
                        max(0, int(query.get("offset", ["0"])[0])),
                        min(25, max(1, int(query.get("limit", ["10"])[0]))),
                        winner=query.get("winner", [None])[0] or None,
                        sort=query.get("sort", ["newest"])[0] or "newest",
                    ),
                )
            elif url.path == "/api/v1/game":
                epoch, game_id = query["epoch"][0], query["id"][0]
                if not re.fullmatch(r"[0-9]{8}", epoch) or not re.fullmatch(
                    r"[0-9a-f]{1,64}", game_id
                ):
                    raise ValueError("Invalid game identity")
                directories = self.app.archive_directories(query["job_id"][0])
                if epoch not in directories:
                    raise ValueError("Unknown game epoch")
                path = replays.find_game(directories[epoch], game_id)
                self.reply(200, replays.load(path))
            elif url.path == "/api/v1/schema":
                from .spec import COMMON, KINDS

                self.reply(
                    200,
                    {
                        "kinds": {
                            k: {
                                name: (
                                    "text"
                                    if t == "grid"
                                    else "boolean"
                                    if t is bool
                                    else "integer"
                                    if t is int
                                    else "number"
                                    if t is float
                                    else list(t)
                                )
                                for name, t in (COMMON | v).items()
                            }
                            for k, v in KINDS.items()
                        }
                    },
                )
            elif url.path.startswith("/api/"):
                self.reply(404, {"error": "Unknown endpoint"})
            else:
                path = (self.app.assets / url.path.lstrip("/")).resolve()
                if not path.is_relative_to(self.app.assets):
                    raise PermissionError("Invalid asset path")
                status = 200
                if not path.is_file():
                    status = (
                        200
                        if url.path
                        in ("/", "/experiments", "/games", "/analyze", "/play")
                        else 404
                    )
                    path = self.app.assets / "index.html"
                content = path.read_bytes()
                if path.name == "index.html":
                    content = content.replace(
                        b"<head>", b'<head><meta name="alz-service" content="1">'
                    )
                self.send_response(status)
                self.send_header(
                    "Content-Type",
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
                )
                self.end_headers()
                self.wfile.write(content)
        except (ValueError, KeyError, TypeError) as error:
            self.reply(400, {"error": str(error)})
        except PermissionError as error:
            self.reply(403, {"error": str(error)})
        except FileNotFoundError:
            self.reply(404, {"error": "Not found"})
        except Exception:
            logging.exception("Read request failed")
            self.reply(500, {"error": "Internal request failure; see service log"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/job-service"))
    parser.add_argument("--assets", type=Path, default=Path("web/dist"))
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--origin",
        help="Exact browser-facing origin, including HTTPS scheme when proxied",
    )
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--host-memory-mb", type=int, default=16384)
    parser.add_argument("--gpu-memory-mb", type=int, default=0)
    parser.add_argument("--analysis-checkpoint", type=Path)
    parser.add_argument(
        "--analysis-binary",
        type=Path,
        help="Optional separate executable for interactive analysis",
    )
    parser.add_argument("--analysis-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--analysis-gpu-memory-mb", type=int, default=0)
    parser.add_argument("--analysis-host-memory-mb", type=int, default=2048)
    parser.add_argument("--set-password", action="store_true")
    parser.add_argument(
        "--demo-url",
        help="Public address of the interactive demo, linked from the dashboard header",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Serve historical data without login, mutations, training, or analysis",
    )
    args = parser.parse_args()
    if args.read_only and (
        args.binary or args.analysis_binary or args.analysis_checkpoint or args.set_password
    ):
        parser.error("--read-only cannot be combined with workers or --set-password")
    if args.read_only and not (args.root / "jobs.sqlite3").is_file():
        parser.error("--read-only requires an existing jobs.sqlite3 archive")
    args.root.mkdir(parents=True, exist_ok=True)
    password = args.root / "admin-password.hash"
    if args.set_password:
        value = Auth.hash_password(getpass.getpass("Administrator password: "))
        import os

        fd = os.open(password, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write(value)
        return
    store = Store(args.root / "jobs.sqlite3")
    analysis = None
    if args.analysis_checkpoint:
        from .analysis import AnalysisWorker

        if (
            not (args.analysis_binary or args.binary)
            or args.analysis_host_memory_mb >= args.host_memory_mb
            or args.analysis_gpu_memory_mb > args.gpu_memory_mb
        ):
            parser.error(
                "Analysis needs a binary and a resource reservation within the host budget"
            )
        if args.analysis_device == "cuda" and args.analysis_gpu_memory_mb <= 0:
            parser.error("CUDA analysis requires an explicit GPU memory reservation")
        args.host_memory_mb -= args.analysis_host_memory_mb
        args.gpu_memory_mb -= args.analysis_gpu_memory_mb
        analysis = AnalysisWorker(
            args.analysis_binary or args.binary,
            args.analysis_checkpoint,
            args.analysis_device,
        )
    scheduler = (
        Scheduler(
            store,
            args.root,
            args.binary,
            {
                "slots": args.slots,
                "host_memory_mb": args.host_memory_mb,
                "gpu_memory_mb": args.gpu_memory_mb,
            },
            Path.cwd(),
        )
        if args.binary
        else None
    )
    app = Application(
        store,
        args.root,
        args.assets,
        Auth(
            password.read_text().strip()
            if password.exists() and not args.read_only
            else None,
            sessions_path=args.root / "admin-sessions.sqlite3",
        ),
        args.origin or f"http://{args.listen}:{args.port}",
        scheduler,
        analysis,
        read_only=args.read_only,
        demo_url=args.demo_url,
    )
    server = ThreadingHTTPServer((args.listen, args.port), Handler)
    server.app = app
    stop = threading.Event()

    def schedule():
        while not stop.is_set():
            try:
                if scheduler:
                    scheduler.tick()
            except Exception:
                logging.exception("Scheduler tick failed")
            stop.wait(1)

    thread = threading.Thread(target=schedule, daemon=True)
    thread.start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        thread.join(timeout=5)
        server.server_close()
        app.auth.close()
        if analysis:
            analysis.close()


if __name__ == "__main__":
    main()
