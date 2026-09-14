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


class Application:
    def __init__(
        self, store, root, assets, auth, origin, scheduler=None, analysis=None
    ):
        self.store, self.root, self.assets = (
            store,
            Path(root).resolve(),
            Path(assets).resolve(),
        )
        self.archive_availability = {}
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

    def game_paths(self, job_id, epoch=None):
        archive = self.archive(job_id)
        if epoch is not None and not re.fullmatch(r"[0-9]{8}", epoch):
            raise ValueError("Invalid epoch")
        directory = archive / epoch if epoch else archive
        return sorted(
            directory.glob("[0-9a-f]*.json" if epoch else "*/[0-9a-f]*.json"),
            key=lambda p: (-int(p.parent.name), p.name),
        )

    def has_games(self, job_id):
        cached = self.archive_availability.get(job_id)
        if cached and (cached[1] or time.monotonic() - cached[0] < 10):
            return cached[1]
        found = next(self.archive(job_id).glob("*/[0-9a-f]*.json"), None) is not None
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
        game = json.loads(path.read_text())
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
                session = self.app.auth.get(self.token())
                self.reply(
                    200,
                    {
                        "authenticated": bool(session),
                        **(session or {}),
                        "login_enabled": bool(self.app.auth.password_hash),
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
                    ),
                )
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
                job = query["job_id"][0]
                epoch = query.get("epoch", [None])[0]
                archive = self.app.archive(job)
                directories = sorted(
                    (
                        p
                        for p in archive.glob("*")
                        if p.is_dir() and re.fullmatch(r"[0-9]{8}", p.name)
                    ),
                    reverse=True,
                )
                paths = self.app.game_paths(job, epoch)
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = min(25, max(1, int(query.get("limit", ["10"])[0])))
                self.reply(
                    200,
                    {
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
                                "id": p.stem,
                                "epoch": p.parent.name,
                                "job_id": job,
                                **self.app.game_summary(p, p.stat().st_mtime_ns),
                            }
                            for p in paths[offset : offset + limit]
                        ],
                    },
                )
            elif url.path == "/api/v1/game":
                epoch, game_id = query["epoch"][0], query["id"][0]
                if not re.fullmatch(r"[0-9]{8}", epoch) or not re.fullmatch(
                    r"[0-9a-f]{1,64}", game_id
                ):
                    raise ValueError("Invalid game identity")
                path = (
                    self.app.archive(query["job_id"][0]) / epoch / (game_id + ".json")
                )
                self.reply(200, json.loads(path.read_text()))
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
    args = parser.parse_args()
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
            password.read_text().strip() if password.exists() else None,
            sessions_path=args.root / "admin-sessions.sqlite3",
        ),
        args.origin or f"http://{args.listen}:{args.port}",
        scheduler,
        analysis,
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
