#!/usr/bin/env python3
"""Read-only loopback dashboard backed by an authenticated SSH connection."""

import argparse
import base64
from collections import OrderedDict
import json
import mimetypes
import os
from pathlib import Path
import select
import shlex
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit
from experiment_metadata import apply_experiment_metadata


class SSHBridge:
    def __init__(self, connection):
        self.config = connection
        self.lock = threading.Lock()
        self.process = None
        self.buffer = b""
        self.serial = 0

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        self.buffer = b""

    def receive(self, marker, timeout=20):
        deadline = time.monotonic() + timeout
        while marker not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("The pod did not respond within 20 seconds")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise ConnectionError("The SSH connection to the pod closed")
            self.buffer += data
            if len(self.buffer) > 6_000_000:
                raise ValueError("Unexpectedly large SSH response")
        before, self.buffer = self.buffer.split(marker, 1)
        return before

    def connect(self):
        config = self.config
        command = ["ssh", "-tt", "-e", "none", "-i", config["identity_file"],
                   "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                   "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + config["host_key_file"],
                   "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
                   "-o", "ServerAliveCountMax=3", config["ssh_target"]]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
        self.receive(b"# ")
        remote = ["python3", "-u", config["remote_agent"], "--activation-dir", config["remote_activation"],
                  "--queue-dir", config["remote_queue"]]
        bootstrap = "unset HISTFILE; stty -echo; exec " + shlex.join(remote) + "\n"
        self.process.stdin.write(bootstrap.encode())
        self.process.stdin.flush()
        # The RunPod gateway can translate LF to CRLF even after tty.setraw().
        self.receive(b"ALZ_EXPERIMENT_AGENT_READY")
        self.receive(b"\n")

    def call(self, method, **arguments):
        if method not in ("snapshot", "logs", "game_image"):
            raise ValueError("The dashboard SSH bridge is read-only")
        with self.lock:
            try:
                if self.process is None or self.process.poll() is not None:
                    self.close()
                    self.connect()
                self.serial += 1
                request = {"id": self.serial, "method": method, **arguments}
                self.process.stdin.write((json.dumps(request) + "\n").encode())
                self.process.stdin.flush()
                response = json.loads(self.receive(b"\n"))
                if response.get("id") != self.serial:
                    raise ValueError("SSH response does not match the request")
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response["result"]
            except Exception:
                self.close()
                raise


class ArchiveBridge:
    """Serve the final snapshot and saved log views without any SSH connection."""
    def __init__(self, path):
        self.lock = threading.Lock()
        self.archive = json.loads(Path(path).read_text())
        metadata_path = Path(path).with_suffix('.metadata.json')
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        self.archive['snapshot'] = apply_experiment_metadata(self.archive['snapshot'], metadata)
        self.archive['snapshot']['archived'] = True

    def close(self):
        pass

    def call(self, method, **arguments):
        if method == 'snapshot':
            return self.archive['snapshot']
        if method == 'logs':
            identity = arguments.get('phase_id')
            if identity in self.archive['logs']:
                return self.archive['logs'][identity]
            raise ValueError('Unknown experiment')
        raise ValueError('The archive is read-only')


class Dashboard:
    def __init__(self, bridge, metadata_path=None):
        self.bridge = bridge
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.lock = threading.Lock()
        self.state = {"connected": False, "error": None, "snapshot": None, "received_at": None}
        self.stop = threading.Event()
        self.image_cache = OrderedDict()
        self.image_lock = threading.Lock()

    def game_image(self, phase_id, epoch, sample):
        state = self.snapshot().get('snapshot') or {}
        phase = next((p for p in state.get('phases', []) if p['id'] == phase_id), {})
        if not any(row['epoch'] == epoch and row['sample'] == sample for row in phase.get('game_samples', [])):
            raise ValueError('Unknown game sample')
        selected = next(row for row in phase['game_samples'] if row['epoch'] == epoch and row['sample'] == sample)
        key = (phase_id, epoch, sample, selected['width'], selected['height'])
        with self.image_lock:
            if key in self.image_cache:
                self.image_cache.move_to_end(key)
                return self.image_cache[key]
            result = self.bridge.call('game_image', phase_id=phase_id, epoch=epoch, sample=sample)
            if len(result['data']) > 5_333_336:
                raise ValueError('Game image exceeds size limit')
            data = base64.b64decode(result['data'], validate=True)
            if not data.startswith(b'\x89PNG\r\n\x1a\n') or len(data) > 4_000_000:
                raise ValueError('Invalid game image')
            self.image_cache[key] = data
            while len(self.image_cache) > 8:
                self.image_cache.popitem(last=False)
            return data

    def refresh(self):
        try:
            snapshot = self.bridge.call("snapshot")
            metadata = (json.loads(self.metadata_path.read_text())
                        if self.metadata_path and self.metadata_path.exists() else None)
            if 'phases' in snapshot:
                snapshot = apply_experiment_metadata(snapshot, metadata)
            with self.lock:
                self.state = {"connected": True, "error": None, "snapshot": snapshot,
                              "received_at": time.time()}
        except Exception as error:
            with self.lock:
                self.state = {**self.state, "connected": False, "error": str(error)}

    def poll(self):
        while not self.stop.is_set():
            self.refresh()
            self.stop.wait(5)

    def snapshot(self):
        with self.lock:
            return self.state.copy()


def handler(dashboard, assets):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Periodic polling should not flood the terminal.

        def respond(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 405:
                self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/api/experiments":
                return self.respond(200, dashboard.snapshot())
            if url.path == "/api/experiments/logs":
                identity = parse_qs(url.query).get("phase_id", [""])[0]
                try:
                    return self.respond(200, dashboard.bridge.call("logs", phase_id=identity))
                except Exception as error:
                    return self.respond(502, {"error": str(error)})
            if url.path == '/api/experiments/game-image':
                query = parse_qs(url.query)
                try:
                    data = dashboard.game_image(query.get('phase_id', [''])[0],
                        int(query.get('epoch', [''])[0]), int(query.get('sample', [''])[0]))
                except ValueError as error:
                    return self.respond(404, {'error':str(error)})
                except Exception as error:
                    return self.respond(502, {'error':str(error)})
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'private, max-age=3600')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                if self.command != 'HEAD':
                    self.wfile.write(data)
                return
            if url.path.startswith("/api/"):
                return self.respond(404, {"error": "Unknown API endpoint"})
            if url.path == "/":
                self.send_response(302)
                self.send_header("Location", "/experiments")
                self.end_headers()
                return
            name = "index.html" if url.path == "/experiments" else unquote(url.path).lstrip("/")
            path = (assets / name).resolve()
            if not path.is_relative_to(assets) or not path.is_file():
                return self.respond(404, {"error": "Not found"})
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        do_HEAD = do_GET

        def reject_mutation(self):
            self.respond(405, {"error": "This dashboard is read-only"})

        do_POST = do_PUT = do_PATCH = do_DELETE = reject_mutation
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--connection", type=Path)
    source.add_argument("--archive", type=Path)
    parser.add_argument("--assets", type=Path, default=Path(__file__).resolve().parents[1] / "web/dist")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--metadata", type=Path,
                        help="Local identity registry; defaults to <connection-stem>.metadata.json")
    args = parser.parse_args()
    if not (args.assets / "index.html").is_file():
        parser.error("Build the frontend first with npm --prefix web run build")
    bridge = (ArchiveBridge(args.archive) if args.archive else
              SSHBridge(json.loads(args.connection.read_text())))
    metadata_path = args.metadata or (args.connection.with_suffix('.metadata.json') if args.connection else None)
    dashboard = Dashboard(bridge, metadata_path)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(dashboard, args.assets.resolve()))
    threading.Thread(target=dashboard.poll, daemon=True).start()
    print(f"Experiment dashboard: http://127.0.0.1:{server.server_port}/experiments", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop.set()
        server.server_close()
        with dashboard.bridge.lock:
            dashboard.bridge.close()


if __name__ == "__main__":
    main()
