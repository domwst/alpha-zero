"""One bounded, pinned-checkpoint analysis process. Independent from trainer executors."""

import json
import os
import queue
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from .store import Conflict, encode


class AnalysisRequestError(ValueError):
    """Native rejected one framed request without touching its cached search."""


class AnalysisWorker:
    def __init__(
        self, binary, checkpoint, device="cpu", max_simulations=20000, timeout=900
    ):
        checkpoint = Path(checkpoint).resolve()
        if not (checkpoint / "metadata.json").exists():
            directory = (
                checkpoint / "checkpoints"
                if (checkpoint / "checkpoints").is_dir()
                else checkpoint
            )
            candidates = [
                p
                for p in directory.iterdir()
                if p.name.isdigit()
                and (p / "metadata.json").is_file()
                and (p / "model.safetensors").is_file()
            ]
            if not candidates:
                raise ValueError("No complete analysis checkpoint")
            checkpoint = max(candidates, key=lambda p: int(p.name))
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        self.digest = metadata["model_sha256"]
        self.argv = [
            str(Path(binary).resolve()),
            "analyze",
            "--checkpoint-dir",
            str(Path(checkpoint).resolve()),
            "--device",
            device,
            "--max-simulations",
            str(max_simulations),
        ]
        self.checkpoint, self.device = str(checkpoint), device
        self.maximum, self.timeout = max_simulations, timeout
        self.lock = threading.Lock()
        self.process = None
        self.activity_lock = threading.Lock()
        self.active = None
        self.cancelled = {}
        # Linux parent-death signals follow the thread that created the child.
        # A request handler exits after one response, so launch from a persistent
        # owner thread instead; gateway death still kills the native child.
        self.launches = queue.SimpleQueue()
        self.launcher = threading.Thread(target=self._launch_loop, daemon=True)
        self.launcher.start()

    def _launch_loop(self):
        while True:
            kwargs, result = self.launches.get()
            try:
                result.put(subprocess.Popen(**kwargs))
            except Exception as error:
                result.put(error)

    def _launch(self, **kwargs):
        result = queue.SimpleQueue()
        self.launches.put((kwargs, result))
        process = result.get()
        if isinstance(process, Exception):
            raise process
        return process

    def close(self):
        if self.process:
            self.process.kill()
            self.process.wait(timeout=10)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def cancel(self, request_id, owner):
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("Invalid analysis request identity")
        with self.activity_lock:
            now = time.monotonic()
            self.cancelled = {k: t for k, t in self.cancelled.items() if t > now}
            if len(self.cancelled) >= 256:
                self.cancelled.pop(next(iter(self.cancelled)))
            # Also cover cancellation arriving just before the search request.
            self.cancelled[(owner, request_id)] = now + self.timeout
            active = self.active
            if not active or active[:2] != (owner, request_id):
                return {"cancelled": True}
            active[2].set()
        if not active[3].wait(self.timeout):
            raise TimeoutError("Analysis cancellation exceeded its time budget")
        return {"cancelled": True}

    def request(self, body, on_update=None, owner=None):
        if not self.lock.acquire(blocking=False):
            raise Conflict("Analysis worker is busy; retry when this search finishes")
        cancel, finished = threading.Event(), threading.Event()
        try:
            target = body.get("simulations", 0)
            if not isinstance(target, int) or not 0 <= target <= self.maximum:
                raise ValueError("Invalid simulation budget")
            if len(encode(body)) > 65536:
                raise ValueError("Analysis request is too large")
            body = dict(body)
            request_id = body.pop("request_id", None)
            if request_id is not None and (
                not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
            ):
                raise ValueError("Invalid analysis request identity")
            with self.activity_lock:
                if self.cancelled.pop((owner, request_id), 0) > time.monotonic():
                    raise Conflict("Analysis request was cancelled")
                self.active = (owner, request_id, cancel, finished)
            if self.process is None or self.process.poll() is not None:
                self.close()
                self.process = self._launch(
                    args=[
                        sys.executable,
                        "-m",
                        "scripts.job_service.exec_child",
                        str(os.getpid()),
                        *self.argv,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    bufsize=0,
                )
            started = time.monotonic()
            deadline = started + self.timeout
            step, carried = 0, None
            initial = True
            activations = []
            while True:
                # First expose the retained root (or initialize a fresh one) and
                # inspect the position before spending the remaining search budget.
                # Bounded targets let us yield between searches without killing
                # the native worker or discarding its retained MCTS tree.
                native = {
                    **body,
                    "simulations": step,
                    "stream": on_update is not None and not initial,
                    "inspect": body.get("inspect", False) and initial,
                }
                for result in self._exchange(native, deadline):
                    stats = result["result"]
                    completed = stats.get("searched_simulations", target)
                    if stats.get("activations"):
                        activations = stats["activations"]
                    if carried is None:
                        carried = stats.get("carried_visits", 0)
                    step_done = stats.get("complete", True)
                    final = step_done and (
                        cancel.is_set()
                        or completed >= target
                        or stats.get("terminal") is not None
                    )
                    if final and activations:
                        stats["activations"] = activations
                    elapsed = (time.monotonic() - started) * 1000
                    stats.update(
                        complete=final,
                        cancelled=cancel.is_set(),
                        target_simulations=target,
                        carried_visits=carried,
                        elapsed_ms=elapsed,
                        simulations_per_second=max(0, completed - carried)
                        / max(elapsed / 1000, 0.001),
                    )
                    if on_update:
                        on_update(result)
                if final:
                    return result
                initial = False
                step = min(target, completed + 32)
        except AnalysisRequestError:
            raise
        except BaseException:
            # Invalid/cancelled requests before native submission leave its tree intact.
            if self.active and self.active[3] is finished:
                self.close()
            raise
        finally:
            with self.activity_lock:
                if self.active and self.active[3] is finished:
                    self.active = None
            self.lock.release()
            finished.set()

    def _exchange(self, body, deadline):
        self.process.stdin.write((encode(body) + "\n").encode())
        data = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([self.process.stdout], [], [], remaining)[0]
            ):
                raise TimeoutError("Analysis exceeded its time budget")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Analysis worker stopped")
            data.extend(chunk)
            if len(data) > 16 * 1024 * 1024:
                raise ValueError(
                    "Activation output exceeds its limit; select fewer layers"
                )
            while b"\n" in data:
                line, _, remainder = data.partition(b"\n")
                data = bytearray(remainder)
                result = json.loads(line)
                if "error" in result:
                    if result.get("error_kind") == "invalid_request":
                        raise AnalysisRequestError(result["error"])
                    raise ValueError(result["error"])
                if result.get("checkpoint", {}).get("model_sha256") != self.digest:
                    raise ValueError("Analysis checkpoint identity changed")
                complete = result["result"].get("complete", True)
                yield result
                if complete:
                    return
