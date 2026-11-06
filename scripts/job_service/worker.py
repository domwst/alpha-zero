"""Detached attempt owner. Outlives the HTTP service; owns the child's process group."""

from __future__ import annotations

import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .store import encode
from .resources import memory_pressure, gpu_memory


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        f.write(encode(data) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def emit(directory, kind, payload):
    # The owner is the single writer; reconciliation writes only after it exits.
    # Discard a torn final record before appending a recovered completion receipt.
    with (Path(directory) / "worker-events.jsonl").open("a+b") as f:
        f.seek(0)
        existing = f.read()
        if existing and not existing.endswith(b"\n"):
            f.truncate(existing.rfind(b"\n") + 1)
        f.write(
            (
                encode({"time": time.time(), "kind": kind, "payload": payload}) + "\n"
            ).encode()
        )
        f.flush()
        os.fsync(f.fileno())


def process_identity(pid):
    """PID + boot + Linux start ticks prevent PID reuse from implying ownership."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": stat[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def parent_death_signal(parent):
    # The attempt owner outlives service restarts. If that owner itself crashes,
    # kill its native child instead of allowing an untracked orphan to consume GPU.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGKILL)


def run(directory):
    directory = Path(directory).resolve()
    lock = (directory / "owner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    # An attempt executes at most once, even if duplicate launch requests race.
    if (directory / "started.json").exists():
        return 0
    request = json.loads((directory / "request.json").read_text())
    atomic_json(
        directory / "started.json",
        {"owner": process_identity(os.getpid()), "time": time.time()},
    )
    child = None
    stopped = False
    returncode = 1
    reason = None
    try:
        environment = os.environ.copy()
        environment.update(request.get("environment_overrides", {}))
        environment["ALZ_JOB_DIR"] = str(directory)
        environment["ALZ_JOB_ID"] = request["job_id"]
        environment["ALZ_ATTEMPT_ID"] = request["attempt_id"]
        with (directory / "output.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                request["argv"],
                cwd=request["cwd"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=lambda parent=os.getpid(): parent_death_signal(parent),
            )
            atomic_json(
                directory / "process.json",
                {
                    "owner": process_identity(os.getpid()),
                    "child": process_identity(child.pid),
                },
            )
            emit(
                directory,
                "worker_started",
                {
                    "pid": child.pid,
                    "owner": os.getpid(),
                    "binary_sha256": request["binary_sha256"],
                },
            )
            terminate_at = None
            gpu_bytes, next_gpu = None, 0
            while child.poll() is None:
                control_path = directory / "control.json"
                control = (
                    json.loads(control_path.read_text())
                    if control_path.exists()
                    else None
                )
                if (
                    control
                    and control.get("mode") == "immediate"
                    and terminate_at is None
                ):
                    os.killpg(child.pid, signal.SIGTERM)
                    terminate_at = time.monotonic()
                    stopped = True
                if terminate_at is not None and time.monotonic() - terminate_at > 30:
                    os.killpg(child.pid, signal.SIGKILL)
                memory = {}
                try:
                    for line in (
                        Path(f"/proc/{child.pid}/status").read_text().splitlines()
                    ):
                        if line.startswith(("VmRSS:", "VmHWM:")):
                            key, value = line.split(":")
                            memory[key] = int(value.split()[0]) * 1024
                except FileNotFoundError:
                    pass
                if time.monotonic() >= next_gpu:
                    gpu_bytes, next_gpu = gpu_memory(child.pid), time.monotonic() + 10
                observed = memory_pressure()
                atomic_json(
                    directory / "heartbeat.json",
                    {
                        "time": time.time(),
                        "memory": memory,
                        "gpu_memory_bytes": gpu_bytes,
                        "host": observed,
                        "child": process_identity(child.pid),
                    },
                )
                reservation = request["resources"]["host_memory_mb"] * 1024 * 1024
                over_budget = reservation > 0 and memory.get("VmRSS", 0) > reservation
                host_critical = (
                    observed
                    and observed.get("pressure_bytes", observed["used_bytes"])
                    > observed["limit_bytes"] * 0.95
                )
                if (over_budget or host_critical) and terminate_at is None:
                    # Stop admission/finish the current recovery boundary first.
                    if control is None:
                        atomic_json(
                            directory / "control.json",
                            {
                                "action": "pause",
                                "mode": "boundary",
                                "reason": "memory reservation exceeded",
                            },
                        )
                        emit(
                            directory,
                            "memory_guard",
                            {
                                "rss_bytes": memory.get("VmRSS"),
                                "reservation_bytes": reservation,
                                "host": observed,
                            },
                        )
                    if host_critical or memory.get("VmRSS", 0) > reservation * 1.1:
                        os.killpg(child.pid, signal.SIGTERM)
                        terminate_at, stopped = time.monotonic(), True
                        reason = "Memory guard interrupted work; completed games and checkpoints are retained"

                time.sleep(1)
            returncode = child.wait()
            # Cooperative stops carry an explicit acknowledgement from the native process.
            ack = directory / "stopped.json"
            if ack.exists():
                stopped = True
                reason = json.loads(ack.read_text()).get("reason")
    except BaseException as error:
        reason = f"{type(error).__name__}: {error}"
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    result = {"returncode": returncode, "stopped": stopped, "reason": reason}
    atomic_json(directory / "result.json", result)
    emit(directory, "worker_finished", result)
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
