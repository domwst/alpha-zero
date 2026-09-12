"""Durable queue controls shared by the worker and SSH dashboard agent."""

import datetime
import fcntl
import json
from pathlib import Path


def read_control(root):
    path = Path(root) / "control.json"
    if not path.exists():
        return {"paused": False, "updated_at": None}
    value = json.loads(path.read_text())
    if type(value.get("paused")) is not bool:
        raise ValueError("Queue control file has an invalid paused flag")
    return value


def set_paused(root, paused):
    if type(paused) is not bool:
        raise ValueError("paused must be a boolean")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".control.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = {"paused": paused, "updated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat()}
        temporary = root / "control.json.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(root / "control.json")
        return value
