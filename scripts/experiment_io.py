"""Shared experiment records, checksums, and checkpoint validation."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fixed(path, value):
    if path.exists():
        require(read(path) == value, f"Configuration or inputs changed: {path}")
    else:
        write(path, value)


def checkpoint(path):
    path = Path(path).resolve()
    metadata = read(path / "metadata.json")
    require(digest(path / "model.safetensors") == metadata["model_sha256"],
            f"Model checksum mismatch: {path}")
    return {"path": str(path), **{key: metadata[key] for key in (
        "format_version", "epoch", "model", "model_sha256", "tensor_schema_sha256")}}


def canonical_descriptor(descriptor):
    return {**descriptor, "path": str(Path(descriptor["path"]).resolve())}


def validate_battle(report, first, second, games, simulations, temperature):
    require(canonical_descriptor(report["first_checkpoint"]) == canonical_descriptor(first)
            and canonical_descriptor(report["second_checkpoint"]) == canonical_descriptor(second),
            "Battle checkpoint identities differ from the requested checkpoints")
    require(len(report["games"]) == games and report["config"]["games"] == games,
            "Battle has the wrong number of games")
    require(report["config"]["simulations"] == simulations, "Wrong simulation budget")
    require(report["first_temperature"] == report["second_temperature"] == temperature,
            "Wrong match temperature")
    require(sorted(game["game"] for game in report["games"]) == list(range(1, games + 1)),
            "Battle contains missing or duplicate games")
    for label, opponent in [("first_checkpoint", "second_checkpoint"),
                            ("second_checkpoint", "first_checkpoint")]:
        stats = report[label + "_result"]
        wins = sum(game["winner"] == label for game in report["games"])
        draws = sum(game["winner"] is None for game in report["games"])
        require((stats["wins"], stats["losses"], stats["draws"]) ==
                (wins, games - wins - draws, draws), "Inconsistent match totals")
        require(stats["score"] == wins + draws * 0.5, "Inconsistent match score")
        require(sum(game["first_seat"] == label for game in report["games"]) == games // 2,
                "Match must have equal games in each seat")
        require(all(game["winner"] in (label, opponent, None) for game in report["games"]),
                "Unknown game winner")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def tail(path, limit=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            start = max(0, stream.tell() - limit)
            stream.seek(start)
            lines = stream.read().decode(errors="replace").splitlines()
            return lines[1:] if start else lines
    except FileNotFoundError:
        return []


def locked(path):
    if not path.exists():
        return False
    with path.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True

