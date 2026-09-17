"""Register trusted local checkpoint/history inputs; not exposed as a public path API."""

import argparse
import hashlib
import json
from pathlib import Path

from .store import Store, encode
from .worker import atomic_json


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def file_signature(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def register_checkpoint(store, directory, job_id=None, attempt_id=None):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    model = directory / "model.safetensors"
    signatures = {
        p.name: file_signature(p)
        for p in directory.iterdir()
        if p.name
        in {"model.safetensors", "metadata.json", "optimizer.ot", "replay.bin.zst"}
    }
    with store.transaction(write=False) as db:
        cached = db.execute(
            "SELECT * FROM artifacts WHERE path=? AND sha256=? AND kind='checkpoint'",
            (str(model), metadata["model_sha256"]),
        ).fetchone()
    if cached and (job_id is None or cached["job_id"] == job_id):
        if json.loads(cached["metadata"]).get("file_signatures") == signatures:
            return dict(cached)
    if digest(model) != metadata["model_sha256"]:
        raise ValueError("Checkpoint model checksum mismatch")
    metadata = metadata | {
        "files": {
            p.name: digest(p)
            for p in directory.iterdir()
            if p.name
            in {"model.safetensors", "metadata.json", "optimizer.ot", "replay.bin.zst"}
        }
    }
    if any(
        file_signature(directory / name) != signature
        for name, signature in signatures.items()
    ):
        raise ValueError("Checkpoint changed while being cataloged")
    return store.artifact(
        job_id,
        attempt_id,
        "checkpoint",
        model,
        metadata | {"file_signatures": signatures},
    )


def register_history(store, root, directory, job_id=None, attempt_id=None):
    directory = Path(directory).resolve()
    files = sorted(directory.glob("checkpoints/*/replay.bin.zst"))
    if not files:
        raise ValueError("History contains no replay checkpoints")
    hashes = {}
    for replay in files:
        with store.transaction(write=False) as db:
            row = db.execute(
                "SELECT metadata FROM artifacts WHERE path=? AND kind='checkpoint' ORDER BY rowid DESC LIMIT 1",
                (str(replay.parent / "model.safetensors"),),
            ).fetchone()
        cached = json.loads(row[0]) if row else {}
        for name in ("replay.bin.zst", "metadata.json"):
            path = replay.parent / name
            cached_digest = cached.get("files", {}).get(name)
            verified = cached.get("file_signatures", {}).get(name) == file_signature(
                path
            )
            hashes[str(path.relative_to(directory))] = (
                cached_digest if verified and cached_digest else digest(path)
            )
        stats = directory / "stats" / (replay.parent.name + ".json")
        hashes[str(stats.relative_to(directory))] = digest(stats)
    manifest = {"directory": str(directory), "files": hashes}
    content = encode(manifest).encode()
    destination = (
        Path(root) / "catalog" / (hashlib.sha256(content).hexdigest() + ".json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A restarted finalizer may encounter a torn manifest from an older release.
    try:
        intact = json.loads(destination.read_text()) == manifest
    except (FileNotFoundError, ValueError):
        intact = False
    if not intact:
        atomic_json(destination, manifest)
    return store.artifact(
        job_id, attempt_id, "history", destination, {"epochs": len(files)}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--kind", choices=["checkpoint", "history", "binary"], required=True
    )
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    store = Store(args.root / "jobs.sqlite3")
    if args.kind == "binary":
        import os

        if not args.path.is_file() or not os.access(args.path, os.X_OK):
            parser.error("Binary must be an executable file")
        identifier = store.artifact(
            None, None, "binary", args.path.resolve(), {"label": args.path.name}
        )
        print(encode({"artifact_id": identifier["id"]}))
        return
    identifier = (
        register_checkpoint(store, args.path, args.job_id)
        if args.kind == "checkpoint"
        else register_history(store, args.root, args.path, args.job_id)
    )
    print(encode({"artifact_id": identifier["id"]}))


if __name__ == "__main__":
    main()
