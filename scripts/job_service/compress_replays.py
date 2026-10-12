"""Convert a trusted local jobs tree; leave manifests and directory aliases alone.

Stop archive writers first. Readers may remain live after deploying dual-format
support. Dry-run by default; --apply publishes verified compressed games, and
--remove-json removes originals only after a durable hash journal entry.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import replays


def inventory(root):
    groups = {}
    for base, directories, names in os.walk(root, followlinks=False):
        directory = Path(base)
        if directory.parent.name != "archive" or not re.fullmatch(
            r"\d{8}", directory.name
        ):
            continue
        for name in sorted(names):
            path = directory / name
            if (
                path.is_symlink()
                or path.suffix != ".json"
                or replays.game_id(path) is None
            ):
                continue
            stat = path.stat()
            groups.setdefault((stat.st_dev, stat.st_ino), []).append(str(path))
    return list(groups.values())


def identity(path):
    stat = Path(path).lstat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def prepare(paths):
    source = Path(paths[0])
    before = identity(source)
    raw = source.read_bytes()
    game = json.loads(raw)
    if game["game_id"] != replays.game_id(source):
        raise ValueError(f"Replay identity mismatch: {source}")
    target = replays.save(source.parent, game)
    encoded = target.read_bytes()
    if replays.unpack(encoded) != replays.normalize(game):
        raise ValueError(f"Published replay verification failed: {target}")
    if identity(source) != before:
        raise ValueError(f"Source changed: {source}")
    targets = []
    for name in paths:
        alias = Path(name)
        if identity(alias) != before:
            raise ValueError(f"Source alias changed: {alias}")
        destination = alias.with_name(replays.game_id(alias) + replays.SUFFIX)
        if destination != target:
            try:
                os.link(target, destination)
                replays.sync_directory(destination.parent)
            except FileExistsError:
                if destination.read_bytes() != encoded:
                    raise ValueError(f"Existing alias differs: {destination}")
        targets.append(str(destination))
    return {
        "sources": paths,
        "targets": targets,
        "source_identity": before,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "target_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_bytes": len(raw),
        "target_bytes": len(encoded),
        "source_allocated": source.stat().st_blocks * 512,
        "target_allocated": target.stat().st_blocks * 512,
    }


def commit(batch, journal, remove_json):
    # The receipt is durable BEFORE any removal. Re-running safely handles a
    # crash between publication, journaling and removal without duplicate games.
    for entry in batch:
        journal.write(json.dumps(entry, separators=(",", ":")) + "\n")
    journal.flush()
    os.fsync(journal.fileno())
    directories = set()
    if remove_json:
        for entry in batch:
            # Validate every hard-link alias before unlink changes their ctime.
            for path in entry["sources"]:
                if identity(path) != entry["source_identity"]:
                    raise ValueError(f"Source changed before removal: {path}")
            for path in entry["sources"]:
                Path(path).unlink()
                directories.add(Path(path).parent)
        for directory in directories:
            replays.sync_directory(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Jobs tree to convert")
    parser.add_argument("--journal", type=Path, help="Append-only migration receipt")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--remove-json", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.remove_json and not args.apply:
        parser.error("--remove-json requires --apply")
    if args.apply and not args.journal:
        parser.error("--apply requires --journal")
    root = args.root.resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor):
        parser.error("root must be a specific existing jobs directory")
    groups = inventory(root)
    if args.limit:
        groups = groups[: args.limit]
    summary = {
        "root": str(root),
        "unique_games": len(groups),
        "paths": sum(map(len, groups)),
        "remove_json": args.remove_json,
        "source_bytes": sum(Path(g[0]).stat().st_size for g in groups),
    }
    print(json.dumps(summary), flush=True)
    if not args.apply:
        return
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    counts = dict.fromkeys(
        ("source_bytes", "target_bytes", "source_allocated", "target_allocated"), 0
    )
    completed = 0
    with (
        args.journal.open("a") as journal,
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        replays.sync_directory(args.journal.parent)
        # Bound both queued work and retained metadata even on huge archives.
        iterator = iter(groups)
        while chunk := list(itertools.islice(iterator, args.workers * 8)):
            batch = list(pool.map(prepare, chunk))
            commit(batch, journal, args.remove_json)
            for entry in batch:
                for key in counts:
                    counts[key] += entry[key]
            previous = completed
            completed += len(batch)
            if completed // 1000 != previous // 1000 or completed == len(groups):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(groups),
                            "seconds": round(time.monotonic() - start, 1),
                            **counts,
                        }
                    ),
                    flush=True,
                )
    print(
        json.dumps(
            summary
            | counts
            | {"completed": completed, "seconds": time.monotonic() - start}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
