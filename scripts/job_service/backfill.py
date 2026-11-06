"""Backfill all distinct historical replay datasets; share identical archived datasets."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
from .catalog import digest
from .importer import import_rows
from .store import Store, encode
from .worker import atomic_json


def backfill(root, exporter, semantics, reuse_root=None):
    root = Path(root).resolve()
    store = Store(root / "jobs.sqlite3")
    groups = {}
    known_semantics = {}
    with store.transaction(write=False) as db:
        jobs = list(db.execute("SELECT id FROM jobs WHERE state='archived'"))
        registered = {
            str(Path(row["path"]).parent / "replay.bin.zst"): json.loads(
                row["metadata"]
            )
            .get("files", {})
            .get("replay.bin.zst")
            for row in db.execute(
                "SELECT path,metadata FROM artifacts WHERE kind='checkpoint'"
            )
        }
    for row in jobs:
        job = row["id"]
        directory = root / "jobs" / job
        unique = {}
        for replay in sorted(directory.glob("checkpoints/*/replay.bin.zst")):
            unique.setdefault(
                registered.get(str(replay)) or digest(replay), replay.parent
            )
        if not unique:
            continue
        policy = semantics.get(job, "unknown_legacy_policy")
        if policy != "unknown_legacy_policy":
            for replay_hash in unique:
                if known_semantics.setdefault(replay_hash, policy) != policy:
                    raise ValueError(
                        "Identical replay bytes have conflicting policy semantics"
                    )
        key = hashlib.sha256(
            encode({"replays": sorted(unique), "policy_semantics": policy}).encode()
        ).hexdigest()
        group = groups.setdefault(
            key,
            {
                "jobs": [],
                "checkpoints": list(unique.values()),
                "semantics": policy,
                "replay_hashes": sorted(unique),
            },
        )
        group["jobs"].append(job)
    reports = {}
    for key, group in groups.items():
        directory = root / "backfill" / key
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "receipt.json"
        representative = group["jobs"][0]
        print(
            f"Backfill {key[:12]}: {len(group['checkpoints'])} distinct buffers, {len(group['jobs'])} jobs",
            flush=True,
        )
        if not receipt.exists() and reuse_root:
            source_receipt = Path(reuse_root) / "backfill" / key / "receipt.json"
            if source_receipt.exists():
                previous = json.loads(source_receipt.read_text())
                source_archive = (
                    Path(reuse_root) / "jobs" / representative / "games" / "archive"
                )
                if previous["dataset_key"] != key or not source_archive.is_dir():
                    raise ValueError("Reused dataset identity mismatch")
                source_rows = Path(previous["retained_source"])
                if digest(source_rows) != previous["source_sha256"]:
                    raise ValueError("Reused source checksum mismatch")
                target_rows = root / "sources" / source_rows.name
                if not target_rows.exists():
                    os.link(source_rows, target_rows)
                elif digest(target_rows) != previous["source_sha256"]:
                    raise ValueError("Existing retained source checksum mismatch")
                target_archive = root / "jobs" / representative / "games" / "archive"
                target_archive.parent.mkdir(parents=True, exist_ok=True)

                def link_existing(source, target):
                    try:
                        os.link(source, target)
                    except FileExistsError:
                        if not os.path.samefile(source, target) and digest(
                            source
                        ) != digest(target):
                            raise ValueError("Existing reused archive file differs")
                    return target

                # Complete a partially copied dataset after an interrupted migration.
                shutil.copytree(
                    source_archive,
                    target_archive,
                    copy_function=link_existing,
                    dirs_exist_ok=True,
                )
                previous["retained_source"] = str(target_rows)
                atomic_json(receipt, previous)
        if not receipt.exists():
            rows = directory / "positions.jsonl"
            with (directory / "export.log").open("ab", buffering=0) as log:
                subprocess.run(
                    [
                        str(Path(exporter).resolve()),
                        str(rows),
                        *[str(p) for p in group["checkpoints"]],
                    ],
                    stdout=log,
                    stderr=log,
                    check=True,
                )
            metadata = json.loads(rows.with_suffix(".metadata.json").read_text())
            if (
                sorted({source["replay_sha256"] for source in metadata["sources"]})
                != group["replay_hashes"]
            ):
                raise ValueError("Replay input changed since catalog registration")
            result = import_rows(store, root, representative, rows, group["semantics"])
            result["dataset_key"] = key
            result["source_metadata"] = json.loads(
                rows.with_suffix(".metadata.json").read_text()
            )
            atomic_json(receipt, result)
            # import_rows retained a verified source copy; remove only this generated duplicate.
            if digest(rows) != result["source_sha256"]:
                raise ValueError("Export changed after import")
            rows.unlink()
        archive = root / "jobs" / representative / "games" / "archive"
        artifact = store.artifact(representative, None, "replay_dataset", receipt)
        for job in group["jobs"]:
            target = root / "jobs" / job / "games" / "archive"
            if job != representative:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.resolve() != archive.resolve():
                        raise ValueError("Refusing to replace another existing archive")
                else:
                    target.symlink_to(
                        os.path.relpath(archive, target.parent),
                        target_is_directory=True,
                    )
            atomic_json(
                root / "jobs" / job / "replay-dataset.json",
                {
                    "artifact_id": artifact["id"],
                    "dataset_key": key,
                    "archive_job_id": representative,
                    "policy_semantics": group["semantics"],
                },
            )
        reports[key] = json.loads(receipt.read_text()) | {"jobs": group["jobs"]}
    atomic_json(root / "backfill-receipt.json", reports)
    store.artifact(None, None, "backfill_receipt", root / "backfill-receipt.json")
    return {
        key: {"games": value["imported_games"], "jobs": value["jobs"]}
        for key, value in reports.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument(
        "--semantics",
        type=Path,
        required=True,
        help="Job ID to known policy semantics; missing entries stay unknown",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            backfill(
                args.root,
                args.exporter,
                json.loads(args.semantics.read_text()),
                args.reuse_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
