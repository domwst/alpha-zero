"""Migrate a trusted inventory into a fresh service root, preserving source evidence.

Inventory paths are administrative/local inputs, never accepted by the HTTP API.
The caller must stop writers before the final inventory is copied and verified.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from .catalog import register_checkpoint, register_history
from .importer import identity, import_snapshot, retain_source, converted_battle_game
from .store import Store
from .worker import atomic_json


def verified_copy(source, destination, shared=None):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def digest(path):
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    expected = digest(source)
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"Existing migration destination differs: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".copying")
        reusable = (
            (shared or {}).get(expected)
            if source.name in ("model.safetensors", "optimizer.ot", "replay.bin.zst")
            else None
        )
        if temporary.exists():
            if digest(temporary) != expected:
                if temporary.stat().st_nlink != 1:
                    raise ValueError(
                        "Partial migration file unexpectedly shares storage"
                    )
                shutil.copy2(source, temporary)
        elif reusable:
            os.link(reusable, temporary)
        else:
            shutil.copy2(source, temporary)
        if digest(temporary) != expected or digest(source) != expected:
            raise ValueError(f"Source changed during migration: {source}")
        temporary.replace(destination)
    if shared is not None:
        shared.setdefault(expected, destination)
    return expected


def migrate(root, snapshot_path, inventory_path, namespace="legacy"):
    root = Path(root).resolve()
    inventory = json.loads(Path(inventory_path).read_text())
    data = json.loads(Path(snapshot_path).read_text())
    snapshot = data.get("snapshot", data)
    phases = {p["id"]: p for p in snapshot["phases"]}
    if set(inventory["jobs"]) != set(phases):
        raise ValueError("Inventory must cover exactly every snapshot job")
    retain_source(root, snapshot_path)
    retain_source(root, inventory_path)
    # Full native reports retain individual game rows omitted by dashboard caches.
    for name, entry in inventory["jobs"].items():
        if entry.get("report"):
            report = json.loads(Path(entry["report"]).read_text())
            previous = phases[name].get("result") or {}
            for key in ("first_checkpoint", "second_checkpoint"):
                expected = previous.get(key, {}).get("model_sha256")
                if expected and report.get(key, {}).get("model_sha256") != expected:
                    raise ValueError(f"Report checkpoint mismatch for {name}")
            if (
                previous.get("first_checkpoint_result")
                and report.get("first_checkpoint_result")
                != previous["first_checkpoint_result"]
            ):
                raise ValueError(f"Report outcome mismatch for {name}")
            phases[name]["result"] = previous | report
        elif entry.get("partial_statistics"):
            phases[name]["result"] = {
                "game_statistics": entry["partial_statistics"],
                "partial": True,
            }
    enriched = root / "migration-snapshot.json"
    atomic_json(enriched, snapshot)
    store = Store(root / "jobs.sqlite3")
    result = import_snapshot(store, root, enriched, namespace)
    files = {}
    shared = {}
    checkpoints = 0
    jobs = {}
    for name, entry in inventory["jobs"].items():
        print(f"Migrating {name}", flush=True)
        job_id = identity(namespace, name)
        output = root / "jobs" / job_id
        source = Path(entry["directory"]) if entry.get("directory") else None
        if source:
            for path in sorted(source.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    target = output / path.relative_to(source)
                    files[str(target.relative_to(root))] = verified_copy(
                        path, target, shared
                    )
        for key, target_name in [("report", "result.json"), ("log", "legacy.log")]:
            if entry.get(key):
                target = output / target_name
                files[str(target.relative_to(root))] = verified_copy(
                    entry[key], target, shared
                )
        report = phases[name].get("result") or {}
        if report.get("games") and all("moves" in game for game in report["games"]):
            digest = files[str((output / "result.json").relative_to(root))]
            directory = output / "games" / "archive" / "00000000"
            directory.mkdir(parents=True, exist_ok=True)
            for game in report["games"]:
                archive = converted_battle_game(game, report, digest)
                atomic_json(directory / (archive["game_id"] + ".json"), archive)
            atomic_json(
                directory / "manifest.json",
                {
                    "schema_version": 1,
                    "label": "Recorded comparison games",
                    "source_sha256": digest,
                },
            )
        for model in output.glob("checkpoints/*/model.safetensors"):
            register_checkpoint(store, model.parent, job_id)
            checkpoints += 1
        replays = list(output.glob("checkpoints/*/replay.bin.zst"))
        if replays and all(
            (output / "stats" / (p.parent.name + ".json")).exists() for p in replays
        ):
            register_history(store, root, output, job_id)
        jobs[name] = {
            "job_id": job_id,
            "directory": str(output),
            "replay_checkpoints": len(replays),
        }
    # Inputs not produced by one of the displayed jobs (e.g. original checkpoint69).
    for index, path in enumerate(inventory.get("additional_checkpoints", [])):
        source = Path(path)
        output = root / "inputs" / f"{index:04}" / source.name
        for file in source.iterdir():
            if file.is_file():
                target = output / file.name
                files[str(target.relative_to(root))] = verified_copy(
                    file, target, shared
                )
        register_checkpoint(store, output)
        checkpoints += 1
    receipt = result | {
        "schema_version": 1,
        "checkpoints": checkpoints,
        "files": files,
        "jobs": jobs,
    }
    atomic_json(root / "migration-receipt.json", receipt)
    store.artifact(None, None, "migration_receipt", root / "migration-receipt.json")
    return {k: v for k, v in receipt.items() if k != "files"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--namespace", default="legacy")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        inventory = json.loads(args.inventory.read_text())
        print(json.dumps({"jobs": list(inventory["jobs"]), "apply": False}))
        return
    print(
        json.dumps(
            migrate(args.root, args.snapshot, args.inventory, args.namespace), indent=2
        )
    )


if __name__ == "__main__":
    main()
