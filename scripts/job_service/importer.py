"""Best-effort historical import. Originals retained; missing evidence stays unavailable.

Run without --apply to inspect a snapshot. This never adopts a live process.
"""

from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

from .store import Store, encode
from .worker import atomic_json


def identity(namespace, name):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, namespace + ":" + name))


def timestamp(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return None


def retain_source(root, path):
    path = Path(path)
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    target = Path(root) / "sources" / (digest + path.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
    with target.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != digest:
            raise ValueError(
                "Retained source checksum mismatch; refusing partial evidence"
            )
    return digest, target


def inspect_snapshot(path):
    data = json.loads(Path(path).read_text())
    snapshot = data.get("snapshot", data)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("phases"), list):
        raise ValueError("Expected a dashboard snapshot with phases")
    phases = snapshot["phases"]
    return snapshot, {
        "jobs": len(phases),
        "training_epochs": sum(len(p.get("metrics", [])) for p in phases),
        "comparison_games": sum(
            len((p.get("result") or {}).get("games", [])) for p in phases
        ),
        "live_jobs_requiring_separate_handoff": [
            p["id"]
            for p in phases
            if p.get("state") in ("running", "paused", "queued", "pending")
        ],
    }


def import_snapshot(store, root, path, namespace="legacy"):
    snapshot, report = inspect_snapshot(path)
    digest, original = retain_source(root, path)
    import_key = "snapshot-import:" + digest
    now = time.time()
    with store.transaction() as db:
        db.execute("PRAGMA cache_size=-65536")
        if db.execute("SELECT 1 FROM settings WHERE key=?", (import_key,)).fetchone():
            return report | {"already_imported": True}
        for phase in snapshot["phases"]:
            job_id = identity(namespace, phase["id"])
            experiment_id = identity(namespace, "experiment:" + phase["id"])
            existing = db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone()
            if existing:
                # Don't silently overwrite history from a different source snapshot.
                raise ValueError(
                    f"{phase['id']} already imported from another snapshot"
                )
            timing = phase.get("timing") or {
                key: phase[key]
                for key in (
                    "started_at",
                    "ended_at",
                    "duration_seconds",
                    "timing_source",
                )
                if phase.get(key) is not None
            }
            started = timestamp(timing.get("started_at") or phase.get("started_at"))
            ended = timestamp(timing.get("ended_at") or phase.get("ended_at"))
            effective = {
                "resources": {"slots": 0, "host_memory_mb": 0, "gpu_memory_mb": 0},
                "provenance": "imported",
                "source_sha256": digest,
                "original_id": phase["id"],
                "historical_state": phase.get("state"),
                "historical_details": {
                    key: phase[key]
                    for key in (
                        "architecture",
                        "completed",
                        "total",
                        "unit",
                        "note",
                        "training_performance",
                        "match_settings",
                        "baseline_checkpoint",
                    )
                    if key in phase
                },
                "participants": phase.get("participants", {}),
                "recorded_timing": timing,
                "resolved_config": (phase.get("result") or {}).get("config"),
            }
            spec = {
                "kind": "historical_" + phase.get("kind", "unknown"),
                "options": {},
                "inputs": {},
                "resources": effective["resources"],
            }
            title = phase.get("title", phase["id"])
            db.execute(
                "INSERT INTO experiments VALUES (?,?,?,?)",
                (experiment_id, title, encode(effective), started or now),
            )
            db.execute(
                "INSERT INTO jobs(id,experiment_id,title,spec,state,created,updated,reason) VALUES (?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    experiment_id,
                    title,
                    encode(spec),
                    "archived",
                    started or now,
                    ended or now,
                    f"Historical status: {phase.get('state', 'unknown')}",
                ),
            )
            attempt = identity(namespace, "attempt:" + phase["id"])
            db.execute(
                "INSERT INTO attempts(id,job_id,number,state,created,started,ended,effective,result) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    attempt,
                    job_id,
                    1,
                    "archived",
                    started or now,
                    started,
                    ended,
                    encode(effective),
                    encode({"historical_state": phase.get("state")}),
                ),
            )
            store.event(db, job_id, attempt, "imported", effective)
            for metric in phase.get("metrics", []):
                # Dashboard epochs are displayed one-based; native protocol is zero-based.
                store.event(
                    db,
                    job_id,
                    attempt,
                    "epoch_completed",
                    metric
                    | {"epoch": int(metric["epoch"]) - 1, "provenance": "imported"},
                )
            result = phase.get("result") or {}
            for game in result.get("games", []):
                store.event(
                    db,
                    job_id,
                    attempt,
                    "comparison_game",
                    {k: v for k, v in game.items() if k != "moves"}
                    | {"finish_time_unavailable": True},
                )
            if result:
                store.event(
                    db,
                    job_id,
                    attempt,
                    "comparison_completed"
                    if phase.get("kind") == "battle"
                    else "historical_result",
                    {k: v for k, v in result.items() if k != "games"},
                )
        db.execute(
            "INSERT INTO settings VALUES (?,?)",
            (import_key, encode({"source": str(original), "report": report})),
        )
    return report | {"source_sha256": digest}


def packed(stones):
    cells = [0] * 361
    for action, value in stones:
        if not 0 <= action < 361 or value not in (1, 2) or cells[action]:
            raise ValueError("Invalid or duplicate occupied cell")
        cells[action] = value
    result = [0] * 91
    for i, value in enumerate(cells):
        result[i // 4] |= value << (2 * (i % 4))
    return cells, {"state": result}


def converted_game(rows, semantics, source_sha256):
    if semantics not in (
        "unknown_legacy_policy",
        "temperature_adjusted_policy",
        "normalized_root_visits",
    ):
        raise ValueError("Explicit stored-policy semantics are required")
    plies = []
    for index, row in enumerate(rows):
        if row["ply"] != index:
            raise ValueError("Expected contiguous positions from the start of a game")
        cells, state = packed(row["stones"])
        legal = [i for i, value in enumerate(cells) if value == 0]
        weights = dict(row["policy"])
        legal_set = set(legal)
        if (
            any(
                a not in legal_set or not isinstance(p, (int, float)) or not 0 <= p <= 1
                for a, p in weights.items()
            )
            or abs(sum(weights.values()) - 1) > 1e-4
        ):
            raise ValueError("Invalid stored policy")
        chosen = None
        if index + 1 < len(rows):
            successor, _ = packed(rows[index + 1]["stones"])
            candidates = [i for i in legal if successor[i] == 2]
            verified = [
                i
                for i in candidates
                if [
                    2 if j == i else 2 if n == 1 else 1 if n == 2 else 0
                    for j, n in enumerate(cells)
                ]
                == successor
            ]
            if len(verified) == 1:
                chosen = verified[0]
        plies.append(
            {
                "state": state,
                "action": None
                if chosen is None
                else {"x": chosen // 19, "y": chosen % 19},
                "actor": "First" if index % 2 == 0 else "Second",
                "turn_change": "SwitchPlayer",
                "value_actual": row["value"],
                "decision": {
                    "move_index": legal.index(chosen) if chosen is not None else None,
                    "training_policy": [weights.get(a, 0) for a in legal],
                    "diagnostics": {
                        "network_prior": None,
                        "root_visits": None,
                        "sampling_policy": None,
                        "value_estimate": None,
                        "search_value": None,
                        "search": None,
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "game_type": "gomoku19_five_v1",
        "game_id": rows[0]["game"],
        "seed": None,
        "model_identity": None,
        "policy_semantics": semantics,
        "provenance": "imported",
        "source_sha256": source_sha256,
        "containing_checkpoint_epoch": rows[0]["source_epoch"],
        "generation_epoch": None,
        "field_provenance": {
            "state": "recorded",
            "training_policy": "recorded",
            "value_actual": "recorded",
            "action": "derived_when_transition_verified",
            "network_prior": "unavailable",
            "generating_model": "unavailable",
        },
        "record": {
            "plies": plies,
            "terminal_state": None,
            "terminal_actor": None,
            "terminal_value": None,
        },
    }


def import_rows(store, root, job_id, path, semantics="unknown_legacy_policy"):
    with store.transaction(write=False) as db:
        if not db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
            raise ValueError("Unknown destination job")
    digest, original = retain_source(root, path)
    count = duplicates = 0
    published_manifests = set()

    def flush(rows):
        nonlocal count, duplicates
        if not rows:
            return
        archive = converted_game(rows, semantics, digest)
        game_id = archive["game_id"]
        if not game_id or any(c not in "0123456789abcdef" for c in game_id):
            raise ValueError("Game id must be a hexadecimal content identity")
        partition = f"{int(rows[0]['source_epoch']):08}"
        directory = Path(root) / "jobs" / job_id / "games" / "archive" / partition
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (game_id + ".json")
        if target.exists():
            if json.loads(target.read_text()) != archive:
                raise ValueError("Existing game differs; refusing to overwrite it")
            duplicates += 1
            return
        atomic_json(target, archive)
        if partition not in published_manifests:
            atomic_json(
                directory / "manifest.json",
                {
                    "schema_version": 1,
                    "label": f"Source checkpoint {int(partition)} (generation epoch unknown)",
                    "source_sha256": digest,
                },
            )
            published_manifests.add(partition)
        count += 1

    rows = []
    with Path(path).open() as file:
        for line in file:
            row = json.loads(line)
            if rows and rows[-1]["game"] != row["game"]:
                flush(rows)
                rows = []
            rows.append(row)
        flush(rows)
    return {
        "imported_games": count,
        "duplicate_games": duplicates,
        "total_games": count + duplicates,
        "source_sha256": digest,
        "retained_source": str(original),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--replay-rows", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument(
        "--policy-semantics",
        default="unknown_legacy_policy",
        choices=[
            "unknown_legacy_policy",
            "temperature_adjusted_policy",
            "normalized_root_visits",
        ],
    )
    parser.add_argument("--namespace", default="legacy")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        if not args.snapshot:
            parser.error("Dry run currently requires --snapshot")
        print(json.dumps(inspect_snapshot(args.snapshot)[1], indent=2))
        return
    store = Store(args.root / "jobs.sqlite3")
    if args.snapshot:
        print(encode(import_snapshot(store, args.root, args.snapshot, args.namespace)))
    if args.replay_rows:
        if not args.job_id:
            parser.error("--replay-rows requires --job-id")
        print(
            encode(
                import_rows(
                    store,
                    args.root,
                    args.job_id,
                    args.replay_rows,
                    args.policy_semantics,
                )
            )
        )


def converted_battle_game(game, report, source_sha256):
    """Reconstruct canonical states from a complete recorded move sequence."""
    moves = game.get("moves", [])
    if len(moves) != game["plies"]:
        raise ValueError("Incomplete comparison move history")
    cells = [0] * 361
    plies = []
    models = {
        seat: report[game[key]]["model_sha256"]
        for seat, key in [("First", "first_seat"), ("Second", "second_seat")]
    }
    for index, move in enumerate(moves):
        row, column = move["row"] - 1, move["column"] - 1
        if not (
            isinstance(row, int)
            and isinstance(column, int)
            and 0 <= row < 19
            and 0 <= column < 19
        ):
            raise ValueError("Invalid comparison move coordinates")
        action = row * 19 + column
        side = game["first_seat" if index % 2 == 0 else "second_seat"]
        if cells[action] or move["checkpoint"] != side or move["ply"] != index + 1:
            raise ValueError("Invalid comparison move sequence")
        _, state = packed([(i, n) for i, n in enumerate(cells) if n])
        legal = [i for i, n in enumerate(cells) if not n]
        plies.append(
            {
                "state": state,
                "action": {"x": row, "y": column},
                "actor": "First" if index % 2 == 0 else "Second",
                "turn_change": "SwitchPlayer",
                "value_actual": 0
                if not game["winner"]
                else 1
                if game["winner"] == side
                else -1,
                "decision": {
                    "move_index": legal.index(action),
                    "training_policy": None,
                    "diagnostics": {
                        "network_prior": None,
                        "root_visits": None,
                        "sampling_policy": None,
                        "value_estimate": move.get("value_estimate"),
                        "search_value": None,
                        "search": None,
                    },
                },
            }
        )
        cells = [
            2 if i == action else 2 if n == 1 else 1 if n == 2 else 0
            for i, n in enumerate(cells)
        ]
    _, terminal = packed([(i, n) for i, n in enumerate(cells) if n])
    side = game["first_seat" if len(plies) % 2 == 0 else "second_seat"]
    return {
        "schema_version": 1,
        "game_type": "gomoku19_five_v1",
        "game_id": f"{game['game'] - 1:08}",
        "seed": None,
        "model_identity": None,
        "models_by_seat": models,
        "duration_seconds": game.get("duration_seconds"),
        "policy_semantics": "unavailable",
        "provenance": "imported",
        "source_sha256": source_sha256,
        "field_provenance": {
            "state": "derived_from_recorded_moves",
            "action": "recorded",
            "network_prior": "unavailable",
            "generating_model": "recorded",
        },
        "record": {
            "plies": plies,
            "terminal_state": terminal,
            "terminal_actor": "First" if len(plies) % 2 == 0 else "Second",
            "terminal_value": 0
            if not game["winner"]
            else 1
            if game["winner"] == side
            else -1,
        },
    }


if __name__ == "__main__":
    main()
