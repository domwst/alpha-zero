"""Bounded journal reads outside SQLite write transactions."""

import json

MAX_RECORDS = 500
MAX_BYTES = 4 * 1024 * 1024


def read_batch(path, cursor):
    records = []
    end = cursor
    if not path.exists():
        return records, end, True
    with path.open("rb") as file:
        file.seek(cursor)
        while len(records) < MAX_RECORDS and end - cursor < MAX_BYTES:
            line = file.readline(MAX_BYTES + 1)
            if len(line) > MAX_BYTES:
                raise ValueError(f"Oversized journal record in {path.name}")
            if not line or not line.endswith(b"\n"):
                # A killed writer can leave an incomplete final record. Never
                # advance its cursor; a live writer may still finish it.
                return records, end, True
            records.append(json.loads(line))
            end += len(line)
    return records, end, False


def ingest_native(store, attempt, path):
    key = "native_cursor:" + attempt["id"]
    with store.transaction(write=False) as db:
        row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        cursor = int(row[0]) if row else 0
    records, end, caught_up = read_batch(path, cursor)
    if not records:
        return caught_up
    with store.transaction() as db:
        row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if (int(row[0]) if row else 0) != cursor:
            return False  # Another scheduler committed this batch first.
        for record in records:
            store.event(
                db,
                attempt["job_id"],
                attempt["id"],
                record["kind"],
                record.get("payload", {}),
                timestamp=record.get("time"),
            )
            if record["kind"] == "configured":
                row = db.execute(
                    "SELECT effective FROM attempts WHERE id=?", (attempt["id"],)
                ).fetchone()
                effective = json.loads(row[0]) | {"resolved_config": record["payload"]}
                db.execute(
                    "UPDATE attempts SET effective=? WHERE id=?",
                    (json.dumps(effective), attempt["id"]),
                )
        db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, str(end)))
    return caught_up
