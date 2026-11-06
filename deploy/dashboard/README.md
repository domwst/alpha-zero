# Local historical dashboard

Docker Compose serves the final RunPod experiment archive on **127.0.0.1:8765**.
The existing reverse proxy for `https://dashboard.oleja.dev` can keep this upstream.
No SSH tunnel, remote pod, GPU, or training worker is required.

```sh
docker compose up -d --build --wait --wait-timeout 60
docker compose ps
docker compose logs --tail 50
```

Run these commands from this directory. Optional UID/GID/port overrides are shown
in `.env.example`; old SSH variables in an existing `.env` are unused.
`DASHBOARD_DEMO_URL` (default `https://gomoku.oleja.dev`) is passed as
`--demo-url` so the dashboard header links to the interactive demo and recorded
games offer `Play from here ↗` links into it.

The container mounts `scripts/` and `web/dist/` read-only, and uses
`runs/dashboard-local/data/` for SQLite and historical games. Build frontend
assets with `npm --prefix web run build` from the repository root when needed.
The archive backend uses `--read-only`: login and all API mutations are disabled,
and no scheduler or analysis worker is started. SQLite may still create WAL and
session files, so the data mount is writable. The independent interactive demo
continues to serve its own checkpoint.

## Retained data and restoration

`data/jobs.sqlite3` retains all experiment, event, and epoch history. Each job's
`games/archive/` contains compressed `.msgpack.zst` replays and JSON manifests.
Old `.json` replays remain readable, including archives restored from backup.
Preserve relative
symlinks between job archives. Model weights, checkpoint metadata, result files,
and logs are also retained. Optimizer state, binary replay buffers, imported source
datasets, and old deployment/build snapshots are in the user's external backup.

The artifact catalog retains original `/workspace/...` paths and hashes for
provenance; it does not imply those training artifacts are available locally.
Resume training only after restoring the complete archived training environment.
Cleanup inventories and integrity receipts live in `runs/dashboard-local/`.

The former tunnel deployment remains available in Git history. Do not restart the
old user-systemd SSH tunnel alongside this container.
