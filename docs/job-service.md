# Job execution and analysis service

The replacement was deployed on September 12, 2026. Production history has been
migrated and verified; the batching experiment is complete and self-play has resumed.
See the [cutover record](job-service-cutover-20260912.md) for the exact recovery
boundary and operational paths. Starting the service never adopts arbitrary
existing processes.

## Ownership and storage

The stdlib-only job service supports Python 3.12 and newer; its local formatter
configuration preserves that syntax. The separate torch development environment
uses Python 3.14.

Run the service on the **compute host**, with its database and outputs on persistent
storage. The browser can be anywhere. A local SSH tunnel can expose the pod's
loopback listener at `127.0.0.1:8766`; the scheduler must not depend on a laptop
remaining awake. This first implementation manages one Linux host and its GPU
memory pool, rather than scheduling across multiple hosts.

The deployed local tunnel is managed by [Docker Compose](../deploy/dashboard/README.md),
with a loopback listener at `127.0.0.1:8765`. This replaces the local
`alz-experiment-dashboard.service` user unit. The web backend and scheduler still
run on the compute host; the local container only provides the SSH connection.

`--root` contains:

- `jobs.sqlite3`: experiments, logical jobs, execution attempts, dependencies,
  acknowledged commands, events, and artifact references (SQLite WAL).
- `jobs/<job-id>/`: checkpoints, statistics, comparison results, and rich games.
- `attempts/<attempt-id>/`: immutable launch request, resolved configuration in
  the database, owner/child identities, logs, heartbeats, native/owner journals,
  stop requests, and final receipts.
- `binaries/<sha256>`: the executable pinned for each attempt.
- `catalog/`: immutable manifests for registered replay histories.
- `sources/`: preserved migration inputs identified by content hash.

A detached attempt owner survives HTTP-service restarts. It holds an exclusive
lock, launches an attempt at most once, and records the child's Linux boot ID,
PID and process start ticks. A dead owner kills its native child; reconciliation
fails the attempt instead of silently assuming completion or launching a duplicate.
Restarting a paused/failed logical job creates another attempt and reads its durable
checkpoints and completed games.

Admission first reserves an attempt in `preparing`, using a short database
transaction. A background preparation thread verifies each input reference once,
copies initialization data and pins the executable without holding a database
transaction or blocking event collection. An attempt-specific filesystem lock
prevents duplicate preparation across service instances. Restarting the service
recovers unfinished preparations. Pause/cancel during preparation is acknowledged
as `stopping`; its reservation remains held until preparation exits, and the
native process is never launched for that stopped attempt.

When the native process exits, the attempt enters `finalizing`. A background
finalizer drains bounded journal batches, catalogs published artifacts, recovers
missing epoch metrics, and only then commits completion and releases resources.
A filesystem lock and durable state make finalization recoverable across gateway
restarts. A catalog failure marks that attempt failed with a `finalization_failed`
event; other jobs keep receiving controls and admission continues. Correct the
reported artifact problem before resuming the failed job.

Checkpoint cataloging reuses hashes only when device/inode, size, modification
time, and change time of the published files still match. Changed files are
rechecked. Input resolution still verifies content before execution. Native
journals are read outside write transactions, at most 500 records per batch with
a byte budget, and their cursors commit atomically with events.

Queued jobs display a reason when their request exceeds this host's capacity or
when resources are occupied. These reasons are reevaluated during admission,
including after a host migration. Input roles are checked against registered
artifact types; replay training may use a checkpoint with a registered replay
buffer. Checkpoint selectors must refer to jobs that produce the required data.

The native archive is `games/archive/<zero-based-epoch>/<game-id>.json`.
It contains canonical states, actions, actors, outcomes, raw network priors, root
visits, training policies, actual sampling probabilities, search statistics,
model hashes by seat, and game duration. Collection manifests pin game-generation
settings. Self-play also saves the producer weights for an unfinished epoch.
Completed games publish atomically before being reported; incomplete games can
be regenerated. Completed-collection receipts retain their original duration and
inference counts across a pre-training pause. Earlier failed partial attempts
remain visible in attempt timings; collection throughput describes the successful
collection attempt, not the total cost of all failed work. Recovered self-play games are ordered by stable game ID before
training, independent of completion order.

Checkpoint/optimizer recovery is at completed training epochs. A pause during a
training pass finishes that pass; an immediate interruption repeats that pass from
the preceding checkpoint. This is not mid-minibatch optimizer recovery.

## Start a fresh service

Build the native executable and web assets in the normal repository environment:

```sh
./run.sh cargo build --release --workspace
npm --prefix web ci
npm --prefix web run build
./run.sh python -m scripts.job_service.server --root runs/job-service --set-password
./run.sh python -m scripts.job_service.server \
  --root runs/job-service --binary target/release/alz \
  --listen 127.0.0.1 --port 8766 \
  --slots 2 --host-memory-mb 65536 --gpu-memory-mb 20000
```

The numbers above are **example reservations**, not tested pod capacity. Choose
budgets for the actual host and leave room for the operating system and other
processes. CPU jobs still require host-memory reservations. GPU reservations govern
admission; allocator use is observed separately. The worker reports RSS/high-water
RSS, cgroup working set, per-process GPU memory when available, and heartbeat age.
Host-memory pressure requests a recoverable stop and escalates if necessary.
Admission and the emergency threshold use usage minus estimated clean, unmapped file cache;
shared, mapped, dirty, writeback and unevictable memory remain charged. Conventional
working set and raw cgroup usage are also reported, and worker RSS has its own
reservation limit. This avoids treating read-heavy archive migration as equivalent
to anonymous allocation pressure. The accounting follows the kernel's
[cgroup v1](https://docs.kernel.org/admin-guide/cgroup-v1/memory.html#stat-file) and
[cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory) statistics;
the cache estimate is not a guarantee that every page can be reclaimed immediately.

Without an administrator password, mutations cannot authenticate. The password is
stored as a salted scrypt hash; sessions have no server-side expiry and survive
service restarts. The browser cookie is persistent for one year and renewed on each
page load; browser-side deletion or expiry still requires signing in again.
Active sessions from the previous eight-hour policy are migrated without logout;
already-expired sessions are not revived. The private `admin-sessions.sqlite3` database stores token hashes
and CSRF tokens; logout revokes sessions durably. Changing the administrator
password invalidates existing sessions when the service starts. Public GET views are readable. Scheduling, pause/resume,
configuration changes and analysis require a session cookie, matching Origin and
CSRF token. The HTTP API accepts typed inputs, not shell commands or client paths.

For a public HTTPS reverse proxy, set `--origin https://your-dashboard.example`.
This must match the browser's exact origin and enables Secure cookies. Keep the
service on loopback behind the proxy or SSH tunnel. No production credentials or
proxy settings are installed by the implementation.

To enable the analysis worker, add:

```sh
--analysis-checkpoint /absolute/path/to/an/exact/checkpoint \
--analysis-device cpu --analysis-host-memory-mb 2048
```

Analysis resolves and pins a checkpoint at service startup, loads its model lazily,
and handles one request at a time. Searches stream snapshots through the authenticated
`POST /api/v1/analyze/stream` endpoint. A simulation budget is a target: repeated
requests reuse the current tree, and a larger target extends it. Moving to a
searched successor retains its subtree; unrelated positions replace the cache.
The first response exposes that retained root (or initializes a fresh root) and
includes the selected activation layer before the remaining search starts.
The explorer preserves layer/channel selection across moves; its chart counts new
simulations from zero and reports the retained count separately.
A client can send a `request_id` and cancel its own session's search through
`POST /api/v1/analysis/cancel` with that ID. The gateway yields after at most 32
additional simulations, preserving the native worker and its tree. The board can
change immediately while the next search waits for cancellation; stale snapshots
cannot overwrite the new position.
The analysis executable can be upgraded separately with `--analysis-binary`.
It has a 20,000-simulation cap, a 15-minute request timeout, output limits,
and a separate memory reservation subtracted from scheduler capacity. GPU analysis
also requires `--analysis-gpu-memory-mb`. Activation capture runs only on this
inspection path, not normal self-play/training. The UI fetches one selected layer's
values at a time; layer metadata is cheap to browse. The current inspector supports
Kata network variants; the legacy ResNet reports that inspection is unsupported.

## Routes and API

- `/experiments`: newest-first paginated queue, job attempts, timing, stage,
  progress, losses, self-play statistics, confidence intervals, game outcomes,
  resource observations, and event history. Logged-in controls schedule/pause/
  resume/cancel jobs and pause queue admission.
- `/games?job=<id>`: paginated recorded games, selected epoch and ply, saved
  distributions, and links into branch exploration.
- `/analyze` (`/play` redirects here): shared board exploration, search, manual alternative
  moves, undo, and activation/channel inspection. Position links include game
  identity and a game-defined state encoding.

Native analysis uses `PositionEnvelope { game_type, state }`; the game adapter
owns state validation and semantics. The implemented view/adapter is Gomoku. Go
will need its own rules, history/ko state, non-board actions and rendering adapter;
it does not need changes to the job database or worker protocol.

Useful public endpoints are `GET /api/v1/jobs?offset=0&limit=25`,
`jobs/<id>`, `events?job_id=<id>&after=<cursor>`, `artifacts`, `games`, `game`,
`job-summary?job_id=<id>`, `event-history?job_id=<id>&before=<cursor>`,
`schema`, `session`, and `analysis`. Job detail includes a `runtime` snapshot of
current stage and progress, independent of paginated event history. Events have
durable numeric cursors. Native
move updates stay in bounded Rust watch channels; durable heartbeats contain
aggregates, not a growing history of all active-game snapshots.

`POST /api/v1/commands` takes `{ "command_id": "unique-client-id", "request": ... }`.
Retries with the same ID and body return the same acknowledgment; a different body
under that ID conflicts. Requests include:

```json
{"action":"pause","job_id":"...","mode":"epoch"}
{"action":"pause","job_id":"...","mode":"boundary"}
{"action":"resume","job_id":"..."}
{"action":"cancel","job_id":"...","mode":"boundary"}
{"action":"scheduler","paused":true}
{"action":"configure","job_id":"...","options":{"games-parallelism":400}}
```

`epoch` stops after a committed training epoch. `boundary` stops after self-play
collection, a completed comparison game, or a completed training epoch, as
appropriate. `immediate` terminates the child and preserves completed artifacts.
Queue pause prevents new admission while running jobs continue. A stop is shown
as **stopping** until the worker acknowledges it; it is not immediately called
paused. A stop arriving after natural completion can still result in success.

`configure` is allowed only while queued/paused/failed and only changes execution
limits (game parallelism, inference batch size, batch timeout, heartbeat interval)
and resource reservations. It cannot silently change architecture, LR, seed or
training data within a logical experiment. Old attempts retain their exact config.
The initial UI exposes scheduling and lifecycle controls; execution-limit edits
are available through this authenticated API.

A create request has a title, typed `spec` and optional dependencies. Job kinds are
`self_play`, `replay_train`, `reconstruction`, `comparison`, `benchmark_executor`,
and `benchmark_self_play`. Benchmark jobs require a registered checkpoint input
and publish a result artifact. Executor benchmarks cover fixed or declining
producer counts, guard reacquisition, output parity, and optional allocator profiling. Obtain allowed
option names/types from `/api/v1/schema`. Every spec explicitly chooses its device
and reserves `slots`, `host_memory_mb`, `gpu_memory_mb`. Unspecified native settings
are captured in the attempt's `resolved_config` event.

Registered checkpoints hash the model, metadata, and available optimizer/replay
files; all are verified before use. Running trainers publish completed snapshots
to the catalog in background scans every 30 seconds; pending snapshot directories
are excluded. Restarting the service backfills any missed snapshots. The scheduling
form refreshes saved choices every 15 seconds without resetting selected inputs.
Checkpoint inputs can be registered artifact IDs or
`{"job_id":"...","selection":"latest"}` /
`{"job_id":"...","selection":"best_value_validation"}`. Deferred selectors wait
for the producing job to succeed, pin the exact source hash, and fail if the chosen
validation evidence is unavailable. They never silently fall back to the last pass.
A comparison dependency can specify `{"job_id":"...","fraction":0.6}` to overlap
a later comparison at 60% completion, subject to resource reservations. Inputs that
require that same job's completed checkpoint still require success.

Register trusted existing inputs locally on the compute host:

```sh
./run.sh python -m scripts.job_service.catalog --root runs/job-service \
  --kind checkpoint --path /absolute/path/to/checkpoints/00000039
./run.sh python -m scripts.job_service.catalog --root runs/job-service \
  --kind history --path /absolute/path/to/self-play-run
```

History registration hashes every included replay, metadata and epoch-stat file.
Reconstruction uses a history artifact and `replay-history-epochs`; native validation
requires matching training/data-order settings. It can continue with new self-play
once the reconstructed prefix finishes. Replay-training jobs can pool up to 100
registered checkpoint buffers (`replay`, `replay_2`, ... inputs); the form's
**Add replay buffer** control adds inputs. Native training deduplicates their games
and keeps the fixed validation split separate from training.

## Migration and cutover

1. Export a complete legacy dashboard snapshot and copy the required checkpoints,
   optimizer states, replays, epoch stats and comparison reports. Keep the originals.
2. Preview the importer against a **new** root:

   ```sh
   ./run.sh python -m scripts.job_service.importer --root runs/job-service-import \
     --snapshot /absolute/path/to/snapshot.json
   ```

3. Apply to that staging root with `--apply`. Import IDs are deterministic within
   `--namespace`; repeated identical imports are idempotent. Imported jobs are
   historical/archived, never runnable legacy recipes. Conflicting newer snapshots
   require a deliberate fresh staging import, not silent overwrite.
4. Export compact replays using `cargo run --example export_replay_policies --
   OUTPUT.jsonl CHECKPOINT...`; import with `--replay-rows OUTPUT.jsonl --job-id ID
   --policy-semantics ... --apply`. Use known policy semantics only. Missing prior,
   visits, sampler, producer identity and final action remain unavailable. Verified
   consecutive states can recover earlier actions. A containing checkpoint is not
   mislabeled as the generation epoch. Sources and checksums remain preserved.
5. Compare job/epoch/game counts, model hashes, WDL, metrics and sample positions.
   Cached old snapshots can contain aggregate comparison statistics without raw
   games; importing them preserves aggregates but does not invent individual games.
6. At cutover, pause the old supervisor at a completed epoch, take the final copy,
   and disable its automatic restarts. Register that exact checkpoint and create a
   new self-play job with it as the initial checkpoint input. Verify epoch/LR/replay
   capacity before admission. Never let two schedulers own the same production run.
7. Start the new service, verify recovery and the new worker's first checkpoint,
   then switch the public proxy. Retain the old database/files for rollback. Remove
   old recipe/collector code and the explicit legacy web entrypoint only after this
   migration is verified.

For backups while running, use SQLite's backup API (not a lone copy of the database
file while its WAL is active) and copy published artifacts. A stopped-service backup
must still account for independent live owners; pausing and waiting for their
receipts gives a consistent experiment boundary.

## Validation and remaining experiments

```sh
./run.sh cargo test --workspace
python -m unittest discover -s scripts -p 'test_*.py'
npm --prefix web test
npm --prefix web run build
ALZ_TEST_BINARY=target/debug/alz OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./run.sh python -m unittest scripts.test_job_service_integration
```

The opt-in integration test uses temporary directories and defaults to CPU.
Set `ALZ_TEST_DEVICE=cuda` for the GPU workflow, and `TMPDIR` to a directory on
the production filesystem to exercise its actual locking and publication behavior. Build the
chosen binary first. It exercises real self-play, collection recovery, replay
training, best-validation selection and comparison recovery through the service.

CPU checks cover native tensor output parity, activity-drop wakeups, cancellation,
three real requests padded to four, tree accounting/pruning, actual training
pause/resume, replay reconstruction, comparison game reuse, command idempotency,
resource/dependency admission, service restart, auth/CSRF and migration provenance.
Web checks cover chart utilities, production build, and browser flows on desktop
and mobile with recorded games and layer inspection.

CUDA padding correctness, memory stability and throughput still need the pod
preflight and [BATCH-001](experiment-backlog.md#batch-001-active-producers-and-fixed-inference-batch-buckets).
The default is unpadded inference. Live per-game streaming UI, automatic resource
tuning, historical search reproduction, checkpoint switching in the analysis UI,
Go rules and transformer attention inspection remain explicit follow-up work.

## Reproducible benchmarks and historical data

The native `benchmark executor` command reports exact request-latency quantiles,
useful throughput, real/physical batch histograms, dispatch reasons and output
parity against single-row inference. Ten maximum-size warmup batches precede its
measurement. Ordinary executor telemetry uses bounded power-of-two latency and
queue-wait histograms; those bins describe upper bounds, not exact quantiles.

`--profile-allocator` loads the trusted local library named by
`ALZ_ALLOCATOR_PROBE` and writes `result.allocator.json`. Build
`scripts/benchmarks/cuda_allocator_probe.cpp` against the deployed libtorch/CUDA
headers. The probe runs separately from primary throughput trials and reads the
same process's CUDA allocator. It does not require changing the tch fork.

`scripts/benchmarks/run_batch_experiment.py` submits sequential typed jobs with
rotated trial order and retains `experiments/<experiment-id>/trials.json`.
A trusted CLI administrator may register an old executable with catalog
`--kind binary`; a spec's optional `binary_artifact` selects that hash-verified
executable. HTTP callers cannot upload executables or provide paths. This allows
an old implementation to be measured without changing the service's default
trainer. Completed legacy comparison reports are recovered into summary events;
unavailable per-game finish times are not invented. Full move histories remain
in result artifacts and game archives, rather than being repeated in event rows.

`python -m scripts.job_service.migrate --root ROOT --snapshot SNAPSHOT
--inventory INVENTORY --apply` imports a trusted inventory covering every job in
the snapshot. It verifies copied bytes, preserves identical immutable checkpoint
files as hard links, registers checkpoints/history, and reconstructs old
comparison boards from their one-based move coordinates. Source documents and
checksums remain available under `sources/` and `migration-receipt.json`.

`python -m scripts.job_service.backfill --root ROOT --exporter EXPORTER
--semantics SEMANTICS` converts all distinct historical replay datasets. The
semantics JSON maps job IDs to known policy-target semantics; unspecified values
remain unknown. Identical datasets share archives. `--reuse-root` can reuse a
verified rehearsal export, avoiding repeated conversion. New training writes
fresh epoch directories; historical datasets remain immutable. A replay-buffer
checkpoint number labels where a game was observed, not its generating epoch.

Native analysis drains oversized input records with bounded memory and reports a
recoverable `invalid_request` response for oversized or malformed JSON records.
The gateway preserves the cached native process for these errors, while resetting
it on analysis/protocol failures. Inference batch grids are passed explicitly
from CLI dispatch to executor construction; no process-global batch-grid state is
used. The executor benchmark's explicit `--grid` overrides the common grid flag.
Comparisons abort on game errors and count only successfully recorded games.

Focused preparation checks: `python -m unittest scripts.test_job_preparation`.

The service navigation currently includes Experiments, Games and Analysis. Play
was deliberately removed pending migration of the original WebSocket demo;
old `/play` URLs preserve their position parameters and open Analysis. Unknown
service paths show a missing-page view. Analysis audit events include the client
request identifier alongside the server audit identifier; cancellation requests,
acknowledgements and failures are recorded without authentication credentials.
Completed collection telemetry is retained separately from live self-play,
and comparison telemetry identifies both checkpoint executors.

Epoch charts load from `/api/v1/epoch-summaries?job_id=…&after=…`, which returns
all matching `epoch_completed` events after the cursor in one response. The
browser polls this feed independently of paginated diagnostic events, so game
and heartbeat backlogs do not delay completed epochs appearing on the charts.


## Frontend boundaries and repeatable browser checks

`JobWorkspace` renders job details; `useJobData` owns independent live status,
epoch and summary feeds, and `JobForm` owns scheduling inputs. `EventHistory`
loads 100 recent events only when opened and pages backward on request. The
browser does not rebuild comparison statistics by replaying diagnostic journals.
`job-summary` returns complete available WDL, seat, length, duration and completion
statistics, preserving imported aggregates when raw game records are incomplete.

`GameExplorer` composes the Gomoku board and move inspection. `ReplayArchive`
renders archive navigation, `usePositionAnalysis` owns search cancellation,
streaming and activation requests, and `ActivationViewer` owns layer/channel
rendering. Pure view conversions and record types live in `gameViewModel`.

```sh
npm --prefix web ci
cd web
npx playwright install chromium firefox
npm run test:browser
```

The checked-in browser suite starts a temporary loopback asset server and mocks
API responses. It needs no credentials, GPU, pod, or experiment data. It covers
terminal board geometry, quiet live refreshes, stale response rejection, lazy
history loading, and making a move while analysis is pending. Run it alongside
`npm test`; the build checks unused TypeScript declarations as well as types.
