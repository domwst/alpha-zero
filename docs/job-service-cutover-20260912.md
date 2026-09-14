# September 12, 2026 service cutover

Status: historical migration verified; [BATCH-001 completed](batching-experiment-20260912.md);
production resumed on September 12 at approximately 13:20 UTC. Its first complete
epoch is being checked at the production simulation budget.

## Recovery boundary

The requested pause was armed for the end of displayed epoch 42. The legacy RAM
guard instead stopped the old process on September 12 at 11:33:23 UTC, after all
1,000 games had been generated but before a training checkpoint was saved. There
was no recorded kernel OOM kill. That old executable did not retain the completed
collection across this interruption.

The last complete checkpoint is native `00000040`, displayed **epoch 41**, model
SHA-256 `12b62bb89dd4b594ab676bd5a1f1abfef98b619b9fba70d06f9ed7a7caa2b282`.
Its model, optimizer, replay buffer, and metadata are preserved. Continuation must
regenerate epoch 42. The imported job has an `epoch_interrupted` event; no completed
epoch or replacement games were invented. New native collection archives publish
completed games before reporting them, so subsequent interruptions can reuse them.

The old guard counted active file cache toward its threshold. Archive reads may
have contributed; this was host RAM pressure accounting, not evidence of CUDA
allocator fragmentation. New admission and worker guards separately report raw
usage, working set, estimated clean unmapped cache, and remaining pressure. Shared,
mapped, dirty, writeback, and unevictable pages remain charged; worker RSS also has
an independent reservation.

## Verified migration

- 39 archived jobs, 364 checkpoint artifacts, and 363 completed training epochs.
- 16,800 complete comparison outcomes, including recorded moves; incomplete
  canceled comparisons retain their available aggregate results.
- Four replay datasets containing 15,750, 20,000, 2,000, and 41,000 games.
  These datasets overlap across experiments; their sum is not a globally distinct
  game count. All retained source hashes and per-dataset counts were checked.
- Historical data marks missing original priors, model identities, and game timing
  as unavailable. Recorded training-policy semantics distinguish old
  temperature-adjusted targets from normalized root visits.
- SQLite integrity verification and a consistent database backup passed.

Code also rejects contradictory policy semantics for identical replay bytes,
recovers interrupted copy operations, and verifies retained source copies.

After verification, 37,776 redundant rehearsal hardlinks were removed, releasing
10,472,416,748 logical bytes while retaining their identical final-root copies.
The cleanup rechecked both inode identity and SHA-256; its audit and receipt are
retained. Earlier duplicate rehearsal replay copies were likewise removed only
after verifying their original counterparts.

The persistent volume was increased from 100 to 150 GB to hold both rollback data
and migrated archives. The resize reset the container; its Python runtime was
backed up and restored. Checkpoints and service state remained under `/workspace`.

## Deployment

- Public origin: `https://dashboard.oleja.dev`.
- Local listener: `127.0.0.1:8765`, forwarded to the pod's `127.0.0.1:8766`
  listener. Initially managed by the user systemd unit, this tunnel was moved to
  [Docker Compose](../deploy/dashboard/README.md) later the same day. The old unit
  is disabled and retained for rollback. The scheduler runs on the pod and
  continues when the laptop disconnects.
- Service root: `/workspace/alz-job-service`.
- Deployed source: `/workspace/alpha-zero-service-20260912`.
- Native binary SHA-256:
  `0f2fd775cd058a623f447fd077bb5f5b00c4f0063f746295e7791a70b59ee30a`.
- Pod tmux sessions: `alz-service` (gateway log), `self-play` (current native log).
  The completed benchmark launcher log is retained in the cutover directory.
  Detailed native logs belong to service attempts and are visible in the dashboard.
- Analysis uses CPU and the pinned replay-trained board-mask checkpoint, native
  pass 13, model `7d79483199e6bcca70955ef293c38832475c813262bfb7955c0f9ac59c0ffbb4`.

Administrator credentials remain in the ignored local cutover directory; the
password is not included in this document or uploaded as plaintext. Public reads,
HTTPS browser login/logout, origin enforcement, CSRF checks, secure cookies,
analysis, recorded-policy replay browsing, and desktop/mobile charts were checked.
Changing the public domain requires changing the exact configured origin.

## Evidence and rollback

The ignored local directory `runs/service-cutover-20260912/` retains raw source
backups, deployment scripts, connection records, and the `service-backup/`
database/receipts. `migration-verification.json` records the checked inventories.
The current-pod raw backup is 3,642,624,000 bytes with SHA-256
`c2e80706920802c3dea3b1ff58336d7719d0d050f40f616d9eb18d4794a98218`.
Earlier pod archives are retained locally as well.

The old trainer and original run directories remain available. Do not restart an
old supervisor alongside a service-owned continuation. Rollback first stops the
new owner at a durable boundary and explicitly selects the desired complete
snapshot. The original local dashboard unit overrides are retained below the new
`40-job-service.conf` override.

## Benchmark gate

See [BATCH-001](experiment-backlog.md#batch-001-active-producers-and-fixed-inference-batch-buckets).
The protocol retains an explicit TF32 diagnosis and precision amendment. Strict
executor checks keep their original numerical tolerances; actual MCTS comparisons
use production precision. Training precision has not been changed.

## Production continuation

[Live job](https://dashboard.oleja.dev/experiments?job=e6d9e519-3869-4b55-8709-f3c5182231e7):
`e6d9e519-3869-4b55-8709-f3c5182231e7`. It copies the complete native epoch-40
snapshot and links the immutable historical game partitions. The 41 earlier
epochs remain visible on its charts. Its first attempt is
`54ddd413-1c10-4bf7-a5e3-38ffd9430525`.

Settings: S3000/P400/B256, 1,000 games per epoch, top-p 0.95, no batch grid,
standard Adam, CPU replay cache, two prefetched batches, and the existing
replay-linked learning-rate schedule. The target remains 100 displayed epochs.
The initial attempt has no precision environment override.
