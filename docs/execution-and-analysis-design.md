# Execution and analysis design decisions

Agreed direction as of 2026-09-12. The replacement is implemented locally; see
[job service operation and migration](job-service.md). The production supervisors
have not yet been migrated. Compatibility adapters are
not a design requirement for the replacement; legacy data will be imported into
the new format on a best-effort basis.

## Job service and Rust controllers

Use one durable job service to manage experiments, logical jobs, execution
attempts, artifacts, resource allocation, and acknowledged lifecycle transitions.
A restart or concurrency adjustment creates a recorded attempt/configuration
change under the same logical run. The service reconciles worker state after its
own restart; a live PID alone is not proof of successful resume.

Keep individual-game scheduling and inference dispatch in Rust. The self-play
controller decides when to admit another game using resource limits and worker
events. The executor dispatches batches using its queue, activity information,
configured batch grid, and deadline. Python and the browser receive aggregated
progress and durable records rather than coordinating every simulation.

Every game has a stable ID, move sequence, and events for moves, search summaries,
completion, and errors. Persist completed games for recovery. High-frequency
telemetry is bounded and coalescible; slow viewers must not block MCTS. Reliable
game/control events have a separate delivery/persistence policy. Capturing live
game events is groundwork; a UI for watching individual live games is deferred.

Track process ownership, heartbeat, stage, host/GPU memory, request latency,
executor queues and dispatch decisions. Start MCTS observability with allocated
child entries and expanded nodes by root-relative depth, simulation leaf-depth
distribution, retained/discarded nodes after moves, and estimated tree bytes.
Keep per-tree counters and update them at safe expansion/pruning boundaries;
avoid a histogram in every node and measure instrumentation overhead.

## Submission handles and batching

An executor handle provides access but does not itself count as active work.
`SubmissionHandle<'a>` mutably borrows an executor handle, registers one active
producer, and is the only interface through which that producer submits work.
Submission takes a mutable borrow so one guard has at most one outstanding
inference operation. Handle clones are passive until they acquire their own guard.
Future parallel simulations must represent additional producer capacity explicitly.

Finishing, cancellation, or guard drop updates the
authoritative producer count and wakes the executor. Pending requests are accounted
for separately. Notifications may coalesce, but activity changes must not be lost
because a notification queue is full. No cross-process messages are required per
guard transition.

- Self-play with one checkpoint: hold a submission guard for the whole game.
- Interactive demo: activity spans a search, excluding human think time and idle
  connected sessions.
- Comparison with separate checkpoint executors: default to activity only for the
  side currently searching. Both passive handles remain allocated; drop and
  reacquire the submission guard per turn. There is no separate suspension API.
  Counting both sides as active
  would make each executor wait for producers that are using the other checkpoint.
  Measure notification overhead before optimizing this lifecycle.

Use a configurable fixed grid of physical inference batch sizes. The real-request
fill target must not exceed the number of requests active producers can supply.
Dispatch when the target is reached, all available producers have submitted, or
the deadline expires. Pad the actual batch upward to a supported bucket and return
only real outputs. For example, three active producers can submit three requests
and execute in a bucket of four without waiting for a fourth producer.

The deadline still handles active producers doing tree traversal instead of
submitting. Inference padding requires evaluation-mode behavior and output-parity
checks; it is not a proposal to pad training batches. The batch-grid decision and
fragmentation hypothesis are queued as [BATCH-001](experiment-backlog.md#batch-001-active-producers-and-fixed-inference-batch-buckets).

## Game archive and replay exploration

Persist a rich game record alongside the compact training representation. New
records should contain complete game state/actions, terminal outcome, acting
player, model identity, and per-move distributions with explicit semantics:

- Raw network prior before search exploration noise.
- Final root visit counts / search policy used as the training target.
- Actual sampling probabilities after temperature and top-p, and the chosen action.
- Available root network/search values and compact search statistics.

These values should be captured while the worker already has them. Exact
historical search reproduction and a separate historical re-evaluation mode are
out of scope. Recording additional data does not require those features.

Representation boundaries:

- `TrainingSample` stores only state, policy target, and terminal value.
- `MoveDecision` carries the chosen move, training policy, priors, visits, values,
  sampling diagnostics and search statistics. Rich game archives preserve them;
  extraction into compact training samples deliberately drops them.
- MCTS counters update on committed expansion and retained-subtree pruning.
  Live game watch channels coalesce updates; the durable protocol emits summaries.

Import old data into the same archive schema; use explicit unavailable fields
instead of separate legacy UI paths. Preserve stored policy semantics: older
temperature-adjusted targets must not be relabeled as normalized search visits.
Recover actions from consecutive states only when the transition can be verified;
the last action generally cannot be uniquely recovered from the existing buffer.
Reconstruct sampling distributions only when policy semantics and the collection
settings are known. Raw historical priors that were never saved remain unavailable.
Distinguish recorded, derived, and unavailable data. A checkpoint containing an
old game does not by itself identify the checkpoint that originally generated it.
Preserve source archives/checksums and make imports idempotent.

The first replay explorer loads a recorded position into the shared interactive
board, runs new analysis with a default checkpoint, and permits alternative move
branches. Its purpose is to assess moves and understand positions. A checkpoint
picker is optional follow-up work. Show the active analysis checkpoint without
requiring reconstruction of the original inference symmetry or search conditions.

## Shared website, network inspection, and additional games

Serve monitoring, recorded games, interactive analysis, and the existing demo under
one application with separate routes. A position URL may embed a small complete
state or refer to a stored state. Preserve original games when users explore branches.

Use game-defined state/rules serialization and action identifiers, with explicit
player perspective and rendering capabilities. Do not assume every action is a
board coordinate or that visible stones fully define a legal position. Gomoku and
future Go adapters should share interfaces without requiring dynamic plugin loading.

An analysis worker uses a pinned checkpoint and separate compute budget. Optional
inspection returns selected named tensors with shapes and semantic axes, supporting
convolution-channel maps, pooled vectors, and future attention heads. Keep activation
capture off the ordinary training/self-play path. Visual patterns suggest hypotheses;
they are not causal explanations of model behavior.

One administrator account enables structured scheduling and pause/resume commands.
Public views remain read-only. Authorization is enforced by the API, including
appropriate session/CSRF protection; hiding buttons is only presentation. Initially
require login for new compute-consuming analysis too. Record acknowledged commands
and resulting transitions in the job history.
