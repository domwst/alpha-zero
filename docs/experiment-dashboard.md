# Read-only experiment dashboard

Open **http://127.0.0.1:8765/experiments** while the local dashboard server is
running. It shows the current activation experiment and the six follow-up
stages: checkpoint-69 match, two fresh value-head training runs, and three
pairwise head comparisons, followed by two pooling stages when the baseline
checkpoint is reused (three for historical paired-training queues). Select a stage for
its timing, loss curves, results, or logs. Result rows identify the training source or
architecture variant, checkpoint number, and model checksum, including matches
between two networks with the same architecture.

The protocol banner describes future matches (1,000 games, up to 300 concurrent).
Each completed result retains its actual game count and concurrency. Match
details also show the inference batch limit of 64 positions per network.

Completed comparison details have a **Confidence level** selector: 50%, 68%,
80%, 90%, 95% (default), 98%, 99% and 99.9%. Both networks' intervals update
together, and the browser remembers the choice across comparisons and reloads.
The browser recalculates the same two-sided Wilson score interval used by the
Rust report from wins, draws and losses. Draws count as half a win, retaining
the report's approximation for non-binary outcomes. Higher confidence gives
wider intervals. Saved reports and experiment selection retain their original
95% intervals; this control only changes the view and sends no writes to the pod.

Completed comparisons also include results by seat for each checkpoint: games,
wins, draws, losses, score and the selected confidence interval. First means the
player making the opening move, independently of which checkpoint is listed first.
Overall starting-player wins, second-player wins and draws summarize seat effects.

Game-length summaries show minimum, mean, median, p90, p95 and maximum, plus a
histogram with 1-, 5- (default), 10- or 20-move bins. Exact per-move counts support
rebucketing without estimating counts from coarse bars. Narrow the move range or
switch to a logarithmic count scale to inspect the tail. Hover or tap anywhere
above a bin, including tiny and empty bars, for its count and percentage. A move
is one ply (one stone placed by either player).
Filter by either checkpoint winning or by draws. Percentiles use nearest rank.
Completion curves and milestones show elapsed match time until 50%, 90% and 99%
of games finished, along with the time taken by the final 10%. These come from
the final complete attempt's progress log and are checked against report runtime.
They are not per-game durations: historical reports do not record game start
times, and games run concurrently. Missing/incomplete logs show an explicit
unavailable state. Curves are sampled to about 300 points while retaining milestones.
The agent caches these summaries and rereads full logs only when inputs change;
full move histories are not sent to the dashboard.

Loss charts show training and validation values to six decimal places at the
inspected pass, with horizontal guides. Hover or tap to inspect points, or focus
a plot and use arrow keys (Home/End jump to the first/last point). Loss and length
charts include expandable exact-data tables. All chart
controls only change the local view. Click anywhere in a scheduled-work row to
inspect it; its title button remains available to keyboard users.

The queue lists each job's wall-clock duration; running jobs show elapsed time.
Job details show start time, end time and duration, with dates and time-zone
labels in the browser's local time zone. Historical starts come from the first
timestamped log entry, and completion times from the published result file's
modification time. This includes setup, validation/checkpoint writing and gaps
between attempts. It is a reconstruction, not a process-level timing trace;
copying or rewriting result files can change the reconstructed end time.
If a match log is unavailable, the recorded match runtime supplies a clearly
marked start estimate. Unavailable times show a dash; active jobs have no end
time. The current training pass timestamp remains separate for ETA calculation.

The local server connects to the pod using the existing SSH identity. The pod
needs no public HTTP port. The SSH agent and local bridge accept only status and known-stage log requests.
The HTTP server binds exclusively to `127.0.0.1` and serves only GET/HEAD.
POST, PUT, PATCH, and DELETE return 405; the queue-control endpoint is removed.

## Start or restart locally

On this computer the dashboard runs as a user service, independently of the
terminal. It starts when the user service manager starts and restarts after a
crash:

```bash
systemctl --user status alz-experiment-dashboard
systemctl --user restart alz-experiment-dashboard
journalctl --user -u alz-experiment-dashboard -n 50 --no-pager
```

The unit is maintained in `scripts/experiment-dashboard.service`. To install or
update it on this computer:

```bash
mkdir -p ~/.config/systemd/user
cp scripts/experiment-dashboard.service ~/.config/systemd/user/alz-experiment-dashboard.service
systemctl --user daemon-reload
systemctl --user enable --now alz-experiment-dashboard
```

Its paths assume the repository is at `~/misc/alpha-zero`; adjust them if moving
the checkout. The existing verified SSH host keys are preserved in
`runs/value-head-preflight-20260906/dashboard-known-hosts`, referenced by the
connection JSON, so reconnecting does not depend on a temporary file.

For a foreground session instead, first stop the service to free the port:

```bash
systemctl --user stop alz-experiment-dashboard
npm --prefix web run build
python3 scripts/experiment_dashboard.py \
  --connection runs/value-head-preflight-20260906/dashboard-connection.json \
  --port 8765
```

The connection file contains paths and the SSH target, not private key contents.
For this pod its remote agent is
`/workspace/alpha-zero-followups/scripts/experiment_agent.py`, with the original
activation run at `/workspace/alpha-zero/runs/relu-vs-gelu-20260906` and the
follow-up run at `/workspace/alpha-zero-followups/runs/value-heads-20260906`.

The queue worker runs independently in the pod's `replay-followups` tmux session.
Closing the webpage or local server does not stop experiments. When the local
computer sleeps, the pod continues; the dashboard reconnects when it can. Last
received data remains visible, marked disconnected, during connection failures.

## Read-only behavior

The dashboard cannot pause, resume, start, stop, or edit experiments. The queue
continues independently on the pod using the existing schedule and control
file. Inspecting stages, switching chart metrics, and selecting a color theme
only change the browser view.

The worker starts follow-ups after the original job publishes a zero exit code
and valid complete results. Nonzero exit, mismatched settings, changed inputs,
or a follow-up failure stop further work. The reviewed schedule is documented
in [replay-followup-experiments.md](replay-followup-experiments.md).

## Reverse proxy

Public exposure is managed separately. Proxy the site root to
`http://127.0.0.1:8765`; the dashboard accepts the proxy's public Host/Origin
headers. The upstream listener remains on loopback. Proxy `/assets/` and `/api/`
along with `/experiments`; no WebSocket connection is needed.

## HTTP API

All endpoints are local, on port 8765:

| Request | Response/action |
| --- | --- |
| `GET /api/experiments` | Connection state, current queue, worker state, telemetry, metrics, results, and protocol |
| `GET /api/experiments/logs?phase_id=kata-v1` | Last 80 log lines for a known stage |
| `HEAD` on either read endpoint | Response headers without a body |

```bash
curl -s http://127.0.0.1:8765/api/experiments
curl -s 'http://127.0.0.1:8765/api/experiments/logs?phase_id=kata-v1'
```

The backend polls every five seconds. Training progress counts published
checkpoints; curves update once per completed pass. Training ETAs use recent
pass durations. Match ETAs become available after games finish and may vary as
game lengths and parallelism change. These are estimates for the current stage,
not a promised completion time for the whole queue.

## Validation

The Python tests cover rejected mutations at the HTTP, bridge, and agent layers,
GET/HEAD responses, proxy headers, restricted log access, and the worker gate. The queue's
separate tests cover predecessor failure, match identity/settings validation,
selection, and immutable configuration. Run with:

```bash
python3 -m unittest discover -s scripts -p 'test_*.py'
npm --prefix web test
```

The local deployment was also tested in Chromium against the live pod: nine
stages, loss-chart switching, recent logs, absent queue controls, rejected write
requests, and unchanged queue state. Light/dark themes and mobile layout were
also verified.

## Local archive after pod decommissioning

The dashboard also accepts `--archive PATH` instead of `--connection PATH`.
It serves the saved snapshot and known log views without SSH, keeps all HTTP
mutation methods disabled, and still binds only to `127.0.0.1`. The page labels
this source “Local archive” and identifies GPU telemetry as the last saved sample.

The 2026-09-08 backup is documented in
`runs/pod-backup-rpii1ijfvmno97-20260908/README.md`. Its systemd drop-in
`~/.config/systemd/user/alz-experiment-dashboard.service.d/20-archive.conf` selects
the archive. Replace or remove that drop-in when reconnecting to a new pod.

## Self-play epoch metrics

The active self-play job's details show its current epoch, completed/total games,
percentage, a progress bar, elapsed game-generation time, evaluation throughput,
and the most recent trainer update. Progress comes from the trainer's 15-second
heartbeats and game-completion logs; the dashboard polls every five seconds.
The panel changes to training or checkpoint saving after game generation ends.
Completed snapshot timestamps, epoch boundaries, and process-restoration markers
prevent counts from a previous epoch or attempt being reused as live progress.
Cancelled and inactive jobs have no live epoch panel.

Self-play jobs show interactive plots of average game length (individual moves /
plies) and first-player win rate. Hover, tap, or use arrow/Home/End keys to inspect
an epoch; the selected point has a horizontal guide. The exact-values table also
shows games, first-player wins, second-player wins and draws. The win-rate
denominator includes all games; draws are not wins.

`average_game_length` comes from each completed epoch's statistics. The legacy
`total_score` is **first-player wins minus second-player wins**, so it cannot alone
identify wins in the presence of draws. `examples/export_self_play_metrics.rs`
reads the newest epoch's games from its saved replay, checks their lengths and
signed outcome sum against the epoch statistics, and exports exact outcome counts.
It refuses epochs whose full set of new games is no longer retained. Missing or
unverified outcome counts are shown as unavailable, never inferred from score.

`scripts/collect_self_play_metrics.py` caches these small derived JSON files in
each run's `epoch-metrics/` directory, with the source statistics SHA-256. On the
pod it runs in the `alz-selfplay:epoch-metrics` tmux window and checks for new
completed epochs every 30 seconds. It reads checkpoints without changing them or
restarting training. `self_play_dashboard.py` verifies the cached source hash and
outcome totals before exposing them to the existing read-only dashboard API.

## Sample games for a selected epoch

Each self-play job has a sample-game viewer with an epoch selector and a game
selector (the board-mask run now exports eight samples per completed epoch). Previous/Next,
First/Last, direct position entry and keyboard arrows/Home/End inspect the saved
pre-move positions. The selection persists across dashboard refreshes. A link
opens the full PNG frame sheet. Red/blue stones represent the first/second
player; green intensity shows the stored search-policy target. The terminal board
is not present in these saved strips.

`self_play_samples.py` lists sample metadata only for completed epochs and serves
only known run IDs and numbered sample PNGs. The new `game_image` SSH RPC and
`GET /api/experiments/game-image?phase_id=…&epoch=…&sample=…` endpoint are read-only.
PNG signature/layout, path containment, numeric identifiers and a 4 MB size limit
are checked; arbitrary paths and symlink samples are rejected. The local server
caches up to eight images, and the browser loads only the selected sample. The
training executable and checkpoint contents are unchanged.

`examples/export_self_play_samples.rs` preserves existing sample IDs, adds distinct
randomly selected games from the epoch's retained trajectories, and creates
`.tiles.png` siblings with at most sixteen columns. Even a full 361-position game
fits within 6,230 × 8,960 pixels; the old horizontal strips could exceed 140,000
pixels in width. The viewer uses the sheet's column count to show each board.
The original strips remain available on disk and determine the true frame count.
`scripts/collect_self_play_samples.py` processes the latest epoch first, backfills
earlier epochs, then watches for newly completed epochs every 30 seconds. It pins
the source stats hash in each `.samples.json` completion marker and verifies that
the recent replay games match the epoch's game lengths and outcomes. It runs in
the `samples` window of the pod's `alz-boardmask-selfplay` tmux session.

## Unified epoch charts and training statistics

One metric selector now switches a single chart between value/policy losses,
average game length, first-player win rate, self-play duration, training duration,
training sample count, and learning rate. The learning-rate chart uses the actual
`scheduled_learning_rate` saved for each completed epoch, including reconstructed
epochs; missing values remain missing. Hover/tap or keyboard inspection shows the
selected value, and the exact table retains its full recorded precision.
The adjacent **LR × steps** chart multiplies each recorded rate by that epoch's
`training.batches` (one optimizer update per batch, including the final partial
batch). The inspection readout shows both factors, and the exact table includes
the step count and product. Missing rates or step counts are omitted.
The exact epoch statistics table includes both durations
and sample counts for every completed self-play epoch. Hover, tap, and keyboard
inspection work on the new charts as on the existing game metrics.

Durations come directly from `self_play_seconds` and `training.duration_seconds`
in each saved epoch's stats. Training duration measures batch processing and
optimizer updates, excluding initial replay encoding, saving, and rendering.
`training.samples` counts the examples processed, including all eight board
symmetries per replay position; it is not the count of newly generated positions.
Missing durations remain unavailable rather than being estimated from elapsed job
time. Replay-only experiments retain the two loss choices.

## Live comparison statistics

While a comparison is running, the dashboard summarizes completed
`battle game complete` events from the current log attempt on its five-second
refresh. It shows checkpoint win/draw/loss totals, scores and selectable
confidence intervals, results by seat, and game-length distributions including
exact-length frequencies. Partial results are labeled: short games finish
first, so early proportions and intervals may be biased by unfinished games.
Completed reports remain authoritative and replace the live summary at the end.

The parser deduplicates game IDs, ignores partial or invalid log records, and
clears previous events on restart. Log modification time and size cache results.
Individual game wall times are not recorded; final match completion timing
continues to appear only when a complete report and timing log are available.

## Experiment identities

Checkpoint display names now come from [versioned experiment metadata](experiment-metadata.md),
including a backfill for historical comparisons. Adding a new comparison does not
require editing a frontend job-ID map. Keep the connection's `.metadata.json`,
the queue's `experiment-metadata.json` when present, and archive `.metadata.json`
sidecars with the experiment data. Local metadata supports existing remote readers
without a pod deployment.
