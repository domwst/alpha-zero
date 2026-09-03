# AlphaZero Playground

The playground is a native Rust inference server with a Preact browser client. The server owns one
Gomoku position and MCTS tree per WebSocket session while all sessions share the existing batched
network executor.

```text
Preact UI <-- JSON WebSocket --> session actor --> MCTS --> batched evaluator --> tch/device
```

The frontend displays the raw network policy, current root visit fraction, and
temperature-adjusted move probability independently. Hovering or focusing a legal board cell shows
all three values, its visit count, and mean action value. Search snapshots are retained in the
browser so the evolution chart and board can be scrubbed without stopping live search.

Move guidance can be shown always, only on the network's turn, or never. The setting applies to board
markers and tooltips, candidate ranking, search history, and last-move policy details. Select an empty
cell once to inspect it and activate the selected cell again to play it. No cell is selected initially,
search never changes selection, and a submitted move clears it immediately. This works on either
side's turn: on the network's turn the interface labels the current color and treats the move as a
manual override, while `Let network choose` samples from MCTS normally. The evolution chart shows
numeric axis ticks and exact vertical-slice values on hover; clicking the chart pins that snapshot. A
move keeps the color assigned when it first enters the leading set, even when its rank later changes.
Black always moves first. The segmented next-game control states the move order for both colors and
makes clear that changing it does not alter the game already in progress.

The interface follows the system color scheme by default and exposes persistent System, Light, and
Dark choices. The board, grid, stones, controls, and chart surfaces all use the selected theme. The
connection indicator reports the executor's actual device, such as `CPU ready` or `CUDA 0 ready`.

## Run

Build and start both parts with a snapshot or directory containing snapshots:

```bash
scripts/run_demo.sh \
  --checkpoint-dir /workspace/alpha-zero/runs/kata/checkpoints \
  --device cuda
```

The safe default is `127.0.0.1:8080`. From another machine, forward it over SSH:

```bash
ssh -L 8080:127.0.0.1:8080 -p 40031 root@213.192.2.124
```

Then open `http://127.0.0.1:8080`. Use `--listen 0.0.0.0:8080` only on a trusted network until
authentication is added.

For frontend development, run the Rust server on port 8080 and `npm --prefix web run dev`; Vite
proxies `/api` and WebSocket upgrades to the server.

## Resource limits

The defaults are deliberately conservative for an unauthenticated service:

- at most 10,000 requested simulations per position;
- at most four simultaneous browser sessions;
- 32 simulations per actor chunk;
- at most ten streamed snapshots per second;
- 16 KiB WebSocket messages from clients.

The first two limits are operator-controlled through `--max-search-simulations` and
`--max-sessions`. A position's budget counts new simulations; visits inherited by subtree reuse are
reported separately as carried work.

## Cancellation model

The session actor owns its tree for the full connection. While a search chunk is awaiting network
inference, the actor also polls the WebSocket. A new command drops that in-flight search future,
leaving completed tree statistics intact; the batched executor filters its abandoned inference
response. Search therefore stops promptly without moving the tree behind a mutex or discarding it.

Every position and search generation has a monotonically increasing identifier. Commands for an old
position are rejected and the browser ignores stale snapshots.

## Process shutdown

`SIGINT` (Ctrl-C) and `SIGTERM` stop accepting requests and broadcast cancellation to every active
WebSocket session before the inference executor is joined. This prevents Axum's graceful-shutdown
wait from being held open by an upgraded connection. A second signal exits immediately; a five-second
watchdog provides the same hard-stop guarantee if a native inference call does not return.

## Reconnect and restart recovery

Protocol version 2 initializes every WebSocket from a browser-supplied move sequence. On the first
connection that sequence is empty. On reconnect, the browser sends the last server-confirmed moves
and human color while continuing to display the existing board. The server validates and replays the
moves from an empty board before publishing the restored position.

Only game state is restored. The MCTS tree, root visits, search snapshots, and last-move
analysis are discarded, and the browser starts a fresh search using its current budget. This works
across a dropped connection or server process restart while the page remains open; reloading the page
still starts a new game because game state is not stored in browser storage.
