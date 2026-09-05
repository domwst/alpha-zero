import { useSignal } from '@preact/signals';
import type { JSX } from 'preact';
import { useEffect, useState } from 'preact/hooks';

import { seriesColor, seriesColorIndexes, trackedMoves } from './chartSeries';
import { formatEta, favorsText, percent, signed } from './format';
import { Board, type Overlay } from './Board';
import { InfoTip } from './InfoTip';
import { SearchChart } from './SearchChart';
import { ThemePicker } from './ThemePicker';
import {
  type PolicyVisibility,
  revealsMoveGuidance,
} from './policyVisibility';
import {
  type Cell,
  type ErrorMessage,
  type HelloMessage,
  type MoveStats,
  type PositionMessage,
  type SearchSnapshotMessage,
  type SearchStatusMessage,
  type ServerMessage,
  type StoneColor,
  BOARD_SIZE,
  PROTOCOL_VERSION,
  cellKey,
  moveName,
  restoreGameCommand,
  undoMoves,
  temperatureProbabilities,
  visitFraction,
} from './protocol';

type ConnectionState = 'connecting' | 'restoring' | 'connected' | 'disconnected';

interface MoveJudgment {
  name: string;
  rank: number | null;
  legalMoves: number;
  prior: number | null;
  visitShare: number | null;
  preMoveValue: number | null;
  expectedPositionId: number;
  valueDelta: number | null;
}

const MAX_SNAPSHOTS = 240;
const RETARGET_DELAY_MS = 250;

function websocketUrl(): string {
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${window.location.host}/api/ws`;
}

function titleCase(value: string): string {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function outcomeText(position: PositionMessage | null): string | null {
  if (!position?.outcome) return null;
  if (position.outcome === 'draw') return 'Game over · draw';
  const winner = position.outcome === 'black_won' ? 'Black' : 'White';
  return `Game over · ${winner} wins`;
}

function downsampleSnapshots(
  snapshots: SearchSnapshotMessage[],
  keepSimulations: number | null,
): SearchSnapshotMessage[] {
  if (snapshots.length <= MAX_SNAPSHOTS) return snapshots;
  return snapshots.filter(
    (snapshot, index) =>
      index === 0
      || index === snapshots.length - 1
      || index % 2 === 0
      || snapshot.searched_simulations === keepSimulations,
  );
}

function inspectedSnapshotIndex(
  snapshots: SearchSnapshotMessage[],
  simulations: number | null,
): number | null {
  if (simulations === null || snapshots.length === 0) return null;
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  snapshots.forEach((snapshot, index) => {
    const distance = Math.abs(snapshot.searched_simulations - simulations);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  });
  return best;
}

export function App(): JSX.Element {
  const connection = useSignal<ConnectionState>('connecting');
  const hello = useSignal<HelloMessage | null>(null);
  const position = useSignal<PositionMessage | null>(null);
  const status = useSignal<SearchStatusMessage | null>(null);
  const snapshots = useSignal<SearchSnapshotMessage[]>([]);
  const inspectedSimulations = useSignal<number | null>(null);
  const selected = useSignal<Cell | null>(null);
  const overlay = useSignal<Overlay>('visits');
  const policyVisibility = useSignal<PolicyVisibility>('network_turn');
  const temperature = useSignal(0.5);
  const budget = useSignal(2_000);
  const requestedBudget = useSignal(0);
  const newHumanColor = useSignal<StoneColor>('black');
  const playInFlight = useSignal<number | null>(null);
  const error = useSignal<ErrorMessage | null>(null);
  const lastJudgment = useSignal<MoveJudgment | null>(null);
  const [connectionGeneration, setConnectionGeneration] = useState(0);
  const socket = useSignal<WebSocket | null>(null);

  useEffect(() => {
    const resumePosition = position.value;
    const restoreMessage = restoreGameCommand(resumePosition, newHumanColor.value);
    let handshakeAccepted = false;
    let awaitingRestoredPosition = true;
    const ws = new WebSocket(websocketUrl());
    socket.value = ws;
    connection.value = 'connecting';
    error.value = null;

    const sendOnThisSocket = (message: object) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
    };

    ws.addEventListener('message', (event) => {
      if (socket.value !== ws) return;
      let message: ServerMessage;
      try {
        message = JSON.parse(String(event.data)) as ServerMessage;
      } catch {
        error.value = {
          type: 'error',
          code: 'invalid_server_message',
          message: 'The server returned an unreadable message.',
          recoverable: false,
        };
        return;
      }

      if (message.type === 'hello') {
        if (
          message.protocol_version !== PROTOCOL_VERSION
          || message.board_size !== BOARD_SIZE
        ) {
          error.value = {
            type: 'error',
            code: 'incompatible_server',
            message: `Incompatible server protocol ${message.protocol_version} for a ${message.board_size}×${message.board_size} board.`,
            recoverable: false,
          };
          socket.value = null;
          connection.value = 'disconnected';
          ws.close();
          return;
        }
        handshakeAccepted = true;
        hello.value = message;
        if (!resumePosition) budget.value = message.default_search_simulations;
        if (resumePosition) connection.value = 'restoring';
        sendOnThisSocket(restoreMessage);
        return;
      }

      if (!handshakeAccepted) return;

      if (message.type === 'position') {
        if (awaitingRestoredPosition) {
          awaitingRestoredPosition = false;
          if (resumePosition) lastJudgment.value = null;
        }
        connection.value = 'connected';
        position.value = message;
        status.value = null;
        snapshots.value = [];
        inspectedSimulations.value = null;
        selected.value = null;
        playInFlight.value = null;
        requestedBudget.value = 0;
        error.value = null;

        if (message.outcome === null) {
          window.setTimeout(() => {
            if (position.value?.position_id !== message.position_id) return;
            const target = budget.value;
            requestedBudget.value = target;
            sendOnThisSocket({
              type: 'start_search',
              position_id: message.position_id,
              simulations: target,
            });
          }, 0);
        }
        return;
      }

      if (message.type === 'search_status') {
        if (message.position_id === position.value?.position_id) status.value = message;
        return;
      }

      if (message.type === 'search_snapshot') {
        if (message.position_id !== position.value?.position_id) return;
        if (status.value && message.analysis_id < status.value.analysis_id) return;
        status.value = {
          type: 'search_status',
          position_id: message.position_id,
          analysis_id: message.analysis_id,
          searched_simulations: message.searched_simulations,
          target_simulations: message.target_simulations,
          running: !message.complete,
        };
        const previous = snapshots.value;
        const last = previous[previous.length - 1];
        const next =
          last?.searched_simulations === message.searched_simulations
            ? [...previous.slice(0, -1), message]
            : [...previous, message];
        snapshots.value = downsampleSnapshots(next, inspectedSimulations.value);

        const judgment = lastJudgment.value;
        if (
          judgment &&
          judgment.valueDelta === null &&
          judgment.preMoveValue !== null &&
          judgment.expectedPositionId === message.position_id
        ) {
          lastJudgment.value = {
            ...judgment,
            valueDelta: -message.network_value - judgment.preMoveValue,
          };
        }
        return;
      }

      error.value = message;
      playInFlight.value = null;
      if (awaitingRestoredPosition) ws.close();
      if (!message.recoverable) requestedBudget.value = 0;
    });
    ws.addEventListener('close', () => {
      if (socket.value === ws) {
        socket.value = null;
        connection.value = 'disconnected';
        if (!error.value) {
          error.value = {
            type: 'error',
            code: 'connection_closed',
            message: 'The analysis server disconnected.',
            recoverable: true,
          };
        }
      }
    });
    ws.addEventListener('error', () => {
      if (socket.value !== ws) return;
      error.value = {
        type: 'error',
        code: 'connection_failed',
        message: 'Could not connect to the analysis server.',
        recoverable: true,
      };
      playInFlight.value = null;
    });

    return () => {
      ws.close();
      if (socket.value === ws) socket.value = null;
    };
  }, [connectionGeneration]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      if (selected.value) selected.value = null;
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  useEffect(() => {
    const simulations = budget.value;
    const timer = window.setTimeout(() => {
      const pos = position.value;
      const live = status.value;
      if (!pos || pos.outcome !== null || live?.position_id !== pos.position_id) return;
      if (!live.running || live.target_simulations === simulations) return;
      requestedBudget.value = simulations;
      send({ type: 'start_search', position_id: pos.position_id, simulations });
    }, RETARGET_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [budget.value]);

  const send = (message: object): boolean => {
    const ws = socket.value;
    if (connection.value !== 'connected' || !ws || ws.readyState !== WebSocket.OPEN) {
      error.value = {
        type: 'error',
        code: 'not_connected',
        message: connection.value === 'restoring'
          ? 'The game is still being restored.'
          : 'The analysis server is not connected.',
        recoverable: true,
      };
      return false;
    }
    ws.send(JSON.stringify(message));
    return true;
  };

  const allSnapshots = snapshots.value;
  const liveSnapshot = allSnapshots[allSnapshots.length - 1] ?? null;
  const inspectedIndex = inspectedSnapshotIndex(allSnapshots, inspectedSimulations.value);
  const displaySnapshot =
    inspectedIndex === null
      ? liveSnapshot
      : (allSnapshots[inspectedIndex] ?? liveSnapshot);
  const currentPosition = position.value;
  const colorIndexes = currentPosition
    ? seriesColorIndexes(
        currentPosition.position_id,
        trackedMoves(allSnapshots.map((snapshot) => snapshot.moves)),
      )
    : new Map<string, number>();
  const dotColorFor = (key: string): string | undefined => {
    const index = colorIndexes.get(key);
    return index === undefined ? undefined : seriesColor(index);
  };
  const currentStatus = status.value;
  const isHumanTurn =
    currentPosition?.outcome === null &&
    currentPosition.to_move === currentPosition.human_color;
  const isNetworkTurn = Boolean(
    currentPosition
    && currentPosition.outcome === null
    && currentPosition.to_move !== currentPosition.human_color,
  );
  const showMoveGuidance = revealsMoveGuidance(
    policyVisibility.value,
    isNetworkTurn,
  );
  const running = currentStatus?.running ?? false;
  const moveProbabilities = temperatureProbabilities(
    displaySnapshot?.moves ?? [],
    temperature.value,
  );
  const topMoves = [...(displaySnapshot?.moves ?? [])]
    .sort((left, right) => right.visits - left.visits)
    .slice(0, 5);
  const selectedMove = selected.value
    ? displaySnapshot?.moves.find((move) => cellKey(move) === cellKey(selected.value!)) ?? null
    : null;
  const progress = currentStatus?.searched_simulations ?? liveSnapshot?.searched_simulations ?? 0;
  const target = currentStatus?.target_simulations ?? requestedBudget.value;
  const remaining = Math.max(0, target - progress);
  const simsPerSecond = liveSnapshot?.simulations_per_second ?? 0;
  const etaSeconds =
    running && simsPerSecond > 0 && remaining > 0 ? remaining / simsPerSecond : null;
  const searchUnavailableReason = !currentPosition
    ? 'Waiting for the server — search unlocks once a position arrives.'
    : currentPosition.outcome !== null
      ? 'The game is over. Start a new game to keep exploring.'
      : !running && progress >= budget.value
        ? 'The search budget is exhausted — increase it to resume the search from the current tree.'
        : null;
  const maximumBudget = hello.value?.max_search_simulations ?? 10_000;
  const budgetStep = maximumBudget < 100 ? 1 : 100;
  const turnColor = titleCase(currentPosition?.to_move ?? 'black');
  const opponentColor = currentPosition?.to_move === 'white' ? 'Black' : 'White';
  const turnLabel = currentPosition
    ? outcomeText(currentPosition) ?? `${turnColor} to move`
    : 'Preparing board';
  const turnOwner = !currentPosition
    ? 'Waiting for server'
    : currentPosition.outcome
      ? 'Game complete'
      : isHumanTurn
        ? 'Your turn'
        : 'Network turn';
  const canPlaySelected = Boolean(
    currentPosition
    && currentPosition.outcome === null
    && selected.value,
  );
  const undoTarget = undoMoves(currentPosition);
  const documentTitle = currentPosition
    ? `${turnLabel} — AlphaZero Playground`
    : 'AlphaZero Playground';
  useEffect(() => {
    document.title = documentTitle;
  }, [documentTitle]);

  const startOrStopSearch = () => {
    if (!currentPosition || currentPosition.outcome !== null) return;
    if (running) {
      if (
        send({
          type: 'stop_search',
          position_id: currentPosition.position_id,
        })
      ) requestedBudget.value = 0;
      return;
    }
    const targetBudget = budget.value;
    if (
      send({
        type: 'start_search',
        position_id: currentPosition.position_id,
        simulations: targetBudget,
      })
    ) requestedBudget.value = targetBudget;
  };

  const playCell = (cell: Cell) => {
    if (!currentPosition || currentPosition.outcome !== null) return;
    if (playInFlight.value === currentPosition.position_id) return;
    const judgingHumanMove = isHumanTurn;
    const latestMove = liveSnapshot?.moves.find(
      (move) => cellKey(move) === cellKey(cell),
    );
    const ranked = [...(liveSnapshot?.moves ?? [])].sort(
      (left, right) => right.visits - left.visits,
    );
    const rank = latestMove
      ? ranked.findIndex((move) => cellKey(move) === cellKey(latestMove)) + 1
      : null;
    const sent = send({
      type: 'play',
      position_id: currentPosition.position_id,
      row: cell.row,
      column: cell.column,
    });
    if (!sent) return;

    playInFlight.value = currentPosition.position_id;
    if (judgingHumanMove) {
      lastJudgment.value = {
        name: moveName(cell),
        rank,
        legalMoves: ranked.length,
        prior: latestMove?.prior ?? null,
        visitShare:
          latestMove && liveSnapshot ? visitFraction(latestMove, liveSnapshot) : null,
        preMoveValue: liveSnapshot?.network_value ?? null,
        expectedPositionId: currentPosition.position_id + 1,
        valueDelta: null,
      };
    }
  };

  const selectOrPlayCell = (cell: Cell) => {
    if (selected.value && cellKey(selected.value) === cellKey(cell)) {
      playCell(cell);
    } else {
      selected.value = cell;
    }
  };

  const playSelected = () => {
    if (selected.value) playCell(selected.value);
  };

  const letNetworkChoose = () => {
    if (!currentPosition || currentPosition.outcome !== null || isHumanTurn || !liveSnapshot) return;
    if (running) {
      send({ type: 'stop_search', position_id: currentPosition.position_id });
    }
    send({
      type: 'choose_network_move',
      position_id: currentPosition.position_id,
      temperature: temperature.value,
    });
  };

  const stepInspection = (delta: number) => {
    const count = allSnapshots.length;
    if (count === 0) return;
    const current = inspectedIndex ?? count - 1;
    const next = Math.min(count - 1, Math.max(0, current + delta));
    inspectedSimulations.value =
      next === count - 1
        ? null
        : (allSnapshots[next]?.searched_simulations ?? null);
  };

  const undoLastMove = () => {
    if (!currentPosition || !undoTarget) return;
    lastJudgment.value = null;
    send({
      type: 'restore_game',
      human_color: currentPosition.human_color,
      moves: undoTarget,
    });
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div>
            <h1>AlphaZero Playground</h1>
            <p>Play Gomoku and inspect the MCTS search as it unfolds</p>
          </div>
        </div>
        <div className="topbar-actions">
          <span className={`connection connection-${connection.value}`}>
            <i aria-hidden="true" />
            {connection.value === 'connected'
              ? hello.value?.compute_device
                ? `${hello.value.compute_device} ready`
                : 'Connected'
              : titleCase(connection.value)}
          </span>
          {hello.value && (
            <span className="checkpoint">
              {hello.value.checkpoint.architecture} · epoch {hello.value.checkpoint.epoch}
              <small>{hello.value.checkpoint.model_digest}</small>
            </span>
          )}
          <ThemePicker />
          <fieldset className="player-picker">
            <legend className="visually-hidden">
              Your color for the next game; Black moves first
            </legend>
            <span aria-hidden="true" className="player-picker-caption">Next game</span>
            <label title="Play Black and move first">
              <input
                checked={newHumanColor.value === 'black'}
                name="next-human-color"
                onChange={() => { newHumanColor.value = 'black'; }}
                type="radio"
                value="black"
              />
              <span>Black</span>
              <small>First</small>
            </label>
            <label title="Play White and move second">
              <input
                checked={newHumanColor.value === 'white'}
                name="next-human-color"
                onChange={() => { newHumanColor.value = 'white'; }}
                type="radio"
                value="white"
              />
              <span>White</span>
              <small>Second</small>
            </label>
          </fieldset>
          <button
            className="button"
            disabled={connection.value !== 'connected'}
            onClick={() => {
              if (send({ type: 'new_game', human_color: newHumanColor.value })) {
                selected.value = null;
                lastJudgment.value = null;
              }
            }}
            type="button"
          >
            New game
          </button>
        </div>
      </header>

      {error.value && (
        <div className={`notice${error.value.recoverable ? '' : ' notice-danger'}`} role="alert">
          <span>{error.value.message}</span>
          {connection.value === 'disconnected' && (
            <button className="text-button" onClick={() => setConnectionGeneration((v) => v + 1)} type="button">
              Reconnect
            </button>
          )}
          <button className="notice-close" aria-label="Dismiss error" onClick={() => { error.value = null; }} type="button">×</button>
        </div>
      )}

      <main className="workspace">
        <div className="board-column">
          <section className="panel board-panel" aria-labelledby="board-title">
          <div className="panel-heading">
            <div>
              <div className="turn-heading">
                <span className={`status ${isHumanTurn ? 'status--info' : 'status--neutral'}`}>
                  {turnOwner}
                </span>
                <h2 id="board-title">{turnLabel}</h2>
              </div>
              <p>Move {(currentPosition?.ply ?? 0) + 1}{selected.value ? ` · ${moveName(selected.value)} selected` : ''}</p>
            </div>
            <div className="search-readout">
              <span className={running ? 'pulse-dot' : 'status-dot'} aria-hidden="true" />
              <strong>{progress.toLocaleString()}</strong>
              <span>/ {target.toLocaleString()} simulations</span>
              {liveSnapshot && <small>{Math.round(liveSnapshot.simulations_per_second).toLocaleString()} sims/s</small>}
              {etaSeconds !== null && <small>~{formatEta(etaSeconds)} left</small>}
            </div>
          </div>
          {currentPosition && currentPosition.outcome === null && target > 0 && (
            <div
              aria-label="Search progress"
              aria-valuemax={target}
              aria-valuemin={0}
              aria-valuenow={Math.min(progress, target)}
              className={`search-progress${!running && progress >= target ? ' is-complete' : ''}`}
              role="progressbar"
            >
              <span style={{ inlineSize: `${Math.min(100, (progress / target) * 100)}%` }} />
            </div>
          )}
          {currentPosition && currentPosition.outcome === null && currentPosition.ply === 0 && (
            <div className={`turn-guidance ${isHumanTurn ? 'turn-guidance-human' : 'turn-guidance-network'}`}>
              <strong>{isHumanTurn ? `You are ${turnColor}.` : `The network is ${turnColor}.`}</strong>
              <span>
                {isHumanTurn
                  ? ' Select an empty cell, then click the selected cell again to play.'
                  : ` Let it choose after search, or select and click again to make a manual ${turnColor} move.`}
              </span>
            </div>
          )}
          <div className="board-frame">
            {currentPosition ? (
              <Board
                onNavigate={(cell) => {
                  selected.value = cell;
                }}
                onSelect={selectOrPlayCell}
                overlay={overlay.value}
                position={currentPosition}
                selected={selected.value}
                showPolicy={showMoveGuidance}
                snapshot={displaySnapshot}
                temperature={temperature.value}
              />
            ) : (
              <div aria-hidden="true" className="board-skeleton" />
            )}
          </div>
          {currentPosition && (
            <div className="board-legend">
              <span><i className="legend-stone legend-black" />Black</span>
              <span><i className="legend-stone legend-white" />White</span>
              {showMoveGuidance ? (
                <>
                  <span><i className="legend-policy" />{
                    overlay.value === 'prior'
                      ? 'Network policy'
                      : overlay.value === 'visits'
                        ? 'Visit fraction'
                        : `Move probability at T=${temperature.value.toFixed(2)}`
                  }</span>
                  <span className="legend-note">larger square = more likely · 1 = top move</span>
                </>
              ) : (
                <span className="legend-note">Move guidance is hidden by your settings.</span>
              )}
            </div>
          )}
          </section>

          {showMoveGuidance && (
            <section className="panel evolution" aria-labelledby="evolution-title">
              <div className="section-heading evolution-heading">
                <div>
                  <h2 id="evolution-title">How search changes its mind</h2>
                  <p>Root visit fraction as simulations accumulate. Hover for exact values; click to pin a snapshot.</p>
                </div>
              </div>
              <SearchChart
                onSelectIndex={(index) => {
                  inspectedSimulations.value =
                    index === null ? null : (allSnapshots[index]?.searched_simulations ?? null);
                }}
                selectedIndex={inspectedIndex}
                snapshots={allSnapshots}
              />
            </section>
          )}
        </div>

        <aside className="panel inspector" aria-label="Position analysis">
          <section className="inspector-section">
            <div className="section-heading">
              <h2>Search controls</h2>
              <div className="search-actions">
                <button
                  className={`button${running ? '' : ' button-primary'}`}
                  disabled={
                    !currentPosition
                    || currentPosition.outcome !== null
                    || (!running && progress >= budget.value)
                  }
                  onClick={startOrStopSearch}
                  type="button"
                >
                  {running ? 'Stop search' : progress > 0 ? 'Continue search' : 'Run search'}
                </button>
                {searchUnavailableReason && (
                  <InfoTip
                    id="tip-search-button"
                    title="Search unavailable"
                    triggerLabel="Why is search unavailable?"
                    variant="trigger"
                  >
                    {searchUnavailableReason}
                  </InfoTip>
                )}
              </div>
            </div>
            <div className="control-grid">
              <label className="range-control">
                <span><b>Search budget</b><output>{budget.value.toLocaleString()}</output></span>
                <input
                  max={maximumBudget}
                  min={Math.min(budgetStep, maximumBudget)}
                  onInput={(event) => { budget.value = Number(event.currentTarget.value); }}
                  step={budgetStep}
                  type="range"
                  value={budget.value}
                />
                <small>Live while a search runs · hard limit {maximumBudget.toLocaleString()}</small>
              </label>
              <label className="range-control">
                <span><b>Temperature</b><output>{temperature.value.toFixed(2)}</output></span>
                <input
                  max={2}
                  min={0}
                  onInput={(event) => { temperature.value = Number(event.currentTarget.value); }}
                  step={0.05}
                  type="range"
                  value={temperature.value}
                />
                <small>Changes move sampling, not search</small>
              </label>
            </div>
            <div className="overlay-controls">
              <div className="overlay-control-group">
                <div className="control-copy">
                  <b>Show probabilities and ranking</b>
                  <small>Applies to board hints, candidate table, and search history</small>
                </div>
                <fieldset className="segmented">
                  <legend className="visually-hidden">Show probabilities and ranking</legend>
                  {([
                    ['always', 'Always'],
                    ['network_turn', 'Network turn only'],
                    ['never', 'Never'],
                  ] as const).map(([value, label]) => (
                    <label key={value}>
                      <input
                        checked={policyVisibility.value === value}
                        name="policy-visibility"
                        onChange={() => { policyVisibility.value = value; }}
                        type="radio"
                        value={value}
                      />
                      <span>{label}</span>
                    </label>
                  ))}
                </fieldset>
              </div>
              <div className="overlay-control-group">
                <div className="control-copy">
                  <b>Board overlay metric</b>
                  <small>What the square size represents</small>
                </div>
                <fieldset className="segmented" disabled={!showMoveGuidance}>
                  <legend className="visually-hidden">Board policy overlay</legend>
                  {([
                    ['prior', 'Network prior'],
                    ['visits', 'Search visits'],
                    ['move', 'Move probability'],
                  ] as const).map(([value, label]) => (
                    <label key={value}>
                      <input
                        checked={overlay.value === value}
                        name="board-overlay"
                        onChange={() => { overlay.value = value; }}
                        type="radio"
                        value={value}
                      />
                      <span>{label}</span>
                    </label>
                  ))}
                </fieldset>
              </div>
            </div>
          </section>

          <section className="inspector-section">
            <div className="section-heading">
              <h2>Position</h2>
            </div>
            <div className="value-grid">
              <div>
                <span>
                  Network value V
                  <InfoTip
                    id="tip-network-value"
                    title="Network value V"
                    triggerLabel="What network value V means"
                  >
                    The value head’s raw read of the position on the board, before any search.
                    It runs from −1 to +1 and is always measured from the perspective of the side
                    to move: positive favors the current player, negative favors the opponent —
                    so which color that is flips with every move.
                    <span className="info-tip-live">
                      {favorsText(displaySnapshot?.network_value ?? null, turnColor, opponentColor)}
                    </span>
                  </InfoTip>
                </span>
                <strong>{signed(displaySnapshot?.network_value ?? null)}</strong>
              </div>
              <div>
                <span>
                  Search value Q
                  <InfoTip
                    id="tip-search-value"
                    title="Search value Q"
                    triggerLabel="What search value Q means"
                  >
                    The search’s estimate of the same position: a visit-weighted average over
                    the root moves explored so far, using the same convention as V — positive
                    favors the side to move. It sharpens toward the best line as visits
                    accumulate.
                    <span className="info-tip-live">
                      {favorsText(displaySnapshot?.search_value ?? null, turnColor, opponentColor)}
                    </span>
                  </InfoTip>
                </span>
                <strong>{signed(displaySnapshot?.search_value ?? null)}</strong>
              </div>
            </div>
            <p className="visits-note">
              {displaySnapshot
                ? `${displaySnapshot.total_visits.toLocaleString()} tree visits`
                : 'No search yet'}
              {displaySnapshot && displaySnapshot.carried_visits > 0 && (
                ` · ${displaySnapshot.carried_visits.toLocaleString()} carried from previous search`
              )}
            </p>
            {showMoveGuidance ? (
              <>
                <div className="table-wrap">
                  <table aria-label="Top candidate moves">
                    <thead><tr><th>Move</th><th>Prior</th><th>Visits</th><th>P(move)</th></tr></thead>
                    <tbody>
                      {topMoves.map((move) => {
                        const key = cellKey(move);
                        const color = dotColorFor(key);
                        return (
                          <tr
                            className={selected.value && key === cellKey(selected.value) ? 'selected-row' : ''}
                            key={key}
                          >
                            <td>
                              <button
                                aria-pressed={selected.value != null && key === cellKey(selected.value)}
                                className="table-move-button"
                                onClick={() => { selected.value = { row: move.row, column: move.column }; }}
                                type="button"
                              >
                                <i
                                  className="series-dot"
                                  style={color ? { background: color } : undefined}
                                />{moveName(move)}
                              </button>
                            </td>
                            <td>{percent(move.prior)}</td>
                            <td>{displaySnapshot ? percent(visitFraction(move, displaySnapshot)) : '—'}</td>
                            <td>{percent(moveProbabilities.get(key) ?? null)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {selectedMove && displaySnapshot && (
                  <div className="selected-summary">
                    <span><b>{moveName(selectedMove)}</b> has {selectedMove.visits.toLocaleString()} visits</span>
                    <span>Q {signed(selectedMove.mean_value, 3)}</span>
                  </div>
                )}
              </>
            ) : (
              <div className="move-guidance-hidden">
                <b>Move guidance hidden</b>
              </div>
            )}
            <div className="action-row">
              <button
                className={`button${isHumanTurn ? ' button-primary' : ''}`}
                disabled={!canPlaySelected}
                onClick={playSelected}
                type="button"
              >
                {selected.value
                  ? isHumanTurn
                    ? `Play ${moveName(selected.value)}`
                    : `Play ${moveName(selected.value)} manually as ${turnColor}`
                  : 'Select a move'}
              </button>
              <button
                className={`button${!isHumanTurn ? ' button-primary' : ''}`}
                disabled={Boolean(isHumanTurn) || !liveSnapshot || liveSnapshot.target_simulations === 0}
                onClick={letNetworkChoose}
                type="button"
              >
                Let network choose
              </button>
              <button
                className="button"
                disabled={!undoTarget}
                onClick={undoLastMove}
                type="button"
              >
                Undo move
              </button>
            </div>
          </section>

          <section className="inspector-section judgment-section">
            <h2>How the network judged your last move</h2>
            {!showMoveGuidance ? null : lastJudgment.value ? (
              <div className="judgment-card">
                <div>
                  <strong>{lastJudgment.value.name}</strong>
                  <span>
                    {lastJudgment.value.rank === 1
                      ? '— its first choice'
                      : lastJudgment.value.rank !== null && lastJudgment.value.rank <= 5
                        ? '— one of its leading choices'
                        : '— outside its leading candidates'}
                  </span>
                </div>
                <dl>
                  <dt>Policy rank before move</dt><dd>{lastJudgment.value.rank ? `${lastJudgment.value.rank} / ${lastJudgment.value.legalMoves}` : '—'}</dd>
                  <dt>Raw network policy</dt><dd>{percent(lastJudgment.value.prior)}</dd>
                  <dt>Search visit fraction</dt><dd>{percent(lastJudgment.value.visitShare)}</dd>
                  <dt>Network value change</dt><dd>{signed(lastJudgment.value.valueDelta)}</dd>
                </dl>
              </div>
            ) : (
              <p className="empty-copy">Your move’s policy rank and value impact will appear here.</p>
            )}
          </section>
        </aside>
      </main>

      {showMoveGuidance && allSnapshots.length > 0 && (
        <div className="inspect-bar" role="group" aria-label="Search snapshot inspection">
          <span className="live-label">
            <i aria-hidden="true" className="status-dot" />
            {inspectedSimulations.value === null ? 'Live' : 'Inspecting history'}
          </span>
          <button
            aria-label="Previous snapshot"
            className="step-button"
            disabled={(inspectedIndex ?? allSnapshots.length - 1) <= 0}
            onClick={() => stepInspection(-1)}
            type="button"
          >
            ‹
          </button>
          <input
            aria-label="Snapshot position in search history"
            max={allSnapshots.length - 1}
            min={0}
            onInput={(event) => {
              const index = Number(event.currentTarget.value);
              inspectedSimulations.value =
                index === allSnapshots.length - 1
                  ? null
                  : (allSnapshots[index]?.searched_simulations ?? null);
            }}
            step={1}
            type="range"
            value={inspectedIndex ?? allSnapshots.length - 1}
          />
          <button
            aria-label="Next snapshot"
            className="step-button"
            disabled={inspectedIndex === null}
            onClick={() => stepInspection(1)}
            type="button"
          >
            ›
          </button>
          <output>{(displaySnapshot?.searched_simulations ?? 0).toLocaleString()} sims</output>
          <button
            className="text-button"
            disabled={inspectedIndex === null}
            onClick={() => { inspectedSimulations.value = null; }}
            type="button"
          >
            Live
          </button>
        </div>
      )}
    </div>
  );
}
