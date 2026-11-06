import { useSignal } from '@preact/signals';
import type { JSX } from 'preact';
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';

import { seriesColor, seriesColorIndexes, trackedMoves } from './chartSeries';
import { formatEta, favorsText, percent, signed } from './format';
import { Board, type Overlay } from './Board';
import { InfoTip } from './InfoTip';
import { SearchChart } from './SearchChart';
import { ThemePicker } from './ThemePicker';
import {
  type Disclosure,
  type DisclosureTopic,
  DISCLOSURE_TOPICS,
  parseDisclosure,
  reveals,
} from './disclosure';
import { ConcealedNote, ConcealedVeil, DisclosureSwitches, EyeIcon } from './DisclosureControls';
import { usePopover } from './usePopover';
import {
  type Cell,
  type ErrorMessage,
  type GameLink,
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
  gameFragment,
  moveName,
  parseGameFragment,
  restoreGameCommand,
  undoMoves,
  temperatureProbabilities,
  visitFraction,
} from './protocol';

type ConnectionState =
  | 'connecting'
  | 'restoring'
  | 'connected'
  | 'reconnecting'
  | 'disconnected';

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
const AUTO_REPLY_KEY = 'alz-playground-auto-reply';
const DISCLOSURE_KEY = 'alz-playground-analysis';
const OVERLAY_KEY = 'alz-playground-overlay';

const OVERLAY_OPTIONS: readonly { value: Overlay; label: string; title: string; description: string }[] = [
  {
    value: 'prior',
    label: 'Prior',
    title: 'Network prior',
    description: "The network's raw policy for each move, before any search.",
  },
  {
    value: 'visits',
    label: 'Visits',
    title: 'Visit fraction',
    description: 'Share of search visits each move received: where the search focused.',
  },
  {
    value: 'move',
    label: 'Probability',
    title: 'Move probability',
    description: 'Chance the network plays each move when sampling at the current temperature',
  },
];

/** Off unless the player turned it on, so each search can be inspected first. */
function storedAutoReply(): boolean {
  try {
    return localStorage.getItem(AUTO_REPLY_KEY) === 'on';
  } catch {
    return false;
  }
}

function storedDisclosure(): Disclosure {
  try {
    return parseDisclosure(localStorage.getItem(DISCLOSURE_KEY));
  } catch {
    return parseDisclosure(null);
  }
}

function storedOverlay(): Overlay {
  try {
    const stored = localStorage.getItem(OVERLAY_KEY);
    return OVERLAY_OPTIONS.find(({ value }) => value === stored)?.value ?? 'visits';
  } catch {
    return 'visits';
  }
}

/** Most-visited first; priors break ties, and rank everything in instant play. */
function byStrength(left: MoveStats, right: MoveStats): number {
  return right.visits - left.visits || right.prior - left.prior;
}

/** Keeps the address bar on the current game, so it can be copied or reloaded. */
function writeGameFragment(game: PositionMessage | null) {
  if (!game) return;
  const fragment = gameFragment(game.stones, game.human_color);
  const hash = fragment ? `#${fragment}` : '';
  if (window.location.hash === hash) return;
  history.replaceState(history.state, '', `${window.location.pathname}${window.location.search}${hash}`);
}

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
  const focus = useSignal<Cell | null>(null);
  const overlay = useSignal(storedOverlay());
  const disclosure = useSignal(storedDisclosure());
  // Groups revealed for one position only; a new position hides them again.
  const peek = useSignal<{ positionId: number; topics: DisclosureTopic[] } | null>(null);
  const temperature = useSignal(0.5);
  const budget = useSignal(2_000);
  const requestedBudget = useSignal(0);
  // Position whose search or evaluation has been requested; an instant reply
  // must not overtake that request, or the server would reject it as stale.
  const searchRequested = useSignal<number | null>(null);
  const newHumanColor = useSignal<StoneColor>('black');
  const autoReply = useSignal(storedAutoReply());
  // Position for which an automatic reply has been requested; the board stays
  // playable so a manual override is still possible while the server answers.
  const autoReplied = useSignal<number | null>(null);
  // Position whose search the user stopped by hand: the snapshot that follows
  // a stop is also flagged complete, but it must not trigger a reply.
  const autoReplyStopped = useSignal<number | null>(null);
  const newGamePopover = usePopover();
  const playInFlight = useSignal<number | null>(null);
  const error = useSignal<ErrorMessage | null>(null);
  const lastJudgment = useSignal<MoveJudgment | null>(null);
  // A game opened from the address bar waits here until the server restores it.
  const openedLink = useMemo(() => parseGameFragment(window.location.hash), []);
  const pendingLink = useRef<GameLink | null>(openedLink?.ok ? openedLink.link : null);
  const restoringLink = useRef(false);
  const linkNotice = useSignal<string | null>(
    openedLink && !openedLink.ok
      ? `This link isn't a playable game: ${openedLink.reason}. Started a new game instead.`
      : null,
  );
  const linkCopied = useSignal(false);
  const [connectionGeneration, setConnectionGeneration] = useState(0);
  const socket = useSignal<WebSocket | null>(null);
  const reconnectAttempt = useSignal(0);
  const connectedOnce = useSignal(false);
  const retryTimer = useRef<number | null>(null);

  // A search reply needs a positive target that was reached: the snapshot that
  // arrives with a new position is complete too, but holds only carried visits.
  // An instant reply needs only the evaluation, once this position asked for it.
  const replyReady = (snapshot: SearchSnapshotMessage, instant: boolean) =>
    snapshot.complete
    && (instant
      ? searchRequested.value === snapshot.position_id
      : snapshot.target_simulations > snapshot.carried_visits);

  const tryReconnectNow = () => {
    if (retryTimer.current != null) {
      window.clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }
    reconnectAttempt.value = 0;
    setConnectionGeneration((v) => v + 1);
  };

  useEffect(() => {
    const resumePosition = position.value;
    const link = resumePosition ? null : pendingLink.current;
    const restoreMessage = link
      ? { type: 'restore_game' as const, human_color: link.humanColor, moves: link.moves }
      : restoreGameCommand(resumePosition, newHumanColor.value);
    let handshakeAccepted = false;
    let awaitingRestoredPosition = true;
    const ws = new WebSocket(websocketUrl());
    socket.value = ws;
    connection.value = reconnectAttempt.value > 0 ? 'reconnecting' : 'connecting';
    error.value = null;

    const scheduleRetry = () => {
      const attempt = reconnectAttempt.value + 1;
      reconnectAttempt.value = attempt;
      const delay =
        Math.min(500 * 2 ** (attempt - 1), 15000) * (0.75 + Math.random() * 0.5);
      retryTimer.current = window.setTimeout(() => {
        retryTimer.current = null;
        setConnectionGeneration((v) => v + 1);
      }, delay);
    };

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
        reconnectAttempt.value = 0;
        connectedOnce.value = true;
        if (retryTimer.current != null) {
          window.clearTimeout(retryTimer.current);
          retryTimer.current = null;
        }
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
        restoringLink.current = link !== null;
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
        pendingLink.current = null;
        restoringLink.current = false;
        writeGameFragment(message);
        status.value = null;
        snapshots.value = [];
        inspectedSimulations.value = null;
        focus.value = null;
        playInFlight.value = null;
        autoReplied.value = null;
        autoReplyStopped.value = null;
        requestedBudget.value = 0;
        error.value = null;

        if (message.outcome === null) {
          window.setTimeout(() => {
            if (position.value?.position_id !== message.position_id) return;
            const target = budget.value;
            requestedBudget.value = target;
            searchRequested.value = message.position_id;
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
          carried_visits: message.carried_visits,
          running: !message.complete,
        };
        const previous = snapshots.value;
        const last = previous[previous.length - 1];
        const next =
          last?.searched_simulations === message.searched_simulations
            ? [...previous.slice(0, -1), message]
            : [...previous, message];
        snapshots.value = downsampleSnapshots(next, inspectedSimulations.value);

        const current = position.value;
        const instantReply = budget.value === 0;
        if (
          replyReady(message, instantReply)
          && autoReply.value
          && current
          && current.position_id === message.position_id
          && current.outcome === null
          && current.to_move !== current.human_color
          && playInFlight.value !== current.position_id
          && autoReplied.value !== current.position_id
          && autoReplyStopped.value !== current.position_id
        ) {
          // Auto-reply: the network answers as soon as its search finishes.
          autoReplied.value = current.position_id;
          sendOnThisSocket({
            type: 'choose_network_move',
            position_id: current.position_id,
            temperature: temperature.value,
            instant: instantReply,
          });
        }

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

      playInFlight.value = null;
      if (restoringLink.current) {
        // The server refused the linked moves, e.g. play that continues after
        // five in a row. Fall back instead of retrying the same restore.
        restoringLink.current = false;
        pendingLink.current = null;
        linkNotice.value = `This link isn't a playable game: ${message.message}. ${
          awaitingRestoredPosition ? 'Started a new game instead.' : 'Kept the current game.'
        }`;
        if (awaitingRestoredPosition) sendOnThisSocket(restoreGameCommand(null, newHumanColor.value));
        else writeGameFragment(position.value);
        return;
      }
      if (
        message.code === 'invalid_command'
        && autoReplied.value !== null
        && autoReplied.value === position.value?.position_id
      ) {
        // An automatic reply raced a retargeted search; the next completed
        // snapshot retries, so this is not worth a banner.
        autoReplied.value = null;
        return;
      }
      error.value = message;
      if (awaitingRestoredPosition) ws.close();
      if (!message.recoverable) requestedBudget.value = 0;
    });
    ws.addEventListener('close', () => {
      if (socket.value !== ws) return;
      socket.value = null;
      playInFlight.value = null;
      // The game state survives a drop: the restore handshake replays it on
      // the next connection, so retry automatically instead of dead-ending.
      connection.value = 'reconnecting';
      scheduleRetry();
    });
    ws.addEventListener('error', () => {
      if (socket.value !== ws) return;
      playInFlight.value = null;
    });

    return () => {
      if (retryTimer.current != null) {
        window.clearTimeout(retryTimer.current);
        retryTimer.current = null;
      }
      ws.close();
      if (socket.value === ws) socket.value = null;
    };
  }, [connectionGeneration]);

  useEffect(() => {
    const openLinkedGame = () => {
      const parsed = parseGameFragment(window.location.hash);
      const current = position.value;
      if (!parsed?.ok) {
        if (parsed) linkNotice.value = `This link isn't a playable game: ${parsed.reason}. Kept the current game.`;
        writeGameFragment(current);
        return;
      }
      const { moves, humanColor } = parsed.link;
      if (!current) {
        pendingLink.current = parsed.link;
        return;
      }
      if (gameFragment(moves, humanColor) === gameFragment(current.stones, current.human_color)) return;
      if (send({ type: 'restore_game', human_color: humanColor, moves })) {
        restoringLink.current = true;
        lastJudgment.value = null;
        linkNotice.value = null;
      } else {
        writeGameFragment(current);
      }
    };
    window.addEventListener('hashchange', openLinkedGame);
    return () => window.removeEventListener('hashchange', openLinkedGame);
  }, []);

  useEffect(() => {
    const simulations = budget.value;
    const timer = window.setTimeout(() => {
      const pos = position.value;
      const live = status.value;
      if (!pos || pos.outcome !== null || live?.position_id !== pos.position_id) return;
      if (!live.running || live.target_simulations - live.carried_visits === simulations) {
        return;
      }
      // A move already sent leaves this position; retargeting it would be stale.
      if (playInFlight.value === pos.position_id || autoReplied.value === pos.position_id) return;
      requestedBudget.value = simulations;
      searchRequested.value = pos.position_id;
      send({ type: 'start_search', position_id: pos.position_id, simulations });
    }, RETARGET_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [budget.value]);

  const setDisclosure = (next: Disclosure) => {
    disclosure.value = next;
    try {
      localStorage.setItem(DISCLOSURE_KEY, JSON.stringify(next));
    } catch {
      // The setting still applies for this page.
    }
  };

  const setOverlay = (next: Overlay) => {
    overlay.value = next;
    try {
      localStorage.setItem(OVERLAY_KEY, next);
    } catch {
      // The choice still applies for this page.
    }
  };

  const setAutoReply = (enabled: boolean) => {
    autoReply.value = enabled;
    try {
      localStorage.setItem(AUTO_REPLY_KEY, enabled ? 'on' : 'off');
    } catch {
      // The preference still applies for this page.
    }
    const current = position.value;
    const live = snapshots.value[snapshots.value.length - 1];
    if (
      enabled
      && current
      && current.outcome === null
      && current.to_move !== current.human_color
      && live
      && replyReady(live, budget.value === 0)
      && !(status.value?.running ?? false)
    ) letNetworkChoose();
  };

  const send = (message: object): boolean => {
    const ws = socket.value;
    if (connection.value === 'reconnecting') return false;
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
  const peekedTopics =
    currentPosition && peek.value?.positionId === currentPosition.position_id
      ? peek.value.topics
      : [];
  const revealed = (topic: DisclosureTopic) =>
    reveals(disclosure.value, topic, isHumanTurn) || peekedTopics.includes(topic);
  const showEvaluation = revealed('evaluation');
  const showReview = revealed('review');
  const showMoveGuidance = revealed('moves');
  const hiddenTurn = isHumanTurn ? 'on your turn' : "on the network's turn";
  const hiddenMessage = (topic: DisclosureTopic) =>
    `${DISCLOSURE_TOPICS.find(({ id }) => id === topic)?.hidden} ${hiddenTurn}`;
  const judged = (text: string) =>
    showReview ? text : <span aria-hidden="true" className="skeleton-line" />;
  const peekAt = (topic: DisclosureTopic) => () => {
    if (!currentPosition) return;
    peek.value = { positionId: currentPosition.position_id, topics: [...peekedTopics, topic] };
  };
  const running = currentStatus?.running ?? false;
  // A budget of zero is instant play: the network replies from its raw policy.
  const instant = budget.value === 0;
  const probabilityBasis = instant ? 'prior' : 'visits';
  const boardOverlay: Overlay = instant && overlay.value === 'visits' ? 'move' : overlay.value;
  const instantReady = instant && Boolean(liveSnapshot?.complete);
  const moveProbabilities = temperatureProbabilities(
    displaySnapshot?.moves ?? [],
    temperature.value,
    probabilityBasis,
  );
  const evaluation = displaySnapshot?.search_value ?? displaySnapshot?.network_value ?? null;
  const moverShare = evaluation === null ? null : (evaluation + 1) / 2;
  const blackShare =
    moverShare === null || !currentPosition
      ? null
      : currentPosition.to_move === 'black'
        ? moverShare
        : 1 - moverShare;
  const topMoves = [...(displaySnapshot?.moves ?? [])].sort(byStrength).slice(0, 5);
  const topVisitShare = topMoves[0] && displaySnapshot ? visitFraction(topMoves[0], displaySnapshot) : 0;
  const carriedVisits =
    currentStatus?.carried_visits ?? liveSnapshot?.carried_visits ?? 0;
  const progress = Math.max(
    0,
    (currentStatus?.searched_simulations
      ?? liveSnapshot?.searched_simulations
      ?? 0) - carriedVisits,
  );
  const target = Math.max(
    0,
    (currentStatus?.target_simulations
      ?? requestedBudget.value + carriedVisits) - carriedVisits,
  );
  const remaining = Math.max(0, target - progress);
  const simsPerSecond = liveSnapshot?.simulations_per_second ?? 0;
  const etaSeconds =
    running && simsPerSecond > 0 && remaining > 0 ? remaining / simsPerSecond : null;
  const searchUnavailableReason = !currentPosition
    ? 'Waiting for the server — search unlocks once a position arrives.'
    : currentPosition.outcome !== null
      ? 'The game is over. Start a new game to keep exploring.'
      : instant
        ? 'Instant mode: the network replies from its policy without searching. Raise the budget above 0 to search.'
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
      ) {
        requestedBudget.value = 0;
        autoReplyStopped.value = currentPosition.position_id;
      }
      return;
    }
    autoReplyStopped.value = null;
    const targetBudget = budget.value;
    if (
      send({
        type: 'start_search',
        position_id: currentPosition.position_id,
        simulations: targetBudget,
      })
    ) {
      requestedBudget.value = targetBudget;
      searchRequested.value = currentPosition.position_id;
    }
  };

  const playCell = (cell: Cell) => {
    if (!currentPosition || currentPosition.outcome !== null) return;
    if (playInFlight.value === currentPosition.position_id) return;
    const judgingHumanMove = isHumanTurn;
    const latestMove = liveSnapshot?.moves.find(
      (move) => cellKey(move) === cellKey(cell),
    );
    const ranked = [...(liveSnapshot?.moves ?? [])].sort(byStrength);
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

  const letNetworkChoose = () => {
    if (!currentPosition || currentPosition.outcome !== null || isHumanTurn || !liveSnapshot) return;
    // One reply per position: a manual request and the automatic one on
    // search completion must not both be sent, or the second arrives after
    // the game has moved on and is rejected as stale.
    if (autoReplied.value === currentPosition.position_id) return;
    if (running) {
      send({ type: 'stop_search', position_id: currentPosition.position_id });
    }
    if (
      send({
        type: 'choose_network_move',
        position_id: currentPosition.position_id,
        temperature: temperature.value,
        instant,
      })
    ) autoReplied.value = currentPosition.position_id;
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

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      linkCopied.value = true;
      window.setTimeout(() => { linkCopied.value = false; }, 2000);
    } catch {
      linkNotice.value = 'Copying is blocked here; copy the address bar to share this position.';
    }
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
            <span
              className="checkpoint"
              title={`${hello.value.checkpoint.architecture} · epoch ${hello.value.checkpoint.epoch} · ${hello.value.checkpoint.model_digest}`}
            >
              <span className="checkpoint-name">
                <span className="checkpoint-architecture">
                  {hello.value.checkpoint.architecture}
                </span>
                <span className="checkpoint-epoch">
                  epoch {hello.value.checkpoint.epoch}
                </span>
              </span>
            </span>
          )}
          {hello.value?.dashboard_url && (
            <a className="topbar-link" href={hello.value.dashboard_url}>
              Research dashboard ↗
            </a>
          )}
          <ThemePicker />
          <div
            className="new-game"
            ref={newGamePopover.container}
            onFocusOut={newGamePopover.onFocusOut}
          >
            <button
              className="button button-primary"
              disabled={connection.value !== 'connected'}
              onClick={() => {
                if (send({ type: 'new_game', human_color: newHumanColor.value })) {
                  focus.value = null;
                  lastJudgment.value = null;
                  newGamePopover.setOpen(false);
                }
              }}
              type="button"
            >
              New game
            </button>
            <button
              aria-controls="new-game-options"
              aria-expanded={newGamePopover.open}
              aria-label="Choose your colour for the next game"
              className="button new-game-toggle"
              ref={newGamePopover.toggle}
              onClick={() => newGamePopover.setOpen((open) => !open)}
              title="Your colour for the next game"
              type="button"
            >
              <i aria-hidden="true" className={`legend-stone legend-${newHumanColor.value} new-game-stone`} />
              <span aria-hidden="true">▾</span>
            </button>
            <div className="new-game-popover" hidden={!newGamePopover.open} id="new-game-options">
              <span className="player-picker-caption">Your colour for the next game</span>
              <fieldset className="player-picker player-picker-group">
                <legend className="visually-hidden">
                  Your color for the next game; Black moves first
                </legend>
                <label title="Play Black and move first">
                  <input
                    checked={newHumanColor.value === 'black'}
                    name="next-human-color"
                    onChange={() => { newHumanColor.value = 'black'; }}
                    type="radio"
                    value="black"
                  />
                  <span>Black</span>
                  <small>moves first</small>
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
                  <small>moves second</small>
                </label>
              </fieldset>
              <p className="muted">Applies when you start the next game; the current game is not affected.</p>
            </div>
          </div>
        </div>
      </header>

      {error.value && (
        <div className={`notice${error.value.recoverable ? '' : ' notice-danger'}`} role="alert">
          <span>{error.value.message}</span>
          {connection.value === 'disconnected' && (
            <button className="text-button" onClick={tryReconnectNow} type="button">
              Reconnect
            </button>
          )}
          <button className="notice-close" aria-label="Dismiss error" onClick={() => { error.value = null; }} type="button">×</button>
        </div>
      )}

      {linkNotice.value && (
        <div className="notice" role="status">
          <span>{linkNotice.value}</span>
          <button className="notice-close" aria-label="Dismiss" onClick={() => { linkNotice.value = null; }} type="button">×</button>
        </div>
      )}

      {connection.value === 'reconnecting'
        && (reconnectAttempt.value >= 4 || !connectedOnce.value)
        && (
          <div className="notice" role="status">
            <span>
              {connectedOnce.value
                ? 'Lost the connection to the analysis server.'
                : 'Waiting for the analysis server.'}{' '}
              Retrying automatically — attempt{' '}
              {Math.max(1, reconnectAttempt.value)}.
            </span>
            <button className="text-button" onClick={tryReconnectNow} type="button">
              Try now
            </button>
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
              <p>
                Move {(currentPosition?.ply ?? 0) + 1}
                {currentPosition?.outcome === null && currentPosition.ply === 0
                  ? isHumanTurn
                    ? " · Click an empty intersection to play"
                    : " · Let it choose, or click an intersection to play a manual move"
                  : ""}
              </p>
            </div>
            {instant ? (
              <div className="search-readout">
                <span className={running ? 'pulse-dot' : 'status-dot'} aria-hidden="true" />
                <strong>Instant</strong>
                <span>raw network policy, no search</span>
                <small>{instantReady ? 'Evaluated' : running ? 'Evaluating…' : '\u00a0'}</small>
              </div>
            ) : (
              <div className="search-readout">
                <span className={running ? 'pulse-dot' : 'status-dot'} aria-hidden="true" />
                <strong>{progress.toLocaleString()}</strong>
                <span>/ {target.toLocaleString()} simulations</span>
                <small>{liveSnapshot ? `${Math.round(liveSnapshot.simulations_per_second).toLocaleString()} sims/s` : "\u00a0"}</small>
                <small>
                  {etaSeconds !== null
                    ? `~${formatEta(etaSeconds)} left`
                    : target > 0 && progress >= target
                      ? "Complete"
                      : target > 0
                        ? "Paused"
                        : "\u00a0"}
                </small>
              </div>
            )}
          </div>
          {currentPosition && currentPosition.outcome === null && (
            <div
              aria-label="Search progress"
              aria-valuemax={Math.max(target, 1)}
              aria-valuemin={0}
              aria-valuenow={Math.min(progress, Math.max(target, 1))}
              className={`search-progress${instantReady || (!running && target > 0 && progress >= target) ? ' is-complete' : ''}`}
              role="progressbar"
            >
              <span style={{ inlineSize: `${instantReady ? 100 : target > 0 ? Math.min(100, (progress / target) * 100) : 0}%` }} />
            </div>
          )}
          {currentPosition && currentPosition.outcome === null && (showEvaluation ? (
            <div
              aria-label={
                blackShare === null
                  ? 'No evaluation yet'
                  : `Black ${Math.round(blackShare * 100)}%, White ${Math.round((1 - blackShare) * 100)}%`
              }
              className="win-bar reveal"
              key="shown"
              role="img"
              title="Winning chances from the search value (network value before any search), from Black's side"
            >
              <span><i className="legend-stone legend-black" />Black {blackShare === null ? '—' : `${Math.round(blackShare * 100)}%`}</span>
              <div className="win-bar-track"><span style={{ inlineSize: `${(blackShare ?? 0.5) * 100}%` }} /></div>
              <span>White {blackShare === null ? '—' : `${Math.round((1 - blackShare) * 100)}%`}<i className="legend-stone legend-white" /></span>
            </div>
          ) : (
            <div className="win-bar is-concealed" key="hidden">
              <ConcealedNote message={hiddenMessage('evaluation')} onPeek={peekAt('evaluation')} />
            </div>
          ))}
          <div className="board-frame">
            {currentPosition ? (
              <Board
                canPlay={currentPosition.outcome === null}
                focus={focus.value}
                onFocusCell={(cell) => {
                  focus.value = cell;
                }}
                footer={
                  currentPosition ? (
                    <div className="action-row">
                      <button
                        className={`button${!isHumanTurn ? ' button-primary' : ''}`}
                        disabled={
                          currentPosition.outcome !== null
                          || Boolean(isHumanTurn)
                          || !liveSnapshot
                          || (!instant && liveSnapshot.target_simulations === 0)
                          || autoReplied.value === currentPosition.position_id
                        }
                        onClick={letNetworkChoose}
                        type="button"
                      >
                        {autoReplied.value === currentPosition.position_id
                          ? 'Network is replying…'
                          : 'Let network choose'}
                      </button>
                      <button
                        className="button"
                        disabled={!undoTarget}
                        onClick={undoLastMove}
                        type="button"
                      >
                        Undo move
                      </button>
                      <button
                        className="button"
                        onClick={copyLink}
                        title="Copy a link that opens this game at the current position"
                        type="button"
                      >
                        <span className="stable-label">
                          <span className={linkCopied.value ? '' : 'is-shown'}>Copy link</span>
                          <span className={linkCopied.value ? 'is-shown' : ''}>Link copied</span>
                        </span>
                      </button>
                      <label className="auto-reply" title="When on, the network plays its move as soon as its search finishes. Turn off to inspect the search before each reply.">
                        <input
                          checked={autoReply.value}
                          onChange={(event) => setAutoReply(event.currentTarget.checked)}
                          type="checkbox"
                        />
                        <span>Auto-reply</span>
                      </label>
                    </div>
                  ) : null
                }
                onPlay={playCell}
                overlay={boardOverlay}
                position={currentPosition}
                probabilityBasis={probabilityBasis}
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
                  <fieldset className="metric-switch reveal">
                    <legend className="visually-hidden">Square size shows</legend>
                    <i aria-hidden="true" className="legend-policy" />
                    {OVERLAY_OPTIONS.map((option) => (
                      <span className="has-tip" key={option.value}>
                        <input
                          aria-describedby={`metric-tip-${option.value}`}
                          checked={boardOverlay === option.value}
                          disabled={instant && option.value === 'visits'}
                          id={`metric-${option.value}`}
                          name="board-overlay"
                          onChange={() => setOverlay(option.value)}
                          type="radio"
                          value={option.value}
                        />
                        <label htmlFor={`metric-${option.value}`}>{option.label}</label>
                        <span className="hover-tip" id={`metric-tip-${option.value}`} role="tooltip">
                          <strong>{option.title}</strong>
                          {option.description}
                          {option.value === 'move' && `, T=${temperature.value.toFixed(2)}.`}
                          {instant && option.value === 'visits' && ' Unavailable in instant mode: nothing is searched.'}
                        </span>
                      </span>
                    ))}
                  </fieldset>
                  <span className="legend-note reveal">larger square = more likely · 1 = top move</span>
                </>
              ) : (
                <ConcealedNote message={hiddenMessage('moves')} onPeek={peekAt('moves')} />
              )}
            </div>
          )}
          </section>

          <section className="panel evolution" aria-labelledby="evolution-title">
            <div className="section-heading evolution-heading">
              <div>
                <h2 id="evolution-title">How search changes its mind</h2>
                <p>Root visit fraction as simulations accumulate. Hover for exact values; click to pin a snapshot.</p>
              </div>
            </div>
            {instant ? (
              <p className="empty-copy evolution-idle">
                No search in instant mode: the network replies from its policy alone. Raise the
                search budget above 0 to see how the search changes its mind.
              </p>
            ) : (
              <div className={`concealable${showMoveGuidance ? '' : ' is-concealed'}`}>
                <SearchChart
                  concealed={!showMoveGuidance}
                  onSelectIndex={(index) => {
                    inspectedSimulations.value =
                      index === null ? null : (allSnapshots[index]?.searched_simulations ?? null);
                  }}
                  onFocusMove={(cell) => {
                    focus.value =
                      focus.value && cellKey(focus.value) === cellKey(cell) ? null : cell;
                  }}
                  selectedIndex={inspectedIndex}
                  focusCell={focus.value}
                  snapshots={showMoveGuidance ? allSnapshots : []}
                />
                {allSnapshots.length > 0 && (
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
                    <output>
                      {Math.max(
                        0,
                        (displaySnapshot?.searched_simulations ?? 0) -
                          (displaySnapshot?.carried_visits ?? 0),
                      ).toLocaleString()}{' '}
                      sims
                    </output>
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
                {!showMoveGuidance && (
                  <ConcealedVeil message={hiddenMessage('moves')} onPeek={peekAt('moves')} />
                )}
              </div>
            )}
          </section>
        </div>

        <aside className="panel inspector" aria-label="Position analysis">
          <section className="inspector-section">
            <div className="section-heading">
              <h2>Search controls</h2>
              <div className="search-actions">
                <button
                  className={`button${running && !instant ? '' : ' button-primary'}`}
                  disabled={
                    !currentPosition
                    || currentPosition.outcome !== null
                    || instant
                    || (!running && progress >= budget.value)
                  }
                  onClick={startOrStopSearch}
                  type="button"
                >
                  {instant ? 'Run search' : running ? 'Stop search' : progress > 0 ? 'Continue search' : 'Run search'}
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
                <span><b>Search budget</b><output>{instant ? 'Instant' : budget.value.toLocaleString()}</output></span>
                <input
                  aria-valuetext={instant ? 'Instant: no search' : `${budget.value.toLocaleString()} simulations`}
                  max={maximumBudget}
                  min={0}
                  onInput={(event) => { budget.value = Number(event.currentTarget.value); }}
                  step={budgetStep}
                  type="range"
                  value={budget.value}
                />
                <small>
                  {instant
                    ? 'At 0 the network replies from its policy, without search'
                    : `Live while a search runs · 0 = instant · limit ${maximumBudget.toLocaleString()}`}
                </small>
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
              <DisclosureSwitches disclosure={disclosure.value} onChange={setDisclosure} />
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
                    {showEvaluation && (
                      <span className="info-tip-live">
                        {favorsText(displaySnapshot?.network_value ?? null, turnColor, opponentColor)}
                      </span>
                    )}
                  </InfoTip>
                </span>
                {showEvaluation ? (
                  <strong className="reveal" key="shown">{signed(displaySnapshot?.network_value ?? null)}</strong>
                ) : (
                  <strong className="value-concealed" key="hidden">
                    <button className="text-button" onClick={peekAt('evaluation')} type="button">
                      <EyeIcon off />Peek
                    </button>
                  </strong>
                )}
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
                    {showEvaluation && (
                      <span className="info-tip-live">
                        {favorsText(displaySnapshot?.search_value ?? null, turnColor, opponentColor)}
                      </span>
                    )}
                  </InfoTip>
                </span>
                {showEvaluation ? (
                  <strong className="reveal" key="shown">{signed(displaySnapshot?.search_value ?? null)}</strong>
                ) : (
                  <strong className="value-concealed" key="hidden">
                    <button className="text-button" onClick={peekAt('evaluation')} type="button">
                      <EyeIcon off />Peek
                    </button>
                  </strong>
                )}
              </div>
            </div>
            <p className="visits-note">
              {!displaySnapshot
                ? 'No search yet'
                : displaySnapshot.total_visits === 0 && instant
                  ? 'Network evaluation only, no search'
                  : `${displaySnapshot.total_visits.toLocaleString()} tree visits`}
              {displaySnapshot && displaySnapshot.carried_visits > 0 && (
                ` · ${displaySnapshot.carried_visits.toLocaleString()} carried from previous search`
              )}
            </p>
            <div className={`concealable${showMoveGuidance ? '' : ' is-concealed'}`}>
              <div className="table-wrap">
                <table aria-label="Top candidate moves" className="candidate-table">
                  <thead><tr><th>Move</th><th>Prior</th><th>Visits</th><th>P(move)</th></tr></thead>
                  <tbody>
                    {!showMoveGuidance ? Array.from({ length: 5 }, (_, row) => (
                      <tr aria-hidden="true" className="skeleton-row" key={row}>
                        {[0, 1, 2, 3].map((column) => <td key={column}><span className="skeleton-line" /></td>)}
                      </tr>
                    )) : topMoves.map((move) => {
                      const key = cellKey(move);
                      const color = dotColorFor(key);
                      const share = displaySnapshot ? visitFraction(move, displaySnapshot) : 0;
                      return (
                        <tr
                          key={key}
                          onMouseEnter={() => { focus.value = { row: move.row, column: move.column }; }}
                          onMouseLeave={() => {
                            if (focus.value && cellKey(focus.value) === key) focus.value = null;
                          }}
                        >
                          <td>
                            <button
                              className="table-move-button"
                              onClick={() => playCell({ row: move.row, column: move.column })}
                              title="Play this move"
                              type="button"
                            >
                              <i
                                className="series-dot"
                                style={color ? { background: color } : undefined}
                              />{moveName(move)}
                            </button>
                          </td>
                          <td>{percent(move.prior)}</td>
                          <td className="visits-cell">
                            <span className="visit-bar" aria-hidden="true" style={{ inlineSize: `${topVisitShare > 0 ? (share / topVisitShare) * 100 : 0}%` }} />
                            {displaySnapshot ? percent(share) : '—'}
                          </td>
                          <td>{percent(moveProbabilities.get(key) ?? null)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {!showMoveGuidance && (
                <ConcealedVeil message={hiddenMessage('moves')} onPeek={peekAt('moves')} />
              )}
            </div>
          </section>

          <section className="inspector-section judgment-section">
            <h2>How the network judged your last move</h2>
            {lastJudgment.value ? (
              <div className={`concealable${showReview ? '' : ' is-concealed'}`}>
                <div className="judgment-card">
                  <div>
                    <strong>{lastJudgment.value.name}</strong>
                    <span>
                      {judged(lastJudgment.value.rank === 1
                        ? '— its first choice'
                        : lastJudgment.value.rank !== null && lastJudgment.value.rank <= 5
                          ? '— one of its leading choices'
                          : '— outside its leading candidates')}
                    </span>
                  </div>
                  <dl>
                    <dt>Policy rank before move</dt><dd>{judged(lastJudgment.value.rank ? `${lastJudgment.value.rank} / ${lastJudgment.value.legalMoves}` : '—')}</dd>
                    <dt>Raw network policy</dt><dd>{judged(percent(lastJudgment.value.prior))}</dd>
                    <dt>Search visit fraction</dt><dd>{judged(percent(lastJudgment.value.visitShare))}</dd>
                    <dt>Network value change</dt><dd>{judged(signed(lastJudgment.value.valueDelta))}</dd>
                  </dl>
                </div>
                {!showReview && (
                  <ConcealedVeil message={hiddenMessage('review')} onPeek={peekAt('review')} />
                )}
              </div>
            ) : (
              <p className="empty-copy">Your move’s policy rank and value impact will appear here.</p>
            )}
          </section>
        </aside>
      </main>

    </div>
  );
}
