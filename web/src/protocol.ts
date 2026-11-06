export type StoneColor = 'black' | 'white';
export type GameOutcome = 'black_won' | 'white_won' | 'draw';

export interface CheckpointInfo {
  architecture: string;
  epoch: number;
  model_digest: string;
}

export interface Cell {
  row: number;
  column: number;
}

export interface Stone extends Cell {
  color: StoneColor;
}

export interface MoveStats extends Cell {
  prior: number;
  visits: number;
  mean_value: number | null;
}

export type ClientMessage =
  | { type: 'new_game'; human_color: StoneColor }
  | { type: 'restore_game'; human_color: StoneColor; moves: Cell[] }
  | { type: 'start_search'; position_id: number; simulations: number }
  | { type: 'stop_search'; position_id: number }
  | { type: 'play'; position_id: number; row: number; column: number }
  | { type: 'choose_network_move'; position_id: number; temperature: number; instant?: boolean };

export interface HelloMessage {
  type: 'hello';
  protocol_version: number;
  board_size: number;
  compute_device: string;
  checkpoint: CheckpointInfo;
  max_search_simulations: number;
  default_search_simulations: number;
  c_puct: number;
  snapshot_interval_ms: number;
  /** Public address of the research dashboard, when the operator configured one. */
  dashboard_url?: string | null;
}

export interface PositionMessage {
  type: 'position';
  position_id: number;
  ply: number;
  human_color: StoneColor;
  to_move: StoneColor;
  stones: Stone[];
  last_move: Cell | null;
  outcome: GameOutcome | null;
  carried_visits: number;
}

export interface SearchStatusMessage {
  type: 'search_status';
  position_id: number;
  analysis_id: number;
  searched_simulations: number;
  target_simulations: number;
  carried_visits: number;
  running: boolean;
}

export interface SearchSnapshotMessage {
  type: 'search_snapshot';
  position_id: number;
  analysis_id: number;
  searched_simulations: number;
  carried_visits: number;
  total_visits: number;
  target_simulations: number;
  elapsed_ms: number;
  simulations_per_second: number;
  network_value: number;
  search_value: number | null;
  moves: MoveStats[];
  complete: boolean;
}

export interface ErrorMessage {
  type: 'error';
  code: string;
  message: string;
  recoverable: boolean;
}

export type ServerMessage =
  | HelloMessage
  | PositionMessage
  | SearchStatusMessage
  | SearchSnapshotMessage
  | ErrorMessage;

export const PROTOCOL_VERSION = 2;
export const BOARD_SIZE = 19;
export const COLUMNS = 'ABCDEFGHJKLMNOPQRST';

export function restoreGameCommand(
  position: PositionMessage | null,
  fallbackHumanColor: StoneColor,
): Extract<ClientMessage, { type: 'restore_game' }> {
  return {
    type: 'restore_game',
    human_color: position?.human_color ?? fallbackHumanColor,
    moves: position?.stones.map(({ row, column }) => ({ row, column })) ?? [],
  };
}

/**
 * Move list that takes back the human's last move — plus the network's reply,
 * when one exists. Returns null when there is no human move to take back.
 */
export function undoMoves(position: PositionMessage | null): Cell[] | null {
  if (!position || position.stones.length === 0) return null;
  return position.stones
    .slice(0, -1)
    .map(({ row, column }) => ({ row, column }));
}

export function cellKey(cell: Cell): string {
  return `${cell.row}:${cell.column}`;
}

export function moveName(cell: Cell): string {
  return `${COLUMNS[cell.column] ?? '?'}${BOARD_SIZE - cell.row}`;
}

export function visitFraction(move: MoveStats, snapshot: SearchSnapshotMessage): number {
  return snapshot.total_visits > 0 ? move.visits / snapshot.total_visits : 0;
}

/**
 * Mirrors alz::engine::apply_temperature, including its last-wins tie break.
 * The server samples search visits, or the raw priors for an instant reply.
 */
export function temperatureProbabilities(
  moves: MoveStats[],
  temperature: number,
  basis: 'visits' | 'prior' = 'visits',
): Map<string, number> {
  const result = new Map<string, number>();
  if (moves.length === 0) return result;

  let maxIndex = 0;
  for (let index = 1; index < moves.length; index += 1) {
    if ((moves[index]?.[basis] ?? 0) >= (moves[maxIndex]?.[basis] ?? 0)) maxIndex = index;
  }
  const maxWeight = moves[maxIndex]?.[basis] ?? 0;
  if (maxWeight === 0) {
    for (const move of moves) result.set(cellKey(move), 0);
    return result;
  }
  if (temperature === 0) {
    moves.forEach((move, index) => result.set(cellKey(move), index === maxIndex ? 1 : 0));
    return result;
  }

  const weights = moves.map((move) =>
    move[basis] === 0 ? 0 : Math.exp(Math.log(move[basis] / maxWeight) / temperature),
  );
  const sum = weights.reduce((total, weight) => total + weight, 0);
  moves.forEach((move, index) => result.set(cellKey(move), (weights[index] ?? 0) / sum));
  return result;
}

// Game links ---------------------------------------------------------------

/**
 * A game carried in the demo's URL fragment, e.g. `#moves=K10.K9.J10&as=white`:
 * the moves in order, in the board's own coordinates, and the side the person
 * opening the link plays. The fragment never reaches the server or its logs.
 */
export interface GameLink {
  moves: Cell[];
  humanColor: StoneColor;
}

export type ParsedGameLink = { ok: true; link: GameLink } | { ok: false; reason: string };

export function sideToMove(moveCount: number): StoneColor {
  return moveCount % 2 ? 'white' : 'black';
}

export function parseMoveName(token: string): Cell | null {
  const match = /^([A-Za-z])(\d{1,2})$/.exec(token.trim());
  if (!match) return null;
  const column = COLUMNS.indexOf(match[1]!.toUpperCase());
  const rank = Number(match[2]);
  if (column < 0 || rank < 1 || rank > BOARD_SIZE) return null;
  return { row: BOARD_SIZE - rank, column };
}

/** Fragment without `#`; empty for a fresh game played as Black. */
export function gameFragment(moves: readonly Cell[], humanColor: StoneColor): string {
  const params = new URLSearchParams();
  if (moves.length > 0) params.set('moves', moves.map(moveName).join('.'));
  if (moves.length > 0 || humanColor !== 'black') params.set('as', humanColor);
  return params.toString();
}

/** Null when the fragment carries no game; otherwise the game or why it is unusable. */
export function parseGameFragment(fragment: string): ParsedGameLink | null {
  const params = new URLSearchParams(fragment.replace(/^#/, ''));
  if (!params.has('moves') && !params.has('as')) return null;

  const tokens = (params.get('moves') ?? '').split(/[.,\s]+/).filter(Boolean);
  if (tokens.length > BOARD_SIZE * BOARD_SIZE) {
    return { ok: false, reason: 'it has more moves than the board has points' };
  }
  const moves: Cell[] = [];
  const occupied = new Set<string>();
  for (const [index, token] of tokens.entries()) {
    const cell = parseMoveName(token);
    if (!cell) return { ok: false, reason: `move ${index + 1}, “${token}”, is not a board point` };
    if (occupied.has(cellKey(cell))) {
      return { ok: false, reason: `move ${index + 1}, ${moveName(cell)}, is played twice` };
    }
    occupied.add(cellKey(cell));
    moves.push(cell);
  }
  const side = params.get('as');
  const humanColor = side === 'black' || side === 'white' ? side : sideToMove(moves.length);
  return { ok: true, link: { moves, humanColor } };
}

/** Demo address that opens a position with the side to move played by the visitor. */
export function demoPositionUrl(demoUrl: string, moves: readonly Cell[]): string {
  const url = new URL(demoUrl);
  url.hash = gameFragment(moves, sideToMove(moves.length));
  return url.toString();
}
