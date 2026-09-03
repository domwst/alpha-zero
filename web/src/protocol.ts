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
  | { type: 'choose_network_move'; position_id: number; temperature: number };

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

export function cellKey(cell: Cell): string {
  return `${cell.row}:${cell.column}`;
}

export function moveName(cell: Cell): string {
  return `${COLUMNS[cell.column] ?? '?'}${BOARD_SIZE - cell.row}`;
}

export function visitFraction(move: MoveStats, snapshot: SearchSnapshotMessage): number {
  return snapshot.total_visits > 0 ? move.visits / snapshot.total_visits : 0;
}

/** Mirrors alz::engine::apply_temperature, including its last-wins tie break. */
export function temperatureProbabilities(
  moves: MoveStats[],
  temperature: number,
): Map<string, number> {
  const result = new Map<string, number>();
  if (moves.length === 0) return result;

  let maxIndex = 0;
  for (let index = 1; index < moves.length; index += 1) {
    if ((moves[index]?.visits ?? 0) >= (moves[maxIndex]?.visits ?? 0)) maxIndex = index;
  }
  const maxVisits = moves[maxIndex]?.visits ?? 0;
  if (maxVisits === 0) {
    for (const move of moves) result.set(cellKey(move), 0);
    return result;
  }
  if (temperature === 0) {
    moves.forEach((move, index) => result.set(cellKey(move), index === maxIndex ? 1 : 0));
    return result;
  }

  const weights = moves.map((move) =>
    move.visits === 0 ? 0 : Math.exp(Math.log(move.visits / maxVisits) / temperature),
  );
  const sum = weights.reduce((total, weight) => total + weight, 0);
  moves.forEach((move, index) => result.set(cellKey(move), (weights[index] ?? 0) / sum));
  return result;
}
