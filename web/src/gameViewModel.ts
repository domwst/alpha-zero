import type { RecordedState } from "./replayPosition";
import type { Cell, PositionMessage, SearchSnapshotMessage } from "./protocol";

export type Diagnostics = {
  network_prior: number[] | null;
  root_visits: number[] | null;
  sampling_policy: number[] | null;
  value_estimate: number | null;
  search_value: number | null;
  search?: {
    expanded_by_depth: number[];
    allocated_by_depth: number[];
    simulation_leaf_depth: number[];
  };
};
export type Ply = {
  state: { state: number[] };
  action: { x: number; y: number } | null;
  actor: "First" | "Second";
  decision: { training_policy: number[] | null; diagnostics: Diagnostics };
};
export type Game = {
  game_type: string;
  game_id: string;
  provenance: string;
  policy_semantics: string;
  model_identity: string | null;
  record: {
    plies: Ply[];
    terminal_state?: RecordedState | null;
    terminal_value: number | null;
    terminal_actor?: string;
  };
};
export type GameRef = {
  id: string;
  epoch: string;
  job_id: string;
  plies: number;
  winner: string | null;
};
export type Activation = {
  name: string;
  shape: number[];
  axes: string[];
  values: number[];
};
export type Analysis = {
  checkpoint: unknown;
  result: {
    complete: boolean;
    searched_simulations: number;
    target_simulations: number;
    carried_visits: number;
    elapsed_ms: number;
    simulations_per_second: number;
    network_value: number;
    search_value: number | null;
    total_visits: number;
    moves: SearchSnapshotMessage["moves"];
    activations: Activation[];
    terminal: number | null;
    search: Diagnostics["search"];
  };
};
export const gameName = (id: string) =>
  /^[0-9]{8}$/.test(id) ? String(Number(id) + 1) : id.slice(0, 10);
export const empty = () => Array(361).fill(0) as number[];
export function encodePosition(cells: number[]) {
  return cells.join("");
}
export function decodePosition(value: string | null) {
  return value && /^[012]{361}$/.test(value) ? [...value].map(Number) : empty();
}
export function position(
  cells: number[],
  ply: number,
  selected: Cell | null,
): PositionMessage {
  const toMove = ply % 2 ? "white" : "black";
  return {
    type: "position",
    position_id: 0,
    ply,
    human_color: toMove,
    to_move: toMove,
    last_move: selected,
    outcome: null,
    carried_visits: 0,
    stones: cells.flatMap((n, i) =>
      n
        ? [
            {
              row: Math.floor(i / 19),
              column: i % 19,
              color: (n === 1
                ? toMove
                : toMove === "black"
                  ? "white"
                  : "black") as "black" | "white",
            },
          ]
        : [],
    ),
  };
}
export function snapshot(
  moves: SearchSnapshotMessage["moves"],
  value: number,
  search: number | null,
): SearchSnapshotMessage {
  const total = moves.reduce((n, m) => n + m.visits, 0);
  return {
    type: "search_snapshot",
    position_id: 0,
    analysis_id: 0,
    searched_simulations: total,
    carried_visits: 0,
    total_visits: total,
    target_simulations: total,
    elapsed_ms: 0,
    simulations_per_second: 0,
    network_value: value,
    search_value: search,
    moves,
    complete: true,
  };
}

export type ArchivePage = {
  games: GameRef[];
  epochs: string[];
  total: number;
  epoch_labels: Record<string, string>;
  offset: number;
  epoch: string;
};
export const PAGE_SIZE = 10;
export function analysisSnapshot(a: Analysis): SearchSnapshotMessage {
  const { activations: _activations, ...result } = a.result;
  return {
    ...snapshot(a.result.moves, a.result.network_value, a.result.search_value),
    ...result,
    type: "search_snapshot",
    position_id: 0,
    analysis_id: 0,
  };
}
