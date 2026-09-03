import type { JSX } from 'preact';
import { useRef, useState } from 'preact/hooks';

import {
  BOARD_SIZE,
  COLUMNS,
  type Cell,
  type MoveStats,
  type PositionMessage,
  type SearchSnapshotMessage,
  cellKey,
  moveName,
  temperatureProbabilities,
  visitFraction,
} from './protocol';
import { percent, signed } from './format';

export type Overlay = 'prior' | 'visits' | 'move';

interface BoardProps {
  position: PositionMessage | null;
  snapshot: SearchSnapshotMessage | null;
  overlay: Overlay;
  showPolicy: boolean;
  temperature: number;
  selected: Cell | null;
  onSelect: (cell: Cell) => void;
}

const ARROW_DELTAS: Record<string, [number, number]> = {
  ArrowUp: [-1, 0],
  ArrowDown: [1, 0],
  ArrowLeft: [0, -1],
  ArrowRight: [0, 1],
};

export function Board({
  position,
  snapshot,
  overlay,
  showPolicy,
  temperature,
  selected,
  onSelect,
}: BoardProps): JSX.Element {
  const gridRef = useRef<HTMLDivElement | null>(null);
  const [activeCell, setActiveCell] = useState<Cell>({ row: 0, column: 0 });
  const stones = new Map(position?.stones.map((stone) => [cellKey(stone), stone]) ?? []);
  const moves = new Map(snapshot?.moves.map((move) => [cellKey(move), move]) ?? []);
  const moveProbabilities = temperatureProbabilities(snapshot?.moves ?? [], temperature);
  const overlayValue = (move: MoveStats): number => {
    if (!snapshot) return 0;
    if (overlay === 'prior') return move.prior;
    if (overlay === 'visits') return visitFraction(move, snapshot);
    return moveProbabilities.get(cellKey(move)) ?? 0;
  };
  let topKey: string | null = null;
  let topValue = 0;
  for (const move of moves.values()) {
    const value = overlayValue(move);
    if (value > topValue) {
      topValue = value;
      topKey = cellKey(move);
    }
  }
  const maximum = topValue;
  const moveOwner = position?.to_move === position?.human_color ? 'you' : 'the network';

  const focusCell = (cell: Cell) => {
    setActiveCell(cell);
    gridRef.current
      ?.querySelector<HTMLElement>(`[data-row="${cell.row}"][data-col="${cell.column}"]`)
      ?.focus();
  };

  const handleCellKeyDown = (cell: Cell, event: KeyboardEvent) => {
    const delta = ARROW_DELTAS[event.key];
    if (!delta) return;
    event.preventDefault();
    const row = Math.min(BOARD_SIZE - 1, Math.max(0, cell.row + delta[0]));
    const column = Math.min(BOARD_SIZE - 1, Math.max(0, cell.column + delta[1]));
    if (row !== cell.row || column !== cell.column) focusCell({ row, column });
  };

  const rows: JSX.Element[] = [];
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    const cells: JSX.Element[] = [];
    for (let column = 0; column < BOARD_SIZE; column += 1) {
      const cell = { row, column };
      const key = cellKey(cell);
      const stone = stones.get(key);
      const move = moves.get(key);
      const isLast = position?.last_move != null && cellKey(position.last_move) === key;
      const isSelected = selected != null && cellKey(selected) === key;
      const isActive = activeCell.row === row && activeCell.column === column;
      const probability = move ? overlayValue(move) : 0;
      const markerSize = maximum > 0 ? 12 + 75 * Math.sqrt(probability / maximum) : 0;
      const visitShare = move && snapshot ? visitFraction(move, snapshot) : 0;
      const moveProbability = moveProbabilities.get(key) ?? 0;
      const isLeading = showPolicy && move != null && key === topKey;
      const tooltipId = `cell-tip-${row}-${column}`;
      const canInspect = !stone;
      const selectionInstruction = isSelected
        ? `, selected; activate again to play for ${moveOwner}`
        : '';
      const label = stone
        ? `${moveName(cell)}, ${stone.color}${isLast ? ', last move' : ''}`
        : move && showPolicy
          ? `${moveName(cell)}${isLeading ? ', leading move' : ''}, network policy ${percent(move.prior)}, visit fraction ${percent(visitShare)}, move probability ${percent(moveProbability)}${selectionInstruction}`
          : `${moveName(cell)}, empty${selectionInstruction}`;

      cells.push(
        <button
          aria-describedby={showPolicy && move ? tooltipId : undefined}
          aria-disabled={!canInspect}
          aria-label={label}
          aria-selected={isSelected}
          class={`board-cell${isSelected ? ' is-selected' : ''}`}
          data-col={column}
          data-row={row}
          key={key}
          onClick={() => {
            setActiveCell(cell);
            if (canInspect) onSelect(cell);
          }}
          onKeyDown={(event) => handleCellKeyDown(cell, event)}
          role="gridcell"
          tabIndex={isActive ? 0 : -1}
          type="button"
        >
          {stone && (
            <span
              aria-hidden="true"
              class={`stone stone-${stone.color}${isLast ? ' is-last' : ''}`}
            />
          )}
          {showPolicy && !stone && move && probability > 0 && (
            <span
              aria-hidden="true"
              className="policy-marker"
              style={{
                height: `${markerSize}%`,
                opacity: 0.38 + 0.56 * (probability / maximum),
                width: `${markerSize}%`,
              }}
            />
          )}
          {isLeading && probability > 0 && (
            <span aria-hidden="true" className="rank-badge">
              1
            </span>
          )}
          {showPolicy && !stone && move && (
            <span
              class={`cell-tooltip${column <= 3 ? ' align-left' : column >= 15 ? ' align-right' : ''}${row <= 2 ? ' below' : ''}`}
              id={tooltipId}
              role="tooltip"
            >
              <strong>{moveName(cell)}</strong>
              <span><em>Network policy</em><b>{percent(move.prior)}</b></span>
              <span><em>Visits</em><b>{move.visits.toLocaleString()} / {(snapshot?.total_visits ?? 0).toLocaleString()}</b></span>
              <span><em>Visit fraction</em><b>{percent(visitShare)}</b></span>
              <span><em>Move probability</em><b>{percent(moveProbability)} <small>at T={temperature.toFixed(2)}</small></b></span>
              <span><em>Action value Q</em><b>{signed(move.mean_value, 3)}</b></span>
            </span>
          )}
        </button>,
      );
    }
    rows.push(
      <div className="board-row" key={row} role="row">
        {cells}
      </div>,
    );
  }

  return (
    <div className="board-with-labels">
      <div aria-hidden="true" className="board-labels board-labels-files">
        {COLUMNS.split('').map((file) => (
          <span key={file}>{file}</span>
        ))}
      </div>
      <div aria-hidden="true" className="board-labels board-labels-ranks">
        {Array.from({ length: BOARD_SIZE }, (_, index) => BOARD_SIZE - index).map((rank) => (
          <span key={rank}>{rank}</span>
        ))}
      </div>
      <div
        aria-label={`19 by 19 Gomoku board${showPolicy ? ' with policy overlay' : ''}`}
        className="board"
        ref={gridRef}
        role="grid"
      >
        {rows}
      </div>
    </div>
  );
}
