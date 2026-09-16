import type { ComponentChildren, JSX } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';

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
  /** Emphasized cell (chart link, pinned inspection); visual focus with no move semantics. */
  focus?: Cell | null;
  /** Reports inspection-focus changes: tap pin on read-only boards, long-press, Escape. */
  onFocusCell?: (cell: Cell | null) => void;
  /** Empty-cell activation commits a move via onPlay (click, Enter, touch tap). */
  canPlay?: boolean;
  onPlay?: (cell: Cell) => void;
  /** Label for the prior-distribution row; recorded views relabel it per distribution. */
  policyLabel?: string;
  recordedSampling?: Map<string, number> | null;
  /** Hides the move-probability row when the emphasized prior row already shows it. */
  hideMoveProbability?: boolean;
  /** Extra controls placed under the board, aligned with the cells column. */
  footer?: ComponentChildren;
}

const ARROW_DELTAS: Record<string, [number, number]> = {
  ArrowUp: [-1, 0],
  ArrowDown: [1, 0],
  ArrowLeft: [0, -1],
  ArrowRight: [0, 1],
};

// A stationary touch held this long pins the tooltip instead of playing.
const LONG_PRESS_MS = 400;

function cellFromEvent(event: { target: EventTarget | null }): Cell | null {
  const cell = (event.target as Element | null)?.closest?.('.board-cell');
  if (!cell) return null;
  const row = Number((cell as HTMLElement).dataset.row);
  const column = Number((cell as HTMLElement).dataset.col);
  if (!Number.isInteger(row) || !Number.isInteger(column)) return null;
  return { row, column };
}

// Touch pointers are implicitly captured by their target, so the element under
// the finger must be resolved geometrically during a gesture.
function cellFromPoint(x: number, y: number): Cell | null {
  return cellFromEvent({ target: document.elementFromPoint(x, y) });
}

function sameCell(a: Cell | null, b: Cell | null): boolean {
  return a != null && b != null && cellKey(a) === cellKey(b);
}

interface TooltipRow {
  label: string;
  value: ComponentChildren;
}

export function Board({
  position,
  snapshot,
  overlay,
  showPolicy,
  temperature,
  focus = null,
  onFocusCell,
  canPlay = false,
  onPlay,
  policyLabel,
  recordedSampling,
  hideMoveProbability = false,
  footer,
}: BoardProps): JSX.Element {
  const gridRef = useRef<HTMLDivElement | null>(null);
  const gesture = useRef<{
    pointerId: number;
    start: Cell;
    moved: boolean;
    held: boolean;
    timer: number | null;
  } | null>(null);
  const suppressPointerClick = useRef(false);
  const [activeCell, setActiveCell] = useState<Cell>({ row: 0, column: 0 });
  const [aim, setAim] = useState<Cell | null>(null);
  const stones = new Map(
    position?.stones.map((stone) => [cellKey(stone), stone]) ?? [],
  );
  const moves = new Map(
    snapshot?.moves.map((move) => [cellKey(move), move]) ?? [],
  );
  const moveProbabilities = temperatureProbabilities(
    snapshot?.moves ?? [],
    temperature,
  );

  useEffect(
    () => () => {
      if (gesture.current?.timer != null) window.clearTimeout(gesture.current.timer);
    },
    [],
  );

  // A pinned inspection clears when the user interacts anywhere outside the
  // board, so a tap elsewhere returns the view to the plain game state.
  useEffect(() => {
    if (focus == null) return;
    const onOutsidePointerDown = (event: PointerEvent) => {
      const within = (event.target as Element | null)?.closest?.('.board-surface');
      if (!within) onFocusCell?.(null);
    };
    document.addEventListener('pointerdown', onOutsidePointerDown);
    return () => document.removeEventListener('pointerdown', onOutsidePointerDown);
  }, [focus]);

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

  const toggleFocus = (cell: Cell) => {
    onFocusCell?.(sameCell(focus, cell) ? null : cell);
  };

  const canInspectCell = (cell: Cell) => {
    return !stones.has(cellKey(cell)) && showPolicy && moves.has(cellKey(cell));
  };

  const moveCursor = (from: Cell, rowDelta: number, columnDelta: number) => {
    let row = from.row + rowDelta;
    let column = from.column + columnDelta;
    while (row >= 0 && row < BOARD_SIZE && column >= 0 && column < BOARD_SIZE) {
      if (!stones.has(cellKey({ row, column }))) {
        const next = { row, column };
        setActiveCell(next);
        gridRef.current
          ?.querySelector<HTMLElement>(`[data-row="${row}"][data-col="${column}"]`)
          ?.focus();
        return;
      }
      row += rowDelta;
      column += columnDelta;
    }
  };

  const activateCell = (cell: Cell) => {
    if (stones.has(cellKey(cell))) return;
    if (canPlay) {
      onPlay?.(cell);
      return;
    }
    if (canInspectCell(cell)) toggleFocus(cell);
  };

  const handleCellKeyDown = (cell: Cell, event: KeyboardEvent) => {
    const delta = ARROW_DELTAS[event.key];
    if (delta) {
      event.preventDefault();
      moveCursor(cell, delta[0], delta[1]);
      return;
    }
    if (event.key === 'Escape') {
      // Un-select: drop the inspection ring/tooltip and leave the grid.
      onFocusCell?.(null);
      (event.currentTarget as HTMLElement).blur();
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (!event.repeat) activateCell(cell);
    }
  };

  const handleCellClick = (cell: Cell, event: MouseEvent) => {
    if (event.button !== 0) return;
    // A held or cancelled touch may still produce a compatibility click.
    // Keyboard/assistive activation has no pointer click count and stays usable.
    if (event.detail > 0 && suppressPointerClick.current) {
      suppressPointerClick.current = false;
      event.preventDefault();
      return;
    }
    activateCell(cell);
  };

  const endGesture = () => {
    const current = gesture.current;
    if (current?.timer != null) window.clearTimeout(current.timer);
    gesture.current = null;
  };

  const handlePointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    const cell = cellFromEvent(event);
    if (!cell) return;
    suppressPointerClick.current = false;
    setActiveCell(cell);
    if (event.pointerType !== 'touch') return;
    endGesture();
    gesture.current = {
      pointerId: event.pointerId,
      start: cell,
      moved: false,
      held: false,
      timer: window.setTimeout(() => {
        if (gesture.current && !gesture.current.moved) {
          gesture.current.held = true;
          if (canInspectCell(cell)) toggleFocus(cell);
          setAim(null);
        }
      }, LONG_PRESS_MS),
    };
    setAim(cell);
  };

  const handlePointerMove = (event: PointerEvent) => {
    const current = gesture.current;
    if (
      event.pointerType !== 'touch'
      || !current
      || current.pointerId !== event.pointerId
    ) {
      return;
    }
    const cell = cellFromPoint(event.clientX, event.clientY);
    if (!cell || !sameCell(cell, current.start)) {
      // The finger left its cell: this reads as a scroll, so the tap no longer
      // commits and the browser may claim the gesture at any moment.
      current.moved = true;
      if (current.timer != null) window.clearTimeout(current.timer);
      setAim(null);
    }
  };

  const handlePointerCancel = () => {
    suppressPointerClick.current = true;
    endGesture();
    setAim(null);
  };

  const handlePointerOver = (event: PointerEvent) => {
    // Touch aiming is driven by pointerdown/move; hover-capable pointers aim here.
    if (event.pointerType === 'touch') return;
    setAim(cellFromEvent(event));
  };

  const handlePointerUp = (event: PointerEvent) => {
    if (event.pointerType === 'touch') {
      const current = gesture.current;
      suppressPointerClick.current = !current
        || current.pointerId !== event.pointerId
        || current.held
        || current.moved
        || !sameCell(cellFromEvent(event), current.start);
      endGesture();
    }
    setAim(null);
  };

  const rows: JSX.Element[] = [];
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    const cells: JSX.Element[] = [];
    for (let column = 0; column < BOARD_SIZE; column += 1) {
      const cell = { row, column };
      const key = cellKey(cell);
      const stone = stones.get(key);
      const move = moves.get(key);
      const isLast =
        position?.last_move != null && cellKey(position.last_move) === key;
      const isFocus = sameCell(focus, cell);
      const isAim = sameCell(aim, cell);
      const probability = move ? overlayValue(move) : 0;
      const markerSize =
        topValue > 0 ? 12 + 75 * Math.sqrt(probability / topValue) : 0;
      const visitShare = move && snapshot ? visitFraction(move, snapshot) : 0;
      const moveProbability =
        recordedSampling === undefined
          ? moveProbabilities.get(key) ?? 0
          : recordedSampling?.get(key) ?? null;
      const isLeading =
        showPolicy && move != null && topKey != null && key === topKey;
      const tooltipId = `cell-tip-${row}-${column}`;
      const canInspect = !stone;
      const priorRowLabel = policyLabel ?? 'Network prior';
      const probabilityLabel = 'Move probability';
      const probabilityValue =
        recordedSampling === undefined
          ? percent(moveProbability)
          : moveProbability == null
            ? 'Unavailable'
            : percent(moveProbability);
      const tooltipRows: TooltipRow[] | null = move
        ? [
            { label: priorRowLabel, value: percent(move.prior) },
            {
              label: 'Visits',
              value: (
                <>
                  {move.visits.toLocaleString()} /{' '}
                  {(snapshot?.total_visits ?? 0).toLocaleString()} ·{' '}
                  {percent(visitShare)}
                </>
              ),
            },
            ...(hideMoveProbability
              ? []
              : [
                  {
                    label: probabilityLabel,
                    value: (
                      <>
                        {probabilityValue}{' '}
                        {recordedSampling === undefined && (
                          <small>at T={temperature.toFixed(2)}</small>
                        )}
                      </>
                    ),
                  },
                ]),
            { label: 'Action value Q', value: signed(move.mean_value, 3) },
          ]
        : null;
      const label = stone
        ? `${moveName(cell)}, ${stone.color}${isLast ? ', last move' : ''}`
        : move && showPolicy
          ? `${moveName(cell)}${isLeading ? ', leading move' : ''}, ${priorRowLabel.toLowerCase()} ${percent(move.prior)}, visit fraction ${percent(visitShare)}, move probability ${moveProbability == null ? 'unavailable' : percent(moveProbability)}`
          : `${moveName(cell)}, empty`;

      cells.push(
        <button
          aria-describedby={showPolicy && move ? tooltipId : undefined}
          aria-disabled={!canInspect}
          aria-label={label}
          class={`board-cell${isFocus ? ' is-focus' : ''}${isAim ? ' is-aim' : ''}`}
          data-col={column}
          data-row={row}
          key={key}
          onClick={(event) => handleCellClick(cell, event)}
          onKeyDown={(event) => handleCellKeyDown(cell, event)}
          role="gridcell"
          tabIndex={
            activeCell.row === row && activeCell.column === column ? 0 : -1
          }
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
                opacity: 0.38 + 0.56 * (probability / topValue),
                width: `${markerSize}%`,
              }}
            />
          )}
          {isLeading && probability > 0 && (
            <span aria-hidden="true" className="rank-badge">
              1
            </span>
          )}
          {showPolicy && !stone && tooltipRows && (
            <span
              class={`cell-tooltip${column <= 3 ? ' align-left' : column >= 15 ? ' align-right' : ''}${row <= 2 ? ' below' : ''}`}
              id={tooltipId}
              role="tooltip"
            >
              <strong>{moveName(cell)}</strong>
              {tooltipRows.map((tooltipRow) => (
                <span key={tooltipRow.label}>
                  <em>{tooltipRow.label}</em>
                  <b>{tooltipRow.value}</b>
                </span>
              ))}
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
        {Array.from({ length: BOARD_SIZE }, (_, index) => BOARD_SIZE - index).map(
          (rank) => (
            <span key={rank}>{rank}</span>
          ),
        )}
      </div>
      <div
        className="board-surface"
        onContextMenu={(event) => event.preventDefault()}
        onPointerCancel={handlePointerCancel}
        onPointerDown={handlePointerDown}
        onPointerLeave={() => setAim(null)}
        onPointerMove={handlePointerMove}
        onPointerOver={handlePointerOver}
        onPointerUp={handlePointerUp}
      >
        {aim && (
          <>
            <span
              aria-hidden="true"
              className="board-aim board-aim-row"
              style={{ insetBlockStart: `${(aim.row * 100) / BOARD_SIZE}%` }}
            />
            <span
              aria-hidden="true"
              className="board-aim board-aim-column"
              style={{ insetInlineStart: `${(aim.column * 100) / BOARD_SIZE}%` }}
            />
          </>
        )}
        {canPlay && aim && !stones.has(cellKey(aim)) && (
          <span
            aria-hidden="true"
            className={`board-ghost stone-${position?.to_move === 'white' ? 'white' : 'black'}`}
            style={{
              insetBlockStart: `${(aim.row * 100) / BOARD_SIZE}%`,
              insetInlineStart: `${(aim.column * 100) / BOARD_SIZE}%`,
            }}
          />
        )}
        <div
          aria-label={`19 by 19 Gomoku board${showPolicy ? ' with policy overlay' : ''}${canPlay ? '; activate an empty intersection to place a stone' : ''}`}
          className="board"
          ref={gridRef}
          role="grid"
        >
          {rows}
        </div>
      </div>
      {footer != null && <div className="board-footer">{footer}</div>}
    </div>
  );
}
