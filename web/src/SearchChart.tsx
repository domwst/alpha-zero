import type { JSX } from 'preact';
import { useState } from 'preact/hooks';

import { seriesColor, seriesColorIndexes, trackedMoves } from './chartSeries';
import { percent } from './format';
import {
  type MoveStats,
  type SearchSnapshotMessage,
  cellKey,
  moveName,
  visitFraction,
} from './protocol';

interface SearchChartProps {
  snapshots: SearchSnapshotMessage[];
  selectedIndex: number | null;
  onSelectIndex: (index: number | null) => void;
}

interface HoverSlice {
  index: number;
  viewX: number;
  tooltipX: number;
  placeBefore: boolean;
}

const WIDTH = 1000;
const HEIGHT = 260;
const MARGIN = { top: 18, right: 20, bottom: 48, left: 64 };

function compactNumber(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(value >= 10_000 ? 0 : 1)}k` : String(value);
}

function elapsed(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds.toLocaleString()} ms`;
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

export function SearchChart({
  snapshots,
  selectedIndex,
  onSelectIndex,
}: SearchChartProps): JSX.Element {
  const [hoverSlice, setHoverSlice] = useState<HoverSlice | null>(null);

  if (snapshots.length === 0) {
    return (
      <div className="chart-empty">
        Search snapshots will appear here as the network explores this position.
      </div>
    );
  }

  const latest = snapshots[snapshots.length - 1]!;
  const inspectedIndex = selectedIndex ?? snapshots.length - 1;
  const inspected = snapshots[inspectedIndex] ?? latest;
  const hovered = hoverSlice === null ? null : (snapshots[hoverSlice.index] ?? null);
  const candidates = trackedMoves(snapshots.map((snapshot) => snapshot.moves));
  const colorIndexes = seriesColorIndexes(latest.position_id, candidates);
  const colorFor = (candidate: MoveStats, fallbackIndex: number): string =>
    seriesColor(colorIndexes.get(cellKey(candidate)) ?? fallbackIndex);
  const maximumNodes = Math.max(1, latest.target_simulations, latest.searched_simulations);
  const maximumShare = Math.max(
    0.25,
    ...snapshots.flatMap((snapshot) =>
      candidates.map((candidate) => {
        const move = snapshot.moves.find((entry) => cellKey(entry) === cellKey(candidate));
        return move ? visitFraction(move, snapshot) : 0;
      }),
    ),
    ...candidates.map((candidate) => candidate.prior),
  );
  const yMaximum = Math.min(1, Math.ceil(maximumShare * 10) / 10);
  const innerWidth = WIDTH - MARGIN.left - MARGIN.right;
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  const x = (nodes: number) => MARGIN.left + (nodes / maximumNodes) * innerWidth;
  const y = (share: number) => MARGIN.top + (1 - share / yMaximum) * innerHeight;
  const inspectedX = x(inspected.searched_simulations);
  const hoveredX = hovered ? hoverSlice?.viewX ?? null : null;
  const leading = candidates[0];
  const runnerUp = candidates[1];
  const latestShare = (candidate: typeof leading): number => {
    if (!candidate) return 0;
    return visitFraction(candidate, latest);
  };

  const sliceFromPointer = (event: {
    clientX: number;
    clientY: number;
    currentTarget: SVGSVGElement;
  }): HoverSlice => {
    const svg = event.currentTarget;
    const bounds = svg.getBoundingClientRect();
    const matrix = svg.getScreenCTM();
    let viewX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
    if (matrix) {
      const pointer = svg.createSVGPoint();
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      viewX = pointer.matrixTransform(matrix.inverse()).x;
    }
    viewX = Math.max(MARGIN.left, Math.min(WIDTH - MARGIN.right, viewX));
    const targetNodes = Math.max(
      0,
      Math.min(maximumNodes, ((viewX - MARGIN.left) / innerWidth) * maximumNodes),
    );
    let nearestIndex = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;
    snapshots.forEach((snapshot, index) => {
      const distance = Math.abs(snapshot.searched_simulations - targetNodes);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = index;
      }
    });
    const wrapBounds = svg.parentElement?.getBoundingClientRect() ?? bounds;
    const tooltipX = Math.max(
      0,
      Math.min(wrapBounds.width, event.clientX - wrapBounds.left),
    );
    return {
      index: nearestIndex,
      viewX,
      tooltipX,
      placeBefore: tooltipX > wrapBounds.width / 2,
    };
  };

  return (
    <figure className="search-figure">
      <div className="chart-legend" aria-label="Leading move series and inspected values">
        {candidates.map((candidate, index) => {
          const move = inspected.moves.find((entry) => cellKey(entry) === cellKey(candidate));
          const share = move ? visitFraction(move, inspected) : 0;
          const color = colorFor(candidate, index);
          return (
            <span key={cellKey(candidate)}>
              <i style={{ background: color }} />
              {moveName(candidate)}
              <b>{percent(share)}</b>
            </span>
          );
        })}
        <span className="legend-note">Dashed line: raw network prior</span>
      </div>
      <div className="chart-wrap">
        <svg
          aria-describedby="search-chart-reading"
          aria-label="Root visit fraction as search simulations accumulate"
          className="chart"
          onClick={(event) => {
            const slice = sliceFromPointer(event);
            onSelectIndex(slice.index === snapshots.length - 1 ? null : slice.index);
          }}
          onPointerLeave={() => setHoverSlice(null)}
          onPointerMove={(event) => setHoverSlice(sliceFromPointer(event))}
          role="img"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const share = yMaximum * ratio;
            return (
              <g key={ratio}>
                <line
                  className="chart-grid"
                  x1={MARGIN.left}
                  x2={WIDTH - MARGIN.right}
                  y1={y(share)}
                  y2={y(share)}
                />
                <text className="chart-label" textAnchor="end" x={MARGIN.left - 10} y={y(share) + 4}>
                  {Math.round(share * 100)}%
                </text>
              </g>
            );
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const nodes = maximumNodes * ratio;
            return (
              <g key={ratio}>
                <line
                  className="chart-grid"
                  x1={x(nodes)}
                  x2={x(nodes)}
                  y1={MARGIN.top}
                  y2={HEIGHT - MARGIN.bottom}
                />
                <text
                  className="chart-label"
                  textAnchor={ratio === 0 ? 'start' : ratio === 1 ? 'end' : 'middle'}
                  x={x(nodes)}
                  y={HEIGHT - 20}
                >
                  {compactNumber(Math.round(nodes))}
                </text>
              </g>
            );
          })}
          <text
            className="chart-axis-title"
            textAnchor="middle"
            transform={`rotate(-90 15 ${MARGIN.top + innerHeight / 2})`}
            x="15"
            y={MARGIN.top + innerHeight / 2}
          >
            Visit fraction
          </text>
          <text
            className="chart-axis-title"
            textAnchor="middle"
            x={MARGIN.left + innerWidth / 2}
            y={HEIGHT - 3}
          >
            Search simulations
          </text>
          {candidates.map((candidate, index) => (
            <line
              className="chart-prior"
              key={`prior-${cellKey(candidate)}`}
              stroke={colorFor(candidate, index)}
              x1={MARGIN.left}
              x2={WIDTH - MARGIN.right}
              y1={y(candidate.prior)}
              y2={y(candidate.prior)}
            />
          ))}
          {candidates.map((candidate, index) => {
            const points = snapshots.map((snapshot) => {
              const move = snapshot.moves.find((entry) => cellKey(entry) === cellKey(candidate));
              const share = move ? visitFraction(move, snapshot) : 0;
              return `${x(snapshot.searched_simulations)},${y(share)}`;
            });
            const finalShare = latestShare(candidate);
            const color = colorFor(candidate, index);
            return (
              <g key={cellKey(candidate)}>
                <polyline
                  fill="none"
                  points={points.join(' ')}
                  stroke={color}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="3"
                />
                <circle
                  className="chart-endpoint"
                  cx={x(latest.searched_simulations)}
                  cy={y(finalShare)}
                  fill={color}
                  r="4"
                />
              </g>
            );
          })}
          <line
            className="chart-cursor"
            x1={inspectedX}
            x2={inspectedX}
            y1={MARGIN.top}
            y2={HEIGHT - MARGIN.bottom}
          />
          {hovered && hoveredX !== null && (
            <g className="chart-hover-slice">
              <line
                x1={hoveredX}
                x2={hoveredX}
                y1={MARGIN.top}
                y2={HEIGHT - MARGIN.bottom}
              />
              {candidates.map((candidate, index) => {
                const move = hovered.moves.find((entry) => cellKey(entry) === cellKey(candidate));
                const share = move ? visitFraction(move, hovered) : 0;
                return (
                  <circle
                    cx={hoveredX}
                    cy={y(share)}
                    fill={colorFor(candidate, index)}
                    key={cellKey(candidate)}
                    r="5"
                  />
                );
              })}
            </g>
          )}
        </svg>
        {hovered && hoveredX !== null && (
          <div
            className={`chart-tooltip${hoverSlice?.placeBefore ? ' chart-tooltip-before' : ''}`}
            role="tooltip"
            style={{ insetInlineStart: `${hoverSlice?.tooltipX ?? 0}px` }}
          >
            <div className="chart-tooltip-heading">
              <strong>{hovered.searched_simulations.toLocaleString()} simulations</strong>
              <span>{elapsed(hovered.elapsed_ms)} · {Math.round(hovered.simulations_per_second).toLocaleString()} sims/s</span>
            </div>
            <ul>
              {candidates.map((candidate, index) => {
                const move = hovered.moves.find((entry) => cellKey(entry) === cellKey(candidate));
                const share = move ? visitFraction(move, hovered) : 0;
                const color = colorFor(candidate, index);
                return (
                  <li key={cellKey(candidate)}>
                    <i style={{ background: color }} />
                    <span>{moveName(candidate)}</span>
                    <strong>{percent(share)}</strong>
                    <small>{(move?.visits ?? 0).toLocaleString()} visits</small>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
      <div className="scrubber">
        <label htmlFor="snapshot-scrubber">Inspect search snapshot</label>
        <input
          id="snapshot-scrubber"
          max={snapshots.length - 1}
          min={0}
          onInput={(event) => {
            const index = Number(event.currentTarget.value);
            onSelectIndex(index === snapshots.length - 1 ? null : index);
          }}
          step={1}
          type="range"
          value={inspectedIndex}
        />
        <output>{inspected.searched_simulations.toLocaleString()} simulations</output>
        <button
          className="text-button"
          disabled={selectedIndex === null}
          onClick={() => onSelectIndex(null)}
          type="button"
        >
          Live
        </button>
      </div>
      <figcaption id="search-chart-reading">
        <strong>Reading:</strong>{' '}
        {leading
          ? `${moveName(leading)} leads at ${percent(latestShare(leading))} after ${latest.searched_simulations.toLocaleString()} simulations${runnerUp ? `; ${moveName(runnerUp)} follows at ${percent(latestShare(runnerUp))}` : ''}.`
          : 'No root move has accumulated visits yet.'}
        {' '}Dashed lines mark each move’s raw policy prior. Hover for an exact vertical slice; click to keep that snapshot selected.
      </figcaption>
    </figure>
  );
}
