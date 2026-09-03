interface ChartMove {
  row: number;
  column: number;
  visits: number;
}

function moveKey(move: ChartMove): string {
  return `${move.row}:${move.column}`;
}

export const CHART_SERIES_LIMIT = 5;

export function leadingMoves<T extends ChartMove>(moves: readonly T[]): T[] {
  return [...moves]
    .sort((left, right) => right.visits - left.visits)
    .slice(0, CHART_SERIES_LIMIT);
}

/**
 * Every move that led a snapshot at any point, plus the latest leading pack.
 * Keeps the "search changed its mind" story visible even after a leader falls
 * behind. Ordered by standing in the latest snapshot.
 */
export function trackedMoves<T extends ChartMove>(snapshots: readonly (readonly T[])[]): T[] {
  const byKey = new Map<string, T>();
  const consider = (move: T) => {
    const key = moveKey(move);
    if (!byKey.has(key)) byKey.set(key, move);
  };
  for (const moves of snapshots) {
    let leader: T | undefined;
    for (const move of moves) {
      if (leader === undefined || move.visits > leader.visits) leader = move;
    }
    if (leader) consider(leader);
  }
  const latest = snapshots[snapshots.length - 1];
  const latestVisits = new Map(
    (latest ?? []).map((move) => [moveKey(move), move.visits] as const),
  );
  if (latest) for (const move of leadingMoves(latest)) consider(move);
  return [...byKey.values()].sort(
    (left, right) => (latestVisits.get(moveKey(right)) ?? 0) - (latestVisits.get(moveKey(left)) ?? 0),
  );
}

export function assignSeriesColorIndexes(
  existing: ReadonlyMap<string, number>,
  candidates: ChartMove[],
): Map<string, number> {
  const result = new Map(existing);
  for (const candidate of candidates) {
    const key = moveKey(candidate);
    if (!result.has(key)) result.set(key, result.size);
  }
  return result;
}

const SERIES_TOKENS = [
  'var(--ds-chart-series-1)',
  'var(--ds-chart-series-2)',
  'var(--ds-chart-series-3)',
  'var(--ds-chart-series-4)',
  'var(--ds-chart-series-5)',
];

export function seriesColor(index: number): string {
  const token = SERIES_TOKENS[index];
  if (token) return token;
  const hue = (215 + index * 137.508) % 360;
  return `light-dark(hsl(${hue.toFixed(2)} 68% 42%), hsl(${hue.toFixed(2)} 72% 70%))`;
}

let registryPositionId: number | null = null;
let registryIndexes = new Map<string, number>();

/** Stable per-position color indexes, shared by the chart and the candidate table. */
export function seriesColorIndexes(
  positionId: number,
  candidates: readonly ChartMove[],
): ReadonlyMap<string, number> {
  if (registryPositionId !== positionId) {
    registryPositionId = positionId;
    registryIndexes = new Map();
  }
  registryIndexes = assignSeriesColorIndexes(registryIndexes, [...candidates]);
  return registryIndexes;
}
