import type { JSX } from "preact";
import { useState } from "preact/hooks";
import { inspectKey, pointerX, usePlotWidth } from "./InteractiveChart";
import { nearestIndex } from "./experimentChartData";
import { scoreInterval } from "./experimentConfidence";
import {
  comparisonScore,
  settingsKey,
  settingsLabel,
  type ComparisonResult,
  type TrainingRun,
} from "./useOverview";

export type Point = {
  result: ComparisonResult;
  /** Which participant of the match is the moving checkpoint. */
  movingSide: "first" | "second";
  epoch: number;
  score: number;
  low: number;
  high: number;
};
export type Series = {
  key: string;
  label: string;
  baseline: string;
  /** Human form of the shared search settings, empty when unknown. */
  settings: string;
  points: Point[];
};

/** Groups completed comparisons into series: the checkpoint that recurs most
 *  across matches is treated as the fixed baseline, the other side as the
 *  moving checkpoint plotted by its epoch. Continuations of a run share a
 *  series with their source run so one line follows the whole lineage. */
export function strengthSeries(
  comparisons: ComparisonResult[],
  runs: TrainingRun[] = [],
): Series[] {
  const sources = new Map(runs.map((run) => [run.id, run.source?.job_id ?? null]));
  const root = (jobId: string): string => {
    let current = jobId;
    const seen = new Set<string>();
    while (!seen.has(current)) {
      seen.add(current);
      const parent = sources.get(current);
      if (!parent) break;
      current = parent;
    }
    return current;
  };
  const occurrences = new Map<string, number>();
  for (const result of comparisons)
    for (const side of ["first", "second"] as const) {
      const sha = result[side].model_sha256;
      if (sha) occurrences.set(sha, (occurrences.get(sha) ?? 0) + 1);
    }
  const groups = new Map<string, Series>();
  for (const result of comparisons) {
    const firstCount = occurrences.get(result.first.model_sha256 ?? "") ?? 0;
    const secondCount = occurrences.get(result.second.model_sha256 ?? "") ?? 0;
    const baselineSide = firstCount > secondCount ? "first" : "second";
    const movingSide = baselineSide === "first" ? "second" : "first";
    const moving = result[movingSide],
      baseline = result[baselineSide];
    if (moving.epoch == null || !moving.job_id) continue;
    // Matches with different search settings are different experiments.
    const key = `${root(moving.job_id)}|${baseline.model_sha256}|${settingsKey(result.settings)}`;
    const counts =
      movingSide === "first"
        ? { wins: result.wins, draws: result.draws, losses: result.losses }
        : { wins: result.losses, draws: result.draws, losses: result.wins };
    const interval = scoreInterval(counts, 95);
    const score = comparisonScore(result, movingSide);
    const series =
      groups.get(key) ??
      {
        key,
        label: moving.job_title ?? moving.label ?? "Checkpoint",
        baseline: baseline.label ?? baseline.job_title ?? "baseline",
        settings: settingsLabel(result.settings),
        points: [],
      };
    series.points.push({
      result,
      movingSide,
      epoch: moving.epoch + 1,
      score,
      low: interval?.low ?? score,
      high: interval?.high ?? score,
    });
    groups.set(key, series);
  }
  return [...groups.values()]
    .map((series) => {
      const points = series.points.sort((a, b) => a.epoch - b.epoch);
      // Name the lineage after its latest continuation.
      const latest = points.at(-1)!;
      const latestSide =
        latest.result.first.epoch != null && latest.result.first.epoch + 1 === latest.epoch
          ? latest.result.first
          : latest.result.second;
      return { ...series, points, label: latestSide.job_title ?? series.label };
    })
    .sort((a, b) => epochSpan(b) - epochSpan(a) || b.points.length - a.points.length);
}

/** Number of distinct epochs a series covers; the reading sentence and the
 *  legend order prefer series that actually move across epochs. */
export function epochSpan(series: Series): number {
  return new Set(series.points.map((p) => p.epoch)).size;
}

export const SERIES_COLORS = [
  "var(--ds-chart-series-1)",
  "var(--ds-chart-series-4)",
  "var(--ds-chart-series-2)",
  "var(--ds-chart-series-3)",
  "var(--ds-chart-series-5)",
];
const percent = (n: number) => `${(n * 100).toFixed(1)}%`;

/** One lineage against one baseline: score by checkpoint epoch with 95%
 *  intervals, a hover/keyboard readout, and a one-line reading. */
export function StrengthChart({
  series,
  color = SERIES_COLORS[0]!,
}: {
  series: Series;
  color?: string;
}): JSX.Element {
  const { ref, width } = usePlotWidth();
  const [cursor, setCursor] = useState<number | null>(null);
  const points = series.points;
  if (!points.length) return <p className="experiment-empty">No matches.</p>;
  const epochs = points.map((p) => p.epoch);
  const xmin = Math.min(...epochs) - 2,
    xmax = Math.max(...epochs) + 2;
  const ymin = Math.max(0, Math.min(0.45, ...points.map((p) => p.low)) - 0.05),
    ymax = Math.min(1, Math.max(0.55, ...points.map((p) => p.high)) + 0.05);
  const left = 52,
    right = width - 16,
    top = 14,
    bottom = 190;
  const x = (epoch: number) =>
    left + ((epoch - xmin) / Math.max(xmax - xmin, 1)) * (right - left);
  const y = (score: number) =>
    bottom - ((score - ymin) / Math.max(ymax - ymin, 1e-6)) * (bottom - top);
  const ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].filter((t) => t >= ymin && t <= ymax);
  const labelledEpochs: number[] = [];
  for (const epoch of [...new Set(epochs)].sort((a, b) => a - b)) {
    const previous = labelledEpochs.at(-1);
    if (previous == null || x(epoch) - x(previous) >= 30) labelledEpochs.push(epoch);
  }
  const selectedIndex = Math.min(cursor ?? points.length - 1, points.length - 1);
  const selected = points[selectedIndex]!;
  const first = points[0]!,
    last = points.at(-1)!;
  function inspect(
    event:
      | JSX.TargetedPointerEvent<SVGSVGElement>
      | JSX.TargetedMouseEvent<SVGSVGElement>,
  ) {
    setCursor(nearestIndex(points.map((p) => x(p.epoch)), pointerX(event)));
  }
  const parity =
    last.low <= 0.5 && last.high >= 0.5
      ? "the latest interval still includes parity"
      : last.low > 0.5
        ? "the latest interval is above parity"
        : "the latest interval is below parity";
  return (
    <figure ref={ref} className="experiment-chart interactive-chart service-strength" style={{ color }}>
      <div className="chart-readout">
        <strong>epoch {selected.epoch}</strong>
        <span className="chart-training-value" style={{ borderColor: color }}>
          score <b>{percent(selected.score)}</b>
        </span>
        <span>
          95% interval {percent(selected.low)} – {percent(selected.high)}
        </span>
        <span>
          {selected.result.wins}–{selected.result.draws}–{selected.result.losses} over{" "}
          {selected.result.games.toLocaleString()} games
        </span>
        <a href={`/experiments?job=${selected.result.id}`}>open match</a>
      </div>
      <svg
        className="interactive-plot service-strength-plot"
        role="img"
        aria-label={`${series.label} against ${series.baseline}: score by checkpoint epoch with 95% intervals`}
        viewBox={`0 0 ${width} 218`}
        onPointerMove={inspect}
        onClick={inspect}
        tabindex={0}
        onKeyDown={(event) => inspectKey(event, selectedIndex, points.length, setCursor)}
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="chart-grid" x1={left} x2={right} y1={y(tick)} y2={y(tick)} />
            <text x={left - 10} y={y(tick) + 4} text-anchor="end">
              {Math.round(tick * 100)}%
            </text>
          </g>
        ))}
        <line className="chart-parity" x1={left} x2={right} y1={y(0.5)} y2={y(0.5)} />
        <text x={left + 6} y={y(0.5) - 6}>
          parity with the baseline
        </text>
        {points.length > 1 && (
          <polyline
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            points={points.map((p) => `${x(p.epoch)},${y(p.score)}`).join(" ")}
          />
        )}
        {points.map((p) => (
          <g key={p.result.id}>
            <line stroke="currentColor" stroke-width="2" x1={x(p.epoch)} x2={x(p.epoch)} y1={y(p.high)} y2={y(p.low)} />
            <line stroke="currentColor" stroke-width="2" x1={x(p.epoch) - 5} x2={x(p.epoch) + 5} y1={y(p.high)} y2={y(p.high)} />
            <line stroke="currentColor" stroke-width="2" x1={x(p.epoch) - 5} x2={x(p.epoch) + 5} y1={y(p.low)} y2={y(p.low)} />
            <circle fill="currentColor" cx={x(p.epoch)} cy={y(p.score)} r={p.result.id === selected.result.id ? 5.5 : 4} />
          </g>
        ))}
        {labelledEpochs.map((epoch) => (
          <text key={epoch} x={x(epoch)} y={bottom + 18} text-anchor="middle">
            E{epoch}
          </text>
        ))}
        <line className="chart-crosshair" x1={x(selected.epoch)} x2={x(selected.epoch)} y1={top} y2={bottom} />
        <rect className="chart-hit-area" x={left} y={top} width={Math.max(0, right - left)} height={bottom - top} />
      </svg>
      <figcaption className="service-reading">
        <strong>Reading:</strong>{" "}
        {first.epoch === last.epoch
          ? `${percent(last.score)} at epoch ${last.epoch}`
          : `from ${percent(first.score)} at epoch ${first.epoch} to ${percent(last.score)} at epoch ${last.epoch}`}{" "}
        against {series.baseline}; {parity}. Hover, tap, or focus the plot and use arrow keys
        to inspect a match.
      </figcaption>
    </figure>
  );
}
