import type { JSX } from 'preact';
import { useState } from 'preact/hooks';
import { inspectKey, pointerX, usePlotWidth } from './InteractiveChart';
import { nearestIndex } from './experimentChartData';

export type SelfPlayStats = { games: number; average_game_length: number | null;
  first_player_win_rate: number | null; first_player_wins: number | null;
  second_player_wins: number | null; draws: number | null };
type Epoch = { epoch: number; self_play?: SelfPlayStats | null; self_play_seconds?: number | null; scheduled_learning_rate?: number | null;
  training: { samples?: number; batches?: number; duration_seconds?: number } };
export const selfPlayMetricLabels = { average_game_length: 'Average game length', first_player_win_rate: 'First-player win rate',
  self_play_seconds: 'Self-play duration', training_seconds: 'Training duration', training_samples: 'Training samples', learning_rate: 'Learning rate', lr_steps: 'LR × steps' };
export type SelfPlayMetric = keyof typeof selfPlayMetricLabels;
const duration = (n: number) => {
  const tenths = Math.round(n * 10);
  const hours = Math.floor(tenths / 36000), minutes = Math.floor(tenths % 36000 / 600), seconds = (tenths % 600 / 10).toFixed(1);
  return `${hours ? `${hours}h ` : ''}${hours || minutes ? `${minutes}m ` : ''}${seconds}s`;
};
const lrSteps = (lr: number | null | undefined, steps: number | undefined) =>
  lr != null && Number.isFinite(lr) && lr >= 0 && steps != null && Number.isSafeInteger(steps) && steps >= 0
    && Number.isFinite(lr * steps) ? lr * steps : null;
const metricValue = (row: Epoch, metric: SelfPlayMetric) => metric === 'self_play_seconds' ? row.self_play_seconds
  : metric === 'learning_rate' ? row.scheduled_learning_rate
  : metric === 'lr_steps' ? lrSteps(row.scheduled_learning_rate, row.training.batches)
  : metric === 'training_seconds' ? row.training.duration_seconds : metric === 'training_samples' ? row.training.samples : row.self_play?.[metric];
const decimal = (n: number | null | undefined, digits = 2) => n == null ? '—' : n.toFixed(digits);

export function SelfPlayMetrics({ metrics, metric }: { metrics: Epoch[]; metric?: SelfPlayMetric }): JSX.Element {
  const epochs = metrics.filter((row): row is Epoch & { self_play: SelfPlayStats } => !!row.self_play);
  return <section className="self-play-metrics" aria-label="Self-play epoch statistics">
    {metric && <GameMetricChart key={metric} epochs={epochs} metric={metric} />}
    <details className="chart-data"><summary>Exact epoch statistics</summary><div className="chart-data-scroll"><table className="experiment-table">
      <thead><tr><th>Epoch</th><th>Games</th><th>Self-play duration</th><th>Training duration</th><th>Training samples</th><th>Optimizer steps</th><th>Learning rate</th><th>LR × steps</th><th>Average moves</th><th>First-player wins</th><th>Second-player wins</th><th>Draws</th><th>First-player win rate</th></tr></thead>
      <tbody>{epochs.map(({ epoch, self_play: s, training, self_play_seconds, scheduled_learning_rate }) => <tr key={epoch}><td>{epoch}</td><td>{s.games.toLocaleString()}</td>
        <td>{self_play_seconds == null ? '—' : duration(self_play_seconds)}</td>
        <td>{training.duration_seconds == null ? '—' : duration(training.duration_seconds)}</td><td>{training.samples?.toLocaleString() ?? '—'}</td>
        <td>{training.batches?.toLocaleString() ?? '—'}</td>
        <td>{scheduled_learning_rate == null ? '—' : String(scheduled_learning_rate)}</td>
        <td>{lrSteps(scheduled_learning_rate, training.batches) ?? '—'}</td>
        <td>{decimal(s.average_game_length, 3)}</td><td>{s.first_player_wins ?? '—'}</td><td>{s.second_player_wins ?? '—'}</td><td>{s.draws ?? '—'}</td>
        <td>{s.first_player_win_rate == null ? '—' : `${(s.first_player_win_rate * 100).toFixed(1)}%`}</td></tr>)}</tbody>
    </table></div>
      <p className="experiment-muted">Learning rate is the recorded rate used for that epoch's optimizer updates. LR × steps multiplies it by the recorded optimizer batch count, including the final partial batch. Training samples include all eight board symmetries of each replay position. Training duration covers batch processing and optimizer updates; it excludes replay encoding, checkpoint saving, and image rendering. Self-play duration covers game generation. Game lengths count individual moves (plies); first-player win rate includes draws in the denominator.</p>
    </details>
  </section>;
}

function GameMetricChart({ epochs, metric }: { epochs: (Epoch & { self_play: SelfPlayStats })[]; metric: SelfPlayMetric }): JSX.Element {
  const { ref, width } = usePlotWidth();
  const [cursor, setCursor] = useState<number | null>(null);
  const points = epochs.flatMap(row => {
    const n = metricValue(row, metric);
    return n == null || !Number.isFinite(n) ? [] : [{ epoch: row.epoch, n, games: row.self_play.games,
      learningRate: row.scheduled_learning_rate, steps: row.training.batches }];
  });
  const rate = metric === 'first_player_win_rate';
  const label = selfPlayMetricLabels[metric];
  const timing = metric === 'self_play_seconds' || metric === 'training_seconds';
  const samples = metric === 'training_samples';
  const learningRate = metric === 'learning_rate';
  const lrProduct = metric === 'lr_steps';
  const format = (n: number) => learningRate || lrProduct ? Number(n.toPrecision(8)).toString() : rate ? `${(n * 100).toFixed(1)}%` : timing ? duration(n) : samples ? `${n.toLocaleString()} samples` : `${n.toFixed(3)} moves`;
  const tickLabel = (n: number) => learningRate ? (n === 0 ? '0' : n.toExponential(1)) : lrProduct ? Number(n.toPrecision(3)).toString() : rate ? `${(n * 100).toFixed(0)}%` : timing ? `${(n / 60).toFixed(1)}m` : samples ? `${(n / 1000).toFixed(1)}k` : n.toFixed(1);
  const last = points.at(-1);
  const selectedIndex = Math.min(cursor ?? points.length - 1, points.length - 1);
  const selected = points[selectedIndex];
  const firstEpoch = epochs[0]?.epoch ?? 1, lastEpoch = epochs.at(-1)?.epoch ?? firstEpoch;
  const high = rate ? 1 : Math.max(learningRate || lrProduct ? 1e-12 : 1, ...points.map(p => p.n)) * 1.12;
  const ticks = [0, 1, 2, 3, 4].map(tick => {
    const n = high * tick / 4;
    return { n, label: tickLabel(n) };
  });
  // Allow 8px per glyph for the 12px monospace labels, plus the axis gap and inset.
  const left = Math.max(55, ...ticks.map(tick => tick.label.length * 8 + 20)), right = width - 18;
  const x = (epoch: number) => left + (epoch - firstEpoch) / Math.max(lastEpoch - firstEpoch, 1) * (right - left);
  const y = (n: number) => 205 - n / high * 175;
  function inspect(event: JSX.TargetedPointerEvent<SVGSVGElement> | JSX.TargetedMouseEvent<SVGSVGElement>) {
    setCursor(nearestIndex(points.map(p => x(p.epoch)), pointerX(event)));
  }
  // Separate segments prevent a missing epoch from being presented as observed data.
  const segments: typeof points[] = [];
  for (const point of points) {
    const previous = segments.at(-1)?.at(-1);
    if (!previous || point.epoch !== previous.epoch + 1) segments.push([]);
    segments.at(-1)!.push(point);
  }
  return <figure ref={ref} className="experiment-chart interactive-chart">
    {!selected || !last ? <p className="experiment-empty">{label} statistics are not available yet.</p> : <>
      <div className="chart-readout"><strong>Epoch {selected.epoch}</strong><span className="chart-training-value">{label} <b>{format(selected.n)}</b></span>
        <span>{lrProduct ? <>LR {format(selected.learningRate!)} × {selected.steps!.toLocaleString()} steps</> : <>{selected.games.toLocaleString()} games</>}</span></div>
      <svg className="interactive-plot" role="img" aria-label={`${label} by self-play epoch`} viewBox={`0 0 ${width} 245`}
        onPointerMove={inspect} onClick={inspect} tabindex={0} onKeyDown={event => inspectKey(event, selectedIndex, points.length, setCursor)}>
        {ticks.map(({ n, label }, tick) => <g key={tick}>
          <line className="chart-grid" x1={left} x2={right} y1={y(n)} y2={y(n)} />
          <text x={left - 10} y={y(n) + 4} text-anchor="end">{label}</text></g>)}
        {segments.map((segment, i) => <polyline key={i} className="train-line" fill="none" points={segment.map(p => `${x(p.epoch)},${y(p.n)}`).join(' ')} />)}
        {points.map(p => <circle key={p.epoch} className="train-point" cx={x(p.epoch)} cy={y(p.n)} r="2.5" />)}
        <line className="chart-guide train-line" x1={left} x2={right} y1={y(selected.n)} y2={y(selected.n)} />
        <line className="chart-crosshair" x1={x(selected.epoch)} x2={x(selected.epoch)} y1="30" y2="205" />
        <circle className="train-point" cx={x(selected.epoch)} cy={y(selected.n)} r="4" />
        <text x={left} y="229">Epoch {firstEpoch}</text><text x={right} y="229" text-anchor="end">Epoch {lastEpoch}</text>
        <rect className="chart-hit-area" x={left} y="25" width={right-left} height="185" />
      </svg>
      <p className="chart-help">Hover, tap, or focus the plot and use arrow keys to inspect an epoch.</p>
      <figcaption><strong>Reading:</strong> Epoch {last.epoch}: {format(last.n)} {lrProduct ? <>from learning rate {format(last.learningRate!)} × {last.steps!.toLocaleString()} optimizer steps</> : learningRate ? 'learning rate used for optimizer updates' : samples ? 'used for training, including all eight board symmetries per replay position' : timing ? `spent in ${metric === 'self_play_seconds' ? 'self-play' : 'training (batch processing and optimizer updates)'}` : rate ? 'first-player win rate' : 'per game on average'}. {!learningRate && !lrProduct && <>{last.games.toLocaleString()} self-play games were generated.</>}</figcaption>
    </>}
    {points.length < epochs.length && <p className="experiment-muted">Some completed epochs have no verified {label.toLowerCase()} statistics yet; missing values are omitted.</p>}
  </figure>;
}
