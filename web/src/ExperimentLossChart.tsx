import type { JSX } from 'preact';
import { useState } from 'preact/hooks';
import { inspectKey, pointerX, usePlotWidth } from './InteractiveChart';
import { nearestIndex } from './experimentChartData';

type Metric = 'value_loss' | 'policy_loss';
type Epoch = { epoch: number; training: Record<Metric, number>; validation: Record<Metric, number> | null };
const value = (n: number) => n.toFixed(6);

export function LossChart({ metrics, metric }: { metrics: Epoch[]; metric: Metric }): JSX.Element {
  const { ref, width } = usePlotWidth();
  const [cursor, setCursor] = useState<number | null>(null);
  const [focus, setFocus] = useState<'training' | 'validation' | null>(null);
  const last = metrics.at(-1);
  if (!last) return <p>No completed training passes yet.</p>;
  const index = Math.min(cursor ?? metrics.length - 1, metrics.length - 1), selected = metrics[index]!;
  const splits = metrics.every(epoch => epoch.validation) ? (['training', 'validation'] as const) : (['training'] as const);
  const all = metrics.flatMap(epoch => splits.map(split => epoch[split]![metric]));
  const minimum = Math.min(...all), maximum = Math.max(...all), padding = Math.max((maximum - minimum) * .12, .01);
  const low = Math.max(0, minimum - padding), high = maximum + padding;
  const left = 55, right = width - 18;
  const firstPass = metrics[0]!.epoch;
  const x = (epoch: number) => left + (epoch - firstPass) / Math.max(last.epoch - firstPass, 1) * (right - left);
  const y = (n: number) => 205 - (n - low) / (high - low) * 175;
  function inspect(event: JSX.TargetedPointerEvent<SVGSVGElement> | JSX.TargetedMouseEvent<SVGSVGElement>) {
    setCursor(nearestIndex(metrics.map(point => x(point.epoch)), pointerX(event)));
  }
  return <figure ref={ref} className="experiment-chart interactive-chart">
    <div className="chart-readout" aria-live="off">
      <strong>Pass {selected.epoch}</strong>
      <span className="chart-training-value">Training <b>{value(selected.training[metric])}</b></span>
      {selected.validation && <span className="chart-validation-value">Validation <b>{value(selected.validation[metric])}</b></span>}
    </div>
    <div className="experiment-legend">
      {splits.map(split => <button key={split} className={split === 'training' ? 'train-series' : 'validation-series'}
        aria-pressed={focus === split} onClick={() => setFocus(focus === split ? null : split)}>
        {split === 'training' ? 'Training' : 'Validation'}</button>)}
    </div>
    <svg className="interactive-plot" role="img" aria-label={`${metric === 'value_loss' ? 'Value MSE' : 'Policy cross-entropy'} by training pass`}
      viewBox={`0 0 ${width} 245`} onPointerMove={inspect} onClick={inspect} tabindex={0}
      onKeyDown={event => inspectKey(event, index, metrics.length, setCursor)}>
      {[0, 1, 2, 3, 4].map(tick => {
        const n = low + (high - low) * tick / 4;
        return <g key={tick}><line className="chart-grid" x1={left} x2={right} y1={y(n)} y2={y(n)} />
          <text x={left - 10} y={y(n) + 4} text-anchor="end">{n.toFixed(2)}</text></g>;
      })}
      {splits.map(split => <g key={split} className={focus && focus !== split ? 'chart-series-muted' : ''}>
        <polyline className={split === 'training' ? 'train-line' : 'validation-line'} fill="none"
          points={metrics.map(point => `${x(point.epoch)},${y(point[split]![metric])}`).join(' ')} />
        <line className={`chart-guide ${split === 'training' ? 'train-line' : 'validation-line'}`} x1={left} x2={right}
          y1={y(selected[split]![metric])} y2={y(selected[split]![metric])} />
        <circle className={split === 'training' ? 'train-point' : 'validation-point'} cx={x(selected.epoch)} cy={y(selected[split]![metric])} r="4" />
      </g>)}
      <line className="chart-crosshair" x1={x(selected.epoch)} x2={x(selected.epoch)} y1="30" y2="205" />
      <text x={left} y="229">Pass {firstPass}</text><text x={right} y="229" text-anchor="end">Pass {last.epoch}</text>
      <rect className="chart-hit-area" x={left} y="25" width={right - left} height="185" />
    </svg>
    <p className="chart-help">Hover, tap, or focus the plot and use arrow keys to inspect a pass.</p>
    <figcaption><strong>Reading:</strong> After pass {last.epoch}: training {last.training[metric].toFixed(4)}, {last.validation ? `validation ${last.validation[metric].toFixed(4)}.` : 'no held-out validation split.'} Lower is better; playing strength is measured separately.</figcaption>
    <details className="chart-data"><summary>Exact loss values</summary><div className="chart-data-scroll"><table className="experiment-table">
      <thead><tr><th>Pass</th><th className={focus === 'training' ? 'chart-focused-cell' : ''}>Training</th><th className={focus === 'validation' ? 'chart-focused-cell' : ''}>Validation</th></tr></thead>
      <tbody>{metrics.map(point => <tr key={point.epoch}><td>{point.epoch}</td><td className={focus === 'training' ? 'chart-focused-cell' : ''}>{value(point.training[metric])}</td><td className={focus === 'validation' ? 'chart-focused-cell' : ''}>{point.validation ? value(point.validation[metric]) : '—'}</td></tr>)}</tbody>
    </table></div></details>
  </figure>;
}
