import type { JSX } from 'preact';
import { useState } from 'preact/hooks';
import { inspectKey, pointerX, usePlotWidth } from './InteractiveChart';
import { lengthBins } from './experimentChartData';
import type { LengthBin, LengthFrequency } from './experimentChartData';

export type Lengths = { count: number; minimum: number; maximum: number; mean: number; median: number;
  p90: number; p95: number; p99: number; histogram: LengthBin[]; frequencies?: LengthFrequency[] };
const range = (bin: LengthBin) => bin.lower === bin.upper ? `${bin.lower}` : `${bin.lower}–${bin.upper}`;

export function LengthHistogram({ data }: { data: Lengths }): JSX.Element {
  const { ref, width } = usePlotWidth();
  const [size, setSize] = useState(5);
  const [minimum, setMinimum] = useState(0), [maximum, setMaximum] = useState<number | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const [logarithmic, setLogarithmic] = useState(false);
  const fine = !!data.frequencies;
  const lower = Math.min(minimum, data.maximum), upper = Math.max(lower, Math.min(maximum ?? data.maximum, data.maximum));
  const bins = fine ? lengthBins(data.frequencies!, size, lower, upper) : data.histogram;
  const peak = bins.reduce((best, bin, index) => bin.count > bins[best]!.count ? index : best, 0);
  const index = Math.min(cursor ?? peak, bins.length - 1), bin = bins[index]!;
  const visible = bins.reduce((sum, item) => sum + item.count, 0);
  const ceiling = Math.max(...bins.map(item => item.count), 1);
  const scale = (n: number) => logarithmic ? Math.log1p(n) / Math.log1p(ceiling) : n / ceiling;
  const left = 48, right = width - 16, barWidth = (right - left) / bins.length;
  function inspect(event: JSX.TargetedPointerEvent<SVGSVGElement> | JSX.TargetedMouseEvent<SVGSVGElement>) {
    setCursor(Math.max(0, Math.min(bins.length - 1, Math.floor((pointerX(event) - left) / barWidth))));
  }
  return <figure ref={ref} className="experiment-distribution interactive-chart">
    <div className="experiment-confidence-controls chart-toolbar">
      <label>Bin width <select value={fine ? size : 20} onChange={event => { setSize(Number(event.currentTarget.value)); setCursor(null); }}>
        {[1, 5, 10, 20].map(n => <option key={n} value={n} disabled={!fine && n !== 20}>{n} {n === 1 ? 'move' : 'moves'}</option>)}
      </select></label>
      <label>From move <input type="number" min="0" max={upper} value={lower} disabled={!fine}
        onChange={event => { setMinimum(Math.max(0, Math.min(upper, Math.trunc(Number(event.currentTarget.value)) || 0))); setCursor(null); }} /></label>
      <label>To move <input type="number" min={lower} max={data.maximum} value={upper} disabled={!fine}
        onChange={event => { setMaximum(Math.max(lower, Math.min(data.maximum, Math.trunc(Number(event.currentTarget.value)) || 0))); setCursor(null); }} /></label>
      <label>Count scale <select value={logarithmic ? 'log' : 'linear'} onChange={event => setLogarithmic(event.currentTarget.value === 'log')}>
        <option value="linear">Linear</option><option value="log">Log (1 + count)</option>
      </select></label>
      <button onClick={() => { setMinimum(0); setMaximum(null); setCursor(null); }}>Reset range</button>
    </div>
    {!fine && <p className="chart-help">This snapshot has 20-move bins. Finer bins become available when detailed counts arrive.</p>}
    <div className="chart-readout" aria-live="off"><strong>{range(bin)} moves</strong><span><b>{bin.count}</b> games</span>
      <span>{(bin.count / data.count * 100).toFixed(2)}% of selected games</span></div>
    <svg className="interactive-plot" role="img" aria-label="Distribution of game lengths in moves" viewBox={`0 0 ${width} 245`}
      onPointerMove={inspect} onClick={inspect} tabindex={0}
      onKeyDown={event => inspectKey(event, index, bins.length, setCursor)}>
      {[0, .5, 1].map(fraction => {
        const n = logarithmic ? Math.expm1(Math.log1p(ceiling) * fraction) : ceiling * fraction;
        return <g key={fraction}><line className="chart-grid" x1={left} x2={right} y1={205 - fraction * 170} y2={205 - fraction * 170} />
          <text x={left - 8} y={209 - fraction * 170} text-anchor="end">{n.toLocaleString(undefined, { maximumFractionDigits: 1 })}</text></g>;
      })}
      <rect className="chart-band" x={left + index * barWidth} y="30" width={barWidth} height="175" />
      {bins.map((item, i) => <g key={item.lower}>
        <rect className={`distribution-bar ${i === index ? 'inspected-bar' : ''}`} x={left + i * barWidth + Math.min(1, barWidth * .1)}
          y={205 - scale(item.count) * 170} width={Math.max(.2, barWidth - Math.min(2, barWidth * .2))} height={scale(item.count) * 170} />
        {i % Math.max(1, Math.ceil(bins.length / Math.max(2, Math.floor((right - left) / 75)))) === 0 &&
          <text x={left + i * barWidth + barWidth / 2} y="225" text-anchor="middle">{item.lower}</text>}
      </g>)}
      <line className="chart-crosshair" x1={left + (index + .5) * barWidth} x2={left + (index + .5) * barWidth} y1="30" y2="205" />
      <text x={left} y="18">Games{logarithmic ? ' · log scale' : ''}</text><text x={right} y="242" text-anchor="end">Moves per game</text>
      <rect className="chart-hit-area" x={left} y="25" width={right - left} height="185" />
    </svg>
    <p className="chart-help">Hover or tap anywhere above a bin, including tiny or empty bars. One move is one stone placed by either player.</p>
    <figcaption><strong>Reading:</strong> {visible} of {data.count} selected games are in the displayed range. The largest bin is {range(bins[peak]!)} moves, with {bins[peak]!.count} games.</figcaption>
    <details className="chart-data"><summary>Exact bin counts</summary><div className="chart-data-scroll"><table className="experiment-table">
      <thead><tr><th>Moves</th><th>Games</th><th>Share of selected games</th></tr></thead>
      <tbody>{bins.map(item => <tr key={item.lower}><td>{range(item)}</td><td>{item.count}</td><td>{(item.count / data.count * 100).toFixed(2)}%</td></tr>)}</tbody>
    </table></div></details>
  </figure>;
}
