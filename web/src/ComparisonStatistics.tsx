import type { JSX } from 'preact';
import { useState } from 'preact/hooks';
import { networkLabel } from './experimentLabels';
import type { Participants } from './experimentLabels';
import { scoreInterval } from './experimentConfidence';
import type { ConfidenceLevel, MatchCounts } from './experimentConfidence';

import { LengthHistogram } from './GameLengthChart';
import type { Lengths } from './GameLengthChart';
import { inspectKey, pointerX, usePlotWidth } from './InteractiveChart';
import { nearestIndex } from './experimentChartData';

type Competitor = 'first_checkpoint' | 'second_checkpoint';
type LengthFilter = 'all' | Competitor | 'draws';
type Completion = { p50_seconds: number; p90_seconds: number; p95_seconds: number; p99_seconds: number;
  final_10_percent_seconds: number; last_finish_seconds: number; points: { games: number; seconds: number }[] };
export type GameStatistics = { games: number;
  seat_results: Record<Competitor, Record<'first' | 'second', MatchCounts & { games: number }>>;
  outcomes: { first_player_wins: number; second_player_wins: number; draws: number };
  lengths: Record<LengthFilter, Lengths | null>; completion: Completion | null };

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const count = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 1 });

function CompletionCurve({ data, games, formatDuration }: {
  data: Completion; games: number; formatDuration: (seconds: number) => string;
}): JSX.Element {
  const { ref, width } = usePlotWidth();
  const [cursor, setCursor] = useState<number | null>(null);
  const index = Math.min(cursor ?? data.points.length - 1, data.points.length - 1), selected = data.points[index]!;
  const end = Math.max(1, data.last_finish_seconds);
  const right = width - 18;
  const x = (seconds: number) => 45 + seconds / end * (right - 45);
  const y = (completed: number) => 205 - completed / games * 170;
  const path = data.points.map((point, i) => i ? `H${x(point.seconds)}V${y(point.games)}` : `M${x(point.seconds)},${y(point.games)}`).join(' ');
  function inspect(event: JSX.TargetedPointerEvent<SVGSVGElement> | JSX.TargetedMouseEvent<SVGSVGElement>) {
    setCursor(nearestIndex(data.points.map(point => x(point.seconds)), pointerX(event)));
  }
  return <figure ref={ref} className="experiment-distribution interactive-chart">
    <div className="chart-readout" aria-live="off"><strong>{selected.games} / {games} finished ({percent(selected.games / games)})</strong>
      <span>{formatDuration(selected.seconds)} · {selected.seconds.toFixed(3)} seconds</span></div>
    <svg className="interactive-plot" role="img" aria-label="Percentage of games finished over elapsed match time" viewBox={`0 0 ${width} 255`}
      onPointerMove={inspect} onClick={inspect} tabindex={0}
      onKeyDown={event => inspectKey(event, index, data.points.length, setCursor)}>
      {[0, .5, 1].map(fraction => <g key={fraction}>
        <line className="chart-grid" x1="45" x2={right} y1={y(games * fraction)} y2={y(games * fraction)} />
        <text x="37" y={y(games * fraction) + 4} text-anchor="end">{fraction * 100}%</text>
      </g>)}
      <path className="completion-line" d={path} fill="none" />
      <line className="chart-crosshair" x1={x(selected.seconds)} x2={x(selected.seconds)} y1="30" y2="205" />
      <line className="chart-crosshair" x1="45" x2={right} y1={y(selected.games)} y2={y(selected.games)} />
      <circle className="distribution-bar" cx={x(selected.seconds)} cy={y(selected.games)} r="4" />
      {[0, .5, 1].map(fraction => <text key={fraction} x={x(end * fraction)} y="230"
        text-anchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'}>{formatDuration(end * fraction)}</text>)}
      <text x="45" y="18">Games finished</text><text x={right} y="251" text-anchor="end">Elapsed match time</text>
      <rect className="chart-hit-area" x="45" y="25" width={right - 45} height="185" />
    </svg>
    <figcaption><strong>Reading:</strong> 90% finished after {formatDuration(data.p90_seconds)}; all {games} games finished after {formatDuration(data.last_finish_seconds)}.</figcaption>
    <p className="chart-help">Inspect retained completion samples by hovering, tapping, or focusing the plot and using arrow keys. Games run concurrently; individual game durations were not recorded.</p>
  </figure>;
}

export function ComparisonStatistics({ participants, data, confidence, formatDuration, partial = false }: {
  partial?: boolean; participants?: Participants; data: GameStatistics; confidence: ConfidenceLevel; formatDuration: (seconds: number) => string;
}): JSX.Element {
  const [filter, setFilter] = useState<LengthFilter>('all');
  const lengths = data.lengths[filter], completion = data.completion;
  const outcomeOptions: [LengthFilter, string][] = [['all', 'All games'],
    ['first_checkpoint', `${networkLabel(participants, 'first')} wins`],
    ['second_checkpoint', `${networkLabel(participants, 'second')} wins`], ['draws', 'Draws']];
  return <div className="experiment-game-statistics">
    {partial && <><h3>Completed games by checkpoint</h3><div className="experiment-table-scroll"><table className="experiment-table experiment-results experiment-live-results">
      <thead><tr><th>Checkpoint</th><th>Wins</th><th>Draws</th><th>Losses</th><th>Score</th><th>{confidence}% interval</th></tr></thead>
      <tbody>{(['first', 'second'] as const).map(side => {
        const seats = data.seat_results[`${side}_checkpoint`];
        const totals = { wins: seats.first.wins + seats.second.wins, draws: seats.first.draws + seats.second.draws, losses: seats.first.losses + seats.second.losses };
        const interval = scoreInterval(totals, confidence);
        return <tr key={side}><td><strong>{networkLabel(participants, side)}</strong></td>
          <td><span className="experiment-result-label">Wins</span>{totals.wins}</td>
          <td><span className="experiment-result-label">Draws</span>{totals.draws}</td>
          <td><span className="experiment-result-label">Losses</span>{totals.losses}</td>
          <td><span className="experiment-result-label">Score</span>{percent((totals.wins + .5 * totals.draws) / data.games)}</td>
          <td><span className="experiment-result-label">{confidence}% interval</span>{interval ? `${percent(interval.low)} – ${percent(interval.high)}` : '—'}</td></tr>;
      })}</tbody></table></div></>}

    <h3>Results by seat</h3>
    <dl className="experiment-stat-grid">
      {([['Starting-player wins', data.outcomes.first_player_wins], ['Second-player wins', data.outcomes.second_player_wins], ['Draws', data.outcomes.draws]] as const)
        .map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{count(value)} <small>({percent(value / data.games)})</small></dd></div>)}
    </dl>
    <div className="experiment-table-scroll"><table className="experiment-table experiment-seat-results">
      <thead><tr><th>Checkpoint</th><th>Plays</th><th>Games</th><th>Wins</th><th>Draws</th><th>Losses</th><th>Score</th><th>{confidence}% interval</th></tr></thead>
      <tbody>{(['first', 'second'] as const).flatMap(side => (['first', 'second'] as const).map(seat => {
        const totals = data.seat_results[`${side}_checkpoint`][seat], interval = scoreInterval(totals, confidence);
        return <tr key={`${side}-${seat}`}>
          <td><strong>{networkLabel(participants, side)}</strong></td><td>{seat === 'first' ? 'First' : 'Second'}</td>
          <td><span className="experiment-result-label">Games</span>{totals.games}</td>
          <td><span className="experiment-result-label">Wins</span>{totals.wins}</td>
          <td><span className="experiment-result-label">Draws</span>{totals.draws}</td>
          <td><span className="experiment-result-label">Losses</span>{totals.losses}</td>
          <td><span className="experiment-result-label">Score</span>{totals.games ? percent((totals.wins + .5 * totals.draws) / totals.games) : '—'}</td>
          <td className="experiment-data"><span className="experiment-result-label">{confidence}% interval</span>{interval ? `${percent(interval.low)} – ${percent(interval.high)}` : '—'}</td>
        </tr>;
      }))}</tbody>
    </table></div>
    <p className="experiment-muted">First means making the opening move. Score counts draws as half a win; intervals use the confidence level selected above.</p>

    <h3>Game lengths</h3>
    <div className="experiment-confidence-controls">
      <label htmlFor="experiment-length-outcome">Outcome</label>
      <select id="experiment-length-outcome" value={filter} onChange={event => setFilter(event.currentTarget.value as LengthFilter)}>
        {outcomeOptions.map(([value, label]) => <option key={value} value={value}>{label} ({data.lengths[value]?.count ?? 0})</option>)}
      </select>
    </div>
    {lengths ? <><dl className="experiment-stat-grid experiment-length-summary">
      {([['Shortest', lengths.minimum], ['Mean', lengths.mean], ['Median', lengths.median],
         ['90th percentile', lengths.p90], ['95th percentile', lengths.p95], ['Longest', lengths.maximum]] as const)
        .map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{count(value)} <small>moves</small></dd></div>)}
    </dl><LengthHistogram key={filter} data={lengths} /></> : <p className="experiment-empty">No games ended with this outcome.</p>}

    <h3>Time to finish the match</h3>
    {completion ? <><dl className="experiment-stat-grid experiment-completion-summary">
      {([['50% finished after', completion.p50_seconds], ['90% finished after', completion.p90_seconds],
         ['99% finished after', completion.p99_seconds], ['Final 10% took', completion.final_10_percent_seconds]] as const)
        .map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{formatDuration(value)}</dd></div>)}
    </dl><CompletionCurve data={completion} games={data.games} formatDuration={formatDuration} /></>
      : <p className="experiment-empty">{partial ? 'Completion timing will appear when the match finishes.' : 'A complete timing log is unavailable for this match.'} Individual game durations were not recorded.</p>}
  </div>;
}
