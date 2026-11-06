import { useState } from "preact/hooks";
import { dateStamp, duration, stamp, type Session } from "./serviceApi";
import { LoadingStatus } from "./LoadingStatus";
import { SERIES_COLORS, StrengthChart, epochSpan, strengthSeries } from "./StrengthChart";
import { comparisonScore, settingsKey, useOverview, type TrainingRun } from "./useOverview";
import { scoreInterval } from "./experimentConfidence";

const kindLabel = (kind: string) => kind.replaceAll("_", " ");
const percent = (n: number) => `${(n * 100).toFixed(1)}%`;

/** A tiny inline line chart; values are scaled to their own range. */
export function Sparkline({
  values,
  className = "",
}: {
  values: (number | null | undefined)[];
  className?: string;
}) {
  const points = values
    .map((value, index) => [index, value] as const)
    .filter((pair): pair is readonly [number, number] => typeof pair[1] === "number" && Number.isFinite(pair[1]));
  if (points.length < 2) return <span className="service-sparkline service-sparkline-empty" />;
  const width = 160,
    height = 40;
  const minimum = Math.min(...points.map((p) => p[1])),
    maximum = Math.max(...points.map((p) => p[1]));
  const last = values.length - 1;
  const x = (index: number) => 2 + (index / Math.max(last, 1)) * (width - 4);
  const y = (value: number) =>
    height - 3 - ((value - minimum) / ((maximum - minimum) || 1)) * (height - 6);
  const line = points.map(([index, value]) => `${x(index)},${y(value)}`).join(" ");
  const end = points[points.length - 1]!;
  return (
    <svg
      className={`service-sparkline ${className}`}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <polygon
        className="service-sparkline-fill"
        points={`${line} ${x(end[0])},${height} ${x(points[0]![0])},${height}`}
      />
      <polyline className="service-sparkline-line" fill="none" points={line} />
      <circle className="service-sparkline-dot" cx={x(end[0])} cy={y(end[1])} r="2.5" />
    </svg>
  );
}

function runSummary(run: TrainingRun): string {
  const completed = run.epochs.filter((e) => e.value_loss != null);
  const last = completed.at(-1);
  const first = run.epochs[0]?.epoch,
    final = run.epochs.at(-1)?.epoch;
  const range =
    first != null && final != null
      ? first === final
        ? `epoch ${first + 1}`
        : `epochs ${first + 1}–${final + 1}`
      : "no completed epochs";
  const games = run.epochs.reduce((n, e) => n + (e.games ?? 0), 0);
  return [
    range,
    games ? `${games.toLocaleString()} games` : null,
    last?.value_loss != null ? `value MSE ${last.value_loss.toFixed(3)}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

export function Overview({ session }: { session: Session }) {
  const { data, error, pending } = useOverview(!session.read_only);
  const [allRuns, setAllRuns] = useState(false);
  const comparisons = data?.comparisons ?? [];
  const series = strengthSeries(comparisons, data?.runs ?? []);
  const charted = series.filter((s) => epochSpan(s) > 1);
  // One-off matches, pooled when the same two checkpoints were compared
  // repeatedly (for example the batching repeats), so the table lists each
  // pairing once with its match count.
  const singles = [
    ...series
      .filter((s) => epochSpan(s) <= 1)
      .flatMap((s) => s.points.map((point) => ({ series: s, point })))
      .reduce((groups, { series: s, point }) => {
        const moving = point.result[point.movingSide];
        const baseline = point.result[point.movingSide === "first" ? "second" : "first"];
        const key = `${moving.model_sha256}|${baseline.model_sha256}|${settingsKey(point.result.settings)}`;
        const counts =
          point.movingSide === "first"
            ? { wins: point.result.wins, draws: point.result.draws, losses: point.result.losses }
            : { wins: point.result.losses, draws: point.result.draws, losses: point.result.wins };
        const group = groups.get(key) ?? {
          series: s,
          point,
          matches: 0,
          wins: 0,
          draws: 0,
          losses: 0,
        };
        group.matches += 1;
        group.wins += counts.wins;
        group.draws += counts.draws;
        group.losses += counts.losses;
        if (point.result.created > group.point.result.created) group.point = point;
        groups.set(key, group);
        return groups;
      }, new Map<string, { series: (typeof series)[number]; point: (typeof series)[number]["points"][number]; matches: number; wins: number; draws: number; losses: number }>())
      .values(),
  ].map((group) => {
    const games = group.wins + group.draws + group.losses;
    const interval = scoreInterval(group, 95);
    return { ...group, games, score: games ? (group.wins + 0.5 * group.draws) / games : 0, interval };
  });
  const runs = (data?.runs ?? []).filter((run) => run.epochs.length);
  const RUN_LIMIT = 8;
  const shownRuns = allRuns ? runs : runs.slice(0, RUN_LIMIT);
  const runCount = Object.entries(data?.by_kind ?? {})
    .filter(([kind]) => ["self_play", "replay_train", "reconstruction", "historical_training"].includes(kind))
    .reduce((n, [, count]) => n + count, 0);
  const recent = [...comparisons].sort((a, b) => b.created - a.created).slice(0, 8);
  return (
    <main className="service-main">
      <div className="service-heading">
        <div>
          <h1>Overview</h1>
          <p>
            {data?.first_activity && data?.last_activity
              ? `Experiments recorded from ${dateStamp(data.first_activity)} to ${dateStamp(data.last_activity)}.`
              : "What the training programme produced, and where to look next."}
          </p>
        </div>
      </div>
      {error && (
        <p role="alert" className="service-error">
          {error}
        </p>
      )}
      <LoadingStatus pending={pending}>Loading overview…</LoadingStatus>
      {data && (
        <>
          <div className="service-kpis">
            <a className="service-kpi" href="/experiments">
              <b>{data.jobs.toLocaleString()}</b>
              <span>
                jobs · {data.experiments.toLocaleString()} series
              </span>
            </a>
            <a className="service-kpi" href="/experiments?kind=comparison">
              <b>{(data.by_kind.comparison ?? 0).toLocaleString()}</b>
              <span>comparisons</span>
            </a>
            <a className="service-kpi" href="/experiments?kind=self_play">
              <b>{runCount.toLocaleString()}</b>
              <span>training runs</span>
            </a>
            <a className="service-kpi" href="/games">
              <b>{data.recorded_games.toLocaleString()}</b>
              <span>recorded games with search diagnostics</span>
            </a>
            <div className="service-kpi">
              <b>{Math.round(data.attempt_seconds / 3600).toLocaleString()} h</b>
              <span>
                attempt wall time · {data.attempts.toLocaleString()} attempts
              </span>
            </div>
          </div>
          <div className="service-overview-grid">
            <section className="service-panel">
              <div className="service-toolbar">
                <h2>Playing strength across checkpoints</h2>
                <span className="service-runtime">
                  score of each lineage against its recurring baseline · draws count half · 95% Wilson intervals
                </span>
              </div>
              {charted.length === 0 && (
                <p>No lineage has been compared at more than one epoch yet.</p>
              )}
              {charted.map((s, index) => (
                <article className="service-strength-card" key={s.key}>
                  <h3>{s.label}</h3>
                  <p className="service-runtime">
                    vs {s.baseline} · {s.points.length} match{s.points.length === 1 ? "" : "es"}
                    {s.settings && ` · ${s.settings}`}
                  </p>
                  <StrengthChart series={s} color={SERIES_COLORS[index % SERIES_COLORS.length]} />
                </article>
              ))}
              {singles.length > 0 && (
                <>
                  <h3 className="service-strength-singles">Single-epoch matches</h3>
                  <p className="service-runtime">
                    Checkpoint pairs compared at one epoch only; repeated matches of the same
                    pair are pooled.
                  </p>
                  <div className="service-table-scroll">
                    <table className="service-singles">
                      <thead>
                        <tr>
                          <th>Checkpoint</th>
                          <th>Baseline</th>
                          <th>Matches</th>
                          <th>Pooled result</th>
                          <th>Score</th>
                          <th>95% interval</th>
                        </tr>
                      </thead>
                      <tbody>
                        {singles.map((group) => (
                          <tr key={group.point.result.id}>
                            <td className="service-wrap">
                              <a href={`/experiments?job=${group.point.result.id}`}>
                                {group.series.label} · epoch {group.point.epoch}
                              </a>
                            </td>
                            <td className="service-wrap">
                              {group.series.baseline}
                              {group.series.settings && (
                                <small className="service-runtime"> · {group.series.settings}</small>
                              )}
                            </td>
                            <td>{group.matches}</td>
                            <td>
                              {group.wins}–{group.draws}–{group.losses}
                            </td>
                            <td>{percent(group.score)}</td>
                            <td>
                              {group.interval
                                ? `${percent(group.interval.low)} – ${percent(group.interval.high)}`
                                : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </section>
            <section className="service-panel">
              <div className="service-toolbar">
                <h2>Training runs</h2>
                <span className="service-runtime">value loss per epoch</span>
              </div>
              {runs.length === 0 && <p>No training runs with completed epochs.</p>}
              <ul className="service-run-list">
                {shownRuns.map((run) => (
                  <li key={run.id}>
                    <div>
                      <a href={`/experiments?job=${run.id}`}>{run.title}</a>
                      <small>
                        {runSummary(run)}
                        {run.source?.job_title
                          ? ` · continues ${run.source.job_title}${run.source.epoch != null ? ` from epoch ${run.source.epoch + 1}` : ""}`
                          : ""}
                      </small>
                    </div>
                    <span className={`service-state state-${run.state}`}>{run.state}</span>
                    <Sparkline values={run.epochs.map((e) => e.value_loss)} />
                  </li>
                ))}
              </ul>
              {runs.length > RUN_LIMIT && (
                <button className="service-show-more" onClick={() => setAllRuns(!allRuns)}>
                  {allRuns ? "Show fewer runs" : `Show all ${runs.length} runs`}
                </button>
              )}
            </section>
          </div>
          <section className="service-panel">
            <div className="service-toolbar">
              <h2>Recent comparisons</h2>
              <a className="service-button" href="/experiments?kind=comparison">
                All comparisons
              </a>
            </div>
            {recent.length === 0 ? (
              <p>No completed comparisons.</p>
            ) : (
              <div className="service-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Match</th>
                      <th>Finished</th>
                      <th>Games</th>
                      <th>A result</th>
                      <th>A score</th>
                      <th>95% interval</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recent.map((result) => {
                      const interval = scoreInterval(result, 95);
                      return (
                        <tr key={result.id}>
                          <td>
                            <a href={`/experiments?job=${result.id}`}>{result.title}</a>
                            <br />
                            <small className="service-runtime">
                              A · {result.first.label ?? result.first.job_title ?? "?"} · vs ·
                              B · {result.second.label ?? result.second.job_title ?? "?"}
                            </small>
                          </td>
                          <td>{stamp(result.created)}</td>
                          <td>{result.games.toLocaleString()}</td>
                          <td>
                            {result.wins}–{result.draws}–{result.losses}
                          </td>
                          <td>{percent(comparisonScore(result, "first"))}</td>
                          <td>
                            {interval ? `${percent(interval.low)} – ${percent(interval.high)}` : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
          <p className="service-runtime">
            Kinds: {Object.entries(data.by_kind).map(([k, n]) => `${kindLabel(k)} ${n}`).join(" · ")}.
            States: {Object.entries(data.by_state).map(([k, n]) => `${k} ${n}`).join(" · ")}.
            Attempt time {duration(data.attempt_seconds)} in total.
          </p>
        </>
      )}
    </main>
  );
}
