import { useEffect, useRef, useState } from "preact/hooks";
import {
  api,
  duration,
  stamp,
  heartbeatStamp,
  type Attempt,
  type Job,
  type Session,
} from "./serviceApi";
import { ComparisonStatistics } from "./ComparisonStatistics";
import {
  EMPTY_FILTERS,
  JOBS_PER_PAGE,
  jobsQuery,
  useJobData,
  type JobFilters,
} from "./useJobData";
import { JobForm } from "./JobForm";
import { EventHistory } from "./EventHistory";
import {
  confidenceLevels,
  scoreInterval,
  type ConfidenceLevel,
} from "./experimentConfidence";
import {
  SelfPlayMetrics,
  selfPlayMetricLabels,
  type SelfPlayMetric,
} from "./SelfPlayMetrics";
import { LossChart } from "./ExperimentLossChart";
import { LoadingStatus } from "./LoadingStatus";
import { Sparkline } from "./Overview";
import { networkLabel } from "./experimentLabels";
import {
  comparisonScore,
  useOverview,
  type ComparisonResult,
  type Overview,
} from "./useOverview";

const ACTIVE_STATES = ["preparing", "starting", "running", "stopping", "finalizing"];
const KIND_LABELS: Record<string, string> = {
  self_play: "self-play",
  replay_train: "replay training",
  reconstruction: "reconstruction",
  comparison: "comparison",
  benchmark_executor: "benchmark",
  historical_battle: "historical comparison",
  historical_training: "historical training",
};
const KIND_ORDER = Object.keys(KIND_LABELS);
const STATES = [
  "queued",
  "preparing",
  "starting",
  "running",
  "stopping",
  "paused",
  "succeeded",
  "failed",
  "cancelled",
  "archived",
];
const kindLabel = (kind: string) => KIND_LABELS[kind] ?? kind.replaceAll("_", " ");
const percent = (n: number) => `${(n * 100).toFixed(1)}%`;
type Metric = "value_loss" | "policy_loss" | SelfPlayMetric;
type EpochRow = {
  epoch: number;
  training: { value_loss: number; policy_loss: number; samples?: number; batches?: number; duration_seconds?: number };
  validation: { value_loss: number; policy_loss: number } | null;
  scheduled_learning_rate: number | null;
  self_play_seconds?: number | null;
  self_play: {
    games: number;
    average_game_length: number | null;
    first_player_win_rate: number | null;
    first_player_wins: number | null;
    second_player_wins: number | null;
    draws: number | null;
  } | null;
};
type MetricCard = {
  key: Metric;
  label: string;
  values: (number | null)[];
  format: (n: number) => string;
};

function metricCards(metrics: EpochRow[]): MetricCard[] {
  const finite = (n: unknown): number | null =>
    typeof n === "number" && Number.isFinite(n) ? n : null;
  const definitions: [Metric, string, (m: EpochRow) => number | null, (n: number) => string][] = [
    ["value_loss", "Value MSE", (m) => finite(m.training.value_loss), (n) => n.toFixed(4)],
    ["policy_loss", "Policy cross-entropy", (m) => finite(m.training.policy_loss), (n) => n.toFixed(3)],
    ["average_game_length", selfPlayMetricLabels.average_game_length, (m) => finite(m.self_play?.average_game_length), (n) => `${n.toFixed(1)} moves`],
    ["first_player_win_rate", selfPlayMetricLabels.first_player_win_rate, (m) => finite(m.self_play?.first_player_win_rate), percent],
    ["self_play_seconds", selfPlayMetricLabels.self_play_seconds, (m) => finite(m.self_play_seconds), duration],
    ["training_seconds", selfPlayMetricLabels.training_seconds, (m) => finite(m.training.duration_seconds), duration],
    ["training_samples", selfPlayMetricLabels.training_samples, (m) => finite(m.training.samples), (n) => n.toLocaleString()],
    ["learning_rate", selfPlayMetricLabels.learning_rate, (m) => finite(m.scheduled_learning_rate), (n) => Number(n.toPrecision(3)).toString()],
    ["lr_steps", selfPlayMetricLabels.lr_steps, (m) => {
      const rate = finite(m.scheduled_learning_rate), steps = finite(m.training.batches);
      return rate != null && steps != null ? rate * steps : null;
    }, (n) => Number(n.toPrecision(3)).toString()],
  ];
  return definitions
    .map(([key, label, pick, format]) => ({ key, label, values: metrics.map(pick), format }))
    .filter((card) => card.values.some((n) => n != null));
}

function AttemptsTimeline({ attempts, now }: { attempts: Attempt[]; now: number }) {
  const timed = attempts.filter((a) => a.started != null);
  if (!timed.length) return null;
  const start = Math.min(...timed.map((a) => a.started!));
  const finish = (a: Attempt) =>
    a.ended ?? (ACTIVE_STATES.includes(a.state) ? now : a.started!);
  const end = Math.max(...timed.map(finish), start + 1);
  const span = end - start;
  return (
    <div className="service-attempt-track" aria-hidden="true">
      {timed.map((a) => {
        const left = ((a.started! - start) / span) * 100;
        const width = Math.max(((finish(a) - a.started!) / span) * 100, 0.5);
        return (
          <div
            key={a.id}
            className={`service-attempt-bar state-${a.state}`}
            style={{ left: `${left}%`, width: `${width}%` }}
            title={`Attempt ${a.number} · ${a.state} · ${stamp(a.started)} → ${stamp(a.ended)} · ${duration(finish(a) - a.started!)}`}
          >
            {a.number} · {a.state} · {duration(finish(a) - a.started!)}
          </div>
        );
      })}
      <div className="service-attempt-axis">
        <span>{stamp(start)}</span>
        <span>{stamp(end)}</span>
      </div>
    </div>
  );
}

function InferenceRows({
  rows,
  live,
  participants,
}: {
  rows: [string, Record<string, number>][];
  live: boolean;
  participants?: Job["attempts"][number]["effective"]["participants"];
}) {
  return (
    <>
      {rows.map(([seat, inference]) => (
        <div key={seat}>
          {seat && (
            <p className="service-runtime">
              {participants?.[seat as "first" | "second"]?.label ||
                `${seat === "first" ? "First" : "Second"} checkpoint`}
            </p>
          )}
          <div className="service-inference service-inference-row">
            {(
              [
                ["active_producers", "Active", false],
                ["outstanding_requests", "Pending", false],
                ["in_flight_requests", "In flight", false],
                ["invocations", "Batches", false],
                ["completed_requests", "Completed", false],
                ["padded_requests", "Padded", false],
                ["average_batch_size", "Batch size", false],
                ["average_latency_us", "Latency · ms", true],
                ["max_latency_us", "Peak · ms", true],
              ] as const
            )
              .filter(
                ([key]) =>
                  live ||
                  !["active_producers", "outstanding_requests", "in_flight_requests"].includes(key),
              )
              .map(([key, label, milliseconds]) => (
                <div key={key} title={key.replaceAll("_", " ")}>
                  <strong>
                    {Number((inference[key] ?? 0) / (milliseconds ? 1000 : 1)).toLocaleString(
                      undefined,
                      { maximumFractionDigits: 1 },
                    )}
                  </strong>
                  <small>{label}</small>
                </div>
              ))}
          </div>
        </div>
      ))}
    </>
  );
}

function CheckpointStanding({
  job,
  overview,
  select,
}: {
  job: Job;
  overview: Overview | null;
  select: (id: string) => void;
}) {
  const rows = (overview?.comparisons ?? []).filter(
    (c) => c.first.job_id === job.id || c.second.job_id === job.id,
  );
  if (!rows.length) return null;
  const sorted = [...rows].sort((a, b) => {
    const epoch = (c: ComparisonResult) =>
      (c.first.job_id === job.id ? c.first.epoch : c.second.epoch) ?? -1;
    return epoch(b) - epoch(a);
  });
  return (
    <>
      <h3>Where its checkpoints stand</h3>
      <div className="service-score-list">
        {sorted.map((c) => {
          const side = c.first.job_id === job.id ? "first" : "second";
          const own = c[side],
            opponent = c[side === "first" ? "second" : "first"];
          const counts =
            side === "first"
              ? { wins: c.wins, draws: c.draws, losses: c.losses }
              : { wins: c.losses, draws: c.draws, losses: c.wins };
          const score = comparisonScore(c, side);
          const interval = scoreInterval(counts, 95);
          return (
            <div className="service-score-row" key={c.id}>
              <a
                href={`?job=${c.id}`}
                title={c.title}
                onClick={(event) => {
                  event.preventDefault();
                  select(c.id);
                }}
              >
                {own.epoch != null ? `E${own.epoch + 1}` : "checkpoint"} vs{" "}
                {opponent.label ?? opponent.job_title ?? "opponent"}
              </a>
              <div className="service-score-track" aria-hidden="true">
                <span className="par" />
                {interval && (
                  <span
                    className="ci"
                    style={{
                      left: `${interval.low * 100}%`,
                      width: `${(interval.high - interval.low) * 100}%`,
                    }}
                  />
                )}
                <span className="pt" style={{ left: `${score * 100}%` }} />
              </div>
              <span className="num">
                <b>{percent(score)}</b>{" "}
                {interval && (
                  <small className="service-runtime">
                    {percent(interval.low)} – {percent(interval.high)} · {c.games.toLocaleString()} games
                  </small>
                )}
              </span>
            </div>
          );
        })}
      </div>
      <p className="service-runtime">
        Score of this run's checkpoint, draws counting half; bars are 95% Wilson intervals and the tick marks parity.
      </p>
    </>
  );
}

export function JobWorkspace({ session }: { session: Session }) {
  const initial = new URLSearchParams(location.search);
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState(initial.get("job"));
  const [filters, setFilters] = useState<JobFilters>({
    ...EMPTY_FILTERS,
    kind: initial.get("kind") || "",
    state: initial.get("state") || "",
    search: initial.get("q") || "",
    experiment: initial.get("experiment") || "",
  });
  const [searchDraft, setSearchDraft] = useState(filters.search);
  const {
    data,
    selectedJob,
    epochSummaries,
    summary,
    queuePending,
    detailPending,
    epochsPending,
    summaryPending,
    loadedQuery,
    refresh,
    error,
    setError,
  } = useJobData(selected, page, filters);
  const { data: overview } = useOverview(!session.read_only);
  // The list is stale while the response for the current page and filters is
  // still pending; pagination waits so Next cannot run past a shrinking total.
  const listStale = queuePending && loadedQuery !== jobsQuery(page, filters);
  const pageCount = Math.max(1, Math.ceil((data?.total || 0) / JOBS_PER_PAGE));
  useEffect(() => {
    if (!listStale && data && page >= pageCount) setPage(pageCount - 1);
  }, [listStale, data, page, pageCount]);
  const [metric, setMetric] = useState<Metric>("value_loss");
  const [confidence, setConfidence] = useState<ConfidenceLevel>(95);
  const [tab, setTab] = useState<"overview" | "attempts" | "configuration">("overview");
  const [creating, setCreating] = useState(false),
    [busy, setBusy] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => {
      setFilters((old) =>
        old.search === searchDraft ? old : { ...old, search: searchDraft },
      );
      if (searchDraft !== filters.search) setPage(0);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchDraft]);
  useEffect(() => {
    const params = new URLSearchParams();
    if (selected) params.set("job", selected);
    if (filters.kind) params.set("kind", filters.kind);
    if (filters.state) params.set("state", filters.state);
    if (filters.search.trim()) params.set("q", filters.search.trim());
    if (filters.experiment) params.set("experiment", filters.experiment);
    const query = params.toString();
    history.replaceState(null, "", `${location.pathname}${query ? `?${query}` : ""}`);
  }, [selected, filters]);
  useEffect(() => setTab("overview"), [selected]);
  const tabList = useRef<HTMLDivElement | null>(null);
  const tabKeys = ["overview", "attempts", "configuration"] as const;
  function tabKeyDown(event: KeyboardEvent, index: number) {
    const next =
      event.key === "ArrowRight"
        ? (index + 1) % tabKeys.length
        : event.key === "ArrowLeft"
          ? (index + tabKeys.length - 1) % tabKeys.length
          : event.key === "Home"
            ? 0
            : event.key === "End"
              ? tabKeys.length - 1
              : null;
    if (next === null) return;
    event.preventDefault();
    setTab(tabKeys[next]!);
    tabList.current?.querySelectorAll<HTMLButtonElement>("[role=tab]")[next]?.focus();
  }
  const updateFilters = (patch: Partial<JobFilters>) => {
    setFilters((old) => ({ ...old, ...patch }));
    setPage(0);
  };
  const select = (id: string) => setSelected(id);
  async function command(request: unknown) {
    setBusy(true);
    setError("");
    try {
      await api("commands", { command_id: crypto.randomUUID(), request }, session);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const job =
      (selectedJob?.id === selected ? selectedJob : null) ||
      data?.jobs.find((j) => j.id === selected),
    latest =
      job?.runtime?.self_play_progress ||
      job?.runtime?.comparison_progress ||
      job?.runtime?.last_collection;
  const metrics: EpochRow[] = epochSummaries
    .filter((e) => e.kind === "epoch_completed" && e.payload.training)
    .map((e) => ({
      ...e.payload,
      epoch: Number(e.payload.epoch) + 1,
      training: e.payload.training,
      validation: e.payload.validation || null,
      scheduled_learning_rate:
        e.payload.scheduled_learning_rate ?? e.payload.learning_rate ?? null,
      self_play_seconds: e.payload.self_play_seconds,
      self_play:
        e.payload.self_play ||
        (e.payload.games
          ? {
              games: e.payload.games,
              average_game_length: e.payload.average_game_length ?? null,
              first_player_win_rate:
                e.payload.first_player_wins != null
                  ? e.payload.first_player_wins / e.payload.games
                  : null,
              first_player_wins: e.payload.first_player_wins ?? null,
              second_player_wins: e.payload.second_player_wins ?? null,
              draws: e.payload.draws ?? null,
            }
          : null),
    }));
  const cards = metricCards(metrics);
  const activeMetric = cards.some((c) => c.key === metric) ? metric : cards[0]?.key;
  const stage = job?.runtime?.stage,
    trainingProgress = job?.runtime?.training_progress;
  const isActive = !!job && ACTIVE_STATES.includes(job.state);
  const collectionLive =
    !!job &&
    ["running", "stopping"].includes(job.state) &&
    (job.spec.kind === "comparison" || stage?.payload.stage === "self_play") &&
    latest?.payload.games_completed < latest?.payload.games_total;
  const inferenceRows: [string, Record<string, number>][] = latest?.payload.inference
    ? [["", latest.payload.inference]]
    : [
        ["first", latest?.payload.first_inference],
        ["second", latest?.payload.second_inference],
      ].filter((row): row is [string, Record<string, number>] => !!row[1]);
  const comparison = summary?.comparison,
    benchmark = summary?.benchmark;
  const readable = (value: unknown, suffix = "") =>
    typeof value === "number" && Number.isFinite(value)
      ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) + suffix
      : "—";
  const participants =
    job?.attempts.at(-1)?.effective.participants || summary?.participants || undefined;
  const now = data?.server_time ?? Date.now() / 1000;
  const attemptSeconds = job
    ? job.attempts.reduce(
        (total, a) =>
          total +
          (a.started != null
            ? (a.ended ?? (ACTIVE_STATES.includes(a.state) ? now : a.started)) - a.started
            : 0),
        0,
      )
    : 0;
  const startedAttempts = job?.attempts.filter((a) => a.started != null) ?? [];
  const wallSeconds = startedAttempts.length
    ? Math.max(...startedAttempts.map((a) => a.ended ?? now)) -
      Math.min(...startedAttempts.map((a) => a.started!))
    : 0;
  const failedAttempts = job?.attempts.filter((a) => a.state === "failed").length ?? 0;
  const lastActivity = job
    ? Math.max(job.updated ?? job.created, ...job.attempts.map((a) => a.ended ?? 0))
    : null;
  const run = overview?.runs.find((r) => r.id === job?.id);
  const children = overview?.runs.filter((r) => r.source?.job_id === job?.id) ?? [];
  const usedBy =
    overview?.comparisons.filter(
      (c) => c.first.job_id === job?.id || c.second.job_id === job?.id,
    ) ?? [];
  const epochRange = metrics.length
    ? `epochs ${metrics[0]!.epoch}–${metrics.at(-1)!.epoch}`
    : null;
  const kinds = KIND_ORDER.concat(
    Object.keys(overview?.by_kind ?? {}).filter((k) => !KIND_ORDER.includes(k)),
  );
  const experimentFilterTitle =
    filters.experiment &&
    (data?.jobs.find((j) => j.experiment_id === filters.experiment)?.experiment_title ||
      overview?.runs.find((r) => r.experiment_id === filters.experiment)?.experiment_title ||
      filters.experiment.slice(0, 8));
  const tabs: ["overview" | "attempts" | "configuration", string][] = [
    ["overview", "Overview"],
    ["attempts", "Attempts & events"],
    ["configuration", "Configuration"],
  ];
  return (
    <main className="service-main">
      <div className="service-heading">
        <div>
          <h1>Experiments</h1>
          <p>Jobs, attempts, and recorded progress.</p>
        </div>
        {session.authenticated && (
          <div className="service-actions">
            <button
              disabled={busy}
              onClick={() =>
                command({ action: "scheduler", paused: !data?.scheduler_paused })
              }
            >
              {data?.scheduler_paused ? "Resume queue" : "Pause queue"}
            </button>
            <button className="primary" onClick={() => setCreating(!creating)}>
              Schedule job
            </button>
          </div>
        )}
      </div>
      {error && (
        <p role="alert" className="service-error">
          {error}
        </p>
      )}
      {data?.scheduler_paused && !session.read_only && (
        <p className="service-notice">Queue admission is paused. Running jobs continue.</p>
      )}
      {creating && (
        <JobForm
          session={session}
          jobs={data?.jobs || []}
          onCreated={() => {
            setCreating(false);
            refresh();
          }}
        />
      )}
      <div className="service-split">
        <section className="service-panel" aria-busy={listStale}>
          <div className="service-toolbar">
            <h2>Scheduled work</h2>
            <LoadingStatus pending={listStale}>
              Loading jobs…
            </LoadingStatus>
            <span>{data?.total ?? "—"} jobs</span>
          </div>
          <div className="service-filters">
            <label className="service-search">
              Search
              <input
                type="search"
                placeholder="title, series or id"
                value={searchDraft}
                onInput={(e) => setSearchDraft(e.currentTarget.value)}
              />
            </label>
            <label>
              Kind
              <select
                value={filters.kind}
                onChange={(e) => updateFilters({ kind: e.currentTarget.value })}
              >
                <option value="">All kinds</option>
                {kinds.map((kind) => (
                  <option key={kind} value={kind}>
                    {kindLabel(kind)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              State
              <select
                value={filters.state}
                onChange={(e) => updateFilters({ state: e.currentTarget.value })}
              >
                <option value="">All states</option>
                {STATES.map((state) => (
                  <option key={state} value={state}>
                    {state}
                  </option>
                ))}
              </select>
            </label>
            {filters.experiment && (
              <span className="service-filter-chip">
                Series · {experimentFilterTitle}
                <button
                  type="button"
                  aria-label="Clear series filter"
                  onClick={() => updateFilters({ experiment: "" })}
                >
                  ×
                </button>
              </span>
            )}
          </div>
          <div className="service-pagination service-job-pagination">
            <button disabled={!page || listStale} onClick={() => setPage(page - 1)}>
              Previous
            </button>
            <span>
              Page {Math.min(page, pageCount - 1) + 1} of {pageCount}
            </span>
            <button
              disabled={(page + 1) * JOBS_PER_PAGE >= (data?.total || 0) || listStale}
              onClick={() => setPage(page + 1)}
            >
              Next
            </button>
          </div>
          {!data ? (
            queuePending ? null : <p>Jobs could not be loaded.</p>
          ) : !data.total ? (
            <p>
              {filters.kind || filters.state || filters.search || filters.experiment
                ? "No jobs match these filters."
                : "No jobs scheduled yet."}
            </p>
          ) : (
            <div className="service-job-list">
              {data.jobs.map((j) => (
                <a
                  href={`?job=${j.id}`}
                  aria-current={selected === j.id ? "true" : undefined}
                  onClick={(event) => {
                    event.preventDefault();
                    select(j.id);
                  }}
                  key={j.id}
                >
                  <strong>{j.title}</strong>
                  <span className={`service-state state-${j.state}`}>{j.state}</span>
                  <small>
                    {kindLabel(j.spec.kind)} · {stamp(j.created)}
                    {j.experiment_title && j.experiment_title !== j.title && (
                      <span className="service-experiment"> · {j.experiment_title}</span>
                    )}
                  </small>
                  {j.state === "queued" && j.reason && <small>{j.reason}</small>}
                </a>
              ))}
            </div>
          )}
        </section>
        <section className="service-panel">
          {!job ? (
            <p>
              {selected
                ? detailPending
                  ? "Loading job details…"
                  : "Job details unavailable."
                : "Select a job to inspect its progress and attempts."}
            </p>
          ) : (
            <>
              <div className="service-toolbar">
                <h2>{job.title}</h2>
                <LoadingStatus pending={detailPending && !selectedJob}>
                  Loading job details…
                </LoadingStatus>
                <span className={`service-state state-${job.state}`}>{job.state}</span>
              </div>
              <div className="service-chips">
                <span className="service-chip">{kindLabel(job.spec.kind)}</span>
                {job.experiment_title && job.experiment_title !== job.title && (
                  <a
                    className="service-chip"
                    href={`?experiment=${job.experiment_id}`}
                    title="Show only this series"
                    onClick={(event) => {
                      event.preventDefault();
                      updateFilters({ experiment: job.experiment_id ?? "" });
                    }}
                  >
                    series · {job.experiment_title}
                  </a>
                )}
                {epochRange && <span className="service-chip">{epochRange}</span>}
                {job.spec.options.device != null && (
                  <span className="service-chip">{String(job.spec.options.device)}</span>
                )}
                {job.has_games && (
                  <a className="service-chip" href={`/games?job=${job.id}`}>
                    recorded games ↗
                  </a>
                )}
              </div>
              <div className="service-meta">
                <span>
                  Created <b>{stamp(job.created)}</b>
                </span>
                <span>
                  Attempts <b>{job.attempts.length}</b>
                  {failedAttempts > 0 && ` (${failedAttempts} failed)`}
                </span>
                {attemptSeconds > 0 && (
                  <span>
                    Attempt time <b>{duration(attemptSeconds)}</b>
                  </span>
                )}
                {wallSeconds > attemptSeconds + 60 && (
                  <span>
                    Wall time <b>{duration(wallSeconds)}</b>
                  </span>
                )}
                {lastActivity != null && (
                  <span>
                    Last activity <b>{stamp(lastActivity)}</b>
                  </span>
                )}
              </div>
              {(run?.source || children.length > 0 || usedBy.length > 0 || job.dependencies.length > 0) && (
                <div className="service-lineage">
                  <span>Lineage</span>
                  {run?.source && (
                    <>
                      <a
                        className="service-node"
                        href={`?job=${run.source.job_id}`}
                        onClick={(event) => {
                          event.preventDefault();
                          select(run.source!.job_id);
                        }}
                      >
                        {run.source.job_title ?? run.source.job_id.slice(0, 8)}
                        {run.source.epoch != null && ` · epoch ${run.source.epoch + 1}`}
                      </a>
                      <span>→</span>
                    </>
                  )}
                  <span className="service-node me">this job</span>
                  {children.length > 0 && (
                    <>
                      <span>→</span>
                      {children.map((child) => (
                        <a
                          key={child.id}
                          className="service-node"
                          href={`?job=${child.id}`}
                          onClick={(event) => {
                            event.preventDefault();
                            select(child.id);
                          }}
                        >
                          {child.title}
                        </a>
                      ))}
                    </>
                  )}
                  {usedBy.length > 0 && (
                    <span>
                      · {usedBy.length} comparison{usedBy.length === 1 ? "" : "s"} use
                      {usedBy.length === 1 ? "s" : ""} its checkpoints
                    </span>
                  )}
                  {job.dependencies.length > 0 && (
                    <span>
                      · depends on{" "}
                      {job.dependencies.map((id) => (
                        <a
                          key={id}
                          href={`?job=${id}`}
                          onClick={(event) => {
                            event.preventDefault();
                            select(id);
                          }}
                        >
                          {id.slice(0, 8)}{" "}
                        </a>
                      ))}
                    </span>
                  )}
                </div>
              )}
              {job.reason && <p>{job.reason}</p>}
              {isActive && stage && (
                <p>
                  <strong>{String(stage.payload.stage).replaceAll("_", " ")}</strong> ·
                  epoch {Number(stage.payload.epoch) + 1}
                </p>
              )}
              {isActive && job.heartbeat && (
                <p className="service-runtime">
                  Host RSS: {((job.heartbeat.memory.VmRSS || 0) / 2 ** 30).toFixed(2)} GiB ·
                  Peak: {((job.heartbeat.memory.VmHWM || 0) / 2 ** 30).toFixed(2)} GiB
                  {job.heartbeat.gpu_memory_bytes != null &&
                    ` · GPU memory: ${(job.heartbeat.gpu_memory_bytes / 2 ** 30).toFixed(2)} GiB`}{" "}
                  · Last heartbeat {heartbeatStamp(job.heartbeat.time, data?.server_time)}
                </p>
              )}
              {job.state === "running" &&
                stage?.payload.stage === "training" &&
                trainingProgress && (
                  <p>
                    {Number(trainingProgress.payload.samples_completed).toLocaleString()} /{" "}
                    {Number(trainingProgress.payload.samples_total).toLocaleString()} training
                    samples · {trainingProgress.payload.batches_completed} batches
                  </p>
                )}
              {session.authenticated && (
                <div className="service-actions">
                  {job.state === "stopping" && (
                    <button
                      disabled={busy}
                      title="Stop active work immediately. Unfinished games or training work since the last checkpoint will be discarded."
                      onClick={() =>
                        command({
                          action: job.reason === "cancel" ? "cancel" : "pause",
                          job_id: job.id,
                          mode: "immediate",
                        })
                      }
                    >
                      Stop now
                    </button>
                  )}
                  {["paused", "failed"].includes(job.state) ? (
                    <button
                      disabled={busy}
                      onClick={() => command({ action: "resume", job_id: job.id })}
                    >
                      Resume job
                    </button>
                  ) : (
                    ["queued", "preparing", "starting", "running"].includes(job.state) && (
                      <>
                        <button
                          disabled={busy}
                          onClick={() =>
                            command({
                              action: "pause",
                              job_id: job.id,
                              mode: job.spec.kind === "comparison" ? "boundary" : "epoch",
                            })
                          }
                        >
                          {["queued", "preparing"].includes(job.state)
                            ? "Pause job"
                            : job.spec.kind === "comparison"
                              ? "Pause after current game"
                              : "Pause after epoch"}
                        </button>
                        <button
                          disabled={busy}
                          onClick={() =>
                            command({ action: "pause", job_id: job.id, mode: "boundary" })
                          }
                        >
                          Pause at recovery boundary
                        </button>
                      </>
                    )
                  )}
                  {["queued", "preparing", "running", "starting", "paused"].includes(
                    job.state,
                  ) && (
                    <button
                      disabled={busy}
                      onClick={() =>
                        command({ action: "cancel", job_id: job.id, mode: "boundary" })
                      }
                    >
                      Cancel remaining work
                    </button>
                  )}
                </div>
              )}
              {job.state === "stopping" && job.spec.kind === "comparison" && (
                <p className="service-form-hint">
                  A boundary stop waits for a game to finish, which can take a while. Stop now
                  discards unfinished games and keeps saved results.
                </p>
              )}
              <div className="service-tabs" role="tablist" aria-label="Job sections" ref={tabList}>
                {tabs.map(([key, label], index) => (
                  <button
                    key={key}
                    role="tab"
                    id={`job-tab-${key}`}
                    aria-selected={tab === key}
                    aria-controls={`job-panel-${key}`}
                    tabIndex={tab === key ? 0 : -1}
                    onClick={() => setTab(key)}
                    onKeyDown={(event) => tabKeyDown(event, index)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div
                role="tabpanel"
                id="job-panel-overview"
                aria-labelledby="job-tab-overview"
                hidden={tab !== "overview"}
              >
                {collectionLive && inferenceRows.length > 0 && (
                  <section aria-label="Inference activity">
                    <h3>Inference activity</h3>
                    <InferenceRows rows={inferenceRows} live participants={participants} />
                  </section>
                )}
                {latest && (
                  <div className="service-summary">
                    <div>
                      <strong>
                        {latest.payload.games_completed} / {latest.payload.games_total}
                      </strong>
                      <span>
                        {job.runtime?.last_collection ? "Games in last collection" : "Games completed"}
                      </span>
                    </div>
                    {collectionLive && latest.payload.active_games != null && (
                      <div>
                        <strong>{latest.payload.active_games}</strong>
                        <span>Active games</span>
                      </div>
                    )}
                    {latest.payload.completed_moves != null && (
                      <div>
                        <strong>{Number(latest.payload.completed_moves).toLocaleString()}</strong>
                        <span>Moves made</span>
                      </div>
                    )}
                  </div>
                )}
                {latest && (
                  <progress
                    className="service-progress"
                    aria-label="Games completed"
                    value={latest.payload.games_completed}
                    max={latest.payload.games_total || 1}
                  />
                )}
                {job.has_games && (
                  <div className="service-actions service-section-actions">
                    <a className="service-button" href={`/games?job=${job.id}`}>
                      Explore recorded games
                    </a>
                  </div>
                )}
                {["self_play", "replay_train", "reconstruction"].includes(job.spec.kind) &&
                  metrics.length === 0 &&
                  epochsPending && (
                    <LoadingStatus pending>Loading epoch statistics…</LoadingStatus>
                  )}
                {metrics.length > 0 && activeMetric && (
                  <>
                    <div className="service-toolbar">
                      <h3>Epoch statistics</h3>
                      <span className="service-runtime">
                        {metrics.length} completed epochs · select a card for the full chart
                      </span>
                    </div>
                    <div className="service-metric-cards" role="group" aria-label="Epoch metrics">
                      {cards.map((card) => {
                        const present = card.values.filter((n): n is number => n != null);
                        const last = present.at(-1), previous = present.at(-2);
                        return (
                          <button
                            key={card.key}
                            className="service-metric-card"
                            aria-pressed={activeMetric === card.key}
                            onClick={() => setMetric(card.key)}
                          >
                            <small>{card.label}</small>
                            <b>{last != null ? card.format(last) : "—"}</b>
                            <small>
                              {last != null && previous != null
                                ? `${last > previous ? "▲" : last < previous ? "▼" : "="} ${card.format(Math.abs(last - previous))} vs previous epoch`
                                : " "}
                            </small>
                            <Sparkline values={card.values} />
                          </button>
                        );
                      })}
                    </div>
                    {activeMetric === "value_loss" || activeMetric === "policy_loss" ? (
                      <LossChart metrics={metrics} metric={activeMetric} />
                    ) : (
                      <SelfPlayMetrics metrics={metrics} metric={activeMetric} />
                    )}
                  </>
                )}
                {comparison && (
                  <>
                    <div className="service-toolbar">
                      <h3>Playing strength</h3>
                      <label className="service-inline-select">
                        Confidence
                        <select
                          value={confidence}
                          onChange={(e) =>
                            setConfidence(Number(e.currentTarget.value) as ConfidenceLevel)
                          }
                        >
                          {confidenceLevels.map((n) => (
                            <option value={n}>{n}%</option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <dl className="service-aliases">
                      <dt>A</dt>
                      <dd>{networkLabel(participants, "first")}</dd>
                      <dt>B</dt>
                      <dd>{networkLabel(participants, "second")}</dd>
                    </dl>
                    <ComparisonStatistics
                      participants={participants}
                      data={comparison}
                      confidence={confidence}
                      formatDuration={duration}
                      partial
                    />
                  </>
                )}
                {!comparison && job.spec.kind === "comparison" ? (
                  <LoadingStatus pending={summaryPending}>Loading comparison results…</LoadingStatus>
                ) : null}
                {job.spec.kind.startsWith("benchmark_") && (
                  <LoadingStatus pending={summaryPending}>
                    Loading benchmark results… details are incomplete.
                  </LoadingStatus>
                )}
                {benchmark && (
                  <>
                    <h3>Benchmark result</h3>
                    <dl>
                      <div>
                        <dt>Useful evaluations / second</dt>
                        <dd>
                          {readable(benchmark.requests_per_second ?? benchmark.evaluations_per_second)}
                        </dd>
                      </div>
                      <div>
                        <dt>Measured duration</dt>
                        <dd>{duration(benchmark.duration_seconds)}</dd>
                      </div>
                      <div>
                        <dt>Precision</dt>
                        <dd>
                          {job.spec.options["disable-tf32"] ? "FP32 · TF32 disabled" : "Production default"}
                        </dd>
                      </div>
                      <div>
                        <dt>Mean real batch size</dt>
                        <dd>{readable(benchmark.network?.requests / benchmark.network?.invocations)}</dd>
                      </div>
                      {benchmark.latency_us && (
                        <>
                          <div>
                            <dt>Request latency · median</dt>
                            <dd>{readable(benchmark.latency_us.p50 / 1000, " ms")}</dd>
                          </div>
                          <div>
                            <dt>Request latency · p95</dt>
                            <dd>{readable(benchmark.latency_us.p95 / 1000, " ms")}</dd>
                          </div>
                          <div>
                            <dt>Request latency · p99</dt>
                            <dd>{readable(benchmark.latency_us.p99 / 1000, " ms")}</dd>
                          </div>
                        </>
                      )}
                      <div>
                        <dt>Padding rows</dt>
                        <dd>
                          {readable(
                            (100 * benchmark.network?.padded_requests) /
                              (benchmark.network?.requests + benchmark.network?.padded_requests),
                            "%",
                          )}
                        </dd>
                      </div>
                      {benchmark.parity && (
                        <div>
                          <dt>Output parity</dt>
                          <dd>{benchmark.parity.passed ? "Passed" : "Failed"}</dd>
                        </div>
                      )}
                    </dl>
                  </>
                )}
                <CheckpointStanding job={job} overview={overview} select={select} />
              </div>
              <div
                role="tabpanel"
                id="job-panel-attempts"
                aria-labelledby="job-tab-attempts"
                hidden={tab !== "attempts"}
              >
                <h3>Execution attempts</h3>
                <AttemptsTimeline attempts={job.attempts} now={now} />
                <div className="service-table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Attempt</th>
                        <th>State</th>
                        <th>Started</th>
                        <th>Ended</th>
                        <th>Duration</th>
                      </tr>
                    </thead>
                    <tbody>
                      {job.attempts.map((a) => (
                        <tr key={a.id}>
                          <td>{a.number}</td>
                          <td>{a.state}</td>
                          <td>{stamp(a.started)}</td>
                          <td>{stamp(a.ended)}</td>
                          <td>
                            {a.effective.recorded_timing?.duration_seconds != null
                              ? duration(a.effective.recorded_timing.duration_seconds)
                              : a.started != null &&
                                  (a.ended != null || ACTIVE_STATES.includes(a.state))
                                ? duration((a.ended ?? now) - a.started)
                                : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {!isActive && job.heartbeat && (job.heartbeat.memory.VmRSS || 0) > 0 && (
                  <p className="service-runtime">
                    Last recorded host RSS {((job.heartbeat.memory.VmRSS || 0) / 2 ** 30).toFixed(2)} GiB
                    · peak {((job.heartbeat.memory.VmHWM || 0) / 2 ** 30).toFixed(2)} GiB
                    {job.heartbeat.gpu_memory_bytes != null &&
                      ` · GPU memory ${(job.heartbeat.gpu_memory_bytes / 2 ** 30).toFixed(2)} GiB`}{" "}
                    · heartbeat {heartbeatStamp(job.heartbeat.time, data?.server_time)}
                  </p>
                )}
                {!collectionLive && inferenceRows.length > 0 && (
                  <section aria-label="Inference telemetry">
                    <h3>
                      {job.spec.kind === "comparison"
                        ? "Last comparison inference"
                        : "Last collection inference"}
                    </h3>
                    <InferenceRows rows={inferenceRows} live={false} participants={participants} />
                  </section>
                )}
                <EventHistory key={job.id} jobId={job.id} />
              </div>
              <div
                role="tabpanel"
                id="job-panel-configuration"
                aria-labelledby="job-tab-configuration"
                hidden={tab !== "configuration"}
              >
                <h3>Configuration and resources</h3>
                <dl className="service-configuration">
                  {Object.entries(job.spec.options).map(([k, v]) => (
                    <div key={k}>
                      <dt>{k.replaceAll("-", " ")}</dt>
                      <dd>{String(v)}</dd>
                    </div>
                  ))}
                  {Object.entries(job.spec.resources).map(([k, v]) => (
                    <div key={k}>
                      <dt>{k.replaceAll("_", " ")}</dt>
                      <dd>{v}</dd>
                    </div>
                  ))}
                  {participants?.first && (
                    <div>
                      <dt>checkpoint A</dt>
                      <dd>{participants.first.label}</dd>
                    </div>
                  )}
                  {participants?.second && (
                    <div>
                      <dt>checkpoint B</dt>
                      <dd>{participants.second.label}</dd>
                    </div>
                  )}
                  {job.attempts.at(-1)?.effective.binary_sha256 && (
                    <div>
                      <dt>binary</dt>
                      <dd>{job.attempts.at(-1)!.effective.binary_sha256!.slice(0, 16)}…</dd>
                    </div>
                  )}
                </dl>
                {job.dependencies.length > 0 && (
                  <p>
                    Depends on:{" "}
                    {job.dependencies.map((id) => (
                      <a
                        key={id}
                        href={`?job=${id}`}
                        onClick={(event) => {
                          event.preventDefault();
                          select(id);
                        }}
                      >
                        {id.slice(0, 8)}{" "}
                      </a>
                    ))}
                  </p>
                )}
              </div>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
