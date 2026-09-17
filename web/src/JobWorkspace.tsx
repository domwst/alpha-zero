import { useState } from "preact/hooks";
import {
  api,
  duration,
  stamp,
  heartbeatStamp,
  type Session,
} from "./serviceApi";
import { ComparisonStatistics } from "./ComparisonStatistics";
import { useJobData } from "./useJobData";
import { JobForm } from "./JobForm";
import { EventHistory } from "./EventHistory";
import { confidenceLevels, type ConfidenceLevel } from "./experimentConfidence";
import {
  SelfPlayMetrics,
  selfPlayMetricLabels,
  type SelfPlayMetric,
} from "./SelfPlayMetrics";
import { LossChart } from "./ExperimentLossChart";
import { LoadingStatus } from "./LoadingStatus";

export function JobWorkspace({ session }: { session: Session }) {
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState(
    new URLSearchParams(location.search).get("job"),
  );
  const {
    data,
    selectedJob,
    epochSummaries,
    summary,
    queuePending,
    detailPending,
    epochsPending,
    summaryPending,
    loadedPage,
    refresh,
    error,
    setError,
  } = useJobData(selected, page);
  const [metric, setMetric] = useState<
    "value_loss" | "policy_loss" | SelfPlayMetric
  >("value_loss");
  const [confidence, setConfidence] = useState<ConfidenceLevel>(95);
  const [creating, setCreating] = useState(false),
    [busy, setBusy] = useState(false);
  async function command(request: unknown) {
    setBusy(true);
    setError("");
    try {
      await api(
        "commands",
        { command_id: crypto.randomUUID(), request },
        session,
      );
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
  const metrics = epochSummaries
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
  const stage = job?.runtime?.stage,
    trainingProgress = job?.runtime?.training_progress;
  const collectionLive =
    !!job &&
    ["running", "stopping"].includes(job.state) &&
    (job.spec.kind === "comparison" || stage?.payload.stage === "self_play") &&
    latest?.payload.games_completed < latest?.payload.games_total;
  const inferenceRows: [string, Record<string, number>][] = latest?.payload
    .inference
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
    job?.attempts.at(-1)?.effective.participants ||
    summary?.participants ||
    undefined;
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
                command({
                  action: "scheduler",
                  paused: !data?.scheduler_paused,
                })
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
      {data?.scheduler_paused && (
        <p className="service-notice">
          Queue admission is paused. Running jobs continue.
        </p>
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
        <section
          className="service-panel"
          aria-busy={queuePending && loadedPage !== page}
        >
          <div className="service-toolbar">
            <h2>Scheduled work</h2>
            <LoadingStatus pending={queuePending && loadedPage !== page}>
              Loading jobs…
            </LoadingStatus>
            <span>{data?.total ?? "—"} jobs</span>
          </div>
          {!data ? (
            queuePending ? null : (
              <p>Jobs could not be loaded.</p>
            )
          ) : !data.total ? (
            <p>No jobs scheduled yet.</p>
          ) : (
            <div className="service-job-list">
              {data.jobs.map((j) => (
                <a
                  href={`?job=${j.id}`}
                  aria-current={selected === j.id ? "true" : undefined}
                  onClick={(event) => {
                    event.preventDefault();
                    setSelected(j.id);
                    history.replaceState(null, "", `?job=${j.id}`);
                  }}
                  key={j.id}
                >
                  <strong>{j.title}</strong>
                  <span className={`service-state state-${j.state}`}>
                    {j.state}
                  </span>
                  <small>
                    {j.spec.kind.replaceAll("_", " ")} · {stamp(j.created)}
                  </small>
                  {j.state === "queued" && j.reason && (
                    <small>{j.reason}</small>
                  )}
                </a>
              ))}
            </div>
          )}
          <div className="service-pagination">
            <button
              disabled={!page || (queuePending && loadedPage !== page)}
              onClick={() => setPage(page - 1)}
            >
              Previous
            </button>
            <span>
              Page {page + 1} of{" "}
              {Math.max(1, Math.ceil((data?.total || 0) / 7))}
            </span>
            <button
              disabled={
                (page + 1) * 7 >= (data?.total || 0) ||
                (queuePending && loadedPage !== page)
              }
              onClick={() => setPage(page + 1)}
            >
              Next
            </button>
          </div>
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
                <span className={`service-state state-${job.state}`}>
                  {job.state}
                </span>
              </div>
              {job.reason && <p>{job.reason}</p>}
              {["running", "stopping"].includes(job.state) && stage && (
                <p>
                  <strong>
                    {String(stage.payload.stage).replaceAll("_", " ")}
                  </strong>{" "}
                  · epoch {Number(stage.payload.epoch) + 1}
                </p>
              )}
              {job.heartbeat && (
                <p className="service-runtime">
                  Host RSS:{" "}
                  {((job.heartbeat.memory.VmRSS || 0) / 2 ** 30).toFixed(2)} GiB
                  · Peak:{" "}
                  {((job.heartbeat.memory.VmHWM || 0) / 2 ** 30).toFixed(2)} GiB
                  {job.heartbeat.gpu_memory_bytes != null &&
                    ` · GPU memory: ${(job.heartbeat.gpu_memory_bytes / 2 ** 30).toFixed(2)} GiB`}{" "}
                  · Last heartbeat{" "}
                  {heartbeatStamp(job.heartbeat.time, data?.server_time)}
                </p>
              )}
              {job.state === "running" &&
                stage?.payload.stage === "training" &&
                trainingProgress && (
                  <p>
                    {Number(
                      trainingProgress.payload.samples_completed,
                    ).toLocaleString()}{" "}
                    /{" "}
                    {Number(
                      trainingProgress.payload.samples_total,
                    ).toLocaleString()}{" "}
                    training samples ·{" "}
                    {trainingProgress.payload.batches_completed} batches
                  </p>
                )}
              {inferenceRows.length > 0 && (
                <section aria-label="Inference activity">
                  <h3>
                    {collectionLive
                      ? "Inference activity"
                      : job.spec.kind === "comparison"
                        ? "Last comparison inference"
                        : "Last collection inference"}
                  </h3>
                  {inferenceRows.map(([seat, inference]) => (
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
                        ).map(([key, label, milliseconds]) => (
                          <div key={key} title={key.replaceAll("_", " ")}>
                            <strong>
                              {Number(
                                (!collectionLive &&
                                [
                                  "active_producers",
                                  "outstanding_requests",
                                  "in_flight_requests",
                                ].includes(key)
                                  ? 0
                                  : (inference[key] ?? 0)) /
                                  (milliseconds ? 1000 : 1),
                              ).toLocaleString(undefined, {
                                maximumFractionDigits: 1,
                              })}
                            </strong>
                            <small>{label}</small>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </section>
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
                      onClick={() =>
                        command({ action: "resume", job_id: job.id })
                      }
                    >
                      Resume job
                    </button>
                  ) : (
                    ["queued", "preparing", "starting", "running"].includes(
                      job.state,
                    ) && (
                      <>
                        <button
                          disabled={busy}
                          onClick={() =>
                            command({
                              action: "pause",
                              job_id: job.id,
                              mode:
                                job.spec.kind === "comparison"
                                  ? "boundary"
                                  : "epoch",
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
                            command({
                              action: "pause",
                              job_id: job.id,
                              mode: "boundary",
                            })
                          }
                        >
                          Pause at recovery boundary
                        </button>
                      </>
                    )
                  )}
                  {[
                    "queued",
                    "preparing",
                    "running",
                    "starting",
                    "paused",
                  ].includes(job.state) && (
                    <button
                      disabled={busy}
                      onClick={() =>
                        command({
                          action: "cancel",
                          job_id: job.id,
                          mode: "boundary",
                        })
                      }
                    >
                      Cancel remaining work
                    </button>
                  )}
                </div>
              )}
              {job.state === "stopping" && job.spec.kind === "comparison" && (
                <p className="service-form-hint">
                  A boundary stop waits for a game to finish, which can take a
                  while. Stop now discards unfinished games and keeps saved
                  results.
                </p>
              )}
              {latest && (
                <div className="service-summary">
                  <div>
                    <strong>
                      {latest.payload.games_completed} /{" "}
                      {latest.payload.games_total}
                    </strong>
                    <span>
                      {job.runtime?.last_collection
                        ? "Games in last collection"
                        : "Games completed"}
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
                      <strong>
                        {Number(
                          latest.payload.completed_moves,
                        ).toLocaleString()}
                      </strong>
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
              {["self_play", "replay_train", "reconstruction"].includes(
                job.spec.kind,
              ) &&
                metrics.length === 0 &&
                epochsPending && (
                  <LoadingStatus pending>
                    Loading epoch statistics…
                  </LoadingStatus>
                )}
              {metrics.length > 0 && (
                <>
                  <div className="service-toolbar">
                    <h3>Epoch statistics</h3>
                  </div>
                  <div className="service-actions">
                    {Object.entries({
                      value_loss: "Value MSE",
                      policy_loss: "Policy cross-entropy",
                      ...(metrics.some((m) => m.self_play)
                        ? selfPlayMetricLabels
                        : {}),
                    }).map(([key, label]) => (
                      <button
                        aria-pressed={metric === key}
                        onClick={() => setMetric(key as typeof metric)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  {metric === "value_loss" || metric === "policy_loss" ? (
                    <LossChart metrics={metrics} metric={metric} />
                  ) : (
                    <SelfPlayMetrics metrics={metrics} metric={metric} />
                  )}
                </>
              )}
              {comparison && (
                <>
                  <h3>Playing strength</h3>
                  <label>
                    Confidence
                    <select
                      value={confidence}
                      onChange={(e) =>
                        setConfidence(
                          Number(e.currentTarget.value) as ConfidenceLevel,
                        )
                      }
                    >
                      {confidenceLevels.map((n) => (
                        <option value={n}>{n}%</option>
                      ))}
                    </select>
                  </label>
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
                <LoadingStatus pending={summaryPending}>
                  Loading comparison results…
                </LoadingStatus>
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
                        {readable(
                          benchmark.requests_per_second ??
                            benchmark.evaluations_per_second,
                        )}
                      </dd>
                    </div>
                    <div>
                      <dt>Measured duration</dt>
                      <dd>{duration(benchmark.duration_seconds)}</dd>
                    </div>
                    <div>
                      <dt>Precision</dt>
                      <dd>
                        {job.spec.options["disable-tf32"]
                          ? "FP32 · TF32 disabled"
                          : "Production default"}
                      </dd>
                    </div>
                    <div>
                      <dt>Mean real batch size</dt>
                      <dd>
                        {readable(
                          benchmark.network?.requests /
                            benchmark.network?.invocations,
                        )}
                      </dd>
                    </div>
                    {benchmark.latency_us && (
                      <>
                        <div>
                          <dt>Request latency · median</dt>
                          <dd>
                            {readable(benchmark.latency_us.p50 / 1000, " ms")}
                          </dd>
                        </div>
                        <div>
                          <dt>Request latency · p95</dt>
                          <dd>
                            {readable(benchmark.latency_us.p95 / 1000, " ms")}
                          </dd>
                        </div>
                        <div>
                          <dt>Request latency · p99</dt>
                          <dd>
                            {readable(benchmark.latency_us.p99 / 1000, " ms")}
                          </dd>
                        </div>
                      </>
                    )}
                    <div>
                      <dt>Padding rows</dt>
                      <dd>
                        {readable(
                          (100 * benchmark.network?.padded_requests) /
                            (benchmark.network?.requests +
                              benchmark.network?.padded_requests),
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
              <h3>Execution attempts</h3>
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
                            ? duration(
                                a.effective.recorded_timing.duration_seconds,
                              )
                            : a.started != null &&
                                (a.ended != null ||
                                  [
                                    "preparing",
                                    "running",
                                    "starting",
                                    "stopping",
                                  ].includes(a.state))
                              ? duration(
                                  (a.ended ??
                                    data?.server_time ??
                                    Date.now() / 1000) - a.started,
                                )
                              : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <details>
                <summary>Configuration and resources</summary>
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
                </dl>
                {job.dependencies.length > 0 && (
                  <p>
                    Depends on:{" "}
                    {job.dependencies.map((id) => (
                      <a href={`?job=${id}`}>{id.slice(0, 8)} </a>
                    ))}
                  </p>
                )}
              </details>
              <EventHistory key={job.id} jobId={job.id} />
            </>
          )}
        </section>
      </div>
    </main>
  );
}
