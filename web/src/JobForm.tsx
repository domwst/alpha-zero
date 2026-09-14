import { useEffect, useRef, useState } from "preact/hooks";
import {
  api,
  type Artifact,
  type Job,
  type Session,
  type Spec,
} from "./serviceApi";
import { InfoTip } from "./InfoTip";
import { LoadingStatus } from "./LoadingStatus";

type Rule = "integer" | "number" | "boolean" | "text" | string[];
const temperatureScheduleHelp: Record<string, string> = {
  sharp:
    "Moves 1–5: 1.0. Move 6: 0.7. Move 7: 0.6. Move 8 onward: 0.5. The transition deliberately gives the second player lower temperatures.",
  paired:
    "Moves 1–6: 1.0. Decreases in seven equal steps for move pairs 7–8 through 19–20, reaching 0.7. Stays at 0.7 afterward. Both players receive the same temperature within each move pair.",
};
export function JobForm({
  session,
  jobs,
  onCreated,
}: {
  session: Session;
  jobs: Job[];
  onCreated: () => void;
}) {
  const [schema, setSchema] = useState<Record<string, Record<string, Rule>>>(
      {},
    ),
    [kind, setKind] = useState("self_play"),
    [title, setTitle] = useState(""),
    [options, setOptions] = useState<Spec["options"]>({
      device: "cpu",
      "temperature-schedule": "sharp",
    }),
    [inputs, setInputs] = useState<Spec["inputs"]>({}),
    [artifacts, setArtifacts] = useState<Artifact[]>([]),
    [error, setError] = useState(""),
    [dependency, setDependency] = useState(""),
    [busy, setBusy] = useState(false);
  const [schemaPending, setSchemaPending] = useState(true),
    [artifactsPending, setArtifactsPending] = useState(true);
  const [memory, setMemory] = useState(8192),
    [gpu, setGpu] = useState(0),
    [overlap, setOverlap] = useState(100),
    [replayCount, setReplayCount] = useState(1);
  const [showDeferred, setShowDeferred] = useState(false);
  const seriesTitle = (a: Artifact) =>
    a.job_title ||
    jobs.find((j) => j.id === a.job_id)?.title ||
    "Imported checkpoint";
  const savedSeries = new Map<string, Artifact[]>();
  for (const artifact of artifacts) {
    const series = savedSeries.get(artifact.job_id) || [];
    series.push(artifact);
    savedSeries.set(artifact.job_id, series);
  }
  const completionJobs = [
    ...new Set([
      ...(dependency && overlap === 100 ? [dependency] : []),
      ...Object.values(inputs).flatMap((ref) =>
        typeof ref === "object" ? [ref.job_id] : [],
      ),
    ]),
  ];
  const errorRef = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    if (!error) return;
    errorRef.current?.focus({ preventScroll: true });
    errorRef.current?.scrollIntoView({ block: "nearest" });
  }, [error]);
  useEffect(() => {
    api<{ kinds: typeof schema }>("schema")
      .then((x) => setSchema(x.kinds))
      .catch((e) => setError(String(e)))
      .finally(() => setSchemaPending(false));
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const data = await api<{ artifacts: Artifact[] }>(
          "artifacts", undefined, undefined, controller.signal,
        );
        if (!controller.signal.aborted) setArtifacts(data.artifacts);
      } catch (e) {
        if (!controller.signal.aborted) setError(String(e));
      } finally {
        if (!controller.signal.aborted) {
          setArtifactsPending(false);
          timer = setTimeout(refresh, 15000);
        }
      }
    }
    void refresh();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);
  async function submit(event: SubmitEvent) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      await api(
        "commands",
        {
          command_id: crypto.randomUUID(),
          request: {
            action: "create",
            title: title || kind.replaceAll("_", " "),
            spec: {
              kind,
              options: Object.fromEntries(
                Object.entries(options).map(([key, value]) => [
                  key,
                  ["integer", "number"].includes(String(schema[kind]?.[key]))
                    ? Number(value)
                    : value,
                ]),
              ),
              inputs,
              resources: {
                slots: 1,
                host_memory_mb: memory,
                gpu_memory_mb: gpu,
              },
            },
            dependencies: dependency
              ? [
                  overlap === 100
                    ? dependency
                    : { job_id: dependency, fraction: overlap / 100 },
                ]
              : [],
          },
        },
        session,
      );
      onCreated();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <form className="service-panel service-job-form" onSubmit={submit}>
      <h2>Schedule a job</h2>
      <LoadingStatus pending={schemaPending || artifactsPending}>
        {schemaPending
          ? "Loading available settings…"
          : "Loading saved checkpoints…"}
      </LoadingStatus>
      <div className="service-form-grid">
        <label>
          Title
          <input
            value={title}
            onInput={(e) => setTitle(e.currentTarget.value)}
          />
        </label>
        <label>
          Operation
          <select
            value={kind}
            onChange={(e) => {
              setKind(e.currentTarget.value);
              setReplayCount(1);
              setOptions(
                e.currentTarget.value === "self_play"
                  ? { device: "cpu", "temperature-schedule": "sharp" }
                  : { device: "cpu" },
              );
              setInputs({});
            }}
          >
            {Object.keys(schema).map((k) => (
              <option value={k}>{k.replaceAll("_", " ")}</option>
            ))}
          </select>
        </label>
        <div className="service-form-field">
          <div className="service-field-heading">
            <label htmlFor="job-device">Device</label>
            <InfoTip
              id="job-device-help"
              title="Device and resources"
              triggerLabel="How are device resources reserved?"
              variant="trigger"
            >
              Choose an explicit device so resources can be reserved. CUDA and
              MPS require a positive GPU memory reservation.
            </InfoTip>
          </div>
          <select
            id="job-device"
            required
            value={String(options.device)}
            aria-describedby="job-device-help"
            onChange={(e) =>
              setOptions({ ...options, device: e.currentTarget.value })
            }
          >
            {((schema[kind]?.device as string[] | undefined) || ["cpu"])
              .filter((device) => device !== "auto")
              .map((device) => (
                <option value={device}>{device.toUpperCase()}</option>
              ))}
          </select>
        </div>
        <label>
          Host memory reservation (MiB)
          <input
            type="number"
            min="1"
            value={memory}
            onInput={(e) => setMemory(Number(e.currentTarget.value))}
          />
        </label>
        <label>
          GPU memory reservation (MiB)
          <input
            type="number"
            min={options.device === "cpu" ? 0 : 1}
            required
            value={gpu}
            onInput={(e) => setGpu(Number(e.currentTarget.value))}
          />
        </label>
        <label>
          Wait for job completion
          <select
            value={dependency}
            onChange={(e) => {
              setDependency(e.currentTarget.value);
              setOverlap(100);
            }}
          >
            <option value="">
              No dependency — run when resources are free
            </option>
            {jobs.map((j) => (
              <option value={j.id}>
                {j.title} · {j.state}
              </option>
            ))}
          </select>
        </label>
        {jobs.find((j) => j.id === dependency)?.spec.kind === "comparison" && (
          <label>
            Start after comparison completion (%)
            <input
              type="number"
              min="1"
              max="100"
              value={overlap}
              onInput={(e) => setOverlap(Number(e.currentTarget.value))}
            />
          </label>
        )}
      </div>
      <div className="service-form-grid service-input-grid">
        {(kind === "comparison"
          ? ["first", "second"]
          : kind === "reconstruction"
            ? ["history"]
            : kind.startsWith("benchmark_")
              ? ["checkpoint"]
              : kind === "replay_train"
                ? Array.from({ length: replayCount }, (_, i) =>
                    i === 0 ? "replay" : `replay_${i + 1}`,
                  )
                : ["checkpoint"]
        ).map((name) => (
          <label key={name}>
            {name} input
            <select
              required={name !== "checkpoint" || kind.startsWith("benchmark_")}
              value={
                typeof inputs[name] === "object"
                  ? `job:${(inputs[name] as { job_id: string }).job_id}:${(inputs[name] as { selection: string }).selection}`
                  : (inputs[name] as string) || ""
              }
              onChange={(e) => {
                const next = { ...inputs };
                const value = e.currentTarget.value;
                if (value.startsWith("job:")) {
                  const [, id, selection] = value.split(":");
                  next[name] = {
                    job_id: id!,
                    selection: selection as "latest" | "best_value_validation",
                  };
                } else if (value) next[name] = value;
                else delete next[name];
                setInputs(next);
              }}
            >
              <option value="">
                {name === "checkpoint"
                  ? "Fresh network"
                  : "Choose a checkpoint"}
              </option>
              {[...savedSeries].map(([id, series]) => {
                const choices = series
                  .filter(
                    (a) =>
                      a.kind ===
                      (name === "history" ? "history" : "checkpoint"),
                  )
                  .sort(
                    (a, b) =>
                      (JSON.parse(b.metadata).epoch ?? -1) -
                      (JSON.parse(a.metadata).epoch ?? -1),
                  );
                return (
                  choices.length > 0 && (
                    <optgroup
                      key={id}
                      label={`Saved · ${seriesTitle(choices[0]!)}`}
                    >
                      {choices.map((a) => (
                        <option key={a.id} value={a.id}>
                          {seriesTitle(a)} ·{" "}
                          {name === "history"
                            ? "replay history"
                            : `epoch ${(JSON.parse(a.metadata).epoch ?? -1) + 1}`}
                        </option>
                      ))}
                    </optgroup>
                  )
                );
              })}
              {showDeferred && (
                <optgroup label="Wait for training to finish — not its next pause">
                  {jobs
                    .filter(
                      (j) =>
                        name !== "history" &&
                        [
                          "self_play",
                          "replay_train",
                          "reconstruction",
                        ].includes(j.spec.kind),
                    )
                    .flatMap((j) => [
                      <option value={`job:${j.id}:latest`}>
                        {j.title} · final checkpoint after successful completion
                      </option>,
                      <option value={`job:${j.id}:best_value_validation`}>
                        {j.title} · best validation value loss after successful
                        completion
                      </option>,
                    ])}
                </optgroup>
              )}
            </select>
          </label>
        ))}
      </div>
      <label className="service-checkbox">
        <input
          type="checkbox"
          checked={showDeferred}
          onChange={(e) => {
            setShowDeferred(e.currentTarget.checked);
            if (!e.currentTarget.checked)
              setInputs(
                Object.fromEntries(
                  Object.entries(inputs).filter(
                    ([, ref]) => typeof ref !== "object",
                  ),
                ),
              );
          }}
        />
        Show checkpoint choices that wait for training to finish
      </label>
      <p className="service-form-hint">
        To compare while self-play is paused, select a saved checkpoint and no
        completion dependency. Saved checkpoints above are ready now.
      </p>
      {kind === "replay_train" && (
        <button
          type="button"
          disabled={replayCount >= 100}
          onClick={() => setReplayCount(replayCount + 1)}
        >
          Add replay buffer
        </button>
      )}
      <details>
        <summary>Training and search settings</summary>
        <p>
          Unset settings use the trainer defaults. Values are recorded for each
          attempt.
        </p>
        {kind === "comparison" && (
          <p id="comparison-temperature-help">
            Shared temperature applies to both checkpoints (default: 0, choosing
            the most visited move). Leave the overrides blank for equal
            temperatures. An override follows its checkpoint when players swap
            colors; first and second refer to the checkpoint inputs, not the
            player moving first or second.
          </p>
        )}
        <div className="service-form-grid">
          {Object.entries(schema[kind] || {})
            .filter(([key]) => key !== "device" && key !== "top-p")
            .map(([key, rule]) =>
              key === "temperature-schedule" && Array.isArray(rule) ? (
                <div key={key} className="service-form-field">
                  <div className="service-field-heading">
                    <label htmlFor="job-temperature-schedule">
                      Temperature schedule
                    </label>
                    <InfoTip
                      id="job-temperature-schedule-help"
                      title={
                        options[key] === "sharp"
                          ? "Sharp schedule"
                          : "Paired schedule"
                      }
                      triggerLabel="Explain selected temperature schedule"
                      variant="trigger"
                    >
                      {
                        temperatureScheduleHelp[
                          String(options[key] || "paired")
                        ]
                      }
                    </InfoTip>
                  </div>
                  <select
                    id="job-temperature-schedule"
                    aria-describedby="job-temperature-schedule-help"
                    title={
                      temperatureScheduleHelp[String(options[key] || "paired")]
                    }
                    value={String(options[key] ?? "")}
                    onChange={(e) => {
                      const next = { ...options };
                      if (e.currentTarget.value)
                        next[key] = e.currentTarget.value;
                      else delete next[key];
                      setOptions(next);
                    }}
                  >
                    <option value="" title={temperatureScheduleHelp.paired}>
                      Default (paired)
                    </option>
                    {rule.map((v) => (
                      <option
                        key={v}
                        value={v}
                        title={temperatureScheduleHelp[v]}
                      >
                        {v}
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <label key={key}>
                  {kind === "comparison" && key === "temperature"
                    ? "Shared temperature"
                    : kind === "comparison" && key === "first-temperature"
                      ? "First checkpoint temperature override"
                      : kind === "comparison" && key === "second-temperature"
                        ? "Second checkpoint temperature override"
                        : key.replaceAll("-", " ")}
                  {Array.isArray(rule) ? (
                    <select
                      value={String(options[key] ?? "")}
                      onChange={(e) => {
                        const next = { ...options };
                        if (e.currentTarget.value)
                          next[key] = e.currentTarget.value;
                        else delete next[key];
                        setOptions(next);
                      }}
                    >
                      <option value="">
                        {key === "temperature-schedule"
                          ? "Default (paired)"
                          : "Default"}
                      </option>
                      {rule.map((v) => (
                        <option value={v}>{v}</option>
                      ))}
                    </select>
                  ) : rule === "text" ? (
                    <input
                      value={String(options[key] ?? "")}
                      placeholder="1,2,4,8,16,32"
                      onInput={(e) =>
                        setOptions({ ...options, [key]: e.currentTarget.value })
                      }
                    />
                  ) : rule === "boolean" ? (
                    <input
                      type="checkbox"
                      checked={Boolean(options[key])}
                      onChange={(e) =>
                        setOptions({
                          ...options,
                          [key]: e.currentTarget.checked,
                        })
                      }
                    />
                  ) : (
                    <input
                      type="number"
                      min="0"
                      step={rule === "integer" ? 1 : "any"}
                      value={String(options[key] ?? "")}
                      aria-describedby={
                        kind === "comparison" && key.includes("temperature")
                          ? "comparison-temperature-help"
                          : undefined
                      }
                      placeholder={
                        kind === "comparison" && key.endsWith("-temperature")
                          ? "Use shared temperature"
                          : undefined
                      }
                      onInput={(e) => {
                        const next = { ...options };
                        if (e.currentTarget.value !== "")
                          next[key] = e.currentTarget.value;
                        else delete next[key];
                        setOptions(next);
                      }}
                    />
                  )}
                </label>
              ),
            )}
        </div>
      </details>
      {schema[kind]?.["top-p"] && (
        <details>
          <summary>Advanced experimental settings</summary>
          <div className="service-form-grid">
            <label htmlFor="job-top-p">
              Nucleus sampling (top-p)
              <input
                id="job-top-p"
                type="number"
                min="0"
                max="1"
                step="any"
                value={String(options["top-p"] ?? "")}
                placeholder="1.0 (disabled)"
                aria-describedby="job-top-p-help"
                onInput={(e) => {
                  e.currentTarget.setCustomValidity(
                    e.currentTarget.value !== "" && Number(e.currentTarget.value) <= 0
                      ? "Top-p must be greater than 0."
                      : "",
                  );
                  const next = { ...options };
                  if (e.currentTarget.value !== "")
                    next["top-p"] = e.currentTarget.value;
                  else delete next["top-p"];
                  setOptions(next);
                }}
              />
            </label>
          </div>
          <p id="job-top-p-help" className="service-form-hint">
            Default: 1.0, disabled. Retained for reproducing research runs; no
            controlled benefit has been established. Values below 1 filter move
            sampling after temperature, preserving boundary ties. Training policy
            targets remain unchanged. Applies to the new job.
          </p>
        </details>
      )}
      {completionJobs.length > 0 && (
        <div className="service-notice" role="status">
          This job will wait for successful completion of:{" "}
          {completionJobs
            .map((id) => jobs.find((j) => j.id === id)?.title || id)
            .join(", ")}
          . Pausing at an epoch boundary does not satisfy this dependency.
          {Object.values(inputs).some((ref) => typeof ref === "object") &&
            " Your deferred checkpoint selection adds this wait even if no separate dependency is selected."}
        </div>
      )}
      {error && (
        <p ref={errorRef} className="service-error" role="alert" tabIndex={-1}>
          {error}
        </p>
      )}
      <button
        className="primary"
        disabled={busy || schemaPending || artifactsPending}
      >
        {busy ? "Adding to queue…" : "Add to queue"}
      </button>
    </form>
  );
}
