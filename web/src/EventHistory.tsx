import { useEffect, useState } from "preact/hooks";
import { api, stamp, type Event } from "./serviceApi";
import { LoadingStatus } from "./LoadingStatus";

export function EventHistory({ jobId }: { jobId: string }) {
  const [open, setOpen] = useState(false);
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Event history</summary>
      {open && <HistoryPage jobId={jobId} />}
    </details>
  );
}
function HistoryPage({ jobId }: { jobId: string }) {
  const [before, setBefore] = useState<number | null>(null),
    [events, setEvents] = useState<Event[]>([]);
  const [pending, setPending] = useState(true),
    [more, setMore] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setPending(true);
    setError("");
    async function poll() {
      try {
        const data = await api<{ events: Event[]; has_more: boolean }>(
          `event-history?job_id=${encodeURIComponent(jobId)}${before == null ? "" : `&before=${before}`}`,
          undefined,
          undefined,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setEvents(data.events);
          setMore(data.has_more);
        }
      } catch (e) {
        if (!controller.signal.aborted) setError(String(e));
      } finally {
        if (!controller.signal.aborted) {
          setPending(false);
          if (before == null) timer = setTimeout(poll, 3000);
        }
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [jobId, before]);
  return (
    <>
      <div className="service-toolbar">
        <LoadingStatus pending={pending}>
          Loading historical events…
        </LoadingStatus>
        <div className="service-actions">
          <button
            disabled={pending || before == null}
            onClick={() => setBefore(null)}
          >
            Newest
          </button>
          <button
            disabled={pending || !more}
            onClick={() => setBefore(events.at(-1)!.id)}
          >
            Older events
          </button>
        </div>
      </div>
      {error && (
        <p role="alert" className="service-error">
          {error}
        </p>
      )}
      <ol className="service-events">
        {events.map((event) => (
          <li key={event.id}>
            <time>{stamp(event.time)}</time>
            <strong>{event.kind.replaceAll("_", " ")}</strong>
            <pre>{JSON.stringify(event.payload, null, 2)}</pre>
          </li>
        ))}
      </ol>
    </>
  );
}
