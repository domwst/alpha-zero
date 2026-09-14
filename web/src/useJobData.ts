import { useEffect, useState } from "preact/hooks";
import { api, type Event, type Job, type JobPage } from "./serviceApi";
import type { GameStatistics } from "./ComparisonStatistics";
import type { Participants } from "./experimentLabels";

type Summary = {
  comparison: GameStatistics | null;
  benchmark: Record<string, any> | null;
  participants: Participants | null;
};

/** Independent feeds: diagnostic history is never a dependency of live status or charts. */
export function useJobData(selected: string | null, page: number) {
  const [data, setData] = useState<JobPage | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [epochSummaries, setEpochSummaries] = useState<Event[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [queuePending, setQueuePending] = useState(true),
    [loadedPage, setLoadedPage] = useState<number | null>(null);
  const [detailPending, setDetailPending] = useState(true),
    [epochsPending, setEpochsPending] = useState(true),
    [summaryPending, setSummaryPending] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const refresh = () => setRevision((n) => n + 1);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setQueuePending(true);
    async function poll() {
      try {
        const result = await api<JobPage>(
          `jobs?offset=${page * 7}&limit=7`,
          undefined,
          undefined,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setData(result);
          setLoadedPage(page);
        }
      } catch (e) {
        if (!controller.signal.aborted) setError(String(e));
      } finally {
        if (!controller.signal.aborted) {
          setQueuePending(false);
          timer = setTimeout(poll, 3000);
        }
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [page, revision]);

  useEffect(() => {
    setSelectedJob(null);
    setEpochSummaries([]);
    setSummary(null);
    setDetailPending(true);
    setEpochsPending(true);
    setSummaryPending(true);
    if (!selected) return;
    const controller = new AbortController();
    const timers = new Set<ReturnType<typeof setTimeout>>();
    let epochCursor = 0;
    function feed(read: () => Promise<void>, settled: () => void) {
      async function poll() {
        try {
          await read();
        } catch (e) {
          if (!controller.signal.aborted) setError(String(e));
        } finally {
          if (!controller.signal.aborted) {
            settled();
            const timer = setTimeout(() => {
              timers.delete(timer);
              void poll();
            }, 2000);
            timers.add(timer);
          }
        }
      }
      void poll();
    }
    const get = <T>(url: string) =>
      api<T>(url, undefined, undefined, controller.signal);
    feed(
      async () => {
        const result = await get<Job>(`jobs/${selected}`);
        if (!controller.signal.aborted) setSelectedJob(result);
      },
      () => setDetailPending(false),
    );
    feed(
      async () => {
        const result = await get<{ events: Event[] }>(
          `epoch-summaries?job_id=${encodeURIComponent(selected)}&after=${epochCursor}`,
        );
        if (!controller.signal.aborted && result.events.length) {
          epochCursor = result.events.at(-1)!.id;
          setEpochSummaries((old) => [...old, ...result.events]);
        }
      },
      () => setEpochsPending(false),
    );
    feed(
      async () => {
        const result = await get<Summary>(
          `job-summary?job_id=${encodeURIComponent(selected)}`,
        );
        if (!controller.signal.aborted) setSummary(result);
      },
      () => setSummaryPending(false),
    );
    return () => {
      controller.abort();
      timers.forEach(clearTimeout);
    };
  }, [selected]);
  return {
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
  };
}
