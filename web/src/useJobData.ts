import { useEffect, useState } from "preact/hooks";
import { api, type Event, type Job, type JobPage } from "./serviceApi";
import type { GameStatistics } from "./ComparisonStatistics";
import type { Participants } from "./experimentLabels";
import { mergeEpochSummaries } from "./epochSummaries";

type Summary = {
  comparison: GameStatistics | null;
  benchmark: Record<string, any> | null;
  participants: Participants | null;
};

export const JOBS_PER_PAGE = 12;

export type JobFilters = {
  kind: string;
  state: string;
  search: string;
  experiment: string;
};
export const EMPTY_FILTERS: JobFilters = { kind: "", state: "", search: "", experiment: "" };

/** Query string for one page of jobs; filters are omitted when empty so the
 *  default request is unchanged. */
export function jobsQuery(page: number, filters: JobFilters = EMPTY_FILTERS): string {
  const params = new URLSearchParams({
    offset: String(page * JOBS_PER_PAGE),
    limit: String(JOBS_PER_PAGE),
  });
  if (filters.kind) params.set("kind", filters.kind);
  if (filters.state) params.set("state", filters.state);
  if (filters.search.trim()) params.set("q", filters.search.trim());
  if (filters.experiment) params.set("experiment_id", filters.experiment);
  return `jobs?${params}`;
}

/** Independent feeds: diagnostic history is never a dependency of live status or charts. */
export function useJobData(
  selected: string | null,
  page: number,
  filters: JobFilters = EMPTY_FILTERS,
) {
  const [data, setData] = useState<JobPage | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [epochSummaries, setEpochSummaries] = useState<Event[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [queuePending, setQueuePending] = useState(true),
    [loadedPage, setLoadedPage] = useState<number | null>(null),
    [loadedQuery, setLoadedQuery] = useState<string | null>(null);
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
          jobsQuery(page, filters),
          undefined,
          undefined,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setData(result);
          setLoadedPage(page);
          setLoadedQuery(jobsQuery(page, filters));
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
  }, [page, revision, filters.kind, filters.state, filters.search, filters.experiment]);

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
          epochCursor = Math.max(epochCursor, ...result.events.map((event) => event.id));
          setEpochSummaries((old) => mergeEpochSummaries(old, result.events));
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
    loadedQuery,
    refresh,
    error,
    setError,
  };
}
