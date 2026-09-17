export type Session = {
  authenticated: boolean;
  csrf?: string;
  login_enabled?: boolean;
};
export type Resources = {
  slots: number;
  host_memory_mb: number;
  gpu_memory_mb: number;
};
export type Spec = {
  kind: string;
  options: Record<string, string | number | boolean>;
  inputs: Record<
    string,
    string | { job_id: string; selection: "latest" | "best_value_validation" }
  >;
  resources: Resources;
};
export type Attempt = {
  id: string;
  number: number;
  state: string;
  started: number | null;
  ended: number | null;
  created: number;
  result: { reason?: string } | null;
  effective: {
    binary_sha256?: string;
    recorded_timing?: { duration_seconds?: number };
    participants?: import("./experimentLabels").Participants;
    resources: Resources;
  };
};
export type Job = {
  has_games?: boolean;
  runtime?: Record<string, Event>;
  heartbeat?: {
    time: number;
    memory: { VmRSS?: number; VmHWM?: number };
    gpu_memory_bytes?: number | null;
  };
  id: string;
  title: string;
  state: string;
  created: number;
  reason: string | null;
  spec: Spec;
  attempts: Attempt[];
  dependencies: string[];
};
export type JobPage = {
  jobs: Job[];
  total: number;
  scheduler_paused: boolean;
  server_time: number;
};
export type Event = {
  id: number;
  time: number;
  kind: string;
  payload: Record<string, any>;
};
export type Artifact = {
  id: string;
  job_id: string;
  job_title?: string | null;
  kind: string;
  sha256: string;
  metadata: string;
};
async function responseError(
  response: Response,
  fallback: string,
): Promise<Error> {
  let message = `${fallback} (HTTP ${response.status})`;
  if (response.headers.get("content-type")?.includes("json")) {
    try {
      const data = await response.json();
      if (typeof data.error === "string") message = data.error;
    } catch {
      /* Keep the HTTP status if the response is malformed. */
    }
  }
  return new Error(message);
}

export async function api<T>(
  path: string,
  body?: unknown,
  session?: Session,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(
    `/api/v1/${path}`,
    body === undefined
      ? { signal }
      : {
          method: "POST",
          signal,
          headers: {
            "Content-Type": "application/json",
            ...(session?.csrf ? { "X-CSRF-Token": session.csrf } : {}),
          },
          body: JSON.stringify(body),
        },
  );
  if (!response.ok) throw await responseError(response, "Request failed");
  return response.json();
}
export const stamp = (time: number | null) =>
  time == null
    ? "—"
    : new Date(time * 1000).toLocaleString(undefined, { hourCycle: "h23" });
export function duration(seconds: number) {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600)
    return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds / 60) % 60)}m`;
}

export const heartbeatStamp = (time: number, now = Date.now() / 1000) =>
  now - time < 600
    ? new Date(time * 1000).toLocaleTimeString(undefined, { hourCycle: "h23" })
    : stamp(time);

export async function streamAnalysis<T>(
  body: unknown,
  session: Session,
  onUpdate: (value: T) => void,
  signal: AbortSignal,
) {
  const response = await fetch("/api/v1/analyze/stream", {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": session.csrf || "",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await responseError(response, "Search failed");
  const reader = response.body!.getReader(),
    decoder = new TextDecoder();
  let pending = "",
    complete = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      let newline;
      while ((newline = pending.indexOf("\n")) >= 0) {
        const line = JSON.parse(pending.slice(0, newline));
        pending = pending.slice(newline + 1);
        if (line.error) throw new Error(line.error);
        onUpdate(line);
        complete = !!line.result?.complete;
      }
      if (done) break;
    }
    if (!complete)
      throw new Error(
        "Search connection closed before completion. Retry to continue.",
      );
  } finally {
    reader.releaseLock();
  }
}
