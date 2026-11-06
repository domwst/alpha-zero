import { useEffect, useState } from "preact/hooks";
import { api } from "./serviceApi";

export type CheckpointSide = {
  label: string | null;
  model_sha256: string | null;
  job_id: string | null;
  job_title: string | null;
  /** Zero-based checkpoint epoch; display as epoch + 1 like the job pages. */
  epoch: number | null;
};
export type ComparisonResult = {
  id: string;
  title: string;
  created: number;
  state: string;
  games: number;
  /** Search settings that affect strength: simulations, temperatures, c_puct. */
  settings: Record<string, number | string>;
  wins: number;
  draws: number;
  losses: number;
  first: CheckpointSide;
  second: CheckpointSide;
};
export type RunEpoch = {
  epoch: number | null;
  value_loss: number | null;
  policy_loss: number | null;
  first_player_win_rate: number | null;
  games: number | null;
};
export type TrainingRun = {
  id: string;
  title: string;
  kind: string;
  state: string;
  created: number;
  experiment_id: string | null;
  experiment_title: string | null;
  source: { job_id: string; job_title: string | null; epoch: number | null } | null;
  epochs: RunEpoch[];
};
export type Overview = {
  jobs: number;
  experiments: number;
  by_kind: Record<string, number>;
  by_state: Record<string, number>;
  attempts: number;
  attempt_seconds: number;
  first_activity: number | null;
  last_activity: number | null;
  recorded_games: number;
  comparisons: ComparisonResult[];
  runs: TrainingRun[];
};

/** Cross-job aggregates; cached by the server and shared by the landing page
 *  and the job lineage panel. On a live service the page refreshes every
 *  30 seconds so new epochs and comparisons appear without a reload. */
export function useOverview(live = false) {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const result = await api<Overview>("overview", undefined, undefined, controller.signal);
        if (!controller.signal.aborted) {
          setData(result);
          setError("");
        }
      } catch (e) {
        if (!controller.signal.aborted) setError(String(e));
      } finally {
        if (!controller.signal.aborted) {
          setPending(false);
          if (live) timer = setTimeout(poll, 30000);
        }
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [live]);
  return { data, error, pending };
}

/** Short human form of the comparison settings: "4,000 sims · T 0.5". */
export function settingsLabel(settings: Record<string, number | string> | undefined): string {
  if (!settings) return "";
  const parts: string[] = [];
  // Native f32 settings arrive with float noise (0.699999988…); three decimals suffice.
  const tidy = (value: number | string | null | undefined) =>
    value == null ? null : Number(Number(value).toFixed(3));
  if (settings.simulations != null) parts.push(`${Number(settings.simulations).toLocaleString()} sims`);
  const first = tidy(settings.first_temperature ?? settings.temperature);
  const second = tidy(settings.second_temperature ?? settings.temperature);
  if (first != null || second != null)
    parts.push(first === second || second == null ? `T ${first}` : `T ${first ?? "?"}/${second}`);
  const cPuct = tidy(settings.c_puct);
  if (cPuct != null && cPuct !== 1) parts.push(`c ${cPuct}`);
  return parts.join(" · ");
}

/** Identity of the comparison settings for grouping; order-independent. */
export function settingsKey(settings: Record<string, number | string> | undefined): string {
  return JSON.stringify(
    Object.fromEntries(Object.entries(settings ?? {}).sort(([a], [b]) => a.localeCompare(b))),
  );
}

/** Score of a participant, draws counting half. */
export function comparisonScore(
  result: ComparisonResult,
  side: "first" | "second",
): number {
  const wins = side === "first" ? result.wins : result.losses;
  const losses = side === "first" ? result.losses : result.wins;
  const games = wins + losses + result.draws;
  return games ? (wins + 0.5 * result.draws) / games : 0;
}
