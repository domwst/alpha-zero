import { useEffect, useRef, useState } from "preact/hooks";
import { api, streamAnalysis, type Session } from "./serviceApi";
import {
  analysisSnapshot,
  encodePosition,
  type Analysis,
  type Activation,
} from "./gameViewModel";
import type { SearchSnapshotMessage } from "./protocol";

export function usePositionAnalysis(
  cells: number[],
  recorded: boolean,
  session: Session,
  setError: (error: string) => void,
) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null),
    [snapshots, setSnapshots] = useState<SearchSnapshotMessage[]>([]);
  const [activationTensors, setActivationTensors] = useState<Activation[]>([]);
  const [activationLayer, setActivationLayer] = useState("trunk.block_0.conv1");
  const [snapshotIndex, setSnapshotIndex] = useState<number | null>(null);
  const [info, setInfo] = useState<{
    available: boolean;
    checkpoint: string;
    max_simulations: number;
  } | null>(null);
  const [infoPending, setInfoPending] = useState(true);
  const [simulations, setSimulations] = useState(1000),
    [busy, setBusy] = useState(false);
  const [inspect, setInspect] = useState(false),
    [autoAnalyze, setAutoAnalyze] = useState(true);
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const request = useRef<AbortController | null>(null);
  const requestId = useRef<string | null>(null);
  const cancellation = useRef<Promise<void>>(Promise.resolve());
  function stopSearch() {
    const previous = request.current,
      id = requestId.current;
    request.current = null;
    requestId.current = null;
    if (previous && id) {
      const cancel = async () => {
        try {
          await api(
            "analysis/cancel",
            { request_id: id },
            sessionRef.current,
            AbortSignal.timeout(10000),
          );
        } catch (e) {
          setError(
            `Could not confirm search cancellation: ${String(e)}. If the server is still busy, retry shortly.`,
          );
        } finally {
          previous.abort();
        }
      };
      // Always settle the gate: a failed cancellation must not poison later searches.
      cancellation.current = cancellation.current.then(cancel, cancel);
    }
  }
  useEffect(() => {
    api<typeof info>("analysis")
      .then(setInfo)
      .catch((e) => setError(String(e)))
      .finally(() => setInfoPending(false));
    return () => stopSearch();
  }, []);
  async function search() {
    if (busy || !session.authenticated || !info?.available) return;
    if (
      !Number.isInteger(simulations) ||
      simulations < 1 ||
      simulations > info.max_simulations
    ) {
      setError(`Choose 1–${info.max_simulations} simulations.`);
      return;
    }
    if (
      analysis?.result.complete &&
      analysis.result.searched_simulations >= simulations &&
      (!inspect || analysis.result.activations.length)
    )
      return;
    const controller = new AbortController();
    request.current = controller;
    const id = crypto.randomUUID();
    requestId.current = id;
    setBusy(true);
    setError("");
    setSnapshotIndex(null);
    try {
      await cancellation.current;
      if (request.current !== controller) return;
      await streamAnalysis<Analysis>(
        {
          request_id: id,
          position: { game_type: "gomoku19_five_v1", state: { cells } },
          simulations,
          inspect,
          layers: inspect ? [activationLayer] : [],
        },
        session,
        (a) => {
          if (request.current !== controller) return;
          if (a.result.activations.length)
            setActivationTensors(a.result.activations);
          setAnalysis((previous) =>
            previous && !a.result.activations.length
              ? {
                  ...a,
                  result: {
                    ...a.result,
                    activations: previous.result.activations,
                  },
                }
              : a,
          );
          setSnapshots((old) => {
            const next = [...old, analysisSnapshot(a)];
            // Thin long searches while retaining the initial carried-visit count.
            return next.length > 512
              ? next.filter((_, i) => i % 2 === 0 || i === next.length - 1)
              : next;
          });
        },
        controller.signal,
      );
    } catch (e) {
      if (request.current === controller && !controller.signal.aborted)
        setError(String(e));
    } finally {
      if (request.current === controller) {
        request.current = null;
        requestId.current = null;
        setBusy(false);
      }
    }
  }
  useEffect(() => {
    if (!recorded && autoAnalyze && session.authenticated && info?.available)
      void search();
  }, [cells, autoAnalyze, session.authenticated, info?.available]);
  async function loadLayer(name: string) {
    if (busy) return;
    const controller = new AbortController(),
      id = crypto.randomUUID();
    request.current = controller;
    requestId.current = id;
    setBusy(true);
    setError("");
    try {
      await cancellation.current;
      if (request.current !== controller) return;
      const response = await api<Analysis>(
        "analyze",
        {
          request_id: id,
          position: { game_type: "gomoku19_five_v1", state: { cells } },
          simulations: 0,
          inspect: true,
          layers: [name],
        },
        session,
        controller.signal,
      );
      if (request.current !== controller) return;
      setActivationTensors(response.result.activations);
      setAnalysis((previous) =>
        previous
          ? {
              ...previous,
              result: {
                ...previous.result,
                activations: response.result.activations,
              },
            }
          : response,
      );
    } catch (e) {
      if (request.current === controller) setError(String(e));
    } finally {
      if (request.current === controller) {
        request.current = null;
        requestId.current = null;
        setBusy(false);
      }
    }
  }
  const captureAttempt = useRef<string | null>(null);
  useEffect(() => {
    if (!inspect) {
      captureAttempt.current = null;
      return;
    }
    const key = `${encodePosition(cells)}:${activationLayer}`;
    if (
      recorded ||
      busy ||
      request.current ||
      !session.authenticated ||
      !info?.available ||
      captureAttempt.current === key ||
      analysis?.result.activations.some(
        (a) => a.name === activationLayer && a.values.length,
      )
    )
      return;
    captureAttempt.current = key;
    void loadLayer(activationLayer);
  }, [
    inspect,
    cells,
    activationLayer,
    busy,
    session.authenticated,
    info?.available,
  ]);
  const targetReached =
    !!analysis?.result.complete &&
    analysis.result.searched_simulations >= simulations &&
    (!inspect || analysis.result.activations.length > 0);
  function resetAnalysis() {
    stopSearch();
    setAnalysis(null);
    setSnapshots([]);
    setSnapshotIndex(null);
    setBusy(false);
  }
  return {
    analysis,
    snapshots,
    activationTensors,
    activationLayer,
    setActivationLayer,
    snapshotIndex,
    setSnapshotIndex,
    info,
    infoPending,
    simulations,
    setSimulations,
    busy,
    inspect,
    setInspect,
    autoAnalyze,
    setAutoAnalyze,
    search,
    loadLayer,
    targetReached,
    resetAnalysis,
  };
}
