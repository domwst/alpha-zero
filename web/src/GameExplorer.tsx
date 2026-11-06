import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { usePlotWidth, pointerX } from "./InteractiveChart";
import { unpackPosition, terminalPosition } from "./replayPosition";
export { unpackPosition } from "./replayPosition";
import { usePositionAnalysis } from "./usePositionAnalysis";
import { ActivationViewer } from "./ActivationViewer";
import { ReplayArchive } from "./ReplayArchive";
import {
  gameName,
  empty,
  encodePosition,
  decodePosition,
  position,
  snapshot,
  analysisSnapshot,
  PAGE_SIZE,
  type Game,
  type GameRef,
  type ArchivePage,
} from "./gameViewModel";
export { encodePosition, decodePosition } from "./gameViewModel";
import type { JSX } from "preact";
import { Board, type Overlay } from "./Board";
import { InfoTip } from "./InfoTip";
import { LoadingStatus } from "./LoadingStatus";
import { SearchChart } from "./SearchChart";
import { api, duration, type Session } from "./serviceApi";
import { moveName, cellKey, type Cell } from "./protocol";

// This board view is the Gomoku adapter. Other games supply their own state,
// legal actions (including non-board actions), URLs, and renderer here.
export function GameExplorer(props: { session: Session; recorded: boolean }) {
  const gameType =
    new URLSearchParams(location.search).get("game") || "gomoku19_five_v1";
  if (gameType !== "gomoku19_five_v1")
    return (
      <main className="service-main">
        <h1>Unsupported game</h1>
        <p>No view is registered for {gameType}.</p>
      </main>
    );
  return <GomokuExplorer {...props} />;
}
const valueText = (value: number | null | undefined) =>
  value == null ? "Unavailable" : value.toFixed(4);
function Segments<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled = false,
}: {
  label: string;
  value: T;
  options: [T, string][];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="service-segments" role="group" aria-label={label}>
      {options.map(([key, title]) => (
        <button
          key={key}
          aria-pressed={value === key}
          disabled={disabled}
          onClick={() => onChange(key)}
        >
          {title}
        </button>
      ))}
    </div>
  );
}
function GomokuExplorer({
  session,
  recorded,
}: {
  session: Session;
  recorded: boolean;
}) {
  const params = new URLSearchParams(location.search);
  const [job, setJob] = useState(params.get("job") || ""),
    [jobs, setJobs] = useState<{ id: string; title: string }[]>([]);
  const [requested, setRequested] = useState({
    epoch: params.get("epoch") || "",
    offset: 0,
    winner: "",
    sort: "newest",
  });
  const [page, setPage] = useState<ArchivePage | null>(null),
    [listBusy, setListBusy] = useState(false);
  const [gameKey, setGameKey] = useState("");
  const [game, setGame] = useState<Game | null>(null),
    [gameBusy, setGameBusy] = useState(false),
    [ply, setPly] = useState(0);
  const loadId = useRef(0);
  const [playing, setPlaying] = useState(false);
  const [cells, setCells] = useState(() =>
    decodePosition(params.get("position")),
  );
  const [history, setHistory] = useState<
      { cells: number[]; last: Cell | null }[]
    >([]),
    [lastMove, setLastMove] = useState<Cell | null>(null);
  const [focus, setFocus] = useState<Cell | null>(null),
    [overlay, setOverlay] = useState<Overlay>("prior");
  const [policyKind, setPolicyKind] = useState<"prior" | "search" | "sampling">(
    "prior",
  );
  const [error, setError] = useState("");
  const [jobsPending, setJobsPending] = useState(recorded);
  const {
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
  } = usePositionAnalysis(cells, recorded, session, setError);
  useEffect(() => {
    if (recorded)
      api<{ jobs: typeof jobs }>("game-jobs")
        .then((x) => setJobs(x.jobs))
        .catch((e) => setError(String(e)))
        .finally(() => setJobsPending(false));
  }, [recorded]);
  useEffect(() => {
    if (!recorded || !job) return;
    let cancelled = false;
    setListBusy(true);
    api<Omit<ArchivePage, "offset" | "epoch" | "winner" | "sort">>(
      `games?job_id=${encodeURIComponent(job)}&offset=${requested.offset}&limit=${PAGE_SIZE}${requested.epoch ? `&epoch=${requested.epoch}` : ""}${requested.epoch && requested.winner ? `&winner=${requested.winner}` : ""}${requested.epoch && requested.sort !== "newest" ? `&sort=${requested.sort}` : ""}`,
    )
      .then((x) => {
        if (!cancelled) {
          setPage({ ...x, ...requested });
          setListBusy(false);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(String(e));
          setListBusy(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [job, requested, recorded]);
  async function load(ref: GameRef) {
    const id = ++loadId.current;
    setPlaying(false);
    setGameBusy(true);
    setError("");
    try {
      const result = await api<Game>(
        `game?job_id=${ref.job_id}&epoch=${ref.epoch}&id=${ref.id}`,
      );
      if (id !== loadId.current) return;
      if (result.game_type !== "gomoku19_five_v1")
        throw new Error("This archive requires another game adapter");
      setGame(result);
      setGameKey(`${ref.epoch}/${ref.id}`);
      setPly(0);
      setFocus(null);
    } catch (e) {
      if (id === loadId.current) setError(String(e));
    } finally {
      if (id === loadId.current) setGameBusy(false);
    }
  }
  const terminalCells = game ? terminalPosition(game.record) : null;
  const lastPosition = game
    ? Math.max(0, game.record.plies.length - (terminalCells ? 0 : 1))
    : 0;
  const atTerminal =
    !!game && !!terminalCells && ply === game.record.plies.length;
  const entry = game?.record.plies[ply],
    viewCells =
      recorded && atTerminal
        ? terminalCells!
        : recorded && entry
          ? unpackPosition(entry.state.state)
          : cells;
  const actualPly = viewCells.filter((n) => n !== 0).length,
    toMove = actualPly % 2 ? "white" : "black";
  const previousAction =
    recorded && ply > 0 ? game?.record.plies[ply - 1]?.action : null;
  const boardPosition = position(
    viewCells,
    actualPly,
    recorded
      ? previousAction
        ? { row: previousAction.x, column: previousAction.y }
        : null
      : lastMove,
  );
  const legal = viewCells.flatMap((n, i) =>
    n === 0 ? [{ row: Math.floor(i / 19), column: i % 19 }] : [],
  );
  const diagnostics = entry?.decision.diagnostics;
  const distribution =
    policyKind === "prior"
      ? diagnostics?.network_prior
      : policyKind === "search"
        ? entry?.decision.training_policy
        : diagnostics?.sampling_policy;
  const savedSnapshot = distribution
    ? snapshot(
        legal.map((m, i) => ({
          ...m,
          prior: distribution[i] || 0,
          visits: diagnostics?.root_visits?.[i] || 0,
          mean_value: null,
        })),
        diagnostics?.value_estimate ?? 0,
        diagnostics?.search_value ?? null,
      )
    : null;
  const currentSnapshot = recorded
    ? savedSnapshot
    : snapshotIndex != null
      ? snapshots[snapshotIndex] || null
      : analysis
        ? analysisSnapshot(analysis)
        : null;
  const stats = recorded
    ? diagnostics?.search
    : analysis?.result.search || {
        expanded_by_depth: [],
        allocated_by_depth: [],
        simulation_leaf_depth: [],
      };
  const networkValue = recorded
      ? diagnostics?.value_estimate
      : currentSnapshot?.network_value,
    searchValue = recorded
      ? diagnostics?.search_value
      : currentSnapshot?.search_value;
  function changePosition(next: number[], last: Cell | null) {
    resetAnalysis();
    setCells(next);
    setLastMove(last);
    setFocus(null);
    window.history.replaceState(
      null,
      "",
      `${location.pathname}?game=gomoku19_five_v1&position=${encodePosition(next)}`,
    );
  }
  function move(cell: Cell) {
    if (
      recorded ||
      analysis?.result.terminal != null ||
      cells[cell.row * 19 + cell.column] !== 0
    )
      return;
    setHistory((old) => [...old, { cells, last: lastMove }]);
    changePosition(
      cells.map((n, i) =>
        i === cell.row * 19 + cell.column ? 2 : n === 1 ? 2 : n === 2 ? 1 : 0,
      ),
      cell,
    );
  }
  useEffect(() => {
    if (!recorded || !game || gameBusy) return;
    function navigate(event: KeyboardEvent) {
      const target = event.target as HTMLElement;
      if (
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        target.closest("input, textarea, select, [contenteditable=true]")
      )
        return;
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      event.stopPropagation();
      setPly((old) =>
        Math.max(
          0,
          Math.min(lastPosition, old + (event.key === "ArrowRight" ? 1 : -1)),
        ),
      );
      setFocus(null);
    }
    window.addEventListener("keydown", navigate, true);
    return () => window.removeEventListener("keydown", navigate, true);
  }, [recorded, game, gameBusy, lastPosition]);
  const best =
    analysis &&
    [...analysis.result.moves].sort(
      (a, b) => b.visits - a.visits || b.prior - a.prior,
    )[0];
  const choosePage = (
    epoch: string,
    offset: number,
    winner = requested.winner,
    sort = requested.sort,
  ) => {
    setListBusy(true);
    // Outcome and order only apply within one epoch; clearing the epoch
    // clears them too so the controls never show filters that are not applied.
    setRequested(epoch ? { epoch, offset, winner, sort } : { epoch, offset, winner: "", sort: "newest" });
  };
  useEffect(() => {
    if (!playing || !recorded || !game) return;
    const timer = setInterval(() => {
      setPly((old) => {
        if (old >= lastPosition) {
          setPlaying(false);
          return old;
        }
        return old + 1;
      });
      setFocus(null);
    }, 650);
    return () => clearInterval(timer);
  }, [playing, recorded, game, lastPosition]);
  useEffect(() => setPlaying(false), [gameKey]);
  // Black's winning chances at each recorded move, from the search value when
  // it was recorded (the raw network value is optimistic for whoever is to
  // move, which reads as a sawtooth) and the network value otherwise.
  const trajectory = useMemo(
    () =>
      game
        ? game.record.plies.map((entry) => {
            const diagnostics = entry.decision.diagnostics;
            const value = diagnostics.search_value ?? diagnostics.value_estimate;
            if (value == null || !Number.isFinite(value)) return null;
            return ((entry.actor === "First" ? value : -value) + 1) / 2;
          })
        : [],
    [game],
  );
  return (
    <main className="service-main">
      <div className="service-heading">
        <div>
          <h1>{recorded ? "Recorded games" : "Position analysis"}</h1>
          <p>
            {recorded
              ? "Inspect the distributions recorded when a move was played."
              : "Explore branches with the pinned analysis checkpoint."}
          </p>
        </div>
      </div>
      {error && (
        <p className="service-error" role="alert">
          {error}
        </p>
      )}
      <div className="service-game-layout">
        <section className="service-panel" aria-busy={gameBusy}>
          <div className="service-toolbar">
            <h2>
              {recorded
                ? game
                  ? `Game ${gameName(game.game_id)} · ${atTerminal ? "Final position" : `Move ${ply + 1}`}`
                  : "Select a game"
                : `Move ${actualPly + 1} · ${toMove} to play`}
            </h2>
            {recorded && game && (
              <div className="service-actions">
                <button
                  title="First position"
                  aria-label="First position"
                  disabled={!ply || gameBusy}
                  onClick={() => {
                    setPly(0);
                    setFocus(null);
                  }}
                >
                  ⏮
                </button>
                <div className="service-step-buttons">
                  <button
                    disabled={!ply || gameBusy}
                    onClick={() => {
                      setPly(ply - 1);
                      setFocus(null);
                    }}
                  >
                    Previous
                  </button>
                  <button
                    disabled={ply >= lastPosition || gameBusy}
                    onClick={() => {
                      setPly(ply + 1);
                      setFocus(null);
                    }}
                  >
                    Next
                  </button>
                </div>
                <button
                  title="Last recorded position"
                  aria-label="Last recorded position"
                  disabled={ply >= lastPosition || gameBusy}
                  onClick={() => {
                    setPly(lastPosition);
                    setFocus(null);
                  }}
                >
                  ⏭
                </button>
              </div>
            )}
          </div>
          <LoadingStatus pending={gameBusy}>Loading game…</LoadingStatus>
          <div className="service-toolbar">
            {recorded ? (
              <Segments
                label="Distribution"
                disabled={atTerminal}
                value={policyKind}
                onChange={setPolicyKind}
                options={[
                  ["prior", "Network prior"],
                  ["search", "Training policy"],
                  ["sampling", "Move sampling"],
                ]}
              />
            ) : (
              <Segments
                label="Board overlay"
                value={overlay}
                onChange={setOverlay}
                options={[
                  ["prior", "Network prior"],
                  ["visits", "Search visits"],
                  ["move", "Move probability"],
                ]}
              />
            )}
            {recorded && (entry || atTerminal) && (
              <a
                className="service-button"
                href={`/analyze?game=gomoku19_five_v1&position=${encodePosition(viewCells)}`}
              >
                Explore this position
              </a>
            )}
          </div>
          <div className="service-position-layout">
            <Board
              position={boardPosition}
              snapshot={currentSnapshot}
              overlay={recorded ? "prior" : overlay}
              showPolicy={!!currentSnapshot}
              temperature={1}
              focus={focus}
              onFocusCell={setFocus}
              canPlay={!recorded && analysis?.result.terminal == null}
              onPlay={move}
              policyLabel={
                recorded
                  ? policyKind === "prior"
                    ? "Network prior"
                    : policyKind === "search"
                      ? "Training policy"
                      : "Sampling probability"
                  : undefined
              }
              hideMoveProbability={recorded && policyKind === "sampling"}
              recordedSampling={
                recorded
                  ? diagnostics?.sampling_policy
                    ? new Map(
                        legal.map((m, i) => [
                          cellKey(m),
                          diagnostics.sampling_policy![i] || 0,
                        ]),
                      )
                    : null
                  : undefined
              }
            />
            <aside className="service-move-rail">
              {(entry || atTerminal || !recorded) && (
                <>
                  <div className="service-inference service-position-values">
                    <div>
                      <strong>{valueText(networkValue)}</strong>
                      <small>Network value · {toMove}</small>
                    </div>
                    <div>
                      <strong>{valueText(searchValue)}</strong>
                      <small>Search value · {toMove}</small>
                    </div>
                    <div>
                      <strong>
                        {currentSnapshot?.total_visits.toLocaleString() ?? "—"}
                      </strong>
                      <small>Tree visits</small>
                    </div>
                    {!recorded && analysis && (
                      <div>
                        <strong>
                          {Math.round(
                            analysis.result.simulations_per_second,
                          ).toLocaleString()}
                        </strong>
                        <small>
                          Simulations/s ·{" "}
                          {duration(analysis.result.elapsed_ms / 1000)}
                        </small>
                      </div>
                    )}
                  </div>
                  {!recorded && (
                    <progress
                      className="service-progress"
                      aria-label="Search progress"
                      value={Math.max(
                        0,
                        (analysis?.result.searched_simulations ?? 0) -
                          (analysis?.result.carried_visits ?? 0),
                      )}
                      max={Math.max(simulations, 1)}
                    />
                  )}
                  {!recorded && analysis?.result.terminal != null && (
                    <p role="status">
                      {analysis.result.terminal === 0
                        ? "Draw"
                        : `${analysis.result.terminal > 0 ? toMove : toMove === "black" ? "white" : "black"} won`}
                    </p>
                  )}
                </>
              )}

              {atTerminal && game && (
                <p role="status">
                  {game.record.terminal_value == null
                    ? "Final recorded position · outcome unavailable"
                    : game.record.terminal_value === 0
                      ? "Draw"
                      : `${(game.record.terminal_actor ? game.record.terminal_actor === "First" : toMove === "black") === game.record.terminal_value > 0 ? "Black" : "White"} won`}
                </p>
              )}
              {atTerminal && (
                <p className="service-runtime">
                  Game over. No move probabilities or search statistics apply to
                  this final position.
                </p>
              )}
              {recorded && entry && !distribution && (
                <p>This distribution was not recorded.</p>
              )}
              {stats && (
                <details>
                  <summary>Search depth and width</summary>
                  <div className="service-table-scroll">
                    <table className="service-depth-table">
                      <thead>
                        <tr>
                          <th scope="col">Depth</th>
                          <th scope="col">Expanded</th>
                          <th scope="col">Allocated</th>
                          <th scope="col">Leaves visited</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Array.from(
                          {
                            length: Math.max(
                              stats.expanded_by_depth.length,
                              stats.allocated_by_depth.length,
                              stats.simulation_leaf_depth.length,
                            ),
                          },
                          (_, i) => (
                            <tr key={i}>
                              <td>{i}</td>
                              <td>{stats.expanded_by_depth[i] || 0}</td>
                              <td>{stats.allocated_by_depth[i] || 0}</td>
                              <td>{stats.simulation_leaf_depth[i] || 0}</td>
                            </tr>
                          ),
                        )}
                      </tbody>
                    </table>
                  </div>
                </details>
              )}
            </aside>
          </div>
          {recorded && game && (
            <p>
              {atTerminal
                ? "Final position"
                : `Recorded action: ${
                    entry?.action
                      ? moveName({
                          row: entry.action.x,
                          column: entry.action.y,
                        })
                      : "Unavailable"
                  }`}{" "}
              · Policy: {game?.policy_semantics.replaceAll("_", " ")} ·{" "}
              {game?.provenance}
            </p>
          )}
          {recorded && game && (
            <GameTimeline
              trajectory={trajectory}
              ply={ply}
              lastPosition={lastPosition}
              plies={game.record.plies}
              playing={playing}
              busy={gameBusy}
              onSelect={(next) => {
                setPly(next);
                setFocus(null);
              }}
              onTogglePlay={() => {
                if (!playing && ply >= lastPosition) setPly(0);
                setPlaying(!playing);
              }}
            />
          )}
          {!recorded && (
            <div aria-busy={busy && !snapshots.length}>
              <SearchChart
                key={encodePosition(cells)}
                snapshots={snapshots}
                selectedIndex={snapshotIndex}
                onSelectIndex={setSnapshotIndex}
                focusCell={focus}
                onFocusMove={setFocus}
              />
            </div>
          )}
        </section>
        
        <aside className="service-panel">
          {recorded ? (
            <ReplayArchive
              job={job}
              jobs={jobs}
              jobsPending={jobsPending}
              page={page}
              listBusy={listBusy}
              gameBusy={gameBusy}
              gameKey={gameKey}
              choosePage={choosePage}
              load={load}
              onJobChange={(id) => {
                setPlaying(false);
                setJob(id);
                setPage(null);
                setRequested({ epoch: "", offset: 0, winner: "", sort: "newest" });
                setGame(null);
                ++loadId.current;
                setGameBusy(false);
                window.history.replaceState(null, "", `/games?job=${id}`);
              }}
            />
          ) : (
            <>
              <h2>Search</h2>
              <LoadingStatus pending={infoPending}>
                Loading analysis checkpoint…
              </LoadingStatus>
              {!infoPending && (
                <p>
                  {info?.available
                    ? `Checkpoint: ${info.checkpoint.split("/").slice(-2).join("/")}`
                    : info
                      ? "No analysis checkpoint configured."
                      : "Analysis checkpoint could not be loaded."}
                </p>
              )}
              {session.authenticated && info?.available ? (
                <>
                  <label>
                    Target simulations
                    <input
                      type="number"
                      min="1"
                      max={info.max_simulations}
                      value={simulations}
                      onInput={(e) =>
                        setSimulations(Number(e.currentTarget.value))
                      }
                    />
                    <small>
                      Counts new simulations; visits retained from the previous position stay on top.
                    </small>
                  </label>
                  <label className="service-checkbox">
                    <input
                      type="checkbox"
                      checked={autoAnalyze}
                      onChange={(e) => setAutoAnalyze(e.currentTarget.checked)}
                    />
                    Analyze after each move
                  </label>
                  <label className="service-checkbox">
                    <input
                      type="checkbox"
                      checked={inspect}
                      onChange={(e) => setInspect(e.currentTarget.checked)}
                    />
                    Capture activations
                  </label>
                  <div className="service-actions service-section-actions">
                    <button
                      className="primary"
                      disabled={busy || targetReached}
                      aria-describedby={
                        targetReached ? "analysis-target-help" : undefined
                      }
                      onClick={search}
                    >
                      {busy ? "Searching…" : "Analyze position"}
                    </button>
                    {targetReached && (
                      <InfoTip
                        id="analysis-target-help"
                        title="Target reached"
                        triggerLabel="Why is Analyze position disabled?"
                        variant="trigger"
                      >
                        Increase the target to extend this search.
                      </InfoTip>
                    )}
                  </div>
                </>
              ) : (
                <p>Log in to run a search.</p>
              )}
              <div className="service-actions service-section-actions">
                <button
                  disabled={!history.length}
                  onClick={() => {
                    const prev = history.at(-1)!;
                    setHistory(history.slice(0, -1));
                    changePosition(prev.cells, prev.last);
                  }}
                >
                  Undo move
                </button>
                <button
                  onClick={() => {
                    setHistory([]);
                    changePosition(empty(), null);
                  }}
                >
                  New game
                </button>
              </div>
              {
                <button
                  disabled={!best || analysis?.result.terminal != null}
                  onClick={() => best && move(best)}
                >
                  Play most visited move
                </button>
              }
            </>
          )}
        </aside>
      </div>
      {activationTensors.length > 0 && (
        <ActivationViewer
          activations={activationTensors}
          layerName={activationLayer}
          setLayerName={setActivationLayer}
          positionPending={
            !analysis?.result.activations.some((a) => a.values.length)
          }
          loadLayer={loadLayer}
          busy={busy}
          board={boardPosition}
        />
      )}
    </main>
  );
}

function GameTimeline({
  trajectory,
  ply,
  lastPosition,
  plies,
  playing,
  busy,
  onSelect,
  onTogglePlay,
}: {
  trajectory: (number | null)[];
  ply: number;
  lastPosition: number;
  plies: Game["record"]["plies"];
  playing: boolean;
  busy: boolean;
  onSelect: (ply: number) => void;
  onTogglePlay: () => void;
}) {
  const { ref, width } = usePlotWidth();
  const height = 110,
    left = 40,
    right = width - 10,
    top = 8,
    bottom = 88;
  const count = Math.max(lastPosition + 1, 1);
  const x = (index: number) => left + (index / Math.max(count - 1, 1)) * (right - left);
  const y = (share: number) => bottom - share * (bottom - top);
  const points = trajectory
    .map((share, index) => (share == null ? null : [x(index), y(share)] as const))
    .filter((p): p is readonly [number, number] => p != null);
  const line = points.map((p) => p.join(",")).join(" ");
  const half = y(0.5);
  const upper = points.map((p) => `${p[0]},${Math.min(p[1], half)}`);
  const lower = points.map((p) => `${p[0]},${Math.max(p[1], half)}`);
  const select = (event: JSX.TargetedPointerEvent<SVGSVGElement> | JSX.TargetedMouseEvent<SVGSVGElement>) => {
    const position = pointerX(event);
    const index = Math.round(((position - left) / Math.max(right - left, 1)) * (count - 1));
    onSelect(Math.max(0, Math.min(lastPosition, index)));
  };
  const window = Array.from({ length: 7 }, (_, i) => ply - 3 + i).filter(
    (i) => i >= 0 && i <= lastPosition,
  );
  const current = trajectory[ply];
  return (
    <figure className="service-game-timeline" ref={ref}>
      <div className="service-toolbar" style={{ marginBottom: 0 }}>
        <span className="service-runtime">
          Black's winning chances from the recorded search value at each move
          {current != null && ` · now ${Math.round(current * 100)}%`}
        </span>
        <span className="service-runtime">hover to preview · click to jump</span>
      </div>
      {points.length > 1 ? (
        <svg
          className="service-trajectory"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Black's winning chances over the game"
          onClick={select}
          onPointerMove={(event) => {
            if (event.buttons) select(event);
          }}
        >
          {points.length > 1 && (
            <>
              <polygon className="black-area" points={`${upper.join(" ")} ${upper.at(-1)!.split(",")[0]},${half} ${upper[0]!.split(",")[0]},${half}`} />
              <polygon className="white-area" points={`${lower.join(" ")} ${lower.at(-1)!.split(",")[0]},${half} ${lower[0]!.split(",")[0]},${half}`} />
            </>
          )}
          {[0, 0.5, 1].map((tick) => (
            <g key={tick}>
              <line className="chart-grid" x1={left} x2={right} y1={y(tick)} y2={y(tick)} stroke-dasharray={tick === 0.5 ? "4 4" : undefined} />
              <text x={left - 6} y={y(tick) + 4} text-anchor="end">
                {Math.round(tick * 100)}%
              </text>
            </g>
          ))}
          <polyline className="line" points={line} />
          <line className="cursor" x1={x(ply)} x2={x(ply)} y1={top} y2={bottom} />
          {current != null && <circle className="played" cx={x(ply)} cy={y(current)} r="4" />}
          <text x={left} y={height - 2}>1</text>
          <text x={right} y={height - 2} text-anchor="end">{lastPosition + 1}</text>
        </svg>
      ) : (
        <p className="service-runtime">No recorded network values for this game.</p>
      )}
      <input
        type="range"
        min="0"
        max={lastPosition}
        value={ply}
        aria-label="Position in game"
        disabled={busy}
        onInput={(event) => onSelect(Number(event.currentTarget.value))}
      />
      <div className="service-actions">
        <button aria-label="First position" title="First position" disabled={!ply || busy} onClick={() => onSelect(0)}>
          ⏮
        </button>
        <button aria-label="Previous move" title="Previous move" disabled={!ply || busy} onClick={() => onSelect(ply - 1)}>
          ◀
        </button>
        <button className={playing ? "" : "primary"} disabled={busy} onClick={onTogglePlay} aria-pressed={playing}>
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button aria-label="Next move" title="Next move" disabled={ply >= lastPosition || busy} onClick={() => onSelect(ply + 1)}>
          ▶
        </button>
        <button aria-label="Last position" title="Last position" disabled={ply >= lastPosition || busy} onClick={() => onSelect(lastPosition)}>
          ⏭
        </button>
        <span className="service-runtime">
          move {ply + 1} of {lastPosition + 1} · ← → keys step, Home/End jump
        </span>
        <span className="service-ply-strip">
          {window.map((index) => {
            const action = plies[index]?.action;
            return (
              <button
                key={index}
                aria-current={index === ply ? "true" : undefined}
                disabled={busy}
                onClick={() => onSelect(index)}
              >
                {index + 1}{" "}
                {action ? moveName({ row: action.x, column: action.y }) : index === plies.length ? "end" : "?"}
              </button>
            );
          })}
        </span>
      </div>
    </figure>
  );
}
