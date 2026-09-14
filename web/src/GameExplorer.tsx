import { useEffect, useRef, useState } from "preact/hooks";
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
  });
  const [page, setPage] = useState<ArchivePage | null>(null),
    [listBusy, setListBusy] = useState(false);
  const [gameKey, setGameKey] = useState("");
  const [game, setGame] = useState<Game | null>(null),
    [gameBusy, setGameBusy] = useState(false),
    [ply, setPly] = useState(0);
  const loadId = useRef(0);
  const [cells, setCells] = useState(() =>
    decodePosition(params.get("position")),
  );
  const [history, setHistory] = useState<
      { cells: number[]; last: Cell | null }[]
    >([]),
    [lastMove, setLastMove] = useState<Cell | null>(null);
  const [selected, setSelected] = useState<Cell | null>(null),
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
    api<Omit<ArchivePage, "offset" | "epoch">>(
      `games?job_id=${encodeURIComponent(job)}&offset=${requested.offset}&limit=${PAGE_SIZE}${requested.epoch ? `&epoch=${requested.epoch}` : ""}`,
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
      setSelected(null);
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
    setSelected(null);
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
      setSelected(null);
    }
    window.addEventListener("keydown", navigate, true);
    return () => window.removeEventListener("keydown", navigate, true);
  }, [recorded, game, gameBusy, lastPosition]);
  const best =
    analysis &&
    [...analysis.result.moves].sort(
      (a, b) => b.visits - a.visits || b.prior - a.prior,
    )[0];
  const choosePage = (epoch: string, offset: number) => {
    setListBusy(true);
    setRequested({ epoch, offset });
  };
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
                setJob(id);
                setPage(null);
                setRequested({ epoch: "", offset: 0 });
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
                      Applies to the next search; completed visits are retained.
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
                    setSelected(null);
                  }}
                >
                  ⏮
                </button>
                <div className="service-step-buttons">
                  <button
                    disabled={!ply || gameBusy}
                    onClick={() => {
                      setPly(ply - 1);
                      setSelected(null);
                    }}
                  >
                    Previous
                  </button>
                  <button
                    disabled={ply >= lastPosition || gameBusy}
                    onClick={() => {
                      setPly(ply + 1);
                      setSelected(null);
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
                    setSelected(null);
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
              selected={selected}
              onNavigate={setSelected}
              onSelect={(cell) => {
                if (
                  !recorded &&
                  selected &&
                  cellKey(cell) === cellKey(selected)
                )
                  move(cell);
                else setSelected(cell);
              }}
              policyLabel={
                recorded
                  ? policyKind === "prior"
                    ? "Network prior"
                    : policyKind === "search"
                      ? "Training policy"
                      : "Sampling probability"
                  : "Network policy"
              }
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
                  {!recorded && analysis && (
                    <progress
                      className="service-progress"
                      aria-label="Search progress"
                      value={analysis.result.searched_simulations}
                      max={Math.max(
                        simulations,
                        analysis.result.searched_simulations,
                      )}
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
              {atTerminal ? (
                <p className="service-runtime">
                  Game over. No move probabilities or search statistics apply to
                  this final position.
                </p>
              ) : selected ? (
                <div className="service-selected-move">
                  <h3>{moveName(selected)}</h3>
                  <dl>
                    {recorded
                      ? (
                          [
                            "Network prior",
                            "Training policy",
                            "Sampling probability",
                          ] as const
                        ).map((label, i) => {
                          const index = legal.findIndex(
                            (m) => cellKey(m) === cellKey(selected),
                          );
                          const v = [
                            diagnostics?.network_prior,
                            entry?.decision.training_policy,
                            diagnostics?.sampling_policy,
                          ][i]?.[index];
                          return (
                            <div key={label}>
                              <dt>{label}</dt>
                              <dd>
                                {v == null
                                  ? "Unavailable"
                                  : `${(v * 100).toFixed(3)}%`}
                              </dd>
                            </div>
                          );
                        })
                      : (() => {
                          const m = currentSnapshot?.moves.find(
                            (m) => cellKey(m) === cellKey(selected),
                          );
                          return (
                            <>
                              <div>
                                <dt>Network prior</dt>
                                <dd>
                                  {m ? `${(m.prior * 100).toFixed(3)}%` : "—"}
                                </dd>
                              </div>
                              <div>
                                <dt>Visits</dt>
                                <dd>{m?.visits ?? "—"}</dd>
                              </div>
                              <div>
                                <dt>Action value Q</dt>
                                <dd>{valueText(m?.mean_value)}</dd>
                              </div>
                            </>
                          );
                        })()}
                  </dl>
                  {!recorded && (
                    <button onClick={() => move(selected)}>
                      Play {moveName(selected)}
                    </button>
                  )}
                </div>
              ) : (
                <p className="service-runtime">
                  Select a move to inspect its statistics.
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
          {!recorded && (
            <div aria-busy={busy && !snapshots.length}>
              <SearchChart
                key={encodePosition(cells)}
                snapshots={snapshots}
                selectedIndex={snapshotIndex}
                onSelectIndex={setSnapshotIndex}
                selectedCell={selected}
                onSelectMove={setSelected}
              />
            </div>
          )}
        </section>
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
