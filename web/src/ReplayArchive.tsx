import { LoadingStatus } from "./LoadingStatus";
import {
  gameName,
  PAGE_SIZE,
  type ArchivePage,
  type GameRef,
} from "./gameViewModel";

export function ReplayArchive({
  job,
  jobs,
  jobsPending,
  page,
  listBusy,
  gameBusy,
  gameKey,
  onJobChange,
  choosePage,
  load,
}: {
  job: string;
  jobs: { id: string; title: string }[];
  jobsPending: boolean;
  page: ArchivePage | null;
  listBusy: boolean;
  gameBusy: boolean;
  gameKey: string;
  onJobChange: (id: string) => void;
  choosePage: (epoch: string, offset: number) => void;
  load: (ref: GameRef) => void;
}) {
  return (
    <>
      <h2>Game archive</h2>
      <LoadingStatus pending={jobsPending}>
        Loading experiments with recorded games…
      </LoadingStatus>
      <label>
        Experiment
        <select
          disabled={jobsPending}
          value={job}
          onChange={(e) => onJobChange(e.currentTarget.value)}
        >
          <option value="">Select an experiment</option>
          {jobs.map((j) => (
            <option key={j.id} value={j.id}>
              {j.title}
            </option>
          ))}
        </select>
      </label>
      {job && (
        <>
          <label>
            Epoch
            <select
              disabled={listBusy}
              value={page?.epoch || ""}
              onChange={(e) => choosePage(e.currentTarget.value, 0)}
            >
              <option value="">All epochs</option>
              {[...(page?.epochs || [])]
                .sort()
                .reverse()
                .map((v) => (
                  <option key={v} value={v}>
                    {page?.epoch_labels[v] || `Epoch ${Number(v) + 1}`}
                  </option>
                ))}
            </select>
          </label>
          <div className="service-game-list" aria-busy={listBusy}>
            {page?.games.map((r) => (
              <button
                key={`${r.epoch}/${r.id}`}
                disabled={listBusy || gameBusy}
                aria-pressed={gameKey === `${r.epoch}/${r.id}`}
                onClick={() => load(r)}
              >
                <span>
                  {!page.epoch && `Epoch ${Number(r.epoch) + 1} · `}Game{" "}
                  {gameName(r.id)}
                </span>
                <small
                  className={
                    r.winner === "First"
                      ? "winner-black"
                      : r.winner === "Second"
                        ? "winner-white"
                        : ""
                  }
                >
                  {r.plies} moves ·{" "}
                  {r.winner === "Draw"
                    ? "Draw"
                    : r.winner
                      ? `${r.winner === "First" ? "Black" : "White"} won`
                      : "Outcome unavailable"}
                </small>
              </button>
            ))}
          </div>
          <div className="service-pagination">
            <div className="service-actions">
              <button
                disabled={listBusy || !page?.offset}
                onClick={() =>
                  choosePage(page!.epoch, Math.max(0, page!.offset - PAGE_SIZE))
                }
              >
                Previous
              </button>
              <button
                disabled={
                  listBusy || !page || page.offset + PAGE_SIZE >= page.total
                }
                onClick={() =>
                  choosePage(page!.epoch, page!.offset + PAGE_SIZE)
                }
              >
                Next
              </button>
            </div>
            <span>
              {page
                ? `Page ${Math.floor(page.offset / PAGE_SIZE) + 1} of ${Math.max(1, Math.ceil(page.total / PAGE_SIZE))}`
                : "Loading…"}
            </span>
          </div>
          <p role="status">
            {listBusy ? "Loading games…" : `${page?.total || 0} recorded games`}
          </p>
        </>
      )}
    </>
  );
}
