import type { JSX } from 'preact';
import { useState } from 'preact/hooks';

export type GameSample = { epoch: number; sample: number; width: number; height: number; positions: number; columns?: number };

export function SelfPlayGames({ phaseId, samples }: { phaseId: string; samples: GameSample[] }): JSX.Element {
  const [chosenEpoch, setEpoch] = useState(samples.at(-1)?.epoch ?? 1);
  const [chosenSample, setSample] = useState(0);
  const epochs = [...new Set(samples.map(row => row.epoch))].sort((a, b) => b - a);
  const epoch = epochs.includes(chosenEpoch) ? chosenEpoch : epochs[0];
  const options = samples.filter(row => row.epoch === epoch);
  const selected = options.find(row => row.sample === chosenSample) ?? options[0];
  return <section className="self-play-games" aria-label="Sample self-play games">
    <h3>Sample games</h3>
    {!selected ? <p className="experiment-empty">Game samples appear after an epoch finishes.</p> : <>
      <div className="sample-game-controls">
        <label>Epoch <select aria-label="Epoch" value={epoch} onChange={event => { setEpoch(Number(event.currentTarget.value)); setSample(0); }}>
          {epochs.map(value => <option key={value} value={value}>Epoch {value}</option>)}
        </select></label>
        <label>Game <select aria-label="Game" value={selected.sample} onChange={event => setSample(Number(event.currentTarget.value))}>
          {options.map(row => <option key={row.sample} value={row.sample}>Sample {row.sample + 1} · {row.positions} positions</option>)}
        </select></label>
      </div>
      <SampleBoard key={`${epoch}-${selected.sample}`} phaseId={phaseId} sample={selected} />
    </>}
    <p className="experiment-muted">Red stones: first player. Blue stones: second player. Green intensity: search policy target. The larger stone marks the previous move. These saved images show positions before each move; the terminal board is not included.</p>
  </section>;
}

function SampleBoard({ phaseId, sample }: { phaseId: string; sample: GameSample }): JSX.Element {
  const [frame, setFrame] = useState(0);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [retry, setRetry] = useState(0);
  const url = `/api/experiments/game-image?phase_id=${encodeURIComponent(phaseId)}&epoch=${sample.epoch}&sample=${sample.sample}&layout=${sample.width}x${sample.height}&retry=${retry}`;
  const columns = sample.columns ?? sample.positions;
  const choose = (n: number) => { if (Number.isFinite(n)) setFrame(Math.max(0, Math.min(sample.positions - 1, Math.trunc(n)))); };
  return <div className="sample-game-viewer">
    <div className="sample-game-controls" role="group" aria-label="Sample game position">
      <label>Position <input aria-label="Position" type="number" min="1" max={sample.positions} value={frame + 1}
        onChange={event => choose(Number(event.currentTarget.value) - 1)} /></label>
      <span>of {sample.positions}</span>
    </div>
    <div className="sample-game-navigation" role="group" aria-label="Browse game positions">
      <button className="sample-game-jump" disabled={frame === 0} onClick={() => choose(0)} aria-label="First position" title="Jump to first position">
        <svg aria-hidden="true" viewBox="0 0 20 20"><path d="M4 4v12M15 4l-8 6 8 6" /></svg>
      </button>
      <div className="sample-game-steps">
        <button disabled={frame === 0} onClick={() => choose(frame - 1)} aria-label="Previous position">Previous</button>
        <button disabled={frame === sample.positions - 1} onClick={() => choose(frame + 1)} aria-label="Next position">Next</button>
      </div>
      <button className="sample-game-jump" disabled={frame === sample.positions - 1} onClick={() => choose(sample.positions - 1)} aria-label="Last position" title="Jump to last position">
        <svg aria-hidden="true" viewBox="0 0 20 20"><path d="M16 4v12M5 4l8 6-8 6" /></svg>
      </button>
    </div>
    <p className="experiment-muted" aria-live="polite">Epoch {sample.epoch} · Sample {sample.sample + 1} · Before move {frame + 1}</p>
    {status === 'loading' && <p role="status">Loading saved game…</p>}
    {status === 'error' && <p role="alert">Could not load this sample. <button onClick={() => { setStatus('loading'); setRetry(retry + 1); }}>Retry image</button></p>}
    <div className="sample-game-board" tabIndex={0} aria-label="Game board: use arrow keys to change position"
      onKeyDown={event => {
        const next = { ArrowLeft: frame - 1, ArrowRight: frame + 1, Home: 0, End: sample.positions - 1 }[event.key];
        if (next !== undefined) { event.preventDefault(); choose(next); }
      }}>
      <img key={url} src={url} alt={`Sample ${sample.sample + 1}, epoch ${sample.epoch}: board before move ${frame + 1}`}
        onLoad={() => setStatus('ready')} onError={() => setStatus('error')}
        style={{ width: `${sample.width / 380 * 100}%`, height: `${sample.height / 380 * 100}%`,
          left: `${-(frame % columns) * 390 / 380 * 100}%`, top: `${-Math.floor(frame / columns) * 390 / 380 * 100}%`,
          visibility: status === 'ready' ? 'visible' : 'hidden' }} />
    </div>
    <p className="chart-help">Use Previous / Next, enter a position, or focus the board and use arrow keys. <a href={url} target="_blank" rel="noreferrer">Open full game image</a></p>
  </div>;
}
