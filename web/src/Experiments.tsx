import type { JSX } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { ThemePicker } from './ThemePicker';
import { networkLabel } from './experimentLabels';
import type { Participants } from './experimentLabels';
import { confidenceLevels, scoreInterval } from './experimentConfidence';
import type { ConfidenceLevel } from './experimentConfidence';
import { ComparisonStatistics } from './ComparisonStatistics';
import type { GameStatistics } from './ComparisonStatistics';
import { TrainingMetrics } from './TrainingMetrics';
import type { SelfPlayStats } from './SelfPlayMetrics';
import { SelfPlayGames } from './SelfPlayGames';
import type { GameSample } from './SelfPlayGames';
import './experiments.css';

type Metric = { value_loss: number; policy_loss: number; total_loss: number; samples_per_second?: number; samples?: number; batches?: number; duration_seconds?: number };
type Epoch = { epoch: number; training: Metric; validation: Metric | null; self_play?: SelfPlayStats | null; self_play_seconds?: number | null; scheduled_learning_rate?: number | null };
type Score = { wins: number; losses: number; draws: number; score_rate: number;
  score_rate_95_percent_low: number; score_rate_95_percent_high: number };
type Checkpoint = { model: { architecture: string }; epoch: number; model_sha256: string; path: string };
type CurrentEpoch = { epoch: number; stage: 'starting' | 'self_play' | 'training' | 'replay_training' | 'saving';
  games_completed: number | null; games_total: number; elapsed_seconds: number | null;
  evaluations_per_second: number | null; updated_at: string | null };
type Phase = { id: string; title: string; kind: 'training' | 'battle'; state: string;
  architecture: string; completed: number; total: number; unit: string; eta_seconds: number | null;
  note?: string; baseline_checkpoint?: Checkpoint;
  game_samples?: GameSample[];
  participants?: Participants;
  current_epoch?: CurrentEpoch | null;
  live_statistics?: GameStatistics | null;
  started_at?: string | null; ended_at?: string | null; duration_seconds?: number | null;
  timing_source?: 'log_and_result' | 'report_duration_and_result' | null;
  training_performance?: { adam_backend: string; replay_cache: string; prefetch_batches: number } | null;
  match_settings?: { games: number; parallelism: number; inference_batch_size: number; simulations: number; temperature: number };
  metrics: Epoch[]; result: null | { first_checkpoint_result: Score; second_checkpoint_result: Score;
    first_checkpoint: Checkpoint; second_checkpoint: Checkpoint; game_statistics?: GameStatistics | null } };
type Snapshot = { self_play?: { epochs: number; games_per_epoch: number; simulations: number; parallelism: number; parallelisms?: number[]; inference_batch_size: number }; updated_at: string; archived?: boolean; phases: Phase[]; control: { paused: boolean };
  worker: { alive: boolean; stage?: string; error?: string }; activation_failed: boolean;
  gpu: null | { utilization: number; memory_mib: number; temperature_c: number | null; power_w: number };
  dataset: { training_games: number; validation_games: number; training_positions: number; dataset_sha256: string };
  settings: { epochs: number; games: number; parallelism: number; simulations: number; temperature: number; value_width: number };
  selection: null | { activation: string; interval_includes_half: boolean } };
type Response = { connected: boolean; error: string | null; received_at: number | null; snapshot: Snapshot | null };

const labels: Record<string, string> = { running: 'Running', pending: 'Queued', completed: 'Complete',
  paused: 'Paused', failed: 'Failed', unknown: 'Needs attention', cancelled: 'Cancelled' };
const number = (value: number | null | undefined) => value == null ? '—' : value.toLocaleString();
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const CONFIDENCE_KEY = 'alz-experiment-confidence';
const QUEUE_PAGE_SIZE = 10;
const epochStageLabels = { starting: 'Starting', self_play: 'Self-play', training: 'Training',
  replay_training: 'Training on saved replays', saving: 'Saving checkpoint' };
function storedConfidence(): ConfidenceLevel {
  try {
    const stored = localStorage.getItem(CONFIDENCE_KEY);
    return confidenceLevels.find(level => String(level) === stored) ?? 95;
  } catch {
    return 95;
  }
}
function eta(seconds: number | null): string {
  if (seconds === null) return '—';
  if (seconds < 60) return '< 1 min';
  const minutes = Math.ceil(seconds / 60);
  return minutes >= 60 ? `~${Math.floor(minutes / 60)}h ${minutes % 60}m` : `~${minutes} min`;
}

function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—';
  const total = Math.floor(seconds), hours = Math.floor(total / 3600);
  const minutes = Math.floor(total % 3600 / 60), remaining = total % 60;
  return hours ? `${hours}h ${minutes}m ${remaining}s` : minutes ? `${minutes}m ${remaining}s` : `${remaining}s`;
}

function timestamp(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit',
    timeZoneName: 'short',
  });
}

export function Experiments(): JSX.Element {
  const [response, setResponse] = useState<Response | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [queuePage, setQueuePage] = useState(0);
  const [confidence, setConfidence] = useState<ConfidenceLevel>(storedConfidence);
  const [logsOpen, setLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [logError, setLogError] = useState<string | null>(null);
  const snapshot = response?.snapshot;
  const active = snapshot?.phases.find((phase) => phase.state === 'running');
  const inspected = snapshot?.phases.find((phase) => phase.id === selected) ?? active ?? snapshot?.phases.at(-1);
  const gameStatistics = inspected?.result?.game_statistics ?? inspected?.live_statistics;
  const newestFirst = snapshot?.phases.map((phase, index) => ({ phase, index })).reverse() ?? [];
  const pageCount = Math.max(1, Math.ceil(newestFirst.length / QUEUE_PAGE_SIZE));
  const currentPage = Math.min(queuePage, pageCount - 1);
  const pageStart = currentPage * QUEUE_PAGE_SIZE;
  const pageRows = newestFirst.slice(pageStart, pageStart + QUEUE_PAGE_SIZE);
  const inspectedIndex = newestFirst.findIndex(({ phase }) => phase.id === inspected?.id);
  const inspectedPage = inspectedIndex < 0 ? currentPage : Math.floor(inspectedIndex / QUEUE_PAGE_SIZE);

  useEffect(() => { setQueuePage(page => Math.min(page, pageCount - 1)); }, [pageCount]);

  function selectConfidence(value: string): void {
    const level = confidenceLevels.find(level => String(level) === value);
    if (level === undefined) return;
    setConfidence(level);
    try { localStorage.setItem(CONFIDENCE_KEY, String(level)); } catch { /* Keep the view usable without storage. */ }
  }

  async function refresh(): Promise<void> {
    try {
      const result = await fetch('/api/experiments', { cache: 'no-store' });
      if (!result.ok) throw new Error(`Dashboard request failed (${result.status})`);
      setResponse(await result.json() as Response);
      setError(null);
    } catch (reason) {
      setError(String(reason));
    }
  }
  useEffect(() => {
    document.title = 'Experiment queue · AlphaZero';
    void refresh();
    const timer = setInterval(() => { void refresh(); }, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    setLogs([]); setLogError(null);
    if (!logsOpen || !inspected) return;
    let cancelled = false;
    async function updateLogs(): Promise<void> {
      try {
        const result = await fetch(`/api/experiments/logs?phase_id=${encodeURIComponent(inspected!.id)}`);
        const data = await result.json() as { lines?: string[]; error?: string };
        if (!result.ok) throw new Error(data.error ?? 'Could not load logs');
        if (!cancelled) { setLogs(data.lines ?? []); setLogError(null); }
      } catch (reason) { if (!cancelled) setLogError(String(reason)); }
    }
    void updateLogs();
    const timer = setInterval(() => { void updateLogs(); }, 5000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [logsOpen, inspected?.id]);

  const connected = !!response?.connected && !error;
  const completed = snapshot?.phases.filter((phase) => phase.state === 'completed').length ?? 0;
  const cancelled = snapshot?.phases.filter((phase) => phase.state === 'cancelled').length ?? 0;
  const finished = completed + cancelled;
  return <div className="experiments-app">
    <header className="experiment-header"><strong>AlphaZero <span>/ Experiments</span></strong>
      <div><span className={`experiment-status ${connected ? 'completed' : 'unknown'}`}>{connected ? snapshot?.archived ? 'Local archive' : 'Pod connected' : response?.error || error ? 'Disconnected' : 'Connecting'}</span><ThemePicker /></div>
    </header>
    <main className="experiment-main">
      <div className="experiment-heading"><div><p className="experiment-eyebrow">Gomoku · Training experiments</p><h1>Experiment queue</h1>
        <p>Track training progress and compare playing strength.</p></div>
        <span className="experiment-status pending">Read-only</span>
      </div>
      {(error || response?.error) && <div className="experiment-alert danger" role="alert">{error ?? response?.error} {snapshot && 'Showing the last received data until the connection recovers.'}</div>}
      {snapshot?.activation_failed && <div className="experiment-alert danger" role="alert">The activation experiment failed. Follow-up work is blocked; inspect its logs before restarting.</div>}
      {snapshot?.worker.stage === 'failed' && <div className="experiment-alert danger" role="alert">The follow-up worker stopped: {snapshot.worker.error}</div>}
      {!snapshot ? <section className="experiment-panel"><p>Connecting to the pod and loading the experiment queue…</p></section> : <>
        <section className="experiment-summary" aria-label="Current experiment overview">
          <div><span>Running now</span><strong>{active?.title ?? (finished === snapshot.phases.length ? 'All scheduled work finished' : 'Waiting for next stage')}</strong>
            <small>{active ? `${active.completed} / ${active.total} ${active.unit}` : snapshot.control.paused ? 'Follow-ups are paused' : snapshot.worker.stage?.replaceAll('_', ' ') ?? 'Worker not started'}</small></div>
          <div><span>Queue progress</span><strong>{finished} <em>/ {snapshot.phases.length}</em></strong><small>{cancelled ? `${completed} complete · ${cancelled} cancelled` : 'Completed stages'}</small></div>
          <div><span>{snapshot.archived ? 'Last GPU sample' : 'GPU utilization'}</span><strong>{snapshot.gpu ? `${snapshot.gpu.utilization.toFixed(0)}%` : '—'}</strong>
            <small>{snapshot.gpu ? `${(snapshot.gpu.memory_mib / 1024).toFixed(1)} GiB · ${snapshot.gpu.temperature_c == null ? 'temperature unavailable' : snapshot.gpu.temperature_c.toFixed(0) + '°C'}` : 'Waiting for telemetry'}</small></div>
          <div><span>Current stage ETA</span><strong>{eta(active?.eta_seconds ?? null)}</strong><small>Estimate from observed progress</small></div>
        </section>
        {snapshot.self_play ? <div className="experiment-protocol"><span>Current self-play protocol</span>
          <span>{snapshot.self_play.epochs} epochs / model</span><span>{snapshot.self_play.games_per_epoch} games / epoch</span>
          <span>S{snapshot.self_play.simulations} / {(snapshot.self_play.parallelisms ?? [snapshot.self_play.parallelism]).map(p => `P${p}`).join(' · ')} / B{snapshot.self_play.inference_batch_size}</span>
          <span>Temperature 1.0 → 0.7 · Top-p 0.95 and 1.0</span></div> : <div className="experiment-protocol"><span>{number((snapshot.dataset.training_games ?? 0) + (snapshot.dataset.validation_games ?? 0))} unique replay games</span>
          <span>{snapshot.settings.epochs} passes / model</span><span>{snapshot.settings.games} games / future match</span>
          <span>Up to {snapshot.settings.parallelism} concurrent games</span>
          <span>{number(snapshot.settings.simulations)} simulations</span><span>Temperature {snapshot.settings.temperature}</span></div>}
        <section className="experiment-panel experiment-table-panel" aria-labelledby="queue-title">
          <div className="experiment-panel-heading"><h2 id="queue-title">Scheduled work</h2><span className="experiment-muted">Select an experiment for details</span></div>
          <nav className="experiment-pagination" aria-label="Experiment queue pagination">
            <span className="experiment-muted" role="status">{newestFirst.length ? `${pageStart + 1}–${Math.min(pageStart + QUEUE_PAGE_SIZE, newestFirst.length)} of ${newestFirst.length}` : 'No experiments'} · Newest first</span>
            <div className="experiment-page-controls">
              {inspectedPage !== currentPage && <button onClick={() => setQueuePage(inspectedPage)} aria-label="Show selected experiment in queue">Show selected</button>}
              <button disabled={currentPage === 0} onClick={() => setQueuePage(currentPage - 1)} aria-label="Previous experiment page" aria-controls="experiment-queue-table">Previous</button>
              <label className="experiment-page-picker">Page
                <select aria-label="Experiment page" aria-controls="experiment-queue-table" value={currentPage} onChange={event => setQueuePage(Number(event.currentTarget.value))}>
                  {Array.from({ length: pageCount }, (_, page) => <option key={page} value={page}>{page + 1}</option>)}
                </select>
                <span>of {pageCount}</span>
              </label>
              <button disabled={currentPage === pageCount - 1} onClick={() => setQueuePage(currentPage + 1)} aria-label="Next experiment page" aria-controls="experiment-queue-table">Next</button>
            </div>
          </nav>
          <div className="experiment-table-scroll"><table id="experiment-queue-table" className="experiment-table"><thead><tr><th scope="col">Experiment</th><th scope="col">Status</th><th scope="col">Progress</th><th scope="col">Duration</th><th scope="col">ETA</th></tr></thead>
            <tbody>{pageRows.map(({ phase, index }, rowIndex) => <tr key={phase.id} onClick={() => setSelected(phase.id)} className={`${inspected?.id === phase.id ? 'selected' : ''} ${index === 2 && rowIndex > 0 ? 'followup-start' : ''}`}>
              <td><button className="experiment-row-button" onClick={() => setSelected(phase.id)} aria-pressed={inspected?.id === phase.id}>
                <span className="experiment-index">{String(index + 1).padStart(2, '0')}</span><span>{phase.title}<small>{phase.kind === 'training' ? 'Training' : 'Match'}</small></span></button></td>
              <td><span className={`experiment-status ${phase.state}`}>{labels[phase.state] ?? phase.state}</span></td>
              <td><div className="experiment-progress"><span>{phase.completed} / {phase.total} {phase.unit}</span><progress aria-label={`${phase.title} progress`} value={phase.completed} max={phase.total} /></div></td>
              <td className="experiment-data experiment-duration"><span className="experiment-mobile-label">{phase.state === 'running' ? 'Elapsed: ' : 'Duration: '}</span>{duration(phase.duration_seconds)}</td>
              <td className="experiment-data experiment-eta"><span className="experiment-mobile-label">ETA: </span>{eta(phase.eta_seconds)}</td>
            </tr>)}</tbody></table></div>
          <p className="experiment-table-note">Comparisons start after training. Later matches may overlap according to the schedule shown in their details.</p>
        </section>
        {inspected && <section className="experiment-panel" aria-labelledby="detail-title">
          <div className="experiment-panel-heading"><div><p className="experiment-eyebrow">Experiment details</p><h2 id="detail-title">{inspected.title}</h2></div>
            <span className={`experiment-status ${inspected.state}`}>{labels[inspected.state]}</span></div>
          {inspected.current_epoch && <section className="experiment-live-epoch" aria-label="Current epoch progress">
            <div className="experiment-live-heading"><strong>Epoch {inspected.current_epoch.epoch}</strong>
              <span>{epochStageLabels[inspected.current_epoch.stage]}</span></div>
            {inspected.current_epoch.games_completed != null ? <div className="experiment-progress">
              <span><strong>{number(inspected.current_epoch.games_completed)} / {number(inspected.current_epoch.games_total)} games finished</strong>
                <span>{percent(inspected.current_epoch.games_completed / inspected.current_epoch.games_total)}</span></span>
              <progress aria-label={`Epoch ${inspected.current_epoch.epoch} games finished`}
                value={inspected.current_epoch.games_completed} max={inspected.current_epoch.games_total} />
            </div> : <p className="experiment-muted">{inspected.current_epoch.stage === 'replay_training'
              ? 'Using saved games from the original run.' : 'Waiting for the first game-progress update.'}</p>}
            <p className="experiment-muted">{inspected.current_epoch.elapsed_seconds != null && <>Self-play elapsed: {duration(inspected.current_epoch.elapsed_seconds)} · </>}
              {inspected.current_epoch.stage === 'self_play' && inspected.current_epoch.evaluations_per_second != null && <>{number(Math.round(inspected.current_epoch.evaluations_per_second))} evaluations/s · </>}
              {inspected.current_epoch.updated_at && <>Last update: <time dateTime={inspected.current_epoch.updated_at}>{new Date(inspected.current_epoch.updated_at).toLocaleTimeString()}</time> · </>}
              Trainer reports progress every 15 seconds.</p>
          </section>}
          <dl className="experiment-timing" aria-label="Job timing">
            <div><dt>Start time</dt><dd>{inspected.started_at ? <time dateTime={inspected.started_at} title={inspected.started_at}>{timestamp(inspected.started_at)}</time> : '—'}</dd></div>
            <div><dt>End time</dt><dd>{inspected.ended_at ? <time dateTime={inspected.ended_at} title={inspected.ended_at}>{timestamp(inspected.ended_at)}</time> : inspected.state === 'running' && inspected.started_at ? 'In progress' : '—'}</dd></div>
            <div><dt>{inspected.state === 'running' ? 'Elapsed time' : 'Duration'}</dt><dd>{duration(inspected.duration_seconds)}</dd></div>
          </dl>
          {inspected.started_at && <p className="experiment-muted">Times shown in your browser’s time zone. {inspected.timing_source === 'report_duration_and_result'
            ? 'Start estimated from the recorded match runtime and result-file timestamp.'
            : 'Start comes from the first log timestamp; end comes from the result-file timestamp. Elapsed time includes setup and any gaps between attempts.'}</p>}
          <p className="experiment-muted">{inspected.architecture}</p>
          {inspected.note && <p className="experiment-muted">{inspected.note}</p>}
          {inspected.baseline_checkpoint && <p className="experiment-muted">Reused baseline: {inspected.baseline_checkpoint.model.architecture.replaceAll('_', '-')} · Checkpoint {String(inspected.baseline_checkpoint.epoch).padStart(8, '0')} · Model {inspected.baseline_checkpoint.model_sha256.slice(0, 12)}</p>}
          {inspected.training_performance && <p className="experiment-muted">{inspected.training_performance.adam_backend === 'fused' ? 'Fused Adam' : 'Standard Adam'} · Replay cache: {inspected.training_performance.replay_cache} · Prefetched batches: {inspected.training_performance.prefetch_batches}</p>}
          {inspected.match_settings && <p className="experiment-muted">{inspected.match_settings.games} games · Up to {inspected.match_settings.parallelism} concurrent · Inference batches up to {inspected.match_settings.inference_batch_size} per network</p>}
          {inspected.metrics.length > 0 ? <TrainingMetrics key={inspected.id} metrics={inspected.metrics} />
            : inspected.kind === 'training' && <p className="experiment-empty">Training curves appear after the first completed pass.</p>}
          {inspected.game_samples && <SelfPlayGames key={inspected.id} phaseId={inspected.id} samples={inspected.game_samples} />}
          {(inspected.result || inspected.live_statistics) ? <><div className="experiment-confidence-controls">
            <label htmlFor="experiment-confidence">Confidence level</label>
            <select id="experiment-confidence" value={String(confidence)} onChange={event => selectConfidence(event.currentTarget.value)} aria-describedby="experiment-confidence-help">
              {confidenceLevels.map(level => <option key={level} value={String(level)}>{level}%</option>)}
            </select>
            <p id="experiment-confidence-help" className="experiment-muted">Higher confidence gives wider intervals.</p>
          </div>{inspected.result && <><div className="experiment-table-scroll"><table className="experiment-table experiment-results"><thead><tr><th>Network</th><th>Wins</th><th>Draws</th><th>Losses</th><th>Score</th><th>{confidence}% interval</th></tr></thead>
            <tbody>{(['first', 'second'] as const).map((side) => { const result = inspected.result!; const score = result[`${side}_checkpoint_result`]; const checkpoint = result[`${side}_checkpoint`];
              const interval = scoreInterval(score, confidence);
              return <tr key={side}><td className="experiment-network-identity"><strong>{networkLabel(inspected.participants, side)}</strong>
                <small>{checkpoint.model.architecture.replaceAll('_', '-')} · Checkpoint {String(checkpoint.epoch).padStart(8, '0')}</small>
                <small title={`${checkpoint.path}\nSHA-256: ${checkpoint.model_sha256}`}>Model {checkpoint.model_sha256.slice(0, 12)}</small>
              </td><td><span className="experiment-result-label">Wins</span>{score.wins}</td><td><span className="experiment-result-label">Draws</span>{score.draws}</td><td><span className="experiment-result-label">Losses</span>{score.losses}</td><td><span className="experiment-result-label">Score</span>{percent(score.score_rate)}</td><td className="experiment-data"><span className="experiment-result-label">{confidence}% interval</span>{interval ? `${percent(interval.low)} – ${percent(interval.high)}` : '—'}</td></tr>;
            })}</tbody></table></div><p className="experiment-muted">Approximate Wilson score intervals; draws count as half a win. An interval spanning 50% is consistent with an even score.</p></>}</> : inspected.kind === 'battle' && <p className="experiment-empty">Game statistics appear as games finish and update every 5 seconds.</p>}
          {!inspected.result && inspected.live_statistics && <p className="experiment-muted" role="status">Partial results: {inspected.live_statistics.games} / {inspected.total} games finished. Short games finish first, so these outcomes and intervals can be biased until the match completes.</p>}
          {gameStatistics && <ComparisonStatistics key={inspected.id} participants={inspected.participants}
            data={gameStatistics} partial={!inspected.result} confidence={confidence} formatDuration={duration} />}
          <details className="experiment-logs" onToggle={(event) => setLogsOpen(event.currentTarget.open)}><summary>Recent log output</summary>
            {logError && <p role="alert">{logError}</p>}<pre tabIndex={0}>{logs.join('\n') || 'No log output available yet.'}</pre></details>
        </section>}
        {snapshot.selection && <p className="experiment-muted">Value-head comparisons use {snapshot.selection.activation === 'kata-v1' ? 'ReLU' : 'GELU'}, selected by the activation match score.
          {snapshot.selection.interval_includes_half && ' Its original 95% confidence interval includes 50%; this selection does not establish a clear winner.'}</p>}
        <footer className="experiment-footer"><span>{snapshot.archived ? `Archived ${new Date(snapshot.updated_at).toLocaleString()}` : `Updated ${new Date(snapshot.updated_at).toLocaleTimeString()} · Refreshes every 5 seconds`}</span>
          <span>Read-only dashboard · Queue runs on the pod</span></footer>
      </>}
    </main>
  </div>;
}
