"""Read-only self-play progress appended to the existing comparison dashboard."""
from pathlib import Path
import datetime
import hashlib
import math
import re
from experiment_io import locked, now, tail

from self_play_samples import list_game_samples


def current_epoch_progress(lines, epoch, state, games_total=1000, after=0, history_epochs=0):
    if state != 'running':
        return None
    progress = {'epoch':epoch, 'stage':'starting', 'games_completed':None,
                'games_total':games_total, 'elapsed_seconds':None,
                'evaluations_per_second':None, 'updated_at':None}
    for line in lines:
        stamp = re.match(r'^(\d{4}-\d\d-\d\dT\S+)', line)
        if stamp and datetime.datetime.fromisoformat(stamp[1]).timestamp() < after:
            continue
        if 'epoch complete' in line:
            finished = re.search(r'\bepoch=(\d+)', line)
            if finished and int(finished[1])+1 >= epoch:
                return None  # Snapshot publication raced this refresh; next refresh advances the epoch.
            progress.update(stage='starting', games_completed=None, elapsed_seconds=None,
                            evaluations_per_second=None, updated_at=None)
        elif 'restoring complete training snapshot' in line or 'using compute device' in line:
            progress.update(stage='starting', games_completed=None, elapsed_seconds=None,
                            evaluations_per_second=None, updated_at=None)
        elif 'self-play progress' in line:
            fields = dict(re.findall(r'\b([a-z_]+)=([\d.]+)', line))
            try:
                completed, total = int(fields['games_completed']), int(fields['games_total'])
                elapsed = float(fields['elapsed_seconds'])
                evaluations = float(fields['completed_evaluations'])
            except (KeyError, ValueError):
                continue
            if total <= 0 or not 0 <= completed <= total or not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(evaluations) or evaluations < 0:
                continue
            progress.update(stage='self_play', games_completed=completed, games_total=total,
                            elapsed_seconds=elapsed, evaluations_per_second=evaluations/max(elapsed,1))
        elif 'self-play complete' in line:
            progress.update(stage='training', games_completed=games_total)
            duration = re.search(r'\bself_play_seconds=([\d.]+)', line)
            if duration:
                progress['elapsed_seconds'] = float(duration[1])
        elif 'starting replay-history epoch' in line or 'encoding replay cache' in line:
            progress['stage'] = 'replay_training' if epoch <= history_epochs else 'training'
        elif 'training complete' in line:
            progress['stage'] = 'saving'
        else:
            continue
        if stamp:
            progress['updated_at'] = stamp[1]
    return progress


def self_play_duration(row, inherited):
    if row.get('self_play_seconds') is not None:
        return row['self_play_seconds']
    source = inherited.get('epochs', {}).get(str(row['epoch']), {})
    if not row.get('history_stats_sha256') or source.get('source_stats_sha256') != row['history_stats_sha256']:
        return None
    seconds = source.get('self_play_seconds')
    return seconds if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0 else None


def epoch_game_metrics(row, outcomes=None):
    games = row.get('games')
    if not isinstance(games, int) or games <= 0:
        return None
    data = {'games':games, 'average_game_length':row.get('average_game_length'),
            'first_player_win_rate':None, 'first_player_wins':None, 'second_player_wins':None, 'draws':None}
    if outcomes and outcomes.get('epoch') == row['epoch'] and outcomes.get('games') == games:
        wins, losses, draws = (outcomes.get(k) for k in ('first_player_wins','second_player_wins','draws'))
        if all(isinstance(n, int) and n >= 0 for n in (wins, losses, draws)) and wins+losses+draws == games:
            if wins-losses == row.get('total_score') and outcomes.get('total_game_length') == row.get('total_game_length'):
                data.update(first_player_wins=wins, second_player_wins=losses, draws=draws,
                            first_player_win_rate=wins/games)
    return data


def append_self_play(agent, snapshot):
    root = agent.queue.parent/'selfplay-nucleus'
    plan = agent.json(root/'plan.json')
    if not plan:
        return snapshot
    jobs = []
    for saved in plan['jobs']:
        overrides = agent.json(root/saved['id']/'continuation-settings.json', {})
        jobs.append({**saved, **({'parallelism':overrides['games_parallelism']} if overrides else {})})
    parallelisms = list(dict.fromkeys(job.get('parallelism', plan.get('parallelism', 500)) for job in jobs))
    snapshot['self_play'] = {'epochs':plan['epochs'], 'games_per_epoch':1000, 'simulations':3000, 'parallelism':plan.get('parallelism', 500),
                             'parallelisms':parallelisms, 'inference_batch_size':plan['inference_batch_size']}
    status = agent.json(root/'status.json', {})
    alive = locked(root/'.lock')
    for job in jobs:
        name = job['id']
        directory = root/name
        inherited = agent.json(directory/'inherited-self-play.json', {})
        log = directory/'stdout.log'
        state = agent.json(root/'jobs'/(name+'.json'), {'stage':'queued'})
        phase_state = state['stage']
        if phase_state == 'running' and not alive:
            phase_state = 'unknown'
        metrics, durations = [], []
        completed_at = 0
        for path in sorted((directory/'stats').glob('[0-9]*.json')):
            row = agent.json(path)
            if row is None or not (directory/'checkpoints'/path.stem/'metadata.json').exists():
                continue
            outcomes = agent.json(directory/'epoch-metrics'/path.name)
            if outcomes and outcomes.get('stats_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
                outcomes = None
            metrics.append({'epoch':row['epoch']+1,'training':row['training'],'validation':None,
                            'scheduled_learning_rate':row.get('scheduled_learning_rate'),
                            'self_play_seconds':self_play_duration(row, inherited),
                            'self_play':epoch_game_metrics(row, outcomes)})
            durations.append(row['epoch_seconds'])
            completed_at = max(completed_at, path.stat().st_mtime)
        segment = state.get('segment_started_at') or state.get('started_at')
        after = max(completed_at, datetime.datetime.fromisoformat(segment).timestamp() if segment else 0)
        current = current_epoch_progress(tail(log), max((row['epoch'] for row in metrics), default=0)+1,
            phase_state, plan.get('games_per_epoch',1000), after, job.get('replay_history_epochs',0))
        if current and current['epoch'] > plan['epochs']:
            current = None
        note = (f'Fresh GELU / value64x2; BN gamma=1; top-p {job["top_p"]}. '
                f'S3000/P{job.get("parallelism", plan.get("parallelism", 500))}/B{plan["inference_batch_size"]}; 1,000 games/epoch. '
                'Paired temperature 1→0.7; position replay + replay-linked LR. No held-out validation split.')
        if job.get('replay_history_epochs'):
            note = (f"Board-mask architecture. Epochs 1–{job['replay_history_epochs']} retrain the original run's saved buffers in order; "
                    f"their game outcomes belong to that original run. New self-play begins at epoch {job['replay_history_epochs']+1}. ") + note
        if state.get('note'):
            note += ' ' + state['note']
        if inherited.get('epochs'):
            note += ' Self-play durations for reconstructed epochs are inherited from the original run; training timings belong to this run.'
        start, end = state.get('started_at'), state.get('ended_at')
        duration = None
        if start:
            boundary = datetime.datetime.fromisoformat(end or now())
            duration = max(0, (boundary-datetime.datetime.fromisoformat(start)).total_seconds())
        snapshot['phases'].append({'id':name, 'title':job['title'], 'kind':'training', 'state':phase_state,
            'architecture':job.get('architecture','kata-gelu-value64x2-v1'),'completed':len(metrics),'total':plan['epochs'],'unit':'epochs',
            'metrics':metrics,'result':None,'note':note,'eta_seconds':None,'current_epoch':current,
            'game_samples':list_game_samples(directory, {row['epoch'] for row in metrics}),
            'pass_seconds':sum(durations[-3:])/len(durations[-3:]) if durations else None,
            'started_at':start,'ended_at':end,'duration_seconds':duration,
            'training_performance':{'adam_backend':plan.get('adam_backend','standard'),'replay_cache':'cpu','prefetch_batches':2}})
        agent.logs[name] = log
    snapshot['worker'] = {'alive':alive, **status}
    resources = status.get('resources', {})
    if resources:
        snapshot['gpu'] = {'timestamp':status.get('updated_at'),'utilization':resources['gpu_utilization'],
                           'memory_mib':resources['gpu_memory_mib'],'power_w':None,'temperature_c':None}
    return snapshot
