"""Read-only adapter for the auxiliary board-mask experiment."""
from experiment_io import locked, tail



def append_board_mask(agent, snapshot):
    root = agent.queue.parent/'board-mask-20260910'
    plan = agent.json(root/'plan.json')
    if not plan:
        return snapshot
    status = agent.json(root/'status.json', {})
    stage = status.get('stage', 'pending')
    alive = locked(root/'.lock')
    working = stage in ('preflight','initializing','training','battle')
    state = 'running' if working and alive else 'unknown' if working else stage
    training = agent.training('boardmask-training','Board mask · replay training',root/'model',
        root/'training.log',state if stage != 'battle' else 'completed',plan['architecture'])
    training['note'] = ('20 passes on deduplicated checkpoints 60–69; GELU, deep value head, BN gamma=1. '
        'Constant LR 0.001; matched split and seed with recipe-gamma-one-s1. '
        'All shared initial tensors are verified identical; adds 288 board-mask weights. '
        'Checkpoint selection: minimum combined validation loss.')
    training['baseline_checkpoint'] = plan['baseline']
    battle_state = state if status.get('job') == 'boardmask-vs-current' else 'pending'
    battle = agent.battle('boardmask-vs-current','Current vs board mask · selected checkpoints',
        root/'boardmask-vs-current.json',root/'battle.log',battle_state,
        'kata-gelu-value64x2-v1 / '+plan['architecture'],plan['settings'])
    selection = agent.json(root/'selection.json')
    battle['note'] = ('Current: BN gamma=1, seed 1, selected pass 11. Board mask: '
        + (f"selected pass {selection['selected']['epoch']+1}. " if selection else 'selected after training. ')
        + 'P100/B64; 1,000 games, 4,000 simulations, temperature 0.7. '
          'A memory guard pauses only the auxiliary experiment if pod headroom is exhausted.')
    for phase in (training, battle):
        if phase['state'] in ('failed','paused','unknown'):
            phase['note'] += ' '+status.get('reason',status.get('error',''))
    handoff = root/'comparison-handoff'
    schedule = agent.json(handoff/'schedule.json')
    if schedule:
        stages = agent.json(handoff/'status.json', {})
        queue_alive = locked(handoff/'.lock')
        phases = [training]
        for match in schedule['matches']:
            name = match['id']
            job = stages.get('jobs', {}).get(name, {})
            state = job.get('stage', 'pending')
            if state == 'running' and not queue_alive:
                state = 'failed' if stages.get('stage') == 'failed' else 'unknown'
            log = handoff/f'{name}-attempt{job.get("attempt",1)}.log'
            phase = agent.battle(name,match['title'],handoff/(name+'.json'),log,state,
                'Replay baseline / ' + ('board mask' if name == 'boardmask-vs-current' else 'latest self-play'),schedule['settings'])
            phase['baseline_checkpoint'] = schedule['baseline']
            phase['note'] = ('P500/B128; 1,000 games, 4,000 simulations, temperature 0.7. '
                'First checkpoint: replay-trained BN gamma=1 baseline, selected pass 11. '
                + (f"Second checkpoint: board-mask model, selected pass {schedule['boardmask']['epoch']+1}. "
                   if name == 'boardmask-vs-current' else
                   f"Second checkpoint: self-play epoch {schedule['selfplay_epoch']+1}, saved at the pause boundary. ")
                + 'Second match starts at 60% completion; memory pressure defers overlap until the first finishes. '
                + job.get('note',stages.get('error','')))
            if name == 'boardmask-vs-current':
                phase['note'] += ' Previous P100 attempt stopped by the memory guard; its 3 completed games are excluded.'
            phases.append(phase)
        snapshot['phases'].extend(phases)
        if queue_alive or not snapshot.get('worker', {}).get('alive'):
            snapshot['worker'] = {'alive':queue_alive, **stages}
        telemetry = tail(handoff/'resources.jsonl')
        if telemetry and queue_alive:
            try:
                import json
                sample = json.loads(telemetry[-1])
                snapshot['gpu'] = {'timestamp':stages.get('updated_at'),'utilization':sample['gpu_utilization'],
                    'memory_mib':sample['gpu_memory_mib'],'power_w':None,'temperature_c':None}
            except (ValueError,KeyError):
                pass
    else:
        snapshot['phases'].extend([training,battle])
    return snapshot
