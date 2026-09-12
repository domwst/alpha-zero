"""Read-only dashboard view of metadata-defined comparison handoffs."""
from pathlib import Path
from experiment_io import locked
from experiment_metadata import with_experiment_metadata


def append_additional_comparisons(agent, snapshot):
    for entry in agent.json(agent.queue/'additional-comparisons.json', []):
        root = Path(entry['directory'])
        plan = agent.json(root/'plan.json')
        if not plan:
            continue
        status = agent.json(root/'status.json', {})
        state = status.get('stage','queued')
        if state == 'running' and not locked(root/'.lock'):
            state = 'unknown'
        phase = agent.battle(plan['id'],plan['title'],root/'result.json',root/'battle.log',
                            state,' / '.join(plan[side]['model']['architecture'] for side in ('first','second')),
                            plan['settings'])
        phase = with_experiment_metadata(phase, plan['metadata'])
        phase['note'] = 'Self-play pauses for this match and resumes automatically afterward. '+status.get('error','')
        if status.get('resume'):
            phase['note'] += ' Self-play supervisor restarted at '+status['resume']['at']+'.'
        snapshot['phases'].append(phase)
    return snapshot
