"""Versioned overrides for future matches; original queue/training config stays pinned."""
import json
from pathlib import Path

MATCH_NAMES = ('current-vs-wide64', 'current-vs-deep64', 'wide64-vs-deep64',
               'current-vs-katago-pooling')


def read_match_plan(root):
    path = Path(root) / 'match-plan.json'
    if not path.exists():
        return None
    plan = json.loads(path.read_text())
    version = plan.get('schema_version')
    if version not in (1, 2) or set(plan.get('matches', {})) != set(MATCH_NAMES):
        raise ValueError('Invalid match plan: all future value-head and pooling matches are required')
    settings_to_validate = list(plan['matches'].values())
    if version == 2:
        if plan.get('selection') != 'mean-opponent-score-rate' or not isinstance(plan.get('future_settings'), dict):
            raise ValueError('Mixed budgets require equal opponent weighting and explicit future settings')
        settings_to_validate.append(plan['future_settings'])
    for settings in settings_to_validate:
        if set(settings) != {'games', 'parallelism'} or any(type(v) is not int for v in settings.values()):
            raise ValueError('Match plan may change only games and parallelism')
        if settings['games'] <= 0 or settings['games'] % 2 or not 0 < settings['parallelism'] <= settings['games']:
            raise ValueError('Match counts must be positive and balanced; concurrency cannot exceed games')
    if version == 1 and len({tuple(sorted(plan['matches'][name].items())) for name in MATCH_NAMES[:3]}) != 1:
        raise ValueError('All value-head round-robin matches must have equal budgets')
    return plan


def match_settings(plan, name, games, parallelism):
    return dict(plan['matches'][name] if plan and name in plan['matches'] else
                {'games': games, 'parallelism': parallelism})
