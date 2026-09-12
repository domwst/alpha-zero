"""Live recipe battles plus the immutable dashboard history from the retired pod."""
import csv
from experiment_io import locked, now, tail
from experiment_agent import Experiments
from experiment_control import read_control


class RecipeExperiments(Experiments):
    def snapshot(self):
        plan = self.json(self.queue / 'recipe-battle-plan.json')
        archive = self.json(self.queue / 'archive.json')
        snapshot = dict(archive['snapshot'])
        phases = list(snapshot['phases'])
        stage = self.json(self.queue / 'status.json', {})
        alive = locked(self.queue / '.lock')
        for match in plan['matches']:
            name = match['id']
            status = self.json(self.queue / 'jobs' / (name + '.json'), {})
            state = 'queued'
            if status.get('stage') == 'failed':
                state = 'failed'
            elif name in stage.get('active', []):
                state = 'running' if alive else 'unknown'
            elif read_control(self.queue)['paused']:
                state = 'paused'
            phase = self.battle(name, match['title'], self.queue / (name + '.json'),
                self.queue / (name + '.log'), state, 'kata-gelu-value64x2-v1', plan['settings'])
            phase['note'] = 'Minimum combined validation loss; paired training seed. Two matches overlap after 60% completion.'
            phases.append(phase)
        gpu = None
        rows = tail(self.queue / 'gpu.csv', 4096)
        if rows:
            try:
                row = next(csv.reader([rows[-1]], skipinitialspace=True))
                gpu = {'timestamp': row[0], 'utilization': float(row[2]), 'memory_mib': float(row[4]),
                       'power_w': float(row[5]), 'temperature_c': float(row[8])}
            except (ValueError, IndexError):
                pass
        snapshot.update(archived=False, updated_at=now(), phases=phases, gpu=gpu,
                        control=read_control(self.queue), worker={'alive': alive, **stage},
                        settings={**snapshot.get('settings', {}), **plan['settings']})
        from self_play_dashboard import append_self_play
        from board_mask_dashboard import append_board_mask
        from additional_comparisons import append_additional_comparisons
        return append_additional_comparisons(self, append_board_mask(self, append_self_play(self, snapshot)))

    def dispatch(self, request):
        if request.get('method') == 'logs':
            saved = self.json(self.queue / 'archive.json')['logs']
            if request.get('phase_id') in saved:
                return saved[request['phase_id']]
        return super().dispatch(request)
