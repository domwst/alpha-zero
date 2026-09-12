import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_agent import Experiments
from experiment_io import digest, read, write

from archive import run_training_recipe_experiments as recipe
class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name)
        self.root = self.parent / 'training-recipes'
        self.root.mkdir()
        self.models = recipe.schedule()['models']
        self.base = {'sha256': {'bn.weight': 'random', 'conv.weight': 'same'},
                     'bn_gamma_names': ['bn.weight']}
        self.dataset = {'sources': [], 'dataset_sha256': 'data', 'training_games': 10,
                        'validation_games': 2, 'training_positions': 20, 'validation_positions': 4}
        baseline = self.parent / 'baseline'
        write(baseline / 'dataset.json', self.dataset)
        self.binary = self.parent / 'alz'
        self.binary.write_text('test binary')
        write(self.root / 'cuda-ready.json', {'binary_sha256': digest(self.binary)})
        self.args = SimpleNamespace(output_dir=self.root, baseline_dir=baseline, binary=self.binary)
        for model in self.models:
            data = copy.deepcopy(self.base)
            if model['variant'] == 'gamma-one': data['sha256']['bn.weight'] = 'ones'
            write(self.root / model['id'] / 'initial-tensors.json', data)
            write(self.root / model['id'] / 'initial-validation.json', {'total_loss': 1})
            write(self.root / model['id'] / 'dataset.json', self.dataset)

    def test_pairing_rejects_unrelated_weight_change(self):
        recipe.validate_pairing(self.root, self.models)
        path = self.root / 'recipe-gamma-one-s1/initial-tensors.json'
        data = read(path); data['sha256']['conv.weight'] = 'changed'; write(path, data)
        with self.assertRaisesRegex(RuntimeError, 'other initial tensors'):
            recipe.validate_pairing(self.root, self.models)

    def run_fake(self, fail_first=False):
        live, started, peaks = set(), [], []
        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.name = Path(command[command.index('--run-dir') + 1]).name
                self.returncode = None
                self.polls = 0
                started.append(self.name); live.add(self.name); peaks.append(len(live))
            def poll(self):
                self.polls += 1
                if self.polls >= 2:
                    self.returncode = 1 if fail_first and self.name == started[0] else 0
                    live.discard(self.name)
                return self.returncode
        with patch.object(recipe, 'validate_run_config'), \
             patch.object(recipe, 'select_checkpoints', return_value={'selected': 'test'}), \
             patch.object(recipe.subprocess, 'Popen', FakeProcess), \
             patch.object(recipe.time, 'sleep'):
            if fail_first:
                with self.assertRaisesRegex(RuntimeError, 'exited 1'): recipe.run(self.args)
            else: recipe.run(self.args)
        return started, peaks

    def test_six_models_with_maximum_two_active(self):
        started, peaks = self.run_fake()
        self.assertEqual(len(started), 6)
        self.assertEqual(max(peaks), 2)
        self.assertEqual(read(self.root / 'status.json')['stage'], 'complete')
        self.assertEqual(len(read(self.root / 'selection.json')), 6)
        self.assertEqual(read(self.root / 'summary.json')['next_action'], 'migrate before battles')

    def test_failure_drains_current_worker_and_blocks_later_jobs(self):
        started, peaks = self.run_fake(True)
        self.assertEqual(len(started), 2)
        self.assertEqual(read(self.root / 'status.json')['stage'], 'failed')

    def test_dashboard_displays_both_running_and_other_queued(self):
        write(self.root / 'schedule.json', recipe.schedule())
        write(self.root / 'status.json', {'stage': 'training', 'active': [m['id'] for m in self.models[:2]]})
        with patch('experiment_agent.locked', return_value=True):
            phases = Experiments(self.parent / 'activation', self.parent).snapshot()['phases'][-6:]
        self.assertEqual([p['state'] for p in phases], ['running', 'running'] + ['queued'] * 4)
        self.assertEqual(len({p['title'] for p in phases}), 6)


if __name__ == '__main__': unittest.main()
