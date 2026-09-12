import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_control import read_control, set_paused
from experiment_agent import Experiments
from archive.run_capacity_width_parallel import ParallelWidth, hold_queue, release_queue
from experiment_io import read, write

import test_capacity_experiments as fixtures


class GateTests(unittest.TestCase):
    def test_gate_restores_previous_control_and_respects_external_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'parallel-width48'
            work.mkdir()
            previous = read_control(root)
            hold_queue(root, work)
            self.assertTrue(read_control(root)['paused'])
            hold_queue(root, work)
            release_queue(root, work)
            self.assertEqual(read_control(root), previous)
            hold_queue(root, work)
            changed = set_paused(root, True)
            with self.assertRaisesRegex(RuntimeError, 'changed externally'):
                release_queue(root, work)
            self.assertEqual(read_control(root), changed)

    def test_existing_user_pause_is_not_claimed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'parallel-width48'
            work.mkdir()
            set_paused(root, True)
            with self.assertRaisesRegex(RuntimeError, 'already paused'):
                hold_queue(root, work)


class ParallelTests(unittest.TestCase):
    def setUp(self):
        base = fixtures.CapacityTests()
        base.setUp()
        self.addCleanup(base.doCleanups)
        self.base = base
        config_path = base.fixture.current / 'replay-config.json'
        config = read(config_path)
        config.update(batch_size=256, learning_rate=0.001, weight_decay=0.0001, validation_fraction=0.1, seed=20260906)
        write(config_path, config)
        self.queue = ParallelWidth(base.queue.args)
        self.queue.repo = base.queue.repo
        self.queue.work.mkdir(parents=True)
        write(self.queue.root / 'status.json', {'stage': 'train-capacity-depth16'})
        write(self.queue.root / 'architecture-validation.json', {'passed': True})
        write(self.queue.root / 'backend.json', {'performance': {'adam_backend': 'standard', 'replay_cache': 'device', 'prefetch_batches': 0}})
        self.baseline, self.common, _ = self.queue.prepare()

    def test_training_failure_leaves_comparisons_blocked(self):
        with patch('archive.run_capacity_width_parallel.subprocess.check_output', return_value='30000'):
            with patch.object(self.queue, 'preflight'):
                with patch.object(self.queue, 'command', side_effect=RuntimeError('training failed')):
                    with self.assertRaisesRegex(RuntimeError, 'training failed'):
                        self.queue.run()
        self.assertTrue(read_control(self.queue.root)['paused'])
        self.assertFalse((self.queue.work / 'released.json').exists())

    def test_release_occurs_only_after_checkpoint_validation(self):
        def check_selection(*args):
            self.assertTrue(read_control(self.queue.root)['paused'])
            return self.baseline
        with patch('archive.run_capacity_width_parallel.subprocess.check_output', return_value='30000'):
            with patch.object(self.queue, 'preflight'), patch.object(self.queue, 'command'):
                with patch('archive.run_capacity_width_parallel.select_checkpoints', side_effect=check_selection):
                    self.queue.run()
        self.assertFalse(read_control(self.queue.root)['paused'])
        self.assertTrue((self.queue.work / 'selection.json').exists())
        self.assertTrue((self.queue.work / 'released.json').exists())

    def test_dashboard_reports_both_trainers_running(self):
        write(self.queue.work / 'status.json', {'stage': 'train-capacity-width48'})
        fixture = self.base.fixture
        agent = Experiments(fixture.root / 'activation', fixture.parent)
        with patch('experiment_agent.locked', return_value=True):
            snapshot = agent.snapshot()
        states = {p['id']: p['state'] for p in snapshot['phases']}
        self.assertEqual(states['train-capacity-depth16'], 'running')
        self.assertEqual(states['train-capacity-width48'], 'running')
        write(self.queue.work / 'status.json', {'stage': 'failed', 'error': 'width failed'})
        self.assertEqual({p['id']: p['state'] for p in agent.snapshot()['phases']}['train-capacity-width48'], 'failed')


if __name__ == '__main__':
    unittest.main()
