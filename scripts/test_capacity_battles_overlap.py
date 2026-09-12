import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiment_control import read_control
from experiment_agent import Experiments
from archive.run_capacity_width_parallel import hold_queue, release_queue
from archive.run_capacity_battles_overlap import POLICY, OverlapQueue, can_start, install_gate, progress
from archive.run_capacity_experiments import MATCHES
from experiment_io import read, write

import test_capacity_parallel as fixtures


class SchedulingTests(unittest.TestCase):
    def test_exact_threshold_and_two_job_limit(self):
        names = ['a', 'b', 'c']
        self.assertTrue(can_start(0, names, {}, {}, {}, 1000))
        self.assertFalse(can_start(1, names, {}, {'a': 1}, {'a': 599}, 1000))
        self.assertTrue(can_start(1, names, {}, {'a': 1}, {'a': 600}, 1000))
        self.assertFalse(can_start(2, names, {}, {'a': 1, 'b': 1}, {'b': 900}, 1000))
        self.assertTrue(can_start(2, names, {'a': 1}, {'b': 1}, {'b': 600}, 1000))
        self.assertTrue(can_start(2, names, {'b': 1}, {'a': 1}, {}, 1000))

    def test_restart_ignores_old_progress_and_rejects_wrong_game_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'battle.log'
            path.write_text('games_completed=950 games_total=1000\n')
            offset = path.stat().st_size
            self.assertEqual(progress(path, offset, 1000), 0)
            with path.open('a') as f:
                f.write('games_completed=599 games_total=1000\n')
            self.assertEqual(progress(path, offset, 1000), 599)
            with self.assertRaisesRegex(RuntimeError, 'Invalid battle progress'):
                progress(path, offset, 600)

    def test_width_gate_hands_control_to_overlap_instead_of_serial_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            width, overlap = root / 'parallel-width48', root / 'overlap-battles'
            width.mkdir(); overlap.mkdir()
            hold_queue(root, width)
            width_control = read_control(root)
            gate = install_gate(root, overlap)
            self.assertEqual(read_control(root), width_control)
            self.assertEqual(install_gate(root, overlap), gate)
            release_queue(root, width)
            self.assertEqual(read_control(root), gate['held'])
            self.assertTrue(read_control(root)['paused'])
            write(overlap / 'activated.json', {})
            write(root / 'control.json', gate['previous'])
            install_gate(root, overlap)
            self.assertFalse(read_control(root)['paused'])

    def test_threaded_schedule_overlaps_without_starting_three_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / 'plan.json', {})
            write(root / 'overlap-battles/plan.json', {})
            args = SimpleNamespace(output_dir=root, binary=None, games=1000,
                                   simulations=4000, parallelism=300, inference_batch_size=64, device='cpu')
            queue = OverlapQueue(args)
            names = [m['id'] for m in MATCHES]
            second, third, release_first = threading.Event(), threading.Event(), threading.Event()
            active, maximum, started, seeds = set(), [0], [], []
            guard = threading.Lock()

            class FakeJob:
                def __init__(self, _args): pass
                def status(self, _stage): pass
                def run_match(self, match, selections, seed):
                    with guard:
                        active.add(self.name)
                        maximum[0] = max(maximum[0], len(active))
                        started.append(self.name); seeds.append(seed)
                    if self.name == names[0]:
                        assert second.wait(2)
                        assert release_first.wait(2)
                    elif self.name == names[1]:
                        second.set()
                        assert third.wait(2)
                    else:
                        third.set()
                    with guard:
                        active.remove(self.name)
                    return {'match': self.name}

            def simulated_progress(path, offset, total):
                if path.stem == names[1]:
                    release_first.set()
                return 600

            with patch('archive.run_capacity_battles_overlap.BattleJob', FakeJob):
                with patch('archive.run_capacity_battles_overlap.progress', simulated_progress):
                    with patch.dict(POLICY, poll_seconds=0.001):
                        queue.battles({}, {'seed': 20260906})
            self.assertEqual(started, names)
            self.assertEqual(seeds, [20260926, 20260927, 20260928])
            self.assertEqual(maximum[0], 2)
            self.assertEqual(set(read(root / 'summary.json')['matches']), set(names))

    def test_handoff_refuses_to_terminate_worker_with_a_live_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / 'status.json', {'stage': 'paused'})
            write(root / 'parallel-width48/status.json', {'stage': 'complete'})
            for name in ['capacity-depth16', 'capacity-width48']:
                write(root / name / 'result.json', {})
            held = {'paused': True, 'owner': 'overlap-battles'}
            write(root / 'control.json', held)
            (root / 'overlap-battles').mkdir()
            proc = root / 'proc/123/task/123'
            proc.mkdir(parents=True)
            (proc / 'children').write_text('456')
            (root / 'proc/123/cmdline').write_bytes(b'python3\0scripts/archive/run_capacity_experiments.py\0')
            queue = OverlapQueue(SimpleNamespace(output_dir=root, binary=None))
            real_path = Path
            def mapped_path(value):
                return root / str(value).lstrip('/') if str(value).startswith('/proc/') else real_path(value)
            with patch('archive.run_capacity_battles_overlap.Path', side_effect=mapped_path):
                with patch('archive.run_capacity_battles_overlap.identity', return_value='start'):
                    with patch('archive.run_capacity_battles_overlap.os.kill') as kill:
                        with self.assertRaisesRegex(RuntimeError, 'still has a child'):
                            queue.wait_and_take_over({'held': held}, {'pid': 123, 'start_time': 'start'})
                        kill.assert_not_called()
                        (proc / 'children').write_text('')
                        with patch('archive.run_capacity_battles_overlap.subprocess.run', return_value=SimpleNamespace(returncode=1)):
                            queue.wait_and_take_over({'held': held}, {'pid': 123, 'start_time': 'start'})
                        kill.assert_called_once()


class DashboardTests(unittest.TestCase):
    def test_two_active_battles_and_overlap_policy_are_visible(self):
        base = fixtures.ParallelTests()
        base.setUp()
        self.addCleanup(base.doCleanups)
        root = base.queue.root
        write(root / 'overlap-battles/plan.json', {'policy': POLICY})
        write(root / 'overlap-battles/status.json', {'stage': 'running'})
        for match in MATCHES[:2]:
            write(root / 'overlap-battles/jobs' / (match['id'] + '.json'), {'stage': match['id']})
        fixture = base.base.fixture
        agent = Experiments(fixture.root / 'activation', fixture.parent)
        with patch('experiment_agent.locked', return_value=True):
            phases = {p['id']: p for p in agent.snapshot()['phases']}
        self.assertEqual([phases[m['id']]['state'] for m in MATCHES], ['running', 'running', 'pending'])
        self.assertIn('60%', phases[MATCHES[1]['id']]['note'])


if __name__ == '__main__':
    unittest.main()
