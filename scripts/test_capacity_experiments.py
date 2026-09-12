import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiment_agent import Experiments
from archive.run_capacity_experiments import CapacityQueue, MODELS, MATCHES, schedule
from experiment_io import digest, read, write

import test_checkpoint_comparisons as checkpoint_fixtures


class CapacityTests(unittest.TestCase):
    def setUp(self):
        fixture = checkpoint_fixtures.CheckpointComparisonsTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        write(fixture.output / 'cancellation.json', {
            'cancelled_at': '2026-09-07T16:12:07+00:00', 'reason': 'User requested cancellation',
            'jobs': ['current-pooling-selected-vs-final', 'katago-pooling-selected-vs-final']})
        write(fixture.output / 'status.json', {'stage': 'cancelled'})
        source = fixture.root / 'replay'
        source.mkdir()
        (source / 'replay.bin.zst').write_bytes(b'replay fixture')
        write(fixture.current / 'dataset.json', {'dataset_sha256': 'data', 'sources': [
            {'checkpoint': {'path': str(source)}, 'replay_sha256': digest(source / 'replay.bin.zst')}]})
        repo = fixture.root / 'repo'
        for name in ('scripts/archive/run_capacity_experiments.py', 'scripts/archive/run_checkpoint_comparisons.py',
                     'scripts/archive/run_pooling_followup.py', 'scripts/archive/run_replay_followups.py',
                     'scripts/experiment_control.py', 'scripts/run_cuda_tests_runpod.sh', 'run.sh'):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(__file__).resolve().parents[1] / name, path)
        write(repo / 'capacity-build-ready.json', {'binary_sha256': fixture.args.binary_sha256})
        self.queue = CapacityQueue(SimpleNamespace(output_dir=fixture.parent / 'capacity',
            predecessor=fixture.parent, binary=fixture.args.binary, device='cpu'))
        self.queue.repo = repo

    def test_plan_pins_depth_width_data_and_selected_baseline(self):
        baseline, common, replays = self.queue.prepare()
        self.assertEqual(baseline['selected']['epoch'], 17)
        self.assertEqual([(m['blocks'], m['channels']) for m in MODELS], [(16, 32), (10, 48)])
        self.assertEqual(len(MATCHES), 3)
        self.assertEqual(common['adam_backend'], 'standard')
        self.assertEqual(self.queue.args.games, 1000)
        self.assertEqual(self.queue.prepare(), (baseline, common, replays))
        (Path(replays[1]) / 'replay.bin.zst').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Replay source changed'):
            self.queue.prepare()

    def test_changed_binary_blocks_training(self):
        self.fixture.args.binary.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Validated binary changed'):
            self.queue.prepare()

    def test_dashboard_marks_cancelled_jobs_and_appends_capacity_work(self):
        self.fixture.queue.prepare()
        self.queue.prepare()
        agent = Experiments(self.fixture.root / 'activation', self.fixture.parent)
        snapshot = agent.snapshot()
        phases = {p['id']: p for p in snapshot['phases']}
        self.assertEqual(phases['current-pooling-selected-vs-final']['state'], 'cancelled')
        self.assertEqual(phases['katago-pooling-selected-vs-final']['state'], 'cancelled')
        self.assertEqual([p['kind'] for p in snapshot['phases'][-5:]], ['training', 'training', 'battle', 'battle', 'battle'])
        self.assertTrue(all(p['state'] == 'pending' for p in snapshot['phases'][-5:]))
        write(self.queue.root / 'status.json', {'stage': 'failed', 'failed_stage': 'train-capacity-depth16'})
        self.assertEqual(agent.snapshot()['phases'][-5]['state'], 'failed')

    def test_cuda_failure_cannot_start_training(self):
        with patch.object(self.queue, 'configure_backend', side_effect=RuntimeError('CUDA failure')):
            with patch.object(self.queue, 'preflight') as preflight:
                with self.assertRaisesRegex(RuntimeError, 'CUDA failure'):
                    self.queue.run()
                preflight.assert_not_called()


if __name__ == '__main__':
    unittest.main()
