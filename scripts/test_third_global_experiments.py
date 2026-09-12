import fcntl
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_agent import Experiments
from archive.run_checkpoint_comparisons import select_checkpoints
from experiment_io import checkpoint, digest, read, write

from archive.run_third_global_experiments import ThirdGlobalQueue, MODEL, MATCHES, POLICY, schedule
import test_capacity_experiments as fixtures


class ThirdGlobalTests(unittest.TestCase):
    def setUp(self):
        base = fixtures.CapacityTests()
        base.setUp()
        self.addCleanup(base.doCleanups)
        self.base, f = base, base.fixture
        baseline, _, _ = base.queue.prepare()
        depth = base.queue.root / 'capacity-depth16'
        shutil.copytree(f.katago, depth)
        arch = 'kata_gelu_b16c32_value64x2_v1'
        for path in [depth / 'replay-config.json', *depth.glob('checkpoints/*/metadata.json')]:
            data = read(path); data['model'] = {'architecture': arch}; write(path, data)
        for path in depth.glob('epochs/*.json'):
            data = read(path); data['training'] = dict(data['validation']); write(path, data)
        result = read(depth / 'result.json')
        result['checkpoint'] = checkpoint(depth / 'checkpoints/00000019')
        write(depth / 'result.json', result)
        write(base.queue.root / 'selection.json', {'baseline': baseline,
              'capacity-depth16': select_checkpoints(depth, arch)})
        write(base.queue.root / 'status.json', {'stage': 'complete'})
        (base.queue.root / 'exit-code.txt').write_text('0\n')
        self.queue = ThirdGlobalQueue(SimpleNamespace(output_dir=f.parent / 'third-global',
            predecessor=base.queue.root, binary=f.args.binary, device='cpu'))
        self.queue.repo = base.queue.repo
        for name in ['scripts/archive/run_third_global_experiments.py', 'scripts/archive/launch_third_global_runpod.sh']:
            shutil.copyfile(Path(__file__).resolve().parents[1] / name, self.queue.repo / name)
        write(self.queue.repo / 'global3-build-ready.json', {'binary_sha256': f.args.binary_sha256})

    def test_freezes_reused_opponents_and_budget(self):
        sources, common, replays = self.queue.prepare()
        self.assertEqual(MODEL['global_block_indices'], [3, 7, 11])
        self.assertEqual(MODEL['blocks'], 16)
        self.assertEqual(len(MATCHES), 2)
        self.assertEqual(sources['baseline']['selected']['epoch'], 17)
        self.assertEqual(sources['capacity-depth16']['selected']['epoch'], 14)
        self.assertEqual(schedule()['settings'], {'games': 1000, 'simulations': 4000,
                         'parallelism': 300, 'inference_batch_size': 64, 'temperature': .7})
        self.assertEqual(self.queue.prepare(), (sources, common, replays))
        Path(replays[1], 'replay.bin.zst').write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'Replay source changed'): self.queue.prepare()

    def test_cuda_failure_prevents_training(self):
        with patch.object(self.queue, 'configure_backend', side_effect=RuntimeError('CUDA failure')):
            with patch.object(self.queue, 'command') as command:
                with self.assertRaisesRegex(RuntimeError, 'CUDA failure'): self.queue.run()
                command.assert_not_called()

    def test_changed_opponent_blocks_work(self):
        self.queue.prepare()
        p = self.base.queue.root / 'capacity-depth16/checkpoints/00000014/model.safetensors'
        p.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'): self.queue.prepare()

    def test_dashboard_exposes_new_jobs_and_overlap(self):
        self.queue.prepare()
        root = self.queue.root
        write(root / 'status.json', {'stage': 'global3_battles'})
        for m in MATCHES: write(root / 'overlap-battles/jobs' / (m['id'] + '.json'), {'stage': m['id']})
        with (root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            phases = Experiments(self.base.fixture.root / 'activation', self.base.fixture.parent).snapshot()['phases']
            self.assertEqual([p['id'] for p in phases[-3:]], ['train-' + MODEL['id'], *[m['id'] for m in MATCHES]])
            self.assertEqual([p['state'] for p in phases[-2:]], ['running', 'running'])
            self.assertIn('60%', phases[-1]['note'])

    def test_both_matches_overlap_with_distinct_seeds(self):
        _, common, _ = self.queue.prepare()
        second = threading.Event()
        started, seeds = [], []
        class FakeJob:
            def __init__(self, args): pass
            def status(self, stage): pass
            def run_match(self, match, selections, seed):
                started.append(self.name); seeds.append(seed)
                if self.name == MATCHES[0]['id']: assert second.wait(2)
                else: second.set()
                return {'match': self.name}
        with patch('archive.run_third_global_experiments.BattleJob', FakeJob), \
             patch('archive.run_third_global_experiments.progress', return_value=600), \
             patch.dict(POLICY, poll_seconds=.001):
            self.queue.battles({}, common)
        self.assertEqual(started, [m['id'] for m in MATCHES])
        self.assertEqual(seeds, [20260936, 20260937])
        self.assertEqual(len(read(self.queue.root / 'summary.json')['matches']), 2)


if __name__ == '__main__': unittest.main()
