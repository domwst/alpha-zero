import fcntl
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_agent import Experiments
from archive.run_checkpoint_comparisons import CheckpointQueue, select_checkpoints
from experiment_io import checkpoint, digest, read, write

from test_replay_followups import report


class CheckpointComparisonsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent = self.root / 'queue'
        self.output = self.parent / 'pooling-checkpoints'
        self.output.mkdir(parents=True)
        self.current = self.parent / 'current'
        self.katago = self.parent / 'pooling/katago-pooling'
        for directory, architecture, best in [
            (self.current, 'kata_gelu_value64x2_v1', 17),
            (self.katago, 'kata_gelu_pool_value64x2_v1', 14),
        ]:
            write(directory / 'replay-config.json', {'model': {'architecture': architecture},
                  'dataset_sha256': 'data', 'seed': 20260906, 'learning_rate': .001})
            for epoch in range(20):
                model = directory / 'checkpoints' / f'{epoch:08}'
                model.mkdir(parents=True)
                (model / 'model.safetensors').write_bytes(f'{architecture}-{epoch}'.encode())
                write(model / 'metadata.json', {'format_version': 2, 'epoch': epoch,
                      'model': {'architecture': architecture}, 'tensor_schema_sha256': 'schema',
                      'model_sha256': digest(model / 'model.safetensors')})
                value, policy = (.4, 1.6) if epoch == best else (.3, 1.9)
                write(directory / 'epochs' / f'{epoch:08}.json', {'epoch': epoch,
                      'validation': {'value_loss': value, 'policy_loss': policy,
                                     'total_loss': value + policy}})
            write(directory / 'result.json', {'completed_epochs': 20, 'dataset_sha256': 'data',
                  'checkpoint': checkpoint(directory / 'checkpoints/00000019')})
        write(self.parent / 'pooling/selection.json', {
            'selected_checkpoint': checkpoint(self.current / 'checkpoints/00000019')})
        for directory in (self.parent, self.parent / 'pooling'):
            write(directory / 'status.json', {'stage': 'complete'})
            (directory / 'exit-code.txt').write_text('0\n')
        binary = self.root / 'alz'
        binary.write_bytes(b'validated binary')
        self.args = SimpleNamespace(output_dir=self.output, predecessor=self.parent,
            binary=binary, binary_sha256=digest(binary), device='cpu', games=2, simulations=8,
            parallelism=1, inference_batch_size=1)
        self.queue = CheckpointQueue(self.args)

    def test_selection_uses_combined_loss_and_distinguishes_pass_from_epoch(self):
        plan = self.queue.prepare()
        first = plan['matches'][0]
        self.assertEqual(first['first']['epoch'], 17)
        self.assertEqual(first['second']['epoch'], 14)
        self.assertEqual([m['seed'] for m in plan['matches']], [20260912, 20260913, 20260914])
        self.assertEqual(plan['sources']['current']['selected_validation']['value_loss'], .4)
        self.assertEqual(plan['sources']['current']['final_validation']['value_loss'], .3)
        self.assertEqual(plan, self.queue.prepare())
        self.args.games = 4
        with self.assertRaisesRegex(RuntimeError, 'Configuration or inputs changed'):
            self.queue.prepare()

    def test_model_corruption_is_rejected_before_any_match(self):
        (self.katago / 'checkpoints/00000014/model.safetensors').write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            self.queue.prepare()
        self.assertFalse((self.output / 'plan.json').exists())

    def test_changed_selection_and_training_settings_are_rejected(self):
        path = self.katago / 'epochs/00000003.json'
        original = read(path)
        write(path, {'epoch': 3, 'validation': {'value_loss': .1, 'policy_loss': .1, 'total_loss': .2}})
        with self.assertRaisesRegex(RuntimeError, 'proposed passes'):
            self.queue.prepare()
        write(path, original)
        config = self.katago / 'replay-config.json'
        write(config, {**read(config), 'learning_rate': .01})
        with self.assertRaisesRegex(RuntimeError, 'training settings differ'):
            self.queue.prepare()

    def test_nonfinite_loss_and_unfinished_predecessor_are_rejected(self):
        path = self.current / 'epochs/00000001.json'
        path.write_text('{"epoch":1,"validation":{"value_loss":NaN,"policy_loss":1,"total_loss":NaN}}')
        with self.assertRaisesRegex(RuntimeError, 'Invalid validation loss'):
            select_checkpoints(self.current, 'kata_gelu_value64x2_v1')
        write(self.parent / 'pooling/status.json', {'stage': 'running'})
        with self.assertRaisesRegex(RuntimeError, 'completed successfully'):
            self.queue.prepare()

    def test_completed_matches_are_validated_and_reused_without_commands(self):
        plan = self.queue.prepare()
        for match in plan['matches']:
            data = report()
            data.update(first_checkpoint=match['first'], second_checkpoint=match['second'], duration_seconds=12)
            data['config'].update(seed=match['seed'], games_parallelism=1, inference_batch_size=1)
            write(self.output / (match['id'] + '.json'), data)
        with patch.object(self.queue, 'command', side_effect=AssertionError('Must not run GPU work')):
            self.queue.run()
        self.assertEqual(read(self.output / 'status.json')['stage'], 'complete')
        self.assertEqual(len(read(self.output / 'summary.json')['matches']), 3)
        path = self.output / (plan['matches'][0]['id'] + '.json')
        data = read(path)
        data['config']['seed'] += 1
        write(path, data)
        with self.assertRaisesRegex(RuntimeError, 'seed changed'):
            self.queue.run()

    def test_dashboard_shows_running_and_pending_comparisons_and_logs(self):
        plan = self.queue.prepare()
        first = plan['matches'][0]['id']
        write(self.output / 'status.json', {'stage': first})
        (self.output / (first + '.log')).write_text('2026-09-07T12:00:00Z INFO starting\n')
        with (self.output / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            agent = Experiments(self.root / 'activation', self.parent)
            snapshot = agent.snapshot()
            self.assertEqual([p['state'] for p in snapshot['phases'][-3:]], ['running', 'pending', 'pending'])
            self.assertTrue(snapshot['worker']['alive'])
            self.assertEqual(snapshot['worker']['stage'], first)
            self.assertEqual(snapshot['phases'][-3]['match_settings']['games'], 2)
            self.assertIn('starting', agent.dispatch({'method': 'logs', 'phase_id': first})['lines'][0])
        write(self.output / 'status.json', {'stage': 'failed', 'failed_stage': first})
        self.assertEqual(agent.snapshot()['phases'][-3]['state'], 'failed')


if __name__ == '__main__':
    unittest.main()
