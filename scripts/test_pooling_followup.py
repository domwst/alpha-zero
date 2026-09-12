import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_agent import Experiments
from archive.run_pooling_followup import (HEADS, MATCHES, STANDARD, PoolingQueue,
                                  pooling_architecture, predecessor_ready, select_head)
from experiment_io import checkpoint, digest, read, write

from test_replay_followups import report


class PoolingTests(unittest.TestCase):
    def test_extra_games_do_not_give_an_opponent_more_selection_weight(self):
        models, reports = self.fixtures()
        # Every pairing is tied, but the wide/deep pairing has three times as many games.
        match = reports['wide64-vs-deep64']
        original = copy.deepcopy(match['games'])
        match['games'] = [{**original[i % 2], 'game': i + 1} for i in range(6)]
        match['config']['games'] = 6
        for side in ('first', 'second'):
            match[side + '_checkpoint_result'].update(wins=3, losses=3, draws=0, score=3)
        budgets = {'current-vs-wide64': 2, 'current-vs-deep64': 2, 'wide64-vs-deep64': 6}
        selected = select_head(reports, models, budgets, 8)
        self.assertEqual(selected['head'], 'current')
        self.assertEqual(selected['score_rates'], {'current': .5, 'wide64': .5, 'deep64': .5})
        self.assertEqual(selected['games_played'], {'current': 4, 'wide64': 8, 'deep64': 8})
        with self.assertRaisesRegex(RuntimeError, 'wrong number of games'):
            select_head(reports, models, {**budgets, 'wide64-vs-deep64': 4}, 8)

    def fixtures(self):
        template = report()
        models = {name: {**template['first_checkpoint'], 'model_sha256': name} for name in HEADS}
        reports = {}
        for name, first, second in MATCHES:
            data = copy.deepcopy(template)
            data['first_checkpoint'], data['second_checkpoint'] = models[first], models[second]
            reports[name] = data
        return models, reports

    def test_selection_uses_all_matches_and_ties_prefer_smallest(self):
        models, reports = self.fixtures()
        self.assertEqual(select_head(reports, models, 2, 8)['head'], 'current')
        # Deep wins both games against wide, breaking the otherwise tied table.
        match = reports['wide64-vs-deep64']
        for game in match['games']:
            game['winner'] = 'second_checkpoint'
        match['first_checkpoint_result'].update(wins=0, losses=2, draws=0, score=0)
        match['second_checkpoint_result'].update(wins=2, losses=0, draws=0, score=2)
        selection = select_head(reports, models, 2, 8)
        self.assertEqual(selection['head'], 'deep64')
        self.assertEqual(selection['hidden_dims'], [64, 64])
        self.assertEqual(selection['score_rates']['deep64'], 0.75)
        match['first_checkpoint']['model_sha256'] = 'wrong'
        # Detach the expected descriptors from this mutated fixture.
        models, _ = self.fixtures()
        with self.assertRaisesRegex(RuntimeError, 'identities differ'):
            select_head(reports, models, 2, 8)

    def test_architecture_mapping_covers_each_activation_and_head(self):
        for prefix in ('kata', 'kata-gelu'):
            for suffix in ('-v1', '-value64-v1', '-value64x2-v1'):
                self.assertEqual(pooling_architecture(prefix + suffix), prefix + '-pool' + suffix)
        with self.assertRaises(RuntimeError):
            pooling_architecture('legacy-resnet-v1')

    def test_predecessor_must_be_complete_and_successful(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(predecessor_ready(root))
            (root / 'exit-code.txt').write_text('1\n')
            write(root / 'status.json', {'stage': 'paused'})
            self.assertFalse(predecessor_ready(root))  # stale launcher during upgrade
            write(root / 'status.json', {'stage': 'failed'})
            with self.assertRaises(RuntimeError):
                predecessor_ready(root)
            write(root / 'status.json', {'stage': 'complete'})
            with self.assertRaises(RuntimeError):
                predecessor_ready(root)
            (root / 'exit-code.txt').write_text('0\n')
            self.assertTrue(predecessor_ready(root))

    def test_cache_failure_keeps_baseline_optimizer_and_persists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = PoolingQueue(SimpleNamespace(output_dir=root, binary=Path('/tmp/model'), device='cuda'))
            plan = {'architecture_command': ['architecture'], 'cache_command': ['cache']}
            requested = {'adam_backend': 'standard', 'replay_cache': 'device', 'prefetch_batches': 0}
            with patch('archive.run_pooling_followup.validate_backend', side_effect=[
                    {'passed': True}, {'passed': False, 'timed_out': True}]) as validate:
                queue.configure_backend(plan, requested)
                self.assertEqual(validate.call_count, 2)
                self.assertEqual(validate.call_args.args[0], ['cache'])
            self.assertEqual(queue.performance, STANDARD)
            with patch('archive.run_pooling_followup.validate_backend') as validate:
                queue.configure_backend(plan, requested)
                validate.assert_not_called()
            self.assertEqual(read(root / 'backend.json')['performance'], STANDARD)
            with self.assertRaisesRegex(RuntimeError, 'Requested pooling backend changed'):
                queue.configure_backend(plan, {**requested, 'adam_backend': 'fused'})

    def test_cache_success_enables_device_cache_without_changing_adam(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = PoolingQueue(SimpleNamespace(output_dir=root, binary=Path('/tmp/model'), device='cuda'))
            requested = {**STANDARD, 'replay_cache': 'device'}
            with patch('archive.run_pooling_followup.validate_backend', return_value={'passed': True}):
                queue.configure_backend({'architecture_command': [], 'cache_command': []}, requested)
            self.assertEqual(queue.performance, requested)

    def test_inherited_fused_optimizer_cannot_silently_fall_back_to_standard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = PoolingQueue(SimpleNamespace(output_dir=root, binary=Path('/tmp/model'), device='cuda'))
            with patch('archive.run_pooling_followup.validate_backend', side_effect=[{'passed': True}, {'passed': False}]):
                with self.assertRaisesRegex(RuntimeError, 'cannot change reused baseline optimizer'):
                    queue.configure_backend({'architecture_command': [], 'backend_command': []},
                                            {**STANDARD, 'adam_backend': 'fused', 'replay_cache': 'device'})
            self.assertFalse((root / 'backend.json').exists())

    def test_reused_checkpoint_requires_matching_budget_data_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'checkpoints/00000019'
            model.mkdir(parents=True)
            (model / 'model.safetensors').write_bytes(b'test model')
            architecture = {'architecture': 'kata_gelu_v1'}
            write(model / 'metadata.json', {'model': architecture, 'epoch': 19, 'format_version': 2,
                  'model_sha256': digest(model / 'model.safetensors'), 'tensor_schema_sha256': 'test'})
            common = {'schema_version': 1, 'dataset_sha256': 'data', 'seed': 123, 'batch_size': 256}
            write(root / 'replay-config.json', {**common, 'model': architecture})
            descriptor = checkpoint(model)
            result = {'checkpoint': descriptor, 'model': architecture, 'completed_epochs': 20, 'dataset_sha256': 'data'}
            write(root / 'result.json', result)
            self.assertEqual(PoolingQueue.trained_checkpoint(root, 'kata-gelu-v1', common, 20, 'standard'), descriptor)
            for wrong in [{**common, 'dataset_sha256': 'wrong'}, {**common, 'seed': 999}]:
                with self.assertRaisesRegex(RuntimeError, 'budget or data changed'):
                    PoolingQueue.trained_checkpoint(root, 'kata-gelu-v1', wrong, 20, None)
            with self.assertRaisesRegex(RuntimeError, 'final training budget'):
                PoolingQueue.trained_checkpoint(root, 'kata-gelu-v1', common, 21, None)
            with self.assertRaisesRegex(RuntimeError, 'backend changed'):
                PoolingQueue.trained_checkpoint(root, 'kata-gelu-v1', common, 20, 'fused')
            (model / 'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                PoolingQueue.trained_checkpoint(root, 'kata-gelu-v1', common, 20, None)

    def test_bad_architecture_validation_stops_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = PoolingQueue(SimpleNamespace(output_dir=root, binary=Path('/tmp/model'), device='cuda'))
            with patch('archive.run_pooling_followup.validate_backend', return_value={'passed': False}):
                with self.assertRaisesRegex(RuntimeError, 'architecture CUDA checks failed'):
                    queue.configure_backend({'architecture_command': []}, STANDARD)
            self.assertFalse((root / 'backend.json').exists())

    def test_dashboard_appends_pooling_and_tracks_its_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / 'queue'
            write(queue / 'pooling-followup.json', {'directory': 'pooling'})
            write(queue / 'status.json', {'stage': 'complete'})
            write(queue / 'pooling/status.json', {'stage': 'failed', 'failed_stage': 'train-katago-pooling'})
            agent = Experiments(root / 'activation', queue)
            snapshot = agent.snapshot()
            self.assertEqual(len(snapshot['phases']), 12)
            self.assertEqual(snapshot['phases'][-2]['state'], 'failed')
            self.assertEqual(snapshot['worker']['stage'], 'failed')
            self.assertEqual(agent.dispatch({'method': 'logs', 'phase_id': 'train-katago-pooling'})['lines'], [])
            with self.assertRaises(ValueError):
                agent.dispatch({'method': 'resume'})
            write(queue / 'pooling/queue-config.json', {'baseline_policy': 'reuse-selected-checkpoint'})
            snapshot = agent.snapshot()
            self.assertEqual(len(snapshot['phases']), 11)
            self.assertNotIn('train-current-pooling', [p['id'] for p in snapshot['phases']])
            self.assertEqual(snapshot['phases'][-2]['state'], 'failed')
            self.assertIn('Reuses', snapshot['phases'][-2]['note'])


if __name__ == '__main__':
    unittest.main()
