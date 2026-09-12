import json
from pathlib import Path
import tempfile
import unittest

from experiment_dashboard import ArchiveBridge, Dashboard
from unittest.mock import Mock
from experiment_metadata import apply_experiment_metadata, validate_metadata
from migrate_experiment_metadata import migrate


class MetadataTests(unittest.TestCase):
    def phase(self):
        return {'id': 'unknown-future-job', 'title': 'Original title', 'kind': 'battle',
                'result': {'first_checkpoint': {'model': {'architecture': 'kata_gelu_v1'},
                           'epoch': 11, 'model_sha256': 'a'*64}}}

    def test_arbitrary_identity_and_checksum_guard(self):
        phase = self.phase()
        metadata = {'schema_version': 1, 'experiments': {phase['id']: {
            'title': 'Recorded title', 'participants': {
                'first': {'label': 'Candidate', 'model_sha256': 'a'*64},
                'second': {'label': 'Baseline'}}}}}
        shown = apply_experiment_metadata({'phases': [phase]}, metadata)['phases'][0]
        self.assertEqual(shown['participants']['first']['label'], 'Candidate')
        self.assertEqual(shown['title'], 'Recorded title')
        self.assertNotIn('participants', phase)
        metadata['experiments'][phase['id']]['participants']['first']['model_sha256'] = 'b'*64
        shown = apply_experiment_metadata({'phases': [phase]}, metadata)['phases'][0]
        self.assertEqual(shown['participants']['first']['label'], 'kata-gelu-v1 · checkpoint 11')
        self.assertEqual(shown['participants']['first']['model_sha256'], 'a'*64)
        self.assertIn('another checkpoint', shown['note'])

    def test_invalid_schema_and_hash_rejected(self):
        for document in [{'schema_version': 2, 'experiments': {}},
                         {'schema_version': 1, 'experiments': {'x': {'participants': {'first': {'label': 'X', 'model_sha256': 'bad'}}}}}]:
            with self.assertRaises(ValueError):
                validate_metadata(document)

    def test_migration_is_additive_idempotent_and_preserves_custom_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'archive.json'
            target = source.with_suffix('.metadata.json')
            phase = self.phase()
            source.write_text(json.dumps({'snapshot': {'phases': [phase]}, 'logs': {}}))
            original = source.read_bytes()
            catalog = {phase['id']: ['Historical candidate', 'Historical baseline']}
            result = migrate(source, target, catalog)
            self.assertEqual(result['experiments'][phase['id']]['participants']['first']['label'], 'Historical candidate')
            stamp = target.stat().st_mtime_ns
            migrate(source, target, catalog)
            self.assertEqual(target.stat().st_mtime_ns, stamp)
            result['experiments'][phase['id']]['participants']['first']['label'] = 'Hand-edited label'
            target.write_text(json.dumps(result))
            migrate(source, target, catalog)
            self.assertEqual(json.loads(target.read_text())['experiments'][phase['id']]['participants']['first']['label'], 'Hand-edited label')
            self.assertEqual(source.read_bytes(), original)
            bridge = ArchiveBridge(source)
            shown = bridge.call('snapshot')['phases'][0]
            self.assertEqual(shown['participants']['first']['label'], 'Hand-edited label')
            phase['result']['first_checkpoint']['model_sha256'] = 'b'*64
            source.write_text(json.dumps({'phases': [phase]}))
            before = target.read_bytes()
            with self.assertRaisesRegex(ValueError, 'different checkpoint'):
                migrate(source, target, catalog)
            self.assertEqual(target.read_bytes(), before)

    def test_pin_becomes_available_after_queued_phase_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'snapshot.json', Path(directory)/'metadata.json'
            phase = self.phase()
            completed = phase.pop('result')
            source.write_text(json.dumps({'phases': [phase]}))
            catalog = {phase['id']: ['Candidate', 'Baseline']}
            migrate(source, target, catalog)
            phase['result'] = completed
            source.write_text(json.dumps({'phases': [phase]}))
            entry = migrate(source, target, catalog)['experiments'][phase['id']]
            self.assertEqual(entry['participants']['first']['model_sha256'], 'a'*64)

    def test_local_registry_enriches_legacy_remote_snapshots_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'connection.metadata.json'
            phase = self.phase()
            path.write_text(json.dumps({'schema_version': 1, 'experiments': {
                phase['id']: {'participants': {'first': {'label': 'Local historical name'}}}}}))
            bridge = Mock()
            bridge.call.return_value = {'phases': [phase]}
            dashboard = Dashboard(bridge, path)
            dashboard.refresh()
            snapshot = dashboard.snapshot()
            self.assertTrue(snapshot['connected'])
            self.assertEqual(snapshot['snapshot']['phases'][0]['participants']['first']['label'], 'Local historical name')
            bridge.call.assert_called_once_with('snapshot')
            self.assertNotIn('participants', phase)
            path.write_text('{bad json')
            dashboard.refresh()
            self.assertFalse(dashboard.snapshot()['connected'])
            self.assertEqual(dashboard.snapshot()['snapshot'], snapshot['snapshot'])
