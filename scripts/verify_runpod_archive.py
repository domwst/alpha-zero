#!/usr/bin/env python3
"""Extract a deduplicated backup and verify every file before pod termination."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('backup', type=Path)
args = parser.parse_args()
root = args.backup.resolve()
receipt = json.loads((root / 'transfer-receipt.json').read_text())
with (root / 'workspace.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['archive_sha256']
restored = root / 'restored'
restored.mkdir(exist_ok=True)
with tarfile.open(root / 'workspace.tar.gz', 'r:gz') as archive:
    archive.extractall(restored, filter='data')
manifest = json.loads((restored / 'MANIFEST.json').read_text())
assert len(manifest['files']) == receipt['files']
digests = {}
for entry in manifest['files']:
    path = restored / entry['path']
    assert path.is_relative_to(restored) and path.is_file() and not path.is_symlink()
    info = path.stat()
    assert info.st_size == entry['size'], str(path)
    identity = (info.st_dev, info.st_ino)
    if identity not in digests:
        with path.open('rb') as stream:
            digests[identity] = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digests[identity] == entry['sha256'], str(path)

# Verify checkpoint identities against their original, independently saved metadata.
checkpoints = 0
for metadata in restored.glob('workspace/**/checkpoints/*/metadata.json'):
    data = json.loads(metadata.read_text())
    for field, name in [('model_sha256', 'model.safetensors')]:
        if field in data:
            path = metadata.parent / name
            info = path.stat()
            assert digests[(info.st_dev, info.st_ino)] == data[field], str(path)
    checkpoints += 1

queue = restored / 'workspace/alpha-zero-followups/runs/value-heads-20260906/training-recipes'
selections = json.loads((queue / 'selection.json').read_text())
assert len(selections) == 6
assert json.loads((queue / 'status.json').read_text())['stage'] == 'complete'
assert (queue / 'exit-code.txt').read_text().strip() == '0'
for name, selection in selections.items():
    assert len(list((queue / name / 'checkpoints').glob('*/metadata.json'))) == 20
    assert json.loads((queue / name / 'result.json').read_text())['completed_epochs'] == 20
    for which in ['selected', 'final']:
        descriptor = selection[which]
        path = restored / descriptor['path'].lstrip('/') / 'model.safetensors'
        info = path.stat()
        assert digests[(info.st_dev, info.st_ino)] == descriptor['model_sha256']
snapshot = json.loads((restored / 'workspace/dashboard-archive.json').read_text())
assert len(snapshot['snapshot']['phases']) == 28
assert len(snapshot['logs']) == 28
assert all(p['state'] in ('completed', 'cancelled') for p in snapshot['snapshot']['phases'])
verified = {'verified_at': time.time(), 'archive_sha256': receipt['archive_sha256'],
            'files_verified': len(manifest['files']), 'unique_inodes_verified': len(digests),
            'checkpoint_directories': checkpoints, 'recipe_models': 6, 'recipe_checkpoints': 120,
            'dashboard_jobs': 28, 'all_checks_passed': True}
(root / 'verification.json').write_text(json.dumps(verified, indent=2) + '\n')
print(json.dumps(verified), flush=True)
