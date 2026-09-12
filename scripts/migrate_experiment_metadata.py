#!/usr/bin/env python3
"""Backfill checkpoint labels without rewriting result files or archived snapshots."""
import argparse
import hashlib
import json
from pathlib import Path

from experiment_io import read, write
from experiment_metadata import validate_metadata


def migrate(snapshot_path, output, legacy_labels):
    payload = read(snapshot_path)
    snapshot = payload.get('snapshot', payload)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('phases'), list):
        raise ValueError('Input must contain an experiment snapshot')
    previous = read(output) if output.exists() else {'schema_version':1, 'experiments':{}}
    validate_metadata(previous)
    entries = json.loads(json.dumps(previous['experiments']))
    for phase in snapshot['phases']:
        identity = phase['id']
        entry = {'title':phase['title'], 'kind':phase['kind']}
        if phase['kind'] == 'battle':
            participants = {}
            for index, side in enumerate(('first', 'second')):
                checkpoint = (phase.get('result') or {}).get(side+'_checkpoint') or {}
                existing = phase.get('participants', {}).get(side, {})
                label = existing.get('label')
                if not label and identity in legacy_labels:
                    label = legacy_labels[identity][index]
                if not label:
                    architecture = checkpoint.get('model', {}).get('architecture', 'Network').replace('_', '-')
                    label = f"{architecture} · checkpoint {checkpoint['epoch']}" if checkpoint else ('Checkpoint A' if side == 'first' else 'Checkpoint B')
                participants[side] = {'label':label}
                if checkpoint.get('model_sha256'):
                    participants[side]['model_sha256'] = checkpoint['model_sha256']
            entry['participants'] = participants
        if identity in entries:
            # Preserve hand-edited display labels, but never silently repin them.
            for side, participant in entry.get('participants', {}).items():
                saved = entries[identity].setdefault('participants', {}).setdefault(side, dict(participant))
                old = saved.get('model_sha256')
                new = participant.get('model_sha256')
                if old and new and old != new:
                    raise ValueError(f'{identity}/{side} now refers to a different checkpoint')
                if new and not old:
                    saved['model_sha256'] = new
        else:
            entries[identity] = entry
    result = {**previous, 'experiments':entries}
    if 'migration' not in result:
        result['migration'] = {'source_sha256':hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                               'description':'Backfilled display metadata; original experiment data unchanged'}
    validate_metadata(result)
    if result != previous:
        write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-labels', type=Path, default=Path(__file__).parent/'migrations/legacy-experiment-labels.json')
    args = parser.parse_args()
    result = migrate(args.snapshot, args.output, read(args.legacy_labels))
    print(f"{len(result['experiments'])} experiment identities in {args.output}")


if __name__ == '__main__':
    main()
