"""Versioned display identities stored alongside experiment results."""


def validate_metadata(document):
    if document.get('schema_version') != 1 or not isinstance(document.get('experiments'), dict):
        raise ValueError('Expected experiment metadata schema version 1')
    for identity, entry in document['experiments'].items():
        if not isinstance(identity, str) or not identity or not isinstance(entry, dict):
            raise ValueError('Invalid experiment identity')
        if 'title' in entry and (not isinstance(entry['title'], str) or not entry['title'].strip()):
            raise ValueError('Invalid experiment title')
        participants = entry.get('participants', {})
        if not isinstance(participants, dict) or set(participants)-{'first', 'second'}:
            raise ValueError('Unknown checkpoint side')
        for participant in participants.values():
            if not isinstance(participant, dict) or not isinstance(participant.get('label'), str) or not participant['label'].strip():
                raise ValueError('Participant needs a display label')
            digest = participant.get('model_sha256')
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('Invalid participant model checksum')
    return document


def with_experiment_metadata(phase, entry=None):
    result = dict(phase)
    entry = entry or {}
    validate_metadata({'schema_version': 1, 'experiments': {phase['id']: entry}})
    if entry.get('title'):
        result['title'] = entry['title']
    if phase.get('kind') != 'battle':
        return result
    participants = {}
    for side in ('first', 'second'):
        checkpoint = (phase.get('result') or {}).get(side+'_checkpoint') or {}
        saved = entry.get('participants', {}).get(side) or phase.get('participants', {}).get(side) or {}
        expected = saved.get('model_sha256')
        actual = checkpoint.get('model_sha256')
        if expected and actual and expected != actual:
            saved = {}
            result['note'] = (result.get('note') or '') + ' Saved display metadata refers to another checkpoint; showing the current identity.'
        if saved.get('label'):
            participants[side] = dict(saved)
        elif checkpoint:
            architecture = checkpoint.get('model', {}).get('architecture', 'Network').replace('_', '-')
            participants[side] = {'label':f"{architecture} · checkpoint {checkpoint.get('epoch', '?')}"}
        else:
            participants[side] = {'label':'Checkpoint A' if side == 'first' else 'Checkpoint B'}
        if actual:
            participants[side]['model_sha256'] = actual
    result['participants'] = participants
    return result


def apply_experiment_metadata(snapshot, document=None):
    entries = validate_metadata(document)['experiments'] if document else {}
    result = dict(snapshot)
    result['phases'] = [with_experiment_metadata(phase, entries.get(phase['id'])) for phase in snapshot['phases']]
    result['experiment_metadata_schema_version'] = 1
    return result
