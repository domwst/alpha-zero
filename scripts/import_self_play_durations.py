#!/usr/bin/env python3
"""Copy verified source self-play timings alongside reconstructed epoch statistics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from self_play_runtime import write


def import_durations(directory):
    history = json.loads((directory/'stats/replay-history.json').read_text())
    source = Path(history['source'])
    epochs = {}
    for entry in history['sources']:
        epoch = entry['epoch']
        content = (source/'stats'/f'{epoch:08}.json').read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        row = json.loads((directory/'stats'/f'{epoch:08}.json').read_text())
        original = json.loads(content)
        if digest != entry['stats_sha256'] or digest != row.get('history_stats_sha256'):
            raise ValueError(f'Source statistics no longer match reconstructed epoch {epoch}')
        if original['epoch'] != epoch or row['epoch'] != epoch:
            raise ValueError('Epoch identity mismatch')
        seconds = original.get('self_play_seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'Missing or invalid self-play duration at epoch {epoch}')
        epochs[str(epoch)] = {'self_play_seconds':seconds, 'source_stats_sha256':digest}
    if len(epochs) != history['epochs']:
        raise ValueError('Incomplete replay history')
    result = {'schema_version':1, 'source_run':str(source), 'epochs':epochs,
              'note':'Original game-generation durations, not time spent by the reconstruction worker.'}
    write(directory/'inherited-self-play.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    result = import_durations(parser.parse_args().run_dir)
    print(f"Imported self-play durations for {len(result['epochs'])} epochs")
