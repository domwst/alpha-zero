#!/usr/bin/env python3
"""Resume an archive transfer and verify both chunks and the complete archive."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
from experiment_dashboard import SSHBridge

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--connection', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.connection.read_text())
config['remote_agent'] = '/tmp/read_runpod_archive.py'
bridge = SSHBridge(config)
args.output_dir.mkdir(parents=True, exist_ok=True)
try:
    receipt = bridge.call('snapshot')
    manifest_path = args.output_dir / 'transfer-receipt.json'
    if manifest_path.exists(): assert json.loads(manifest_path.read_text()) == receipt
    else: manifest_path.write_text(json.dumps(receipt, indent=2) + '\n')
    pending = args.output_dir / 'workspace.tar.gz.partial'
    offset = pending.stat().st_size if pending.exists() else 0
    assert offset <= receipt['archive_bytes']
    started = last = time.monotonic()
    initial = offset
    with pending.open('ab') as stream:
        failures = 0
        while offset < receipt['archive_bytes']:
            try:
                chunk = bridge.call('logs', offset=offset, stream=True)
                while True:
                    data = base64.b64decode(chunk['data'], validate=True)
                    assert chunk['offset'] == offset and data
                    assert hashlib.sha256(data).hexdigest() == chunk['sha256']
                    stream.write(data); stream.flush()
                    offset += len(data)
                    if time.monotonic() - last > 15:
                        print(json.dumps({'bytes': offset, 'total': receipt['archive_bytes'],
                              'MiB_per_second': (offset - initial) / (time.monotonic() - started) / 2**20}), flush=True)
                        last = time.monotonic()
                    if offset == receipt['archive_bytes']: break
                    response = json.loads(bridge.receive(b'\n'))
                    assert response['id'] == bridge.serial and 'error' not in response
                    chunk = response['result']
            except Exception:
                failures += 1
                if failures >= 5: raise
                bridge.close()
                time.sleep(2)
    with pending.open('rb') as stream: checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert checksum == receipt['archive_sha256'], 'Archive checksum mismatch'
    pending.rename(args.output_dir / 'workspace.tar.gz')
    print(json.dumps({'verified': True, **receipt}), flush=True)
finally:
    bridge.close()
