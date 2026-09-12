#!/usr/bin/env python3
"""Read-only framed archive transport for RunPod's PTY-only SSH gateway."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import tty

ARCHIVE = Path('/tmp/alz-decommission.tar.gz')
tty.setraw(sys.stdin.fileno())
print('ALZ_EXPERIMENT_AGENT_READY', flush=True)
for line in sys.stdin:
    request = json.loads(line)
    try:
        if request['method'] == 'snapshot':
            result = json.loads(ARCHIVE.with_suffix('.receipt.json').read_text())
        elif request['method'] == 'logs':
            offset = request['offset']
            if type(offset) is not int or offset < 0: raise ValueError('Invalid offset')
            with ARCHIVE.open('rb') as stream:
                stream.seek(offset)
                if request.get('stream'):
                    while data := stream.read(512 * 1024):
                        result = {'offset': offset, 'data': base64.b64encode(data).decode(),
                                  'sha256': hashlib.sha256(data).hexdigest()}
                        print(json.dumps({'id': request['id'], 'result': result}), flush=True)
                        offset += len(data)
                    continue
                data = stream.read(512 * 1024)
            result = {'offset': offset, 'data': base64.b64encode(data).decode(),
                      'sha256': hashlib.sha256(data).hexdigest()}
        else: raise ValueError('Unknown method')
        print(json.dumps({'id': request['id'], 'result': result}), flush=True)
    except Exception as error:
        print(json.dumps({'id': request['id'], 'error': str(error)}), flush=True)
