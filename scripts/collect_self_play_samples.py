#!/usr/bin/env python3
"""Backfill sample sheets and export eight samples after each completed epoch."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    with (root/'.sample-export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            # Repair the latest samples first, then backfill earlier epochs.
            for stats in sorted((root/'stats').glob('[0-9]*.json'), reverse=True):
                if not (root/'checkpoints'/stats.stem/'metadata.json').exists():
                    continue
                try:
                    content = stats.read_bytes()
                    json.loads(content)
                    marker = root/'games'/f'{stats.stem}.samples.json'
                    if marker.exists() and json.loads(marker.read_text()).get('stats_sha256') == hashlib.sha256(content).hexdigest():
                        continue
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
                result = subprocess.run(['./run.sh', str(args.binary.resolve()), str(stats)],
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                if result.returncode:
                    print(f'{stats.name}: sample export failed; will retry', flush=True)
            if args.once:
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
