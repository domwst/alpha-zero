#!/usr/bin/env python3
"""Recover and cache exact epoch outcomes without changing training or checkpoints."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from self_play_runtime import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    root=args.run_dir.resolve()
    with (root/'.epoch-metrics.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            plan=json.loads((root/'plan.json').read_text())
            for job in plan['jobs']:
                directory=root/job['id']
                output=directory/'epoch-metrics'
                output.mkdir(exist_ok=True)
                for stats in sorted((directory/'stats').glob('[0-9]*.json')):
                    target=output/stats.name
                    if target.exists() or not (directory/'checkpoints'/stats.stem/'metadata.json').exists():
                        continue
                    # Stats are written after the complete snapshot; skip in-progress JSON.
                    try:
                        content=stats.read_bytes()
                        json.loads(content)
                    except (json.JSONDecodeError,FileNotFoundError):
                        continue
                    result=subprocess.run(['./run.sh',str(args.binary.resolve()),str(stats)],
                        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True)
                    if result.returncode:
                        print(f'{job["id"]}/{stats.stem}: {result.stderr.strip()}',flush=True)
                        continue
                    row=json.loads(result.stdout)
                    row['stats_sha256']=hashlib.sha256(content).hexdigest()
                    write(target,row)
                    print(f'{job["id"]} epoch {row["epoch"]+1}: {row["first_player_wins"]} first wins, {row["draws"]} draws, {row["second_player_wins"]} second wins',flush=True)
            if args.once:
                return
            time.sleep(30)


if __name__=='__main__':
    main()
