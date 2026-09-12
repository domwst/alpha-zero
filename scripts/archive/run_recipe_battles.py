#!/usr/bin/env python3
"""Run the four paired-seed recipe comparisons from verified selected checkpoints."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import time

from experiment_io import now

from experiment_control import read_control
from archive.run_capacity_battles_overlap import can_start, progress
from experiment_io import checkpoint, digest, fixed, read, require, write
from archive.run_replay_followups import Queue


class RecipeBattle(Queue):
    def status(self, stage, **details):
        self.stage = stage
        write(self.root / 'jobs' / (self.name + '.json'),
              {'stage': stage, 'updated_at': now(), 'pid': os.getpid(), **details})

    def run_match(self, match, models):
        try:
            result = self.battle(self.name, models[match['first']], models[match['second']], match['seed'])
            config = read(self.root / (self.name + '.json'))['config']
            require(config['games_parallelism'] == self.args.parallelism and
                    config['inference_batch_size'] == self.args.inference_batch_size,
                    'Battle concurrency settings changed')
            self.status('complete')
            return result
        except Exception as error:
            self.status('failed', error=str(error))
            raise


def run(args):
    root = args.output_dir
    plan = read(root / 'recipe-battle-plan.json')
    preflight = read(root / 'preflight.json')
    require(preflight['passed'] is True, 'GPU preflight has not passed')
    require(preflight['binary_sha256'] == plan['binary_sha256'], 'Preflight binary changed')
    require(digest(args.binary) == plan['binary_sha256'], 'Validated binary changed')
    models = {name: checkpoint(item['selected']['path']) for name, item in plan['models'].items()}
    require(all(models[name] == item['selected'] for name, item in plan['models'].items()),
            'Selected checkpoint identities changed')
    for key in ('games', 'simulations', 'parallelism', 'inference_batch_size'):
        require(getattr(args, key) == plan['settings'][key], 'Match settings changed: ' + key)
    fixed(root / 'worker-config.json', {'plan_sha256': digest(root / 'recipe-battle-plan.json'),
                                      'script_sha256': digest(__file__)})
    names = [match['id'] for match in plan['matches']]
    finished, active, counts = {}, {}, {}
    failure = None
    started = now()
    with ThreadPoolExecutor(max_workers=2) as executor:
        while len(finished) < len(names):
            for name, (future, job) in list(active.items()):
                if future.done():
                    try:
                        finished[name] = future.result()
                        counts[name] = args.games
                        write(root / 'summary.json', {'matches': finished})
                    except Exception as error:
                        failure = str(error)
                    del active[name]
                else:
                    counts[name] = progress(root / (name + '.log'), job.offset, args.games)
            if failure and not active:
                write(root / 'status.json', {'stage': 'failed', 'error': failure, 'updated_at': now(),
                                            'started_at': started, 'ended_at': now()})
                raise RuntimeError(failure)
            for index, match in enumerate(plan['matches']):
                name = match['id']
                if name in finished or name in active:
                    continue
                if failure or read_control(root)['paused'] or not can_start(index, names, finished, active, counts, args.games):
                    break
                job = RecipeBattle(args)
                job.name = name
                log = root / (name + '.log')
                job.offset = log.stat().st_size if log.exists() else 0
                job.status('starting')
                active[name] = (executor.submit(job.run_match, match, models), job)
            write(root / 'status.json', {'stage': 'running' if active else 'paused', 'updated_at': now(),
                  'started_at': started, 'pid': os.getpid(), 'active': list(active), 'completed_games': counts,
                  'failure': failure})
            if len(finished) < len(names):
                time.sleep(5)
    write(root / 'status.json', {'stage': 'complete', 'started_at': started, 'ended_at': now(),
                                'updated_at': now(), 'pid': os.getpid(), 'active': [], 'completed_games': counts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda'])
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--simulations', type=int, default=4000)
    parser.add_argument('--parallelism', type=int, default=300)
    parser.add_argument('--inference-batch-size', type=int, default=64)
    args = parser.parse_args()
    with (args.output_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
