#!/usr/bin/env python3
"""Run the queued width model alongside depth, holding later queue commands."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import subprocess

from experiment_control import read_control
from archive.run_capacity_experiments import CapacityQueue
from archive.run_checkpoint_comparisons import matched_config, select_checkpoints
from experiment_io import digest, fixed, read, require, write



def hold_queue(root, work):
    """Persist ownership before pausing future commands; running commands continue."""
    with (root / '.control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = work / 'gate.json'
        if not path.exists():
            previous = read_control(root)
            require(not previous['paused'], 'Queue already paused by another controller')
            held = {'paused': True, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'owner': 'parallel-width48', 'reason': 'Wait for parallel width training before subsequent commands'}
            write(path, {'previous': previous, 'held': held})
        gate = read(path)
        current = read_control(root)
        require(current in (gate['previous'], gate['held']), 'Queue control changed externally')
        write(root / 'control.json', gate['held'])


def release_queue(root, work):
    with (root / '.control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        gate = read(work / 'gate.json')
        require(read_control(root) == gate['held'], 'Queue control changed externally; leaving it untouched')
        write(root / 'control.json', gate['previous'])
        write(work / 'released.json', {'released_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})


class ParallelWidth(CapacityQueue):
    @property
    def work(self):
        return self.root / 'parallel-width48'

    def status(self, stage, **details):
        self.stage = stage
        value = {'stage': stage, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'pid': os.getpid(), **details}
        write(self.work / 'status.json', value)
        print(value, flush=True)

    def wait_until_resumed(self, next_stage):
        # The parent waits on this gate. Only this separately locked width worker
        # may run commands while it owns the gate.
        require(read_control(self.root) == read(self.work / 'gate.json')['held'],
                'Parallel worker lost ownership of the queue gate')

    def run(self):
        baseline, common, replays = self.prepare()
        fixed(self.work / 'plan.json', {
            'parent_plan_sha256': digest(self.root / 'plan.json'),
            'script_sha256': digest(__file__), 'model': 'kata_gelu_b10c48_value64x2_v1',
            'epochs': 20, 'selection': 'minimum combined validation loss; earlier ties',
            'preflight_timing_context': 'Concurrent with depth training; correctness checks only, not isolated throughput',
        })
        require(read(self.root / 'architecture-validation.json')['passed'], 'Architecture CUDA checks missing')
        backend = read(self.root / 'backend.json')
        self.performance = backend['performance']
        require(self.performance['adam_backend'] == common['adam_backend'] == 'standard', 'Optimizer mismatch')
        if not (self.work / 'gate.json').exists():
            require(read(self.root / 'status.json')['stage'] == 'train-capacity-depth16', 'Depth trainer is no longer active')
            require(not (self.root / 'capacity-width48').exists(), 'Width training already exists')
            free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            require(float(free.splitlines()[0]) > 18000, 'Insufficient GPU headroom')
        hold_queue(self.root, self.work)
        self.status('preflight-capacity-width48')
        self.preflight('capacity-width48', 'kata-gelu-b10c48-value64x2-v1', common, replays)
        directory = self.root / 'capacity-width48'
        if not (directory / 'result.json').exists():
            self.command('train-capacity-width48', ['train-replay', *replays,
                '--run-dir', directory, '--architecture', 'kata-gelu-b10c48-value64x2-v1',
                '--device', self.args.device, '--epochs', 20,
                '--training-batch-size', common['batch_size'], '--learning-rate', common['learning_rate'],
                '--weight-decay', common['weight_decay'], '--validation-fraction', common['validation_fraction'],
                '--seed', common['seed']])
        selected = select_checkpoints(directory, 'kata_gelu_b10c48_value64x2_v1')
        require(matched_config(selected) == common, 'Width training settings changed')
        write(self.work / 'selection.json', selected)
        # Release only after the native process exited and all 20 passes validate.
        # The parent reuses the completed run (or performs a zero-pass resume if
        # it was already waiting inside its train command).
        release_queue(self.root, self.work)
        self.status('complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.predecessor = args.predecessor.resolve()
    args.output_dir = args.output_dir.resolve()
    args.device, args.binary = 'cuda', None
    queue = ParallelWidth(args)
    queue.work.mkdir(parents=True, exist_ok=True)
    with (queue.work / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (queue.work / 'released.json').exists():
            require((queue.root / 'capacity-width48/result.json').exists(), 'Released worker has no result')
            return
        try:
            queue.run()
        except Exception as error:
            queue.status('failed', failed_stage=queue.stage, error=str(error))
            raise


if __name__ == '__main__':
    main()
