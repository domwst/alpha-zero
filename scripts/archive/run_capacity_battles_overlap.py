#!/usr/bin/env python3
"""Hand off the capacity queue after training; overlap battles at 60%, at most two."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import datetime
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from experiment_io import locked

from experiment_control import read_control
from archive.run_capacity_experiments import CapacityQueue, MATCHES, MODELS
from archive.run_checkpoint_comparisons import matched_config, select_checkpoints
from experiment_io import digest, fixed, read, require, write
from archive.run_replay_followups import Queue

POLICY = {'start_next_at_percent': 60, 'max_active_battles': 2, 'poll_seconds': 5}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def install_gate(root, work):
    """Chain a new gate after the width worker's gate, without stopping training."""
    with (root / '.control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = work / 'gate.json'
        if receipt.exists():
            gate = read(receipt)
            if (work / 'activated.json').exists():
                return gate
            current = read_control(root)
            width_path = root / 'parallel-width48/gate.json'
            width = read(width_path)
            if current == width['held']:
                require(width['previous'] in (gate['previous'], gate['held']), 'Width successor gate changed externally')
                write(width_path, {**width, 'previous': gate['held']})
            elif current == gate['previous']:
                write(root / 'control.json', gate['held'])
            else:
                require(current == gate['held'], 'Comparison gate changed externally')
            return gate
        current = read_control(root)
        held = {'paused': True, 'updated_at': now(), 'owner': 'overlap-battles',
                'reason': 'Transfer comparison scheduling after both trainers exit'}
        width_path = root / 'parallel-width48/gate.json'
        width = read(width_path)
        if current == width['held']:
            require(not width['previous']['paused'], 'Another gate already follows width training')
            gate = {'previous': width['previous'], 'held': held, 'original_width_gate': width}
            write(receipt, gate)
            write(width_path, {**width, 'previous': held})
        else:
            require(not current['paused'], 'Queue paused by another controller')
            gate = {'previous': current, 'held': held}
            write(receipt, gate)
            write(root / 'control.json', held)
        return gate


def progress(path, offset, total):
    if not path.exists():
        return 0
    with path.open() as stream:
        stream.seek(offset)
        lines = stream.read().splitlines()
    completed = 0
    for line in lines:
        match = re.search(r'games_completed=(\d+).*games_total=(\d+)', line)
        if match:
            count, actual_total = map(int, match.groups())
            require(actual_total == total and 0 <= count <= total, 'Invalid battle progress')
            completed = max(completed, count)
    return completed


def can_start(index, names, finished, active, counts, total):
    if len(active) >= POLICY['max_active_battles']:
        return False
    return index == 0 or names[index - 1] in finished or counts.get(names[index - 1], 0) * 100 >= total * POLICY['start_next_at_percent']


class BattleJob(Queue):
    def status(self, stage, **details):
        self.stage = stage
        write(self.root / 'overlap-battles/jobs' / (self.name + '.json'),
              {'stage': stage, 'updated_at': now(), 'pid': os.getpid(), 'log_offset': self.offset, **details})

    def run_match(self, match, selections, seed):
        try:
            result = self.battle(self.name, selections[match['first']]['selected'],
                                 selections[match['second']]['selected'], seed)
            config = read(self.root / (self.name + '.json'))['config']
            require(config['games_parallelism'] == 300 and config['inference_batch_size'] == 64,
                    'Battle settings changed')
            self.status('complete')
            return result
        except Exception as error:
            self.status('failed', failed_stage=self.name, error=str(error))
            raise


class OverlapQueue(CapacityQueue):
    @property
    def work(self):
        return self.root / 'overlap-battles'

    def overlap_status(self, stage, **details):
        write(self.work / 'status.json', {'stage': stage, 'updated_at': now(), 'pid': os.getpid(), **details})

    def wait_and_take_over(self, gate, previous_worker):
        if not (self.work / 'handoff.json').exists():
            self.overlap_status('waiting_for_training')
            while True:
                width = read(self.root / 'parallel-width48/status.json')
                require(width['stage'] != 'failed', 'Width trainer failed; battles remain blocked')
                parent = read(self.root / 'status.json')
                ready = all((self.root / model['id'] / 'result.json').exists() for model in MODELS)
                if ready and not locked(self.root / 'parallel-width48/.lock') and parent['stage'] == 'paused':
                    break
                require(parent['stage'] != 'failed', 'Depth queue failed; battles remain blocked')
                time.sleep(5)
            require(read_control(self.root) == gate['held'], 'Comparison gate changed externally')
            pid = previous_worker['pid']
            require(identity(pid) == previous_worker['start_time'], 'Original worker identity changed')
            command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            require(any(arg.endswith(b'/run_capacity_experiments.py') for arg in command), 'Unexpected worker command')
            require(not Path(f'/proc/{pid}/task/{pid}/children').read_text().strip(), 'Original worker still has a child process')
            write(self.work / 'handoff.json', {'stopped_at': now(), 'previous_worker': previous_worker,
                  'previous_status': parent, 'reason': 'Both trainers finished; replace idle serial comparison scheduler'})
            os.kill(pid, signal.SIGTERM)
        elif identity(previous_worker['pid']) == previous_worker['start_time']:
            # Recover a crash between publishing the handoff receipt and SIGTERM.
            pid = previous_worker['pid']
            require(read(self.root / 'status.json')['stage'] == 'paused', 'Original worker is no longer idle')
            require(not Path(f'/proc/{pid}/task/{pid}/children').read_text().strip(), 'Original worker still has a child process')
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while subprocess.run(['tmux', 'has-session', '-t', 'alz-capacity-20260907'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            require(time.monotonic() < deadline, 'Original launcher did not exit')
            time.sleep(1)

    def battles(self, selections, common):
        names = [m['id'] for m in MATCHES]
        finished, active, counts = {}, {}, {}
        summary = {'plan_sha256': digest(self.root / 'plan.json'),
                   'overlap_plan_sha256': digest(self.work / 'plan.json'), 'matches': finished}
        failure = None
        with ThreadPoolExecutor(max_workers=2) as executor:
            while len(finished) < len(names):
                for name, (future, job) in list(active.items()):
                    if future.done():
                        try:
                            finished[name] = future.result()
                            counts[name] = self.args.games
                            write(self.root / 'summary.json', summary)
                        except Exception as error:
                            failure = error
                        del active[name]
                    else:
                        counts[name] = progress(self.root / (name + '.log'), job.offset, self.args.games)
                if failure is not None and not active:
                    raise failure
                for index, match in enumerate(MATCHES):
                    name = match['id']
                    if name in finished or name in active:
                        continue
                    if failure is not None or read_control(self.root)['paused']:
                        break
                    if not can_start(index, names, finished, active, counts, self.args.games):
                        break
                    job = BattleJob(self.args)
                    job.repo, job.name = self.repo, name
                    log = self.root / (name + '.log')
                    job.offset = log.stat().st_size if log.exists() else 0
                    job.status('starting')
                    active[name] = (executor.submit(job.run_match, match, selections, common['seed'] + 20 + index), job)
                self.status('capacity_battles', active_battles=list(active), completed_games=counts)
                self.overlap_status('running', active_battles=list(active), completed_games=counts)
                if len(finished) < len(names):
                    time.sleep(POLICY['poll_seconds'])

    def run(self):
        baseline, common, _ = self.prepare()
        plan_path = self.work / 'plan.json'
        if plan_path.exists():
            previous_worker = read(plan_path)['previous_worker']
        else:
            parent = read(self.root / 'status.json')
            require(parent['stage'] == 'train-capacity-depth16', 'Install overlap while depth training is active')
            previous_worker = {'pid': parent['pid'], 'start_time': identity(parent['pid'])}
            require(previous_worker['start_time'] is not None, 'Original queue worker is not alive')
        fixed(plan_path, {'parent_plan_sha256': digest(self.root / 'plan.json'), 'policy': POLICY,
                         'script_sha256': digest(__file__), 'previous_worker': previous_worker})
        gate = install_gate(self.root, self.work)
        self.wait_and_take_over(gate, previous_worker)
        with ExitStack() as stack:
            for directory in (self.args.predecessor, self.args.predecessor / 'pooling',
                              self.args.predecessor / 'pooling-checkpoints', self.root):
                lock = stack.enter_context((directory / '.lock').open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not (self.work / 'previous-launcher.json').exists():
                write(self.work / 'previous-launcher.json', {name: (self.root / name).read_text()
                      for name in ('status.json', 'exit-code.txt', 'finished-at.txt') if (self.root / name).exists()})
                for name in ('exit-code.txt', 'finished-at.txt'):
                    (self.root / name).unlink(missing_ok=True)
            selections = {'baseline': baseline}
            for model in MODELS:
                selected = select_checkpoints(self.root / model['id'], model['architecture'].replace('-', '_'))
                require(matched_config(selected) == common, 'Training settings changed')
                selections[model['id']] = selected
            write(self.root / 'selection.json', selections)
            if not (self.work / 'activated.json').exists():
                with (self.root / '.control.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    require(read_control(self.root) == gate['held'], 'Comparison gate changed externally')
                    write(self.root / 'control.json', gate['previous'])
                    write(self.work / 'activated.json', {'activated_at': now()})
            monitors = [subprocess.Popen(['bash', str(self.repo / 'scripts' / script),
                        str(self.work / output), '1000']) for script, output in
                        [('monitor_gpu.sh', 'gpu.csv'), ('monitor_host.sh', 'host.csv')]]
            try:
                self.battles(selections, common)
                self.status('complete', summary=str(self.root / 'summary.json'))
                self.overlap_status('complete')
                (self.root / 'exit-code.txt').write_text('0\n')
                (self.root / 'finished-at.txt').write_text(now() + '\n')
            finally:
                for process in monitors:
                    process.terminate()
                    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.predecessor, args.output_dir = args.predecessor.resolve(), args.output_dir.resolve()
    args.device, args.binary = 'cuda', None
    queue = OverlapQueue(args)
    queue.work.mkdir(parents=True, exist_ok=True)
    with (queue.work / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            queue.run()
        except Exception as error:
            queue.overlap_status('failed', error=str(error))
            if (queue.work / 'activated.json').exists():
                queue.status('failed', error=str(error))
                (queue.root / 'exit-code.txt').write_text('1\n')
                (queue.root / 'finished-at.txt').write_text(now() + '\n')
            raise


if __name__ == '__main__':
    main()
