#!/usr/bin/env python3
"""Replace an idle controller after its current training command finishes."""

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
import signal
import subprocess
import time

from experiment_control import read_control, set_paused
from match_plan import MATCH_NAMES, read_match_plan
from experiment_io import digest, read, require, write



def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def boundary_ready(root, plan):
    require(process_start(plan['worker_pid']) == plan['worker_start'], 'Original worker exited or changed')
    state = read(root / 'status.json')
    require(state['pid'] == plan['worker_pid'], 'Unexpected queue writer')
    require(state['stage'] in (plan['active_stage'], 'paused'), 'Worker left the expected training boundary')
    for name in MATCH_NAMES[:3]:
        require(not any((root / (name + suffix)).exists() for suffix in ('.json', '.partial.json', '.log')),
                'A value-head comparison has already started')
    if state['stage'] != 'paused':
        return False
    require(read_control(root)['paused'], 'Queue gate was cleared')
    require(state.get('next_stage', '').startswith(('preflight-', 'train-')) or
            state.get('next_stage') == MATCH_NAMES[0], 'Unexpected next command')
    pid = plan['worker_pid']
    require(not Path(f'/proc/{pid}/task/{pid}/children').read_text().strip(), 'Worker still has an active child')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    root, repo = Path(plan['queue_dir']), Path(plan['repository'])
    monitors = []

    def status(stage, **details):
        value = {'stage': stage, 'pid': os.getpid(), 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), **details}
        write(root / 'match-plan-handoff.json', value)
        print(value, flush=True)

    try:
        with (root / '.match-plan-handoff.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(read_match_plan(root) is not None and read_control(root)['paused'], 'Expected the approved plan and command gate')
            status('waiting_for_training_boundary', active_stage=plan['active_stage'])
            while not boundary_ready(root, plan):
                time.sleep(5)
            for path, expected in plan['sha256'].items():
                require(digest(path) == expected, f'Pinned deployment input changed: {path}')
            require(process_start(plan['launcher_pid']) == plan['launcher_start'], 'Original launcher changed')
            status('replacing_idle_controller')
            os.kill(plan['worker_pid'], signal.SIGTERM)
            deadline = time.monotonic() + 20
            while process_start(plan['worker_pid']) or process_start(plan['launcher_pid']):
                require(time.monotonic() < deadline, 'Original controller or launcher did not exit')
                time.sleep(0.2)
            with (root / '.lock').open('a') as queue_lock:
                fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for name in ('exit-code.txt', 'finished-at.txt', 'launcher-started-at.txt', 'gpu.csv', 'host.csv'):
                    path, backup = root / name, root / ('before-match-plan-' + name)
                    if path.exists():
                        require(not backup.exists(), 'Match-plan archive already exists')
                        path.rename(backup)
            for command in plan.get('monitor_commands', []):
                monitors.append(subprocess.Popen(command, cwd=repo))
            set_paused(root, False)
            status('planned_worker_running')
            result = subprocess.run(plan['worker_command'], cwd=repo).returncode
            (root / 'exit-code.txt').write_text(str(result) + '\n')
            (root / 'finished-at.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + '\n')
            require(result == 0, 'Planned queue worker failed')
            status('complete')
    except Exception as error:
        status('failed', error=str(error))
        raise
    finally:
        for monitor in monitors:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()


if __name__ == '__main__':
    main()
