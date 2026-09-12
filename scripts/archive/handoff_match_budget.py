#!/usr/bin/env python3
"""Apply a new match budget after an active comparison has published its result."""

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
from archive.handoff_match_plan import process_start
from match_plan import read_match_plan
from experiment_io import digest, read, require, write



def boundary_ready(root, plan):
    pid = plan['worker_pid']
    require(process_start(pid) == plan['worker_start'], 'Original controller exited or changed')
    state = read(root / 'status.json')
    require(state['pid'] == pid and state['stage'] in (plan['active_stage'], 'paused'),
            'Controller left the expected match boundary')
    for suffix in ('.log', '.json', '.partial.json'):
        require(not (root / (plan['next_stage'] + suffix)).exists(), 'Next comparison already started')
    if state['stage'] != 'paused':
        return False
    require(read_control(root)['paused'] and state.get('next_stage') == plan['next_stage'],
            'Unexpected command gate')
    require(not Path(f'/proc/{pid}/task/{pid}/children').read_text().strip(), 'Controller still has an active child')
    require((root / (plan['active_stage'] + '.json')).is_file(), 'Active comparison has not published its result')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    plan = read(parser.parse_args().plan)
    root, repo = Path(plan['queue_dir']), Path(plan['repository'])
    monitors = []

    def status(stage, **details):
        value = {'stage': stage, 'pid': os.getpid(), 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), **details}
        write(root / 'budget-handoff.json', value)
        print(value, flush=True)

    try:
        with (root / '.budget-handoff.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(read_control(root)['paused'], 'Command gate must be installed first')
            status('waiting_for_match_boundary', active_stage=plan['active_stage'], next_stage=plan['next_stage'])
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
                for name in ('exit-code.txt', 'finished-at.txt', 'gpu.csv', 'host.csv',
                             'match-plan-worker.json', plan['next_stage'] + '.settings.json'):
                    path, backup = root / name, root / ('before-1000-games-' + name)
                    if path.exists():
                        require(not backup.exists(), 'Match-budget archive already exists')
                        path.rename(backup)
                runner = repo / 'scripts/archive/run_match_plan.py'
                write(root / 'match-plan-worker.json', {
                    'original_queue_config_sha256': digest(root / 'queue-config.json'),
                    'match_plan': read_match_plan(root),
                    'code_sha256': {p.name: digest(p) for p in (runner, runner.parent.parent/'match_plan.py')},
                })
            for command in plan.get('monitor_commands', []):
                monitors.append(subprocess.Popen(command, cwd=repo))
            set_paused(root, False)
            status('updated_worker_running')
            result = subprocess.run(plan['worker_command'], cwd=repo).returncode
            (root / 'exit-code.txt').write_text(str(result) + '\n')
            (root / 'finished-at.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + '\n')
            require(result == 0, 'Updated queue worker failed')
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
