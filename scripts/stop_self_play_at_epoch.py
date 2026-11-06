#!/usr/bin/env python3
"""Stop an identity-pinned self-play supervisor after its next complete epoch."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from run_self_play_experiments import now, process_identity
from self_play_runtime import write


def complete_epochs(directory):
    result = []
    for path in (directory/'stats').glob('[0-9]*.json'):
        try:
            row = json.loads(path.read_text())
            metadata = directory/'checkpoints'/path.stem/'metadata.json'
            if metadata.exists() and row['epoch'] == int(path.stem):
                result.append(row['epoch'])
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--supervisor-pid', type=int, required=True)
    parser.add_argument('--job-id', default='nucleus-p095')
    parser.add_argument('--pause-note', default='Paused after a fully saved epoch at user request')
    args = parser.parse_args()
    root, work = args.run_dir.resolve(), args.work_dir.resolve()
    work.mkdir(exist_ok=True, parents=True)
    with (work/'.boundary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (work/'boundary-stopped.json').exists(), 'Boundary already handled'
        status = json.loads((root/'status.json').read_text())
        assert status['active'] == [args.job_id], 'Expected exactly the selected job active'
        supervisor = process_identity(args.supervisor_pid)
        assert supervisor and any(script in supervisor['argv'] for script in
            ('scripts/run_self_play_experiments.py', 'scripts/run_board_mask_selfplay.py'))
        assert status.get('pid') == args.supervisor_pid, 'Supervisor does not own this status'
        pid = status['jobs'][args.job_id]['pid']
        identity = process_identity(pid)
        directory = root/args.job_id
        assert identity and str(directory/'checkpoints') in identity['argv']
        prior = complete_epochs(directory)
        target = max(prior, default=-1)+1
        write(work/'boundary-request.json', {'at':now(), 'supervisor':supervisor,
              'worker':identity, 'target_epoch':target, 'stage':'waiting_for_epoch'})
        print(f'Waiting for epoch {target+1} to complete; worker {pid}', flush=True)
        while True:
            assert process_identity(args.supervisor_pid) == supervisor, 'Supervisor changed'
            assert process_identity(pid) == identity, 'Worker exited before boundary'
            # The completion log follows snapshot, renders, stats and JSONL writes.
            with (directory/'stdout.log').open('rb') as log:
                log.seek(max(0, os.fstat(log.fileno()).st_size-8192))
                lines = log.read().decode(errors='replace').splitlines()
            import re
            done = any('epoch complete' in line and
                       re.search(rf'\bepoch={target}\b', line) for line in lines)
            if done and target in complete_epochs(directory):
                break
            time.sleep(.25)
        # Freeze the worker first so no additional epoch work starts while the
        # old supervisor shuts down. SIGTERM avoids its worker-cleanup path.
        os.killpg(pid, signal.SIGSTOP)
        assert process_identity(args.supervisor_pid) == supervisor
        os.kill(args.supervisor_pid, signal.SIGTERM)
        deadline = time.monotonic()+15
        while process_identity(args.supervisor_pid) == supervisor:
            assert time.monotonic()<deadline, 'Supervisor did not exit'
            time.sleep(.1)
        os.killpg(pid, signal.SIGINT)
        os.killpg(pid, signal.SIGCONT)
        deadline = time.monotonic()+15
        while process_identity(pid) == identity:
            if time.monotonic()>=deadline:
                os.killpg(pid, signal.SIGKILL)
                break
            time.sleep(.1)
        with (root/'.lock').open('a') as queue_lock:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = json.loads((root/'jobs'/f'{args.job_id}.json').read_text())
            state.update(stage='paused', note=args.pause_note)
            write(root/'jobs'/f'{args.job_id}.json', state)
            status.update(stage='paused', active=[], updated_at=now())
            status['jobs'][args.job_id] = state
            write(root/'status.json', status)
        receipt = {'at':now(), 'completed_epoch':target, 'worker':identity,
                   'stage':'stopped_at_epoch_boundary'}
        write(work/'boundary-stopped.json', receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
