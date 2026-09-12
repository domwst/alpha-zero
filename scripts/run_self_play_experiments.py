#!/usr/bin/env python3
"""Run the approved self-play/control plan, recording status and resource headroom."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from self_play_runtime import command, resource_sample, stop, write


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def completed(root):
    # Metrics follow successful atomic snapshots; no partial epoch counts as complete.
    return sum((root/'checkpoints'/p.stem/'metadata.json').exists() for p in (root/'stats').glob('[0-9]*.json'))


def initial_job_state(job, previous):
    if job.get('cancelled', False):
        return {**previous, 'stage':'cancelled', 'cancelled_at':job['cancelled_at'],
                'ended_at':previous.get('ended_at') or job['cancelled_at'],
                'note':'Cancelled at user request; automatic resume disabled'}
    if job.get('paused', False):
        return {**previous, 'stage':'paused', 'note':'Held at user request; automatic resume disabled'}
    return {'stage':'queued'}


def process_identity(pid):
    """Pin Linux process identity, including its command, before a handoff."""
    try:
        proc = Path(f'/proc/{pid}')
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        argv = (proc/'cmdline').read_bytes()
        if not argv:  # A process may begin exiting between these two proc reads.
            return None
        return {'pid':pid, 'start_ticks':fields[19],
                'argv':argv.decode().rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


class AdoptedProcess:
    """Monitor an existing process group without pretending to have waitpid status."""
    def __init__(self, identity):
        self.pid = identity['pid']
        self.identity = identity
        if process_identity(self.pid) != identity or os.getpgid(self.pid) != self.pid:
            raise RuntimeError('Cannot adopt a changed process or a non-leader')

    def poll(self):
        # Exit success is decided from complete snapshots by the supervisor.
        return None if process_identity(self.pid) == self.identity else 0

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.identity['argv'], timeout)
            time.sleep(.1)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--adopt-running', type=Path,
                        help='Explicit PID/start-time/argv manifest for a supervisor handoff')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    plan = json.loads((root/'plan.json').read_text())
    check = json.loads((root/'preflight.json').read_text())
    binary = Path('target/release/alz')
    assert check['passed'] and check['binary_sha256'] == hashlib.sha256(binary.read_bytes()).hexdigest()
    lock = (root/'.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'TOKIO_WORKER_THREADS':'16', 'ALZ_LOG_ANSI':'never'}
    active = {}
    serial_only = (root/'serial-only.json').exists() or plan['concurrency'] == 1
    (root/'jobs').mkdir(exist_ok=True)
    states = {}
    for job in plan['jobs']:
        previous = root/'jobs'/(job['id']+'.json')
        old = json.loads(previous.read_text()) if previous.exists() else {}
        states[job['id']] = initial_job_state(job, old)
        if job.get('paused', False) or job.get('cancelled', False):
            write(previous, states[job['id']])
    telemetry = (root/'resources.jsonl').open('a', buffering=1)

    def launch(job):
        name = job['id']
        if job.get('paused', False) or job.get('cancelled', False):
            return
        directory = root/name
        directory.mkdir(exist_ok=True)
        if completed(directory) >= plan['epochs']:
            states[name] = {**states[name], 'stage':'completed'}
            return
        log = (directory/'stdout.log').open('a')
        cmd = command(directory, job['top_p'], epochs=plan['epochs'], batch=plan['inference_batch_size'], adam_backend=plan['adam_backend'],
                      parallelism=job.get('parallelism', plan.get('parallelism', 500)))
        write(directory/'command.json', cmd)
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        active[name] = (process, log)
        previous = root/'jobs'/(name+'.json')
        old = json.loads(previous.read_text()) if previous.exists() else {}
        states[name] = {'stage':'running', 'pid':process.pid, 'started_at':old.get('started_at') or now(), 'segment_started_at':now(),
                        'process_identity':process_identity(process.pid), 'parallelism':job.get('parallelism', plan.get('parallelism', 500))}
        write(previous, states[name])
        print(json.dumps({'event':'started','job':name, **states[name]}), flush=True)

    last_report = 0.0
    try:
        manifest = json.loads(args.adopt_running.read_text()) if args.adopt_running else {}
        # Validate all identities before changing ownership or starting any worker.
        adopted = {}
        for name, identity in manifest.items():
            if name not in states or str(root/name/'checkpoints') not in identity['argv']:
                raise RuntimeError(f'Adoption command does not belong to {name}')
            adopted[name] = AdoptedProcess(identity)
        for job in plan['jobs']:
            name = job['id']
            previous = root/'jobs'/(name+'.json')
            old = json.loads(previous.read_text()) if previous.exists() else {}
            if old.get('stage') == 'running' and process_identity(old['pid']) and name not in adopted:
                raise RuntimeError(f'{name} is still running; explicit adoption is required')
        for name, process in adopted.items():
            previous = root/'jobs'/(name+'.json')
            old = json.loads(previous.read_text())
            active[name] = (process, (root/name/'stdout.log').open('a'))
            states[name] = {**old, 'stage':'running', 'process_identity':process.identity, 'adopted_at':now()}
            write(previous, states[name])
            print(json.dumps({'event':'adopted','job':name, **states[name]}), flush=True)
        for job in plan['jobs']:
            if len(active) >= (1 if serial_only else plan['concurrency']):
                break
            if job['id'] not in active:
                launch(job)
        while active or any(s['stage']=='queued' for s in states.values()):
            sample = resource_sample()
            telemetry.write(json.dumps(sample)+'\n')
            if time.monotonic() - last_report > 60:
                print(json.dumps({'event':'resources','active':list(active), **sample}), flush=True)
                last_report = time.monotonic()
            # Keep the primary running. An interrupted control resumes its last
            # complete checkpoint after the primary finishes; never restart-loop.
            if len(active) > 1 and (sample['working_bytes'] > sample['memory_limit']*.82 or sample['gpu_memory_mib'] > 22500):
                name = plan['jobs'][1]['id']
                process, log = active.pop(name)
                stop(process)
                log.close()
                serial_only = True
                write(root/'serial-only.json', {'at':now(), 'reason':'memory headroom', 'sample':sample})
                states[name] = {**states[name], 'stage':'queued', 'note':'Resumes after primary: memory headroom limit reached'}
                write(root/'jobs'/(name+'.json'), states[name])
                sample = resource_sample()
            if len(active) == 1 and sample['working_bytes'] > sample['memory_limit']*.93:
                raise RuntimeError('Single-worker host-memory headroom exhausted; stopping before cgroup OOM')
            for name, (process, log) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del active[name]
                    success = code == 0 and completed(root/name) >= plan['epochs']
                    states[name] = {**states[name], 'stage':'completed' if success else 'failed',
                                    'exit_code':None if isinstance(process, AdoptedProcess) else code, 'ended_at':now()}
                    write(root/'jobs'/(name+'.json'), states[name])
                    if not success:
                        raise RuntimeError(f'{name} exited with {code}; inspect its log')
            if not active:
                next_job = next((j for j in plan['jobs'] if states[j['id']]['stage']=='queued'), None)
                if next_job:
                    launch(next_job)
            stage = 'running' if active else ('paused' if any(s['stage']=='paused' for s in states.values()) else 'completed')
            write(root/'status.json', {'stage':stage,'active':list(active),'jobs':states,'resources':sample,'serial_only':serial_only,'updated_at':now()})
            if active:
                time.sleep(5)
    except BaseException as error:
        for name, (process, log) in active.items():
            stop(process)
            log.close()
            states[name] = {**states[name], 'stage':'failed', 'error':str(error), 'ended_at':now()}
            write(root/'jobs'/(name+'.json'), states[name])
        write(root/'status.json', {'stage':'failed','error':str(error),'jobs':states,'updated_at':now()})
        raise
    finally:
        telemetry.close()
        lock.close()


if __name__ == '__main__':
    main()
