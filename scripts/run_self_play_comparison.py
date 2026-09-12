#!/usr/bin/env python3
"""Run an identity-pinned comparison between pausing and resuming a self-play supervisor."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from experiment_io import checkpoint, digest, now, read, require, validate_battle, write
from run_self_play_experiments import process_identity
from self_play_runtime import resource_sample, stop


def validate_plan(plan):
    require(plan['schema_version'] == 1, 'Unsupported comparison plan')
    for side in ('first', 'second'):
        require(checkpoint(plan[side]['path']) == plan[side], 'Checkpoint changed: '+side)
    require(digest(plan['binary']) == plan['binary_sha256'], 'Validated binary changed')
    require(digest(plan['supervisor_script']) == plan['supervisor_sha256'], 'Supervisor code changed')
    require(process_identity(plan['supervisor']['pid']) == plan['supervisor'], 'Supervisor identity changed')
    require(process_identity(plan['worker']['pid']) == plan['worker'], 'Self-play worker identity changed')
    require(plan['settings']['games'] > 0 and plan['settings']['games'] % 2 == 0, 'Need balanced seats')


def wait_gone(identity, seconds=30):
    deadline = time.monotonic()+seconds
    while process_identity(identity['pid']) == identity:
        require(time.monotonic() < deadline, 'Process did not stop: '+str(identity['pid']))
        time.sleep(.1)


def battle_command(plan, output):
    s = plan['settings']
    argv = [str(Path(plan['repository'])/'run.sh'), plan['binary'], 'battle',
            '--first-checkpoint-dir', plan['first']['path'], '--second-checkpoint-dir', plan['second']['path'],
            '--device', 'cuda', '--no-move-logs', '--output', str(output)]
    for flag, key in [('games','games'), ('simulations','simulations'), ('temperature','temperature'),
                      ('games-parallelism','parallelism'), ('inference-batch-size','inference_batch_size'),
                      ('batch-timeout-us','batch_timeout_us'), ('seed','seed'), ('heartbeat-seconds','heartbeat_seconds')]:
        argv += ['--'+flag, str(s[key])]
    return argv


def run(plan_path):
    plan = read(plan_path)
    root, selfplay = plan_path.parent, Path(plan['self_play_dir'])
    state = {'stage':'preparing', 'started_at':now(), 'pid':os.getpid()}
    def status(**changes):
        state.update(changes, updated_at=now())
        write(root/'status.json', state)
        print(state, flush=True)
    validate_plan(plan)
    require(not (root/'status.json').exists(), 'Handoff already started; inspect its saved status')
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
           'TOKIO_WORKER_THREADS':'16', 'ALZ_LOG_ANSI':'never'}
    paused = False
    battle = None
    status(stage='pausing_self_play')
    try:
        # This supervisor's SIGTERM handler stops and waits for its own child group.
        os.kill(plan['supervisor']['pid'], signal.SIGTERM)
        paused = True
        wait_gone(plan['supervisor'])
        wait_gone(plan['worker'])
        with (Path(plan['self_play_root'])/'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job_path = Path(plan['self_play_root'])/'jobs'/(plan['self_play_id']+'.json')
        job = read(job_path)
        job.update(stage='paused', note='Paused for '+plan['title']+'; automatically resumes afterward')
        write(job_path, job)
        s = plan['settings']
        sample = resource_sample()
        require(sample['working_bytes'] < sample['memory_limit']*.85 and sample['gpu_memory_mib']<22000,
                'Insufficient memory for comparison')
        output = root/'result.partial.json'
        command = battle_command(plan, output)
        write(root/'command.json', command)
        with (root/'battle.log').open('w') as log, (root/'resources.jsonl').open('a',buffering=1) as telemetry:
            import json
            battle = subprocess.Popen(command, cwd=plan['repository'], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            status(stage='running', battle_pid=battle.pid)
            while battle.poll() is None:
                sample = resource_sample()
                telemetry.write(json.dumps(sample)+'\n')
                require(sample['working_bytes'] < sample['memory_limit']*.92 and sample['gpu_memory_mib']<23000,
                        'Comparison stopped for memory pressure')
                time.sleep(2)
            require(battle.returncode == 0, 'Battle failed; see battle.log')
        report = read(output)
        validate_battle(report, plan['first'], plan['second'], s['games'], s['simulations'], s['temperature'])
        for field,key in [('games_parallelism','parallelism'),('inference_batch_size','inference_batch_size'),('seed','seed')]:
            require(report['config'][field] == s[key], 'Battle setting changed: '+field)
        output.replace(root/'result.json')
        status(stage='completed', ended_at=now())
    except BaseException as error:
        status(stage='failed', error=str(error), ended_at=now())
        raise
    finally:
        if battle is not None:
            stop(battle)
        if paused:
            # Never overlap the resumed trainer with either old process group.
            wait_gone(plan['supervisor'])
            wait_gone(plan['worker'])
            require(digest(plan['supervisor_script']) == plan['supervisor_sha256'], 'Resume supervisor changed')
            with (selfplay/'resume-supervisor.log').open('a') as log:
                resumed = subprocess.Popen(plan['supervisor']['argv'], cwd=plan['repository'], env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            time.sleep(3)
            require(resumed.poll() is None, 'Resumed supervisor exited; see resume-supervisor.log')
            status(resume={'stage':'started', 'pid':resumed.pid, 'at':now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Comparison interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    with (args.plan.parent/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.plan.resolve())


if __name__ == '__main__':
    main()
