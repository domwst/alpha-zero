#!/usr/bin/env python3
"""Reconstruct an ordered replay history, then continue guarded self-play."""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import time

from experiment_io import digest, fixed, locked, now, read, require, write

from self_play_runtime import resource_sample, stop

def reconstruction_plan(directory, binary, args):
    path = directory/'reconstruction-plan.json'
    if path.exists():
        plan = read(path)
        require(plan['binary_sha256'] == digest(binary), 'Validated trainer changed')
        for requested, saved in [(args.source_run, plan['source']), (args.history_epochs, plan['history_epochs']),
                                 (args.epochs, plan['continue_to_epoch'])]:
            if requested is not None:
                require(str(requested) == str(saved), 'Requested settings differ from the saved plan')
        require(args.architecture is None and args.parallelism is None and args.wait_for_queue is None,
                'Existing plans resume their saved architecture, parallelism, and queue dependency')
        return plan
    require(args.source_run is not None and args.history_epochs is not None,
            'New reconstruction requires --source-run and --history-epochs')
    require(args.history_epochs > 0, 'History count must be positive')
    source = args.source_run.resolve()
    recipe = read(source/'stats'/f'{args.history_epochs-1:08}.json')['config']
    target = args.epochs if args.epochs is not None else (recipe.get('epochs') or args.history_epochs)
    require(args.parallelism is None or args.parallelism > 0, 'Parallelism must be positive')
    require(target >= args.history_epochs, 'Target epochs precede the history')
    argv = ['./run.sh', str(binary), 'train', '--device', 'cuda', '--architecture',
            args.architecture or 'kata-gelu-boardmask-value64x2-v1', '--epochs', str(args.history_epochs)]
    fields = ['top_p','replay_lr_exponent','adam_backend','replay_cache','prefetch_batches',
              'games_per_epoch','simulations','c_puct','replay_games','replay_positions',
              'replay_position_growth','replay_growth_start_epoch','inference_batch_size',
              'inference_symmetry','games_parallelism','batch_timeout_us','training_batch_size',
              'learning_rate','weight_decay','rendered_games','seed','progress_every_games','heartbeat_seconds']
    if recipe.get('replay_games') is not None:
        fields = [key for key in fields if key not in ('replay_positions', 'replay_position_growth',
                                                     'replay_growth_start_epoch', 'replay_lr_exponent')]
    for key in fields:
        value = args.parallelism if key == 'games_parallelism' and args.parallelism is not None else recipe.get(key)
        if value is not None:
            argv += ['--'+key.replace('_','-'), str(value)]
    if recipe.get('bn_gamma_one'):
        argv.append('--bn-gamma-one')
    for flag, value in [('checkpoint-dir',directory/'checkpoints'),('stats-dir',directory/'stats'),
                        ('games-dir',directory/'games'),('replay-history-dir',source),
                        ('replay-history-epochs',args.history_epochs)]:
        argv += ['--'+flag,str(value)]
    plan = {'schema_version':2,'source':str(source),'history_epochs':args.history_epochs,
            'binary_sha256':digest(binary),'command':argv,'continue_to_epoch':target,
            'continuation_waits_for':str(args.wait_for_queue.resolve()) if args.wait_for_queue else None,
            'note':'Fresh model; exact ordered source buffers and source training recipe; persistent optimizer.'}
    fixed(path, plan)
    return plan


def continuation_command(directory, plan, parallelism=None):
    """Persist execution-only overrides separately from the frozen reconstruction recipe."""
    path = directory/'continuation-settings.json'
    settings = read(path) if path.exists() else {}
    if parallelism is not None:
        require(parallelism > 0, 'Continuation parallelism must be positive')
        requested = {'schema_version':1, 'games_parallelism':parallelism}
        fixed(path, requested)
        settings = requested
    argv = list(plan['command'])
    argv[argv.index('--epochs')+1] = str(plan['continue_to_epoch'])
    if settings:
        require(settings.get('schema_version') == 1 and isinstance(settings.get('games_parallelism'), int)
                and settings['games_parallelism'] > 0, 'Invalid continuation settings')
        argv[argv.index('--games-parallelism')+1] = str(settings['games_parallelism'])
    return argv


class Worker:
    def __init__(self, root, binary, args):
        self.root, self.binary = root.resolve(),binary.resolve()
        self.repo=Path(__file__).resolve().parents[1]
        self.name = args.job_id
        self.directory=self.root/self.name
        self.directory.mkdir(exist_ok=True)
        self.state=read(self.root/'status.json')
        self.state.update(pid=os.getpid(), started_at=now())
        self.plan = reconstruction_plan(self.directory, self.binary, args)
        self.continuation = continuation_command(self.directory, self.plan, args.continuation_parallelism)
        old = read(self.root/'jobs'/(self.name+'.json')) if (self.root/'jobs'/(self.name+'.json')).exists() else {}
        self.state['jobs'][self.name]={'stage':'running','started_at':old.get('started_at') or now(),
                                     'note':'Reconstructing saved epoch buffers'}
        self.telemetry=(self.directory/'resources.jsonl').open('a',buffering=1)
        self.env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKIO_WORKER_THREADS':'16','ALZ_LOG_ANSI':'never'}

    def status(self,stage,note,sample=None,pid=None):
        job=self.state['jobs'][self.name]
        job.update(stage=stage,note=note,updated_at=now())
        if stage in ('completed', 'paused'):
            job['ended_at']=now()
        if pid is not None:job['pid']=pid
        self.state.update(stage=stage,active=[self.name] if stage=='running' else [],updated_at=now())
        if sample:self.state['resources']=sample
        write(self.root/'jobs'/(self.name+'.json'),job)
        write(self.root/'status.json',self.state)

    def execute(self,argv,note,limit):
        sample=resource_sample()
        require(sample['working_bytes'] < sample['memory_limit']*limit,'Insufficient memory headroom to start')
        with (self.directory/'stdout.log').open('a') as log:
            child=subprocess.Popen(argv,cwd=self.repo,env=self.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    sample=resource_sample()
                    self.telemetry.write(json.dumps(sample)+'\n')
                    if sample['working_bytes'] > sample['memory_limit']*limit or sample['gpu_memory_mib']>22000:
                        raise RuntimeError(f"Memory guard: host {sample['working_bytes']/1e9:.2f} GB / "
                                           f"{sample['memory_limit']/1e9:.2f} GB (limit {limit:.0%}); "
                                           f"GPU {sample['gpu_memory_mib']} MiB / 22000 MiB")
                    self.status('running',note,sample,child.pid)
                    time.sleep(2)
                require(child.returncode==0,f'Trainer exited {child.returncode}; see stdout.log')
            finally:
                stop(child)

    def run(self):
        ready=read(self.directory/'cuda-ready.json')
        require(ready['passed'] and ready['binary_sha256']==digest(self.binary),'Missing matching CUDA correctness receipt')
        source=Path(self.plan['source'])
        history=self.plan['history_epochs']
        argv=list(self.plan['command'])
        require(Path(argv[1]).resolve() == self.binary, 'Saved command references another binary')
        require(argv[argv.index('--checkpoint-dir')+1] == str(self.directory/'checkpoints'), 'Saved command references another run')
        self.execute(argv,f'Reconstructing epochs 1–{history} using original saved buffers; no new games generated',.84)
        for epoch in range(history):
            name=f'{epoch:08}.json'
            new=read(self.directory/'stats'/name)
            old=read(source/'stats'/name)
            require(new['training']['samples']==old['training']['samples'],'Sample budget changed')
            require(new['scheduled_learning_rate']==old['scheduled_learning_rate'],'Learning-rate schedule changed')
            require(new['history_replay_sha256']==digest(source/'checkpoints'/f'{epoch:08}'/'replay.bin.zst'),'Source replay changed')
            require(digest(self.directory/'checkpoints'/f'{epoch:08}'/'replay.bin.zst')==new['history_replay_sha256'],'Saved buffer differs from source')
        write(self.directory/'reconstruction-complete.json',{'completed_at':now(),'epochs':history,'verified_exact_buffers':True})
        dependency=self.plan.get('continuation_waits_for')
        if dependency:
            comparison=Path(dependency)
            if not comparison.is_absolute():
                comparison=self.root.parent/comparison
            while locked(comparison/'.lock'):
                self.status('running',f'Reconstruction complete; waiting for comparison before epoch {history+1}')
                time.sleep(5)
            require(read(comparison/'status.json')['stage']=='complete','Comparison stopped unexpectedly; self-play remains held')
        target=self.plan['continue_to_epoch']
        argv=list(self.continuation)
        write(self.directory/'continuation-command.json',argv)
        parallelism=argv[argv.index('--games-parallelism')+1]
        self.execute(argv,f'Self-play P{parallelism}; epochs 1–{history} reconstructed, new games from epoch {history+1}',.93)
        self.status('completed',f'Completed {target} epochs including {history} reconstructed epochs')



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--source-run', type=Path)
    p.add_argument('--history-epochs', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--architecture')
    p.add_argument('--parallelism', type=int)
    p.add_argument('--continuation-parallelism', type=int,
                   help='Persist a concurrency override for subsequent self-play, preserving the reconstruction plan')
    p.add_argument('--wait-for-queue', type=Path)
    args=p.parse_args()
    require(re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', args.job_id), 'Invalid job identifier')
    with (args.run_dir/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker=Worker(args.run_dir,args.binary,args)
        def interrupt(signum,frame):raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,interrupt)
        try:worker.run()
        except BaseException as error:
            worker.status('paused',str(error))
            raise
        finally:worker.telemetry.close()


if __name__=='__main__':main()
