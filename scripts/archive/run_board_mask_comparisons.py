#!/usr/bin/env python3
"""Epoch-boundary handoff to two guarded, overlapping P500/B128 comparisons."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from experiment_io import locked, now

from archive.run_board_mask_experiment import memory_reason
from archive.run_capacity_battles_overlap import progress
from experiment_io import checkpoint, digest, fixed, read, require, validate_battle, write

from run_self_play_experiments import process_identity
from self_play_runtime import resource_sample, stop

IDS = ['boardmask-vs-current', 'selfplay-vs-replay-baseline']
SETTINGS = {'games':1000, 'simulations':4000, 'temperature':.7, 'parallelism':500, 'inference_batch_size':128}


def overlap_ready(completed_games, first_complete, serial):
    return first_complete or (not serial and completed_games >= 600)


class Comparisons:
    def __init__(self, root, binary):
        self.root, self.binary = root.resolve(), binary.resolve()
        self.repo = Path(__file__).resolve().parents[2]
        self.selfplay = self.repo/'runs/selfplay-nucleus'
        self.started = now()
        self.active = {}
        self.jobs = {name:{'stage':'pending','attempt':0} for name in IDS}
        self.serial = False
        self.telemetry = (self.root/'resources.jsonl').open('a',buffering=1)
        self.env = {**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                    'TOKIO_WORKER_THREADS':'16','ALZ_LOG_ANSI':'never'}

    def status(self, stage, **extra):
        data = {'stage':stage,'started_at':self.started,'updated_at':now(),
                'pid':os.getpid(),'jobs':self.jobs,'serial_after_memory_pressure':self.serial,**extra}
        write(self.root/'status.json',data)

    def run(self):
        previous = read(self.root.parent/'plan.json')
        selected = read(self.root.parent/'selection.json')['selected']
        baseline = previous['baseline']
        for item in (selected,baseline):
            require(checkpoint(item['path']) == item, 'Selected checkpoint identity changed')
        require(digest(self.binary) == previous['binary_sha256'], 'Validated binary changed')
        request = read(self.root/'boundary-request.json')
        schedule = {'schema_version':1,'settings':SETTINGS,'overlap_at_completed_games':600,
            'binary_sha256':previous['binary_sha256'],'baseline':baseline,'boardmask':selected,
            'selfplay_epoch':request['target_epoch'],'boundary_request_sha256':digest(self.root/'boundary-request.json'),
            'matches':[
                {'id':IDS[0],'title':'Current vs board mask · selected checkpoints','seed':20260910},
                {'id':IDS[1],'title':f"Latest self-play · epoch {request['target_epoch']+1} vs replay baseline",'seed':20260911}],
            'memory_policy':'Stop newest overlap on pressure and retry it after the first match completes; pause on single-match pressure',
            'self_play_after_comparisons':'remains paused until explicitly resumed'}
        fixed(self.root/'schedule.json',schedule)
        self.status('waiting_for_epoch')
        while not (self.root/'boundary-stopped.json').exists():
            require(process_identity(request['supervisor']['pid']) == request['supervisor'] or
                    locked(self.root/'.boundary.lock'), 'Boundary watcher and supervisor are no longer running')
            time.sleep(1)
        receipt = read(self.root/'boundary-stopped.json')
        require(receipt['completed_epoch'] == request['target_epoch'], 'Stopped at an unexpected epoch')
        require(not locked(self.selfplay/'.lock'), 'Self-play supervisor is still active')
        require(process_identity(receipt['worker']['pid']) is None, 'Self-play worker is still active')
        selfplay_path = self.selfplay/'nucleus-p095/checkpoints'/f"{receipt['completed_epoch']:08}"
        stats = read(self.selfplay/'nucleus-p095/stats'/f"{receipt['completed_epoch']:08}.json")
        require(stats['epoch'] == receipt['completed_epoch'], 'Missing complete self-play stats')
        latest = checkpoint(selfplay_path)
        self.models = {IDS[0]:(baseline,selected), IDS[1]:(baseline,latest)}
        fixed(self.root/'resolved-checkpoints.json',{'baseline':baseline,'boardmask':selected,'selfplay':latest,
              'selfplay_stats_sha256':digest(self.selfplay/'nucleus-p095/stats'/f"{receipt['completed_epoch']:08}.json")})
        self.matches = {m['id']:m for m in schedule['matches']}
        for name in IDS:
            output = self.root/(name+'.json')
            if output.exists():
                self.validate(read(output),name)
                self.jobs[name] = {'stage':'completed','attempt':0,'completed_games':1000}
        try:
            while not all(j['stage']=='completed' for j in self.jobs.values()):
                for name,(process,stream,temporary) in list(self.active.items()):
                    if process.poll() is None:
                        self.jobs[name]['completed_games'] = progress(Path(self.jobs[name]['log']),0,1000)
                        continue
                    stream.close()
                    del self.active[name]
                    require(process.returncode == 0, f'{name} exited {process.returncode}')
                    self.validate(read(temporary),name)
                    temporary.replace(self.root/(name+'.json'))
                    self.jobs[name].update(stage='completed',completed_games=1000,ended_at=now())
                sample = resource_sample()
                self.telemetry.write(json.dumps({'active':list(self.active),**sample})+'\n')
                reason = memory_reason(sample)
                if reason and self.active:
                    if len(self.active)==2:
                        self.stop_job(IDS[1])
                        self.jobs[IDS[1]].update(stage='pending',note=reason+'; overlap deferred until first match finishes',completed_games=0)
                        self.serial = True
                        print('Overlap deferred: '+reason,flush=True)
                    else:
                        name = next(iter(self.active))
                        self.stop_job(name)
                        self.jobs[name].update(stage='paused',note=reason,ended_at=now())
                        self.status('paused',reason=reason)
                        return
                if not reason:
                    if self.jobs[IDS[0]]['stage']=='pending':
                        self.start_job(IDS[0])
                    first = self.jobs[IDS[0]]
                    if self.jobs[IDS[1]]['stage']=='pending' and overlap_ready(first.get('completed_games',0),first['stage']=='completed',self.serial):
                        self.start_job(IDS[1])
                self.status('running' if self.active else 'waiting_for_memory')
                time.sleep(1)
            self.status('complete',ended_at=now())
        finally:
            for name in list(self.active):
                self.stop_job(name)

    def validate(self, report, name):
        first,second = self.models[name]
        validate_battle(report,first,second,1000,4000,.7)
        require(report['config']['games_parallelism']==500 and report['config']['inference_batch_size']==128 and
                report['config']['seed']==self.matches[name]['seed'], 'Battle settings changed')

    def start_job(self, name):
        first,second = self.models[name]
        # A fresh attempt gets its own log and partial result: interrupted games
        # are never mixed into a new, balanced 1,000-game comparison.
        attempt = self.jobs[name]['attempt']+1
        while (self.root/f'{name}-attempt{attempt}.log').exists():
            attempt += 1
        log = self.root/f'{name}-attempt{attempt}.log'
        temporary = self.root/f'{name}-attempt{attempt}.partial.json'
        command = [str(self.repo/'run.sh'),str(self.binary),'battle',
            '--first-checkpoint-dir',first['path'],'--second-checkpoint-dir',second['path'],
            '--device','cuda','--games','1000','--simulations','4000','--temperature','0.7',
            '--games-parallelism','500','--inference-batch-size','128','--batch-timeout-us','1000',
            '--seed',str(self.matches[name]['seed']),'--heartbeat-seconds','30','--no-move-logs','--output',str(temporary)]
        fixed(self.root/f'{name}-attempt{attempt}-command.json',command)
        stream = log.open('w')
        try:
            process = subprocess.Popen(command,cwd=self.repo,env=self.env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            stream.close()
            raise
        self.active[name] = process,stream,temporary
        self.jobs[name] = {'stage':'running','pid':process.pid,'attempt':attempt,'log':str(log),
                          'started_at':now(),'completed_games':0}
        print(json.dumps({'started':name,**self.jobs[name]}),flush=True)

    def stop_job(self, name):
        process,stream,_ = self.active.pop(name)
        stop(process)
        stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    args=parser.parse_args()
    with (args.run_dir/'.lock').open('a') as lock, (args.run_dir.parent/'.lock').open('a') as original:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(original,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker=Comparisons(args.run_dir,args.binary)
        def interrupt(signum,frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,interrupt)
        try:
            worker.run()
        except BaseException as error:
            for name in list(worker.active):
                worker.stop_job(name)
                worker.jobs[name].update(stage='failed',note=str(error),ended_at=now())
            worker.status('failed',error=str(error),ended_at=now())
            raise
        finally:
            worker.telemetry.close()


if __name__=='__main__':
    main()
