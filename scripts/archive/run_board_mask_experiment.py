#!/usr/bin/env python3
"""One guarded replay trainer, then one guarded P100 board-mask comparison."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from archive.run_checkpoint_comparisons import select_checkpoints
from experiment_io import checkpoint, digest, fixed, read, require, validate_battle, write

from self_play_runtime import resource_sample, stop

ARCHITECTURE = 'kata-gelu-boardmask-value64x2-v1'
TRAIN_ID = 'boardmask-training'
BATTLE_ID = 'boardmask-vs-current'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def memory_reason(sample):
    if sample['working_bytes'] > sample['memory_limit'] * .88:
        return 'Host working memory exceeded 88% of the pod limit'
    if sample['gpu_memory_mib'] > 22000:
        return 'GPU memory exceeded 22,000 MiB'
    return None


class CapacityPause(RuntimeError):
    pass


class Experiment:
    def __init__(self, root, binary):
        self.root, self.binary = root.resolve(), binary.resolve()
        self.repo = Path(__file__).resolve().parents[2]
        self.env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
                    'TOKIO_WORKER_THREADS':'8', 'ALZ_LOG_ANSI':'never'}
        self.started = now()
        self.stage = 'preflight'
        self.current_job = TRAIN_ID
        self.telemetry = (self.root/'resources.jsonl').open('a', buffering=1)

    def status(self, stage, **extra):
        self.stage = stage
        data = {'stage':stage, 'job':self.current_job, 'pid':os.getpid(),
                'started_at':self.started, 'updated_at':now(), **extra}
        write(self.root/'status.json', data)
        print(json.dumps(data), flush=True)

    def command(self, stage, argv, log_name):
        reason = memory_reason(resource_sample())
        if reason:
            raise CapacityPause(reason + '; auxiliary job was not started')
        self.status(stage, command=argv)
        with (self.root/log_name).open('a') as log:
            process = subprocess.Popen(argv, cwd=self.repo, env=self.env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                self.status(stage, child_pid=process.pid, command=argv)
                while process.poll() is None:
                    sample = resource_sample()
                    self.telemetry.write(json.dumps({'stage':stage, **sample})+'\n')
                    reason = memory_reason(sample)
                    if reason:
                        raise CapacityPause(reason + '; stopped only this auxiliary process group')
                    time.sleep(1)
                require(process.returncode == 0, f'{stage} exited {process.returncode}; see {log_name}')
            finally:
                stop(process)

    def run(self):
        root = self.root
        source = read(root/'baseline/dataset.json')
        baseline_config = read(root/'baseline/replay-config.json')
        old_plan = read(self.repo/'runs/recipe-battles/recipe-battle-plan.json')
        baseline = old_plan['models']['recipe-gamma-one-s1']['selected']
        require(checkpoint(baseline['path']) == baseline, 'Baseline checkpoint changed')
        require(old_plan['models']['recipe-gamma-one-s1']['config'] == baseline_config, 'Baseline recipe mismatch')
        replays = []
        for item in source['sources']:
            p = root/'replays'/f"{item['checkpoint']['epoch']:08}"
            require(digest(p/'replay.bin.zst') == item['replay_sha256'], 'Replay checksum mismatch')
            replays.extend(['--replay-checkpoint-dir',str(p)])
        train = [str(self.repo/'run.sh'),str(self.binary),'train-replay',*replays,
            '--run-dir',str(root/'model'),'--architecture',ARCHITECTURE,
            '--device','cuda','--epochs','20','--training-batch-size','256',
            '--learning-rate','0.001','--weight-decay','0.0001','--validation-fraction','0.1',
            '--seed','20260908','--split-seed','20260906','--bn-gamma-one',
            '--adam-backend','standard','--replay-cache','device']
        fixed(root/'plan.json', {'schema_version':1,'architecture':ARCHITECTURE,'epochs':20,
            'baseline':baseline,'baseline_config':baseline_config,'dataset_sha256':source['dataset_sha256'],
            'binary_sha256':digest(self.binary),'training_command':train,
            'selection_rule':'minimum validation policy_loss + value_loss; ties prefer earlier pass',
            'settings':{'games':1000,'simulations':4000,'temperature':0.7,'parallelism':100,'inference_batch_size':64},
            'memory_guard':{'working_fraction':.88,'gpu_memory_mib':22000,'action':'stop auxiliary job only; no automatic retry'}})
        ready = read(root/'cuda-ready.json')
        require(ready['passed'] and ready['binary_sha256'] == digest(self.binary), 'Missing matching CUDA correctness receipt')
        if not (root/'model/initial-validation.json').exists():
            self.command('initializing',train+['--initialize-only'],'training.log')
        actual = read(root/'model/replay-config.json')
        require(actual == {**baseline_config,'model':{'architecture':ARCHITECTURE.replace('-','_')}}, 'Training recipe changed')
        dataset = read(root/'model/dataset.json')
        for key in ('dataset_sha256','training_games','training_positions','validation_games','validation_positions','augmentation_count'):
            require(dataset[key] == source[key], 'Replay split mismatch: '+key)
        old = read(root/'baseline/initial-tensors.json')
        new = read(root/'model/initial-tensors.json')
        require(new['bn_gamma_names'] == old['bn_gamma_names'], 'BatchNorm initialization names changed')
        require(set(new['sha256']) == set(old['sha256']) | {'board_mask_conv.weight'}, 'Unexpected additional tensors')
        require(all(new['sha256'][k] == v for k,v in old['sha256'].items()), 'Shared initial tensors differ from baseline')
        write(root/'pairing-verified.json',{'baseline':'recipe-gamma-one-s1','dataset_sha256':source['dataset_sha256'],
            'shared_initial_tensors_identical':True,'additional_tensor':'board_mask_conv.weight'})
        if not (root/'model/result.json').exists():
            self.command('training',train,'training.log')
        selected = select_checkpoints(root/'model',ARCHITECTURE.replace('-','_'))
        write(root/'selection.json',selected)
        self.current_job = BATTLE_ID
        output = root/(BATTLE_ID+'.json')
        temporary = root/(BATTLE_ID+'.partial.json')
        if not output.exists():
            battle = [str(self.repo/'run.sh'),str(self.binary),'battle',
                '--first-checkpoint-dir',baseline['path'],'--second-checkpoint-dir',selected['selected']['path'],
                '--device','cuda','--games','1000','--simulations','4000','--temperature','0.7',
                '--games-parallelism','100','--inference-batch-size','64','--batch-timeout-us','1000',
                '--seed','20260910','--heartbeat-seconds','30','--no-move-logs','--output',str(temporary)]
            fixed(root/'battle-command.json',battle)
            self.command('battle',battle,'battle.log')
            report = read(temporary)
            validate_battle(report,baseline,selected['selected'],1000,4000,.7)
            require(report['config']['games_parallelism'] == 100 and report['config']['inference_batch_size'] == 64,
                    'Battle capacity settings changed')
            temporary.replace(output)
        report = read(output)
        validate_battle(report,baseline,selected['selected'],1000,4000,.7)
        self.status('complete',ended_at=now())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    args=parser.parse_args()
    with (args.run_dir/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        experiment=Experiment(args.run_dir,args.binary)
        def interrupt(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,interrupt)
        try:
            experiment.run()
        except CapacityPause as error:
            experiment.status('paused',reason=str(error),ended_at=now())
        except BaseException as error:
            experiment.status('failed',error=str(error),ended_at=now())
            raise
        finally:
            experiment.telemetry.close()


if __name__ == '__main__':
    main()
