#!/usr/bin/env python3
"""Six paired normalization/LR runs, two trainers maximum; no battle launch."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from contextlib import ExitStack
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from experiment_control import read_control
from archive.run_checkpoint_comparisons import select_checkpoints
from experiment_io import digest, fixed, read, require, write


ARCHITECTURE = 'kata-gelu-value64x2-v1'
SEEDS = [20260908, 20260909]
SPLIT_SEED = 20260906


def schedule():
    variants = [('constant', 'Constant LR 0.001', []),
                ('cosine', 'Cosine LR 0.001 → 0.0001',
                 ['--lr-schedule', 'cosine', '--final-learning-rate', '0.0001']),
                ('gamma-one', 'BN gamma=1 · constant LR 0.001', ['--bn-gamma-one'])]
    models = [{'id': f'recipe-{variant}-s{index + 1}',
               'title': f'{title} · seed {index + 1}', 'variant': variant,
               'seed': seed, 'architecture': ARCHITECTURE, 'extra_args': extra}
              for index, seed in enumerate(SEEDS) for variant, title, extra in variants]
    return {'schema_version': 1, 'models': models, 'epochs': 20, 'concurrency': 2,
            'split_seed': SPLIT_SEED,
            'selection_rule': 'minimum validation policy_loss + value_loss; ties prefer earlier pass',
            'note': '10 blocks × 32 channels, GELU, original pooling, deep value head; 20 passes. '
                    'Two model/batch-order seeds, one fixed replay split. Paired initial tensors; '
                    'standard Adam and device cache. Battles deferred until migration.'}


def validate_pairing(root, models):
    for seed in SEEDS:
        group = {m['variant']: read(root / m['id'] / 'initial-tensors.json')
                 for m in models if m['seed'] == seed}
        base = group['constant']
        require(base == group['cosine'], 'Constant/cosine initial tensors differ')
        other = group['gamma-one']
        gamma = set(base['bn_gamma_names'])
        require(gamma and gamma == set(other['bn_gamma_names']), 'BatchNorm tensor identities differ')
        require(base['sha256'].keys() == other['sha256'].keys(), 'Initial tensor sets differ')
        changed = {key for key in base['sha256'] if base['sha256'][key] != other['sha256'][key]}
        require(changed == gamma, 'Gamma ablation changed other initial tensors or missed gamma')


def command(repo, binary, root, model, replays):
    return [str(repo / 'run.sh'), str(binary), 'train-replay', *replays,
            '--run-dir', str(root / model['id']), '--architecture', ARCHITECTURE,
            '--device', 'cuda', '--epochs', '20', '--training-batch-size', '256',
            '--learning-rate', '0.001', '--weight-decay', '0.0001',
            '--validation-fraction', '0.1', '--seed', str(model['seed']),
            '--split-seed', str(SPLIT_SEED), '--adam-backend', 'standard',
            '--replay-cache', 'device', *model['extra_args']]


def validate_run_config(root, model, dataset_sha256):
    config = read(root / model['id'] / 'replay-config.json')
    expected = {'schema_version': 1, 'model': {'architecture': ARCHITECTURE.replace('-', '_')},
                'dataset_sha256': dataset_sha256, 'seed': model['seed'], 'split_seed': SPLIT_SEED,
                'batch_size': 256, 'learning_rate': .001, 'weight_decay': .0001,
                'validation_fraction': .1, 'adam_backend': 'standard',
                'lr_schedule': 'cosine' if model['variant'] == 'cosine' else 'constant',
                'final_learning_rate': .0001 if model['variant'] == 'cosine' else None,
                'schedule_epochs': 20 if model['variant'] == 'cosine' else None,
                'bn_gamma_one': model['variant'] == 'gamma-one'}
    require(config == expected, 'Unexpected training recipe configuration: ' + model['id'])


def run(args):
    repo = Path(__file__).resolve().parents[2]
    root, binary = args.output_dir, args.binary
    plan = schedule()
    fixed(root / 'schedule.json', plan)
    models = plan['models']
    dataset = read(args.baseline_dir / 'dataset.json')
    replays = []
    for source in dataset['sources']:
        path = Path(source['checkpoint']['path'])
        require(digest(path / 'replay.bin.zst') == source['replay_sha256'], 'Replay source changed')
        replays.extend(['--replay-checkpoint-dir', str(path)])
    fixed(root / 'plan.json', {'schedule': plan, 'baseline_dir': str(args.baseline_dir),
          'dataset_sha256': dataset['dataset_sha256'], 'dataset_file_sha256': digest(args.baseline_dir / 'dataset.json'),
          'binary_sha256': digest(binary),
          'code_sha256': {str(p.relative_to(repo)): digest(p) for p in
                          [Path(__file__).resolve(), repo / 'src/commands/train_replay.rs', repo / 'src/cli.rs']},
          'commands': {m['id']: command(repo, binary, root, m, replays) for m in models}})

    def status(stage, **extra):
        write(root / 'status.json', {'stage': stage, 'pid': os.getpid(),
              'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra})

    active, completed = {}, {}
    failure = None
    try:
        # Fail closed on CUDA architecture/cache checks before any expensive training.
        if not (root / 'cuda-ready.json').exists():
            status('validating_cuda')
            with (root / 'cuda-checks.log').open('a') as log:
                for check in ['capacity', 'cache']:
                    subprocess.run(['bash', 'scripts/run_cuda_tests_runpod.sh', check], cwd=repo,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            write(root / 'cuda-ready.json', {'binary_sha256': digest(binary)})
        require(read(root / 'cuda-ready.json')['binary_sha256'] == digest(binary), 'CUDA receipt binary changed')
        # Initialize and validate sequentially, before launching either production trainer.
        for model in models:
            directory = root / model['id']
            if not (directory / 'initial-validation.json').exists():
                status('preflight', active=[model['id']])
                with (root / ('preflight-' + model['id'] + '.log')).open('a') as log:
                    subprocess.run(command(repo, binary, root, model, replays) + ['--initialize-only'],
                                   cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            validate_run_config(root, model, dataset['dataset_sha256'])
            split = read(directory / 'dataset.json')
            for key in ['training_games', 'training_positions', 'validation_games', 'validation_positions']:
                require(split[key] == dataset[key], 'Frozen data split changed')
        validate_pairing(root, models)
        write(root / 'pairing-verified.json', {'seeds': SEEDS, 'split_seed': SPLIT_SEED,
              'rule': 'constant and cosine identical; gamma-one differs in exactly all BN gamma tensors'})
        while len(completed) < len(models):
            for name, (process, log, model) in list(active.items()):
                if process.poll() is None: continue
                log.close()
                del active[name]
                if process.returncode:
                    failure = f'{name} exited {process.returncode}; see train-{name}.log'
                else:
                    try:
                        validate_run_config(root, model, dataset['dataset_sha256'])
                        completed[name] = select_checkpoints(root / name, ARCHITECTURE.replace('-', '_'))
                        write(root / 'selection.json', completed)
                    except Exception as error:
                        failure = str(error)
            if failure:
                status('draining_after_failure', active=list(active), completed=list(completed), error=failure)
                if not active: raise RuntimeError(failure)
            else:
                for model in models:
                    name = model['id']
                    if name in completed or name in active: continue
                    if len(active) >= plan['concurrency'] or read_control(root)['paused']: break
                    if (root / name / 'result.json').exists():
                        completed[name] = select_checkpoints(root / name, ARCHITECTURE.replace('-', '_'))
                        write(root / 'selection.json', completed)
                        continue
                    log = (root / ('train-' + name + '.log')).open('a')
                    process = subprocess.Popen(command(repo, binary, root, model, replays),
                                               cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    active[name] = (process, log, model)
                status('training' if active else 'paused', active=list(active), completed=list(completed),
                       concurrency=plan['concurrency'])
            if len(completed) < len(models): time.sleep(5)
        write(root / 'summary.json', {'selection': completed, 'next_action': 'migrate before battles'})
        status('complete', completed=list(completed), next_action='migrate before battles')
    except BaseException as error:
        # On unexpected supervisor failure, stop only its own children, avoiding orphaned extra trainers.
        for process, log, _ in active.values():
            if process.poll() is None: os.killpg(process.pid, signal.SIGTERM)
        for process, log, _ in active.values():
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            log.close()
        status('failed', error=str(error), completed=list(completed))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items(): setattr(args, key, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((args.output_dir / '.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
