#!/usr/bin/env python3
"""Train a 16-block trunk with three original global blocks; reuse two opponents."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
from pathlib import Path
import time

from experiment_control import read_control
from archive.run_capacity_battles_overlap import BattleJob, POLICY, can_start, progress
from archive.run_checkpoint_comparisons import matched_config, select_checkpoints
from archive.run_pooling_followup import PoolingQueue
from experiment_io import digest, fixed, read, require, write


MODEL = {'id': 'depth16-global3', 'architecture': 'kata-gelu-b16c32g3-value64x2-v1',
         'title': 'Train 16-block trunk · three global-pooling blocks',
         'blocks': 16, 'channels': 32, 'global_block_indices': [3, 7, 11]}
MATCHES = [
    {'id': 'global3-baseline-vs-depth16', 'title': 'Current trunk vs 16 blocks · three global pools',
     'first': 'baseline', 'second': MODEL['id'], 'seed_offset': 30},
    {'id': 'global3-depth16-two-vs-three', 'title': '16-block trunk · two vs three global pools',
     'first': 'capacity-depth16', 'second': MODEL['id'], 'seed_offset': 31},
]


def schedule():
    return {'schema_version': 1, 'models': [MODEL], 'matches': MATCHES, 'epochs': 20,
            'settings': {'games': 1000, 'simulations': 4000, 'parallelism': 300,
                         'inference_batch_size': 64, 'temperature': 0.7},
            'overlap_policy': POLICY,
            'note': 'Fresh GELU, 16 blocks × 32 channels, original pooling at blocks 4/8/12, deep value head. Same frozen replays and 20 passes; minimum combined validation loss selects the checkpoint. Both opponents are reused.'}


class ThirdGlobalQueue(PoolingQueue):
    def prepare(self):
        fixed(self.root / 'schedule.json', schedule())
        for key, value in schedule()['settings'].items():
            if key != 'temperature': setattr(self.args, key, value)
        predecessor = self.args.predecessor
        require(read(predecessor / 'status.json')['stage'] == 'complete'
                and (predecessor / 'exit-code.txt').read_text().strip() == '0',
                'Capacity experiment must finish successfully')
        previous = read(predecessor / 'selection.json')
        sources = {}
        for name, architecture in [('baseline', 'kata_gelu_value64x2_v1'),
                                   ('capacity-depth16', 'kata_gelu_b16c32_value64x2_v1')]:
            selected = select_checkpoints(Path(previous[name]['directory']), architecture)
            require(selected == previous[name], 'Previous selection or checkpoint changed')
            sources[name] = selected
        common = matched_config(sources['baseline'])
        require(common == matched_config(sources['capacity-depth16']), 'Opponents used different training settings')
        require(common['adam_backend'] == 'standard', 'Expected standard Adam')
        dataset_path = Path(sources['baseline']['directory']) / 'dataset.json'
        dataset = read(dataset_path)
        require(dataset['dataset_sha256'] == common['dataset_sha256'], 'Dataset mismatch')
        replays = []
        for source in dataset['sources']:
            path = Path(source['checkpoint']['path'])
            require(digest(path / 'replay.bin.zst') == source['replay_sha256'], 'Replay source changed')
            replays.extend(['--replay-checkpoint-dir', str(path)])
        ready = read(self.repo / 'global3-build-ready.json')
        require(digest(self.binary) == ready['binary_sha256'], 'Validated binary changed')
        fixed(self.root / 'plan.json', {
            'schedule': schedule(), 'sources': sources, 'common_training': common,
            'dataset_file_sha256': digest(dataset_path), 'build': ready,
            'selection_rule': 'minimum validation policy_loss + value_loss; ties prefer earlier pass',
            'code_sha256': {str(p.relative_to(self.repo)): digest(p) for p in
                           sorted((self.repo / 'scripts').glob('*.py'))},
            'runtime_sha256': {name: digest(self.repo / name) for name in
                               ['run.sh', 'scripts/run_cuda_tests_runpod.sh', 'scripts/archive/launch_third_global_runpod.sh']}})
        return sources, common, replays

    def battles(self, selections, common):
        names = [m['id'] for m in MATCHES]
        finished, active, counts = {}, {}, {}
        summary = {'plan_sha256': digest(self.root / 'plan.json'), 'matches': finished}
        failure = None
        with ThreadPoolExecutor(max_workers=POLICY['max_active_battles']) as executor:
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
                    if name in finished or name in active: continue
                    if failure is not None or read_control(self.root)['paused']: break
                    if not can_start(index, names, finished, active, counts, self.args.games): break
                    job = BattleJob(self.args)
                    job.repo, job.name = self.repo, name
                    log = self.root / (name + '.log')
                    job.offset = log.stat().st_size if log.exists() else 0
                    job.status('starting')
                    active[name] = (executor.submit(job.run_match, match, selections,
                                                   common['seed'] + match['seed_offset']), job)
                self.status('global3_battles', active_battles=list(active), completed_games=counts)
                if len(finished) < len(names): time.sleep(POLICY['poll_seconds'])

    def run(self):
        selections, common, replays = self.prepare()
        self.configure_backend({
            'architecture_command': ['bash', str(self.repo / 'scripts/run_cuda_tests_runpod.sh'), 'capacity'],
            'cache_command': ['bash', str(self.repo / 'scripts/run_cuda_tests_runpod.sh'), 'cache'],
        }, {'adam_backend': 'standard', 'replay_cache': 'device', 'prefetch_batches': 0})
        self.preflight(MODEL['id'], MODEL['architecture'], common, replays)
        directory = self.root / MODEL['id']
        if not (directory / 'result.json').exists():
            self.command('train-' + MODEL['id'], ['train-replay', *replays,
                '--run-dir', directory, '--architecture', MODEL['architecture'], '--device', self.args.device,
                '--epochs', 20, '--training-batch-size', common['batch_size'],
                '--learning-rate', common['learning_rate'], '--weight-decay', common['weight_decay'],
                '--validation-fraction', common['validation_fraction'], '--seed', common['seed']])
        selected = select_checkpoints(directory, MODEL['architecture'].replace('-', '_'))
        require(matched_config(selected) == common, 'New model used different training settings')
        selections[MODEL['id']] = selected
        fixed(self.root / 'selection.json', selections)
        self.battles(selections, common)
        self.status('complete', summary=str(self.root / 'summary.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for directory in [args.predecessor.parent, args.predecessor, args.output_dir]:
            lock = stack.enter_context((directory / '.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = ThirdGlobalQueue(args)
        try:
            if args.prepare_only: queue.prepare()
            else: queue.run()
        except Exception as error:
            queue.status('failed', failed_stage=queue.stage, error=str(error))
            raise


if __name__ == '__main__':
    main()
