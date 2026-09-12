#!/usr/bin/env python3
"""Apply an audited match plan around the original, immutable training worker."""

# Allow direct execution while sharing the repository tooling modules.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace

from match_plan import match_settings, read_match_plan
from experiment_io import digest, fixed, read, require
from archive.run_replay_followups import Queue


class PlannedQueue(Queue):
    def __init__(self, args):
        super().__init__(args)
        self.match_plan = read_match_plan(self.root)
        require(self.match_plan is not None, 'Missing approved match plan')

    def command(self, name, arguments):
        if arguments[0] == 'train-replay':
            directory = Path(arguments[arguments.index('--run-dir') + 1])
            if (directory / 'result.json').exists():
                # Queue.run validates the completed budget, dataset and checkpoint
                # immediately afterwards. Preserve its weights, logs and timestamps.
                return
        return super().command(name, arguments)

    def battle(self, name, first, second, seed):
        previous = self.args.games, self.args.parallelism
        settings = match_settings(self.match_plan, name, *previous)
        fixed(self.root / (name + '.settings.json'), settings)
        self.args.games, self.args.parallelism = settings['games'], settings['parallelism']
        try:
            return super().battle(name, first, second, seed)
        finally:
            self.args.games, self.args.parallelism = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--training-binary', type=Path)
    cli = parser.parse_args()
    root = cli.output_dir.resolve()
    config = read(root / 'queue-config.json')
    require(digest(cli.binary) == config['binary_sha256'], 'Original match executable changed')
    training = cli.training_binary or cli.binary
    require(digest(training) == config.get('training_binary_sha256', config['binary_sha256']),
            'Original training executable changed')
    script = Path(__file__).resolve()
    fixed(root / 'match-plan-worker.json', {
        'original_queue_config_sha256': digest(root / 'queue-config.json'),
        'match_plan': read_match_plan(root),
        'code_sha256': {p.name: digest(p) for p in [script, script.parent.parent/'match_plan.py']},
    })
    performance = config.get('training_performance',
                            {'adam_backend': 'standard', 'replay_cache': 'none', 'prefetch_batches': 0})
    args = SimpleNamespace(output_dir=root, binary=cli.binary.resolve(), training_binary=training.resolve(),
        latest_checkpoint=Path(config['latest_checkpoint']['path']),
        activation_dir=Path(config['activation_dir']),
        require_success_file=Path(config['require_success_file']) if config['require_success_file'] else None,
        poll_seconds=30, **performance,
        **{key: config[key] for key in ('device', 'games', 'simulations', 'parallelism', 'inference_batch_size')})
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = PlannedQueue(args)
        try:
            queue.run()
        except Exception as error:
            queue.status('failed', failed_stage=queue.stage, error=str(error))
            raise


if __name__ == '__main__':
    main()
