#!/usr/bin/env python3
"""Run a disposable two-epoch copy alongside the capacity depth trainer."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def epochs(directory, log):
    starts = {}
    for line in log.read_text().splitlines():
        if 'starting fixed replay training pass' in line:
            starts[int(re.search(r'epoch=(\d+)', line)[1])] = dt.datetime.fromisoformat(line.split()[0]).timestamp()
    result = []
    for path in sorted((directory / 'epochs').glob('*.json')):
        row = read(path)
        row['start'] = starts[row['epoch']]
        row['training_end'] = row['start'] + row['training']['duration_seconds']
        result.append(row)
    return result


def rate(rows):
    return sum(r['training']['samples'] for r in rows) / sum(r['training']['duration_seconds'] for r in rows)


def summarize(root, capacity):
    probe = epochs(root / 'probe', root / 'probe.log')
    main = epochs(capacity / 'capacity-depth16', capacity / 'train-capacity-depth16.log')
    launch = read(root / 'launch.json')
    baseline = [r for r in main if r['training_end'] < launch['unix_time']]
    # Only complete main epochs wholly inside the probe's training window.
    # Boundary epochs include solo time/cache preparation and are excluded.
    overlap = [r for r in main if r['start'] >= probe[0]['start'] and r['training_end'] <= probe[-1]['training_end']]
    result = {'recorded_at': now(), 'baseline_epochs': baseline, 'probe_epochs': probe,
              'fully_overlapping_main_epochs': overlap, 'baseline_samples_per_second': rate(baseline),
              'probe_samples_per_second': rate(probe),
              'probe_process_seconds': read(root / 'exit.json')['unix_time'] - launch['unix_time'],
              'method': 'Training-only rates from native epoch metrics. Sum concurrent worker rates; exclude partially overlapping main epochs. This is a smoke estimate, not an exact same-window sample counter.'}
    if overlap:
        result['concurrent_main_samples_per_second'] = rate(overlap)
        result['aggregate_samples_per_second'] = rate(probe) + rate(overlap)
        result['aggregate_speedup'] = result['aggregate_samples_per_second'] / rate(baseline)
    write(root / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--capacity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, capacity, repo = args.output.resolve(), args.capacity.resolve(), args.repo.resolve()
    root.mkdir(parents=True, exist_ok=False)
    original = capacity / 'capacity-depth16'
    main_log = capacity / 'train-capacity-depth16.log'
    config = read(original / 'replay-config.json')
    assert config['model']['architecture'] == 'kata_gelu_b16c32_value64x2_v1'
    assert read(capacity / 'backend.json')['performance'] == {'adam_backend': 'standard', 'replay_cache': 'device', 'prefetch_batches': 0}
    source = read(original / 'dataset.json')
    command = [str(repo / 'run.sh'), str(repo / 'target/release/alz'), 'train-replay']
    for item in source['sources']:
        command += ['--replay-checkpoint-dir', item['checkpoint']['path']]
    command += ['--run-dir', str(root / 'probe'), '--architecture', 'kata-gelu-b16c32-value64x2-v1',
                '--device', 'cuda', '--epochs', '2', '--training-batch-size', str(config['batch_size']),
                '--learning-rate', str(config['learning_rate']), '--weight-decay', str(config['weight_decay']),
                '--validation-fraction', str(config['validation_fraction']), '--seed', str(config['seed']),
                '--adam-backend', 'standard', '--replay-cache', 'device', '--prefetch-batches', '0']
    write(root / 'plan.json', {'created_at': now(), 'command': command, 'main_config': config,
          'binary_sha256': hashlib.sha256((repo / 'target/release/alz').read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'baseline_completed_epochs_required': 2, 'probe_epochs': 2,
          'production_queue_unchanged': True, 'mps_configuration_unchanged': True})
    write(root / 'status.json', {'stage': 'waiting_for_two_baseline_epochs', 'updated_at': now()})
    deadline = time.monotonic() + 1800
    while len(epochs(original, main_log)) < 2:
        assert read(capacity / 'status.json')['stage'] == 'train-capacity-depth16', 'Main trainer changed stage'
        assert time.monotonic() < deadline, 'Baseline wait timed out'
        time.sleep(5)
    assert read(capacity / 'status.json')['stage'] == 'train-capacity-depth16'
    (root / 'baseline-main.log').write_text(main_log.read_text())
    write(root / 'baseline.json', epochs(original, main_log))
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    assert float(memory.splitlines()[0]) > 18000, 'Insufficient GPU headroom for second cached trainer'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKIO_WORKER_THREADS='16')
    process = None
    try:
        with (root / 'probe.log').open('w') as log, (root / 'telemetry.jsonl').open('w') as telemetry:
            write(root / 'launch.json', {'started_at': now(), 'unix_time': time.time()})
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=log, start_new_session=True)
            write(root / 'status.json', {'stage': 'running', 'updated_at': now(), 'probe_pid': process.pid})
            deadline = time.monotonic() + 3600
            while process.poll() is None:
                assert read(capacity / 'status.json')['stage'] == 'train-capacity-depth16', 'Main trainer changed stage'
                assert time.monotonic() < deadline, 'Probe exceeded one-hour safety limit'
                gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip()
                telemetry.write(json.dumps({'time': now(), 'unix_time': time.time(), 'gpu': gpu}) + '\n')
                telemetry.flush()
                time.sleep(5)
            write(root / 'exit.json', {'finished_at': now(), 'unix_time': time.time(), 'returncode': process.returncode})
            assert process.returncode == 0, 'Probe training failed; inspect probe.log'
        assert read(root / 'probe/replay-config.json') == config, 'Probe settings differ from baseline'
        assert read(root / 'probe/result.json')['completed_epochs'] == 2
        summarize(root, capacity)
        write(root / 'status.json', {'stage': 'complete', 'updated_at': now()})
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        write(root / 'status.json', {'stage': 'failed', 'updated_at': now(), 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
