#!/usr/bin/env python3
"""Bounded production-depth single/dual self-play probe; never uses production outputs."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from self_play_runtime import command, resource_sample, stop, write








def progress(path):
    found = re.findall(r'completed_evaluations=(\d+).*?elapsed_seconds=([\d.]+)', path.read_text(errors='replace'))
    return [{'evaluations': int(n), 'elapsed': float(t)} for n, t in found]




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=150)
    args = parser.parse_args()
    assert 60 <= args.seconds <= 300
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'TOKIO_WORKER_THREADS':'16', 'ALZ_LOG_ANSI':'never'}
    processes, streams, samples = [], [], []
    result = {'kind':'bounded_fresh_self_play_capacity_probe', 'settings':{'simulations':3000,'parallelism':500,'batch':256,'timeout_us':1000}, 'phases':[]}
    try:
        for index, top_p in enumerate([0.95, 1.0]):
            worker = root / f'worker-{index}'
            worker.mkdir()
            log = worker/'stdout.log'
            stream = log.open('w')
            streams.append(stream)
            cmd = command(worker, top_p)
            write(worker/'command.json', cmd)
            process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            processes.append(process)
            started = time.time()
            print(json.dumps({'phase':index+1, 'active':len(processes), 'started':started}), flush=True)
            while time.time()-started < args.seconds:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('A probe worker exited; inspect its log')
                sample = resource_sample()
                sample['active'] = len(processes)
                samples.append(sample)
                write(root/'live.json', sample)
                if sample['working_bytes'] > sample['memory_limit'] * 0.82 or sample['gpu_memory_mib'] > 22500:
                    raise RuntimeError('Probe reached memory headroom limit')
                time.sleep(5)
            ended = time.time()
            result['phases'].append({'active':len(processes), 'started':started,'ended':ended,
                                     'samples': [s for s in samples if s['time'] >= started]})
    except Exception as error:
        result['error'] = str(error)
    finally:
        for process in reversed(processes):
            stop(process)
        for stream in streams:
            stream.close()
        for i in range(len(processes)):
            result[f'worker_{i}_progress'] = progress(root/f'worker-{i}'/'stdout.log')
        write(root/'result.json', result)
    if 'error' in result:
        raise SystemExit(result['error'])


if __name__ == '__main__':
    main()
