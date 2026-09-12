"""Self-play command construction and process/resource helpers."""
import os
from pathlib import Path
import signal
import subprocess
import time

from experiment_io import write


def command(root, top_p, epochs=1, batch=256, adam_backend="fused", parallelism=500):
    return ['./run.sh', 'target/release/alz', 'train', '--device', 'cuda',
            '--architecture', 'kata-gelu-value64x2-v1', '--bn-gamma-one',
            '--epochs', str(epochs), '--games-per-epoch', '1000', '--simulations', '3000',
            '--games-parallelism', str(parallelism), '--inference-batch-size', str(batch),
            '--batch-timeout-us', '1000', '--training-batch-size', '256',
            '--top-p', str(top_p), '--replay-positions', '62500',
            '--replay-position-growth', '3750', '--replay-growth-start-epoch', '15',
            '--learning-rate', '0.001', '--replay-lr-exponent', '1.1',
            '--weight-decay', '0.0001', '--adam-backend', adam_backend,
            '--replay-cache', 'cpu', '--prefetch-batches', '2', '--seed', '20260909',
            '--rendered-games', '2', '--heartbeat-seconds', '15', '--progress-every-games', '10',
            '--checkpoint-dir', str(root/'checkpoints'), '--stats-dir', str(root/'stats'),
            '--games-dir', str(root/'games')]


def resource_sample():
    cg = Path('/sys/fs/cgroup/memory')
    stats = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
    usage = int((cg/'memory.usage_in_bytes').read_text())
    working = usage - int(stats.get('total_inactive_file', stats.get('inactive_file', 0)))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    return {'time': time.time(), 'working_bytes': working,
            'usage_bytes': usage, 'rss_bytes': int(stats.get('total_rss', stats.get('rss', 0))),
            'cache_bytes': int(stats.get('total_cache', stats.get('cache', 0))),
            'active_file_bytes': int(stats.get('total_active_file', stats.get('active_file', 0))),
            'oom_kills': int(dict(line.split() for line in (cg/'memory.oom_control').read_text().splitlines()).get('oom_kill', 0)), 'memory_limit': int((cg/'memory.limit_in_bytes').read_text()),
            'gpu_memory_mib': int(gpu[0]), 'gpu_utilization': int(gpu[1])}


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

