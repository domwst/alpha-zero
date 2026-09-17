"""Best-effort observations, kept separate from declared admission reservations."""

from pathlib import Path
import shutil
import subprocess


def cgroup_memory(current, maximum, stats):
    """Separate conventional working-set telemetry from the guard's cache estimate.

    Keep shared, mapped, dirty, writeback and unevictable pages charged. These
    categories can overlap; subtracting their sum from cache is conservative.
    The RSS reservation remains an independent per-worker limit.
    """

    def count(name):
        return stats.get("total_" + name, stats.get(name, 0))

    inactive = count("inactive_file")
    cache = stats.get("file", count("cache"))
    protected = count("shmem") + stats.get("file_mapped", count("mapped_file"))
    protected += stats.get("file_dirty", count("dirty"))
    protected += stats.get("file_writeback", count("writeback")) + count("unevictable")
    clean_unmapped = max(0, cache - protected)
    return {
        "used_bytes": max(0, current - inactive),
        "raw_used_bytes": current,
        "reclaimable_bytes": inactive,
        "clean_unmapped_file_bytes": clean_unmapped,
        "pressure_bytes": max(0, current - clean_unmapped),
        "pressure_basis": "usage_minus_clean_unmapped_file_cache",
        "limit_bytes": maximum,
        "source": "cgroup_working_set",
    }


def memory_pressure():
    for directory, usage, limit in [
        (Path("/sys/fs/cgroup"), "memory.current", "memory.max"),
        (
            Path("/sys/fs/cgroup/memory"),
            "memory.usage_in_bytes",
            "memory.limit_in_bytes",
        ),
    ]:
        try:
            current, maximum = (
                int((directory / usage).read_text()),
                int((directory / limit).read_text()),
            )
            if 0 < maximum < 2**60:
                stats = {
                    line.split()[0]: int(line.split()[1])
                    for line in (directory / "memory.stat").read_text().splitlines()
                }
                return cgroup_memory(current, maximum, stats)
        except (OSError, ValueError):
            pass
    try:
        info = {
            k: int(v.split()[0]) * 1024
            for k, v in (
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
        }
        return {
            "used_bytes": info["MemTotal"] - info["MemAvailable"],
            "limit_bytes": info["MemTotal"],
            "source": "host",
        }
    except (OSError, ValueError, KeyError):
        return None


def gpu_memory(pid):
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        matches = []
        for line in result.stdout.splitlines():
            process, memory = line.split(",")
            if int(process.strip()) == pid:
                matches.append(int(memory.strip()) * 1024 * 1024)
        return sum(matches) if matches else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
