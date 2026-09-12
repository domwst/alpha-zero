#!/usr/bin/env python3
"""Summarize an Nsight Systems SQLite capture of this project's training loop.

Read-only analysis, specific to the current FP32 19x19 inputs and unfused Adam.
Batch boundaries are state H2D copies. The Adam tail is identified and validated
from its repeating addcdiv sequence, not guessed from a percentage in nsys stats.
CPU sampling is not needed. All times describe the instrumented capture.
"""
import argparse
import bisect
import collections
import json
import pathlib
import sqlite3
import statistics


def union_ns(intervals):
    total = 0
    previous_end = None
    for start, end in sorted(intervals):
        if previous_end is None or start > previous_end:
            total += end - start
        elif end > previous_end:
            total += end - previous_end
        previous_end = max(previous_end or end, end)
    return total


def summarize(path, batch_size):
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    kernels = [dict(row) for row in conn.execute(
        "SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName=s.id ORDER BY start"
    )]
    copies = [dict(row) for row in conn.execute(
        "SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
    )]
    runtime = [dict(row) for row in conn.execute(
        "SELECT r.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId=s.id ORDER BY start"
    )]
    conn.close()
    if len({row["globalPid"] for row in kernels}) != 1:
        raise ValueError("expected kernels from exactly one process")
    boundaries = [row for row in copies if row["copyKind"] == 1
                  and row["bytes"] == batch_size * 2 * 19 * 19 * 4]
    kernel_starts = [row["start"] for row in kernels]
    copy_starts = [row["start"] for row in copies]
    runtime_starts = [row["start"] for row in runtime]
    by_correlation = collections.defaultdict(list)
    for row in runtime:
        by_correlation[row["correlationId"]].append(row)
    batches = []
    representative_data = []
    for left, right in zip(boundaries, boundaries[1:]):
        a, b = left["start"], right["start"]
        batch_kernels = kernels[bisect.bisect_left(kernel_starts, a):
                                bisect.bisect_left(kernel_starts, b)]
        batch_copies = copies[bisect.bisect_left(copy_starts, a):
                              bisect.bisect_left(copy_starts, b)]
        if not batch_kernels or any(row["end"] > b for row in batch_kernels):
            continue
        ends = [i for i, row in enumerate(batch_kernels)
                if "addcdiv_cuda_kernel" in row["name"]]
        if len(ends) < 2:
            continue
        strides = {j - i for i, j in zip(ends, ends[1:])}
        if len(strides) != 1 or ends[-1] != len(batch_kernels) - 1:
            raise ValueError("unrecognized Adam tail; inspect the trace manually")
        stride = strides.pop()
        # Eight operations for Adam; coupled weight decay adds a ninth.
        if stride not in (8, 9):
            raise ValueError(f"unexpected Adam kernel stride {stride}")
        adam_start = ends[0] - stride + 1
        adam = batch_kernels[adam_start:]
        if adam_start < 0 or len(adam) != len(ends) * stride:
            raise ValueError("incomplete Adam tail")
        d2h = [row for row in batch_copies if row["copyKind"] == 2]
        h2d = [row for row in batch_copies if row["copyKind"] == 1]
        if not d2h:
            continue
        # End of the last scalar readback's synchronization to the next state
        # copy. This includes cleanup and encoding; it is NOT pure encoding time.
        next_copy_api = [r for r in by_correlation[right["correlationId"]]
                         if r["name"].startswith("cudaMemcpyAsync")]
        previous_copy_api = [r for r in by_correlation[d2h[-1]["correlationId"]]
                             if r["name"].startswith("cudaMemcpyAsync")]
        preparation_gap = None
        if next_copy_api and previous_copy_api:
            first = previous_copy_api[0]
            next_start = next_copy_api[0]["start"]
            candidates = runtime[bisect.bisect_left(runtime_starts, first["end"]):
                                 bisect.bisect_left(runtime_starts, next_start)]
            syncs = [r for r in candidates if r["name"].startswith("cudaStreamSynchronize")
                     and r["globalTid"] == first["globalTid"]]
            if syncs:
                preparation_gap = (next_start - syncs[-1]["end"]) / 1e6
        api = runtime[bisect.bisect_left(runtime_starts, a):
                      bisect.bisect_left(runtime_starts, b)]
        batch = {
            "wall_ms": (b - a) / 1e6,
            "kernel_count": len(batch_kernels),
            "kernel_active_ms": union_ns((r["start"], r["end"]) for r in batch_kernels) / 1e6,
            "adam_kernel_count": len(adam),
            "adam_parameter_tensors": len(ends),
            "adam_kernels_per_tensor": stride,
            "adam_span_ms": (adam[-1]["end"] - adam[0]["start"]) / 1e6,
            "adam_kernel_active_ms": union_ns((r["start"], r["end"]) for r in adam) / 1e6,
            "h2d_gpu_ms": sum(r["end"] - r["start"] for r in h2d) / 1e6,
            "d2h_gpu_ms": sum(r["end"] - r["start"] for r in d2h) / 1e6,
            "sync_cpu_ms": sum(r["end"] - r["start"] for r in api
                               if r["name"].startswith("cudaStreamSynchronize")) / 1e6,
            "post_readback_to_next_copy_ms": preparation_gap,
            "kernel_launch_cpu_ms": sum(r["end"] - r["start"] for r in api
                                        if "LaunchKernel" in r["name"]) / 1e6,
        }
        batches.append(batch)
        representative_data.append({"start_ns": a, "end_ns": b,
                                    "adam_start_index": adam_start,
                                    "kernels": batch_kernels, "copies": batch_copies})
    if not batches:
        raise ValueError("no complete training batches detected")
    fields = {}
    for key in batches[0]:
        values = [b[key] for b in batches if b[key] is not None]
        if values:
            fields[key] = {"median": statistics.median(values),
                           "min": min(values), "max": max(values),
                           "mean": statistics.mean(values)}
    median = fields["wall_ms"]["median"]
    index = min(range(len(batches)), key=lambda i: abs(batches[i]["wall_ms"] - median))
    return {"source": str(path), "batch_size": batch_size,
            "complete_batches": len(batches), "statistics": fields,
            "representative_batch": batches[index],
            "timeline": representative_data[index],
            "notes": ["Timings include profiler overhead; use unprofiled benchmarks for throughput.",
                      "Adam span is elapsed time between its first and last kernels, not CPU self-time.",
                      "Preparation gap includes host cleanup and encoding; it is an upper bound on encoding."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = summarize(args.sqlite, args.batch_size)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "timeline"}, indent=2))


if __name__ == "__main__":
    main()
