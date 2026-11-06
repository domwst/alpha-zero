"""Read-only, size-stratified replay storage experiment (Python 3.14 + msgpack).

Writes only the specified result JSON, never modifies the source archive.
Run: uv run --no-project --python 3.14 --with msgpack python \
    scripts/benchmark_replay_storage.py --root runs/dashboard-local/data/jobs \
    --output reports/replay-storage-20260922/results.json
"""

import argparse
from collections import Counter
from compression import zstd
import gzip
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import time

import msgpack


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=64)
    args = parser.parse_args()
    if args.per_stratum < 1:
        parser.error("--per-stratum must be positive")
    root = args.root.resolve()
    seen = set()
    files = []
    for base, dirs, names in os.walk(root, followlinks=False):
        if "archive" not in Path(base).parts:
            continue
        for name in names:
            if not re.fullmatch(r"[0-9a-f]+\.json", name):
                continue
            path = Path(base) / name
            if path.is_symlink():
                continue
            st = path.stat()
            identity = (st.st_dev, st.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            files.append((st.st_size, st.st_blocks * 512, path))
    files.sort()
    if not files:
        parser.error("No replay game JSON files found under --root")
    rng = random.Random(20260922)
    groups = [
        files[len(files) * i // 12 : len(files) * (i + 1) // 12] for i in range(12)
    ]
    results = []
    fields = Counter()
    provenance = Counter()
    total_raw = 0
    print(f"Unique games: {len(files)}; bytes: {sum(f[0] for f in files)}", flush=True)
    for index, group in enumerate(groups):
        sample = rng.sample(group, min(args.per_stratum, len(group)))
        for raw_size, allocated, path in sample:
            raw = path.read_bytes()
            game, json_seconds = timed(json.loads, raw)
            provenance[game.get("provenance", "unknown")] += 1
            total_raw += len(raw)
            compact, compact_seconds = timed(
                lambda: json.dumps(
                    game, separators=(",", ":"), ensure_ascii=False
                ).encode()
            )
            binary, binary_seconds = timed(
                lambda: msgpack.packb(game, use_bin_type=True)
            )
            assert msgpack.unpackb(binary, raw=False) == game
            row = {
                "path": str(path.relative_to(root)),
                "stratum": index,
                "weight": len(group) / len(sample),
                "original": len(raw),
                "parse_json_ms": json_seconds * 1000,
                "formats": {},
            }
            formats = {
                "json": (raw, lambda x: x, lambda x: x, json.loads, 0),
                "compact_json": (
                    compact,
                    lambda x: x,
                    lambda x: x,
                    json.loads,
                    compact_seconds,
                ),
                "gzip6_json": (
                    raw,
                    lambda x: gzip.compress(x, compresslevel=6, mtime=0),
                    gzip.decompress,
                    json.loads,
                    0,
                ),
                "zstd3_json": (
                    raw,
                    lambda x: zstd.compress(x, level=3),
                    zstd.decompress,
                    json.loads,
                    0,
                ),
                "zstd6_json": (
                    raw,
                    lambda x: zstd.compress(x, level=6),
                    zstd.decompress,
                    json.loads,
                    0,
                ),
                "zstd9_json": (
                    raw,
                    lambda x: zstd.compress(x, level=9),
                    zstd.decompress,
                    json.loads,
                    0,
                ),
                "msgpack64": (
                    binary,
                    lambda x: x,
                    lambda x: x,
                    lambda x: msgpack.unpackb(x, raw=False),
                    binary_seconds,
                ),
                "zstd6_msgpack64": (
                    binary,
                    lambda x: zstd.compress(x, level=6),
                    zstd.decompress,
                    lambda x: msgpack.unpackb(x, raw=False),
                    binary_seconds,
                ),
            }
            for name, (
                payload,
                compress,
                decompress,
                decode,
                serialize_seconds,
            ) in formats.items():
                packed, encode_seconds = timed(compress, payload)
                restored, decompress_seconds = timed(decompress, packed)
                assert restored == payload
                decoded, decode_seconds = timed(decode, restored)
                assert decoded == game
                row["formats"][name] = {
                    "bytes": len(packed),
                    "blocks_4k": math.ceil(len(packed) / 4096) * 4096,
                    "encode_ms": (serialize_seconds + encode_seconds) * 1000,
                    "decompress_ms": decompress_seconds * 1000,
                    "decode_ms": (decompress_seconds + decode_seconds) * 1000,
                }
            # Serialized field attribution, excluding punctuation between fields.
            for ply in game["record"]["plies"]:
                for key, value in ply.items():
                    if key == "decision":
                        for k, v in value.items():
                            if k == "diagnostics" and isinstance(v, dict):
                                for dk, dv in v.items():
                                    fields["diagnostics." + dk] += len(
                                        json.dumps(dv, separators=(",", ":"))
                                    )
                            else:
                                fields["decision." + k] += len(
                                    json.dumps(v, separators=(",", ":"))
                                )
                    else:
                        fields[key] += len(json.dumps(value, separators=(",", ":")))
            results.append(row)
        print(f"Stratum {index + 1}/12 complete ({len(results)} games)", flush=True)
    summary = {}
    for name in results[0]["formats"]:
        projected = sum(r["weight"] * r["formats"][name]["bytes"] for r in results)
        variance = 0
        for index, group in enumerate(groups):
            sizes = [
                r["formats"][name]["bytes"] for r in results if r["stratum"] == index
            ]
            if len(sizes) > 1:
                variance += (
                    len(group) ** 2
                    * (1 - len(sizes) / len(group))
                    * statistics.variance(sizes)
                    / len(sizes)
                )
        summary[name] = {
            "projected_payload_gib": projected / 2**30,
            "approx_95pct_margin_gib": 1.96 * math.sqrt(variance) / 2**30,
            "projected_blocks_4k_gib": sum(
                r["weight"] * r["formats"][name]["blocks_4k"] for r in results
            )
            / 2**30,
            "encode_ms_mean": statistics.mean(
                r["formats"][name]["encode_ms"] for r in results
            ),
            "decode_ms_mean": statistics.mean(
                r["formats"][name]["decode_ms"] for r in results
            ),
            "decode_ms_p95": sorted(r["formats"][name]["decode_ms"] for r in results)[
                int(len(results) * 0.95)
            ],
            "decompress_ms_mean": statistics.mean(
                r["formats"][name]["decompress_ms"] for r in results
            ),
        }
    report = {
        "python": platform.python_version(),
        "msgpack": msgpack.__version__,
        "root": str(root),
        "sampling": "12 equal-count size strata, seeded random sample within each; inode deduplicated, directory symlinks not traversed",
        "unique_games": len(files),
        "actual_json_gib": sum(f[0] for f in files) / 2**30,
        "actual_allocated_gib": sum(f[1] for f in files) / 2**30,
        "sample_games": len(results),
        "sample_bytes": total_raw,
        "sample_provenance": dict(provenance),
        "field_serialized_bytes": dict(fields.most_common()),
        "summary": summary,
        "games": results,
        "timing_notes": "In-memory single pass; no cold-disk I/O; encoding excludes JSON parse; msgpack retains binary64 values; byte-for-byte decompression and parsed-object equality checked for every sample/format.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "games"}, indent=2))


if __name__ == "__main__":
    main()
