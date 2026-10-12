"""Controlled follow-up to benchmark_replay_storage.py; never modifies replays.

uv run --no-project --python 3.14 --with msgpack==1.2.2 python \
    scripts/benchmark_replay_floats.py
"""

from collections import defaultdict
from compression import zstd
import copy
import json
from pathlib import Path
import struct

import msgpack


def f32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def main():
    report_dir = Path("reports/replay-storage-20260922")
    prior = json.loads((report_dir / "results.json").read_text())
    # Same size strata, first 16 of the seeded random 64 samples in each.
    counts = defaultdict(int)
    samples = []
    totals = defaultdict(lambda: defaultdict(int))
    floats = defaultdict(lambda: defaultdict(float))
    example = None
    for item in prior["games"]:
        stratum = item["stratum"]
        if counts[stratum] >= 16:
            continue
        counts[stratum] += 1
        path = Path(prior["root"]) / item["path"]
        raw = path.read_bytes()
        game = json.loads(raw)
        provenance = game["provenance"]
        normalized = copy.deepcopy(game)
        for ply in normalized["record"]["plies"]:
            decision = ply["decision"]
            diagnostics = decision.get("diagnostics") or {}
            for container, key in (
                (decision, "training_policy"),
                (diagnostics, "network_prior"),
                (diagnostics, "sampling_policy"),
            ):
                values = container.get(key)
                if values is None:
                    continue
                converted = []
                for value in values:
                    if not isinstance(value, float):
                        converted.append(value)
                        continue
                    rounded = f32(value)
                    converted.append(rounded)
                    if value:
                        stat = floats[provenance]
                        stat["nonzero_values"] += 1
                        stat["changed_values"] += value != rounded
                        stat["max_absolute_change"] = max(
                            stat["max_absolute_change"], abs(value - rounded)
                        )
                        if value != rounded and example is None:
                            example = {
                                "decimal": repr(value),
                                "as_f32": rounded,
                                "f64_hex": struct.pack("!d", value).hex(),
                                "f32_promoted_f64_hex": struct.pack(
                                    "!d", rounded
                                ).hex(),
                                "f32_hex": struct.pack("!f", value).hex(),
                            }
                container[key] = converted
        formats = {
            "original_json": raw,
            "msgpack64_original": msgpack.packb(game, use_bin_type=True),
            # Same wire types and lengths as original MessagePack. Only policy
            # numeric values change to the binary32 grid; isolates low-order bits.
            "msgpack64_f32_policies": msgpack.packb(normalized, use_bin_type=True),
            # Width probe only: rounds ALL floats, including f64 durations.
            # Not a lossless migration candidate for arbitrary JSON.
            "msgpack32_all_floats_probe": msgpack.packb(
                game, use_bin_type=True, use_single_float=True
            ),
        }
        assert msgpack.unpackb(formats["msgpack64_original"], raw=False) == game
        assert (
            msgpack.unpackb(formats["msgpack64_f32_policies"], raw=False) == normalized
        )
        assert len(formats["msgpack64_original"]) == len(
            formats["msgpack64_f32_policies"]
        )
        row = {
            "path": item["path"],
            "provenance": provenance,
            "weight": item["weight"] * 4,
            "formats": {},
        }
        for name, data in formats.items():
            compressed = zstd.compress(data, level=6)
            assert zstd.decompress(compressed) == data
            row["formats"][name] = {
                "bytes": len(data),
                "compressed_bytes": len(compressed),
            }
            totals[provenance][name] += len(compressed)
        samples.append(row)
        if len(samples) % 32 == 0:
            print(f"{len(samples)}/192 samples", flush=True)
    report = {
        "sample_games": len(samples),
        "sample_compressed_bytes_by_provenance": dict(totals),
        "policy_float_statistics": dict(floats),
        "example": example,
        "projected_payload_gib": {
            name: sum(
                r["weight"] * r["formats"][name]["compressed_bytes"] for r in samples
            )
            / 2**30
            for name in formats
        },
        "notes": "All formats compressed per game with Zstd-6. Rounding probes are NOT exact JSON-number round-trips. Policy-only normalization targets the three native Vec<f32> fields; scalars remain unchanged. All-float width probe also rounds duration_seconds and is not a production format.",
        "games": samples,
    }
    (report_dir / "float-diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "games"}, indent=2))


if __name__ == "__main__":
    main()
