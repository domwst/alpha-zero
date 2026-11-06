# Replay storage options — 2026-09-22

The raw `results.json` and `float-diagnostics.json` cited below are kept out of the
repository because of their size; they exist only on the machine that produced them.

**Implementation follow-up:** the chosen schema-aware MessagePack/f32 + Zstd
format and its backward-compatible migration are documented in
[migration.md](migration.md). The recommendation and measurements below are the
original investigation, including the binary64 result and its explanation.

## Recommendation

Keep the existing JSON schema and store each game as an independent `.json.zst`
file, initially at Zstandard level 6. Estimated footprint is **8.9 GiB instead of
50.9 GiB**, saving approximately **42 GiB (82%)**. This preserves the original
JSON bytes, every policy/value diagnostic, random game access, and the logical
continuation inheritance introduced earlier. No production archives were changed.

Use compressed JSON as the first step; consider a compact game-summary index as
a separate browsing improvement. A binary schema could be worthwhile later for
decode speed, but plain binary serialization is not automatically smaller after
compression.

**Follow-up:** the controlled float experiment below explains the initial binary
result. A schema-aware f32 representation is also promising for space, not only
decode speed; the initial binary64 comparison must not be generalized to native
binary replay formats. Compressed JSON remains the lowest-risk byte-preserving
option, rather than an established optimum for all encodings.

## Method

The inventory found **198,414 unique game files**, occupying 50.4885 GiB of payload
and 50.8554 GiB of allocated file blocks. Directory symlinks were not followed and
hard links were deduplicated by device/inode. Counts therefore exclude repeated
views of shared/inherited archives. Manifests, producer weights, and other files
are not included in these game-file figures.

The benchmark sampled **768 games / 199.5 MiB**, with 64 random games from each of
12 equal-count file-size strata, seed 20260922. There were 381 imported and 387
natively recorded games. Storage projections expand each stratum separately.
The projected raw JSON payload is 50.3447 GiB versus the measured 50.4885 GiB.

Each format round-tripped to the same parsed Python object; compressed JSON also
round-tripped to identical original bytes. MessagePack used 64-bit floats to retain
the numeric values parsed from JSON. No quantization or diagnostic removal was used.

Timings are single-pass, in-memory CPU measurements on this local machine using
Python 3.14.7 and msgpack 1.2.2. They exclude disk/network I/O and HTTP response
serialization. They are not end-to-end dashboard benchmarks. Output allocation
estimates round every file to 4 KiB and exclude directory/inode metadata.

| Representation | Projected payload GiB | Estimated file blocks GiB | Encode ms/game | Decode to object ms/game |
| --- | ---: | ---: | ---: | ---: |
| Original JSON | 50.49 actual | 50.86 actual | — | 3.55 |
| Re-serialized compact JSON | 49.77 | 50.14 | 9.16 | 3.66 |
| JSON + gzip level 6 | 8.70 | 9.07 | 5.37 | 4.17 |
| JSON + Zstandard level 3 | 9.30 | 9.67 | 0.86 | 3.83 |
| JSON + Zstandard level 6 | 8.56 | 8.94 | 2.37 | 3.84 |
| JSON + Zstandard level 9 | 8.39 | 8.77 | 3.79 | 3.87 |
| MessagePack, binary64 floats | 37.08 | 37.48 | 1.02 | 0.63 |
| MessagePack + Zstandard level 6 | 11.74 | 12.14 | 2.26 | 0.78 |

Compression estimates are sample projections, not the result of migrating the
entire archive. Approximate stratified 95% sampling margins on payload estimates
are ±0.29 GiB for JSON/Zstd-6 and ±0.50 GiB for MessagePack/Zstd-6; they do not
capture every possible source of sampling or timing error. Encode times for
binary/compact JSON exclude the initial JSON parse needed during migration.

Zstd-6 decompression alone averaged **0.215 ms/game**; JSON parsing still dominates
the ~3.84 ms combined time. Level 9 saved only another ~0.16 GiB of file blocks
while taking ~60% more compression CPU time in this sample. Gzip is a reasonable
dependency-light alternative with almost the same space savings, but slower
compression/decompression here.

## What occupies the space?

Serialized field-value attribution in the sample (excluding JSON keys and other
structural punctuation):

- Training policy: 42.3%.
- Network prior: 29.5%.
- Sampling policy: 18.1%.
- Root visits: 4.7%.
- Board states: 2.7%.
- Search statistics: 2.0%.

The current native and imported writers already use compact JSON. Removing
whitespace is therefore not the main opportunity. Avoid dropping stored policy
variants: they describe distinct distributions and are useful in the explorer.
Reconstructing boards from moves would add complexity for a small upper bound on
space savings; compression already captures much of their repeated content.

## Other designs

1. **Per-game compressed JSON:** smallest implementation change. Keep manifests
   plain JSON, support mixed `.json`/`.json.zst` archives, and preserve current
   game IDs and archive ownership. Additional file-block overhead is only about
   0.38 GiB across the compressed dataset.
2. **SQLite with individually compressed game BLOBs and summary columns:** retains
   indexed random access and reduces filesystem entry count. A per-epoch or
   per-job database can preserve archive sharing and provide move counts/winners
   without decoding large policy arrays. It requires a new storage abstraction,
   index migration and recovery design; database-page overhead and actual I/O
   performance were not benchmarked here. Do not merge large payloads into the
   existing control/history database without a separate design review.
3. **Versioned typed binary plus compression:** may improve decode speed and
   permit f32 arrays, packed integers or sparse arrays. The native Rust policy
   fields are f32, while JSON parsing promotes them to Python floats. A schema-aware
   format can preserve original f32 semantics, but that is a different guarantee
   from preserving arbitrary historical JSON numeric values or original bytes.
   MessagePack's binary64 result above is not a benchmark of optimized f32/bincode.
   Cross-language tooling, format versioning and migrations make this a larger change.
4. **Whole-epoch `.tar.zst`/compressed JSONL:** useful for cold storage. A naive
   stream requires decompression up to the selected record (or a cache) for random
   access. An offset index plus independently compressed frames can solve this,
   but then it is effectively a custom indexed container. Not the first choice
   for the active interactive archive.

## Implementation outline, if approved

- Add one shared Python replay reader and discover both extensions. Ensure a
  game temporarily present in both forms appears once during migration.
- Update summary reads, full-game reads and inherited-archive resolution through
  that reader; continue returning the same JSON HTTP API to the browser.
- Update Rust archive save/load and recovery discovery in self-play and battle,
  plus Python import/backfill readers, before changing new training output.
- Rust already depends on `zstd`. The deployed Alpine backend uses Python 3.12,
  so add a Zstandard binding or explicitly move to Python 3.14; do not spawn a
  compression CLI process for every request. Python's stdlib Zstandard module
  was introduced in 3.14 ([documentation](https://docs.python.org/3/library/compression.zstd.html)).
- Migrate one canonical physical file at a time, not each inherited view. Write
  to a temporary file, verify decompressed bytes/hash, sync and publish atomically;
  only then remove the original. Preserve relative archive links and record a
  migration manifest with original content hashes. Audit any manifests holding
  literal filenames/hashes before changing those files.
- Verify listings, pagination, inherited epochs, policy values and sampled browser
  replays. Keep the user's external full backup for recovery.

Zstandard provides configurable lossless compression and a stable published
format ([project documentation](https://facebook.github.io/zstd/)). SQLite's own
[BLOB versus files benchmark](https://www.sqlite.org/fasterthanfs.html) illustrates
why indexed containers deserve measurement, but its reported speedup is not an
estimate for this dataset.

## Reproduce

```sh
uv run --no-project --python 3.14 --with msgpack==1.2.2 python \
  scripts/benchmark_replay_storage.py \
  --root runs/dashboard-local/data/jobs \
  --output reports/replay-storage-20260922/results.json
```

See `results.json` for every sampled path, measurement, aggregate and uncertainty
estimate. The benchmark only reads archive files and writes the requested report.

## Follow-up: why MessagePack + Zstd was larger

`scripts/benchmark_replay_floats.py` repeats a controlled comparison on 192 games
(the first 16 random samples in each original size stratum). All representations
use independent Zstd level-6 frames. No archive files are changed.

The native policy fields are `Vec<f32>` in `src/engine/match_runner.rs`. JSON's
short decimals round-trip to these f32 values, but Python parses them as binary64.
The binary64 encoding of a short decimal is not generally the same as converting
the original f32 value to binary64. The former can have irregular low-order
mantissa bits; the latter has 29 zero low-order mantissa bits. This does not add
real model precision or fundamental information: it changes the byte patterns
visible to the compressor.

An actual sampled value:

```
JSON:                              0.00033333333
Parsed directly as f64:            3f35d867c042a2aa
Rounded to f32, then promoted f64:  3f35d867c0000000
Native f32 encoding:               39aec33e
```

The policy-only experiment changes `training_policy`, `network_prior`, and
`sampling_policy` float values to their nearest f32 values, while still serializing
them as MessagePack f64. Integer values and all other fields stay unchanged.
The script asserts equal uncompressed MessagePack byte lengths: this isolates
numeric bit patterns rather than field removal or smaller float tags.

| Zstd-6 representation | Imported subset bytes | Native subset bytes | Projected archive payload GiB |
| --- | ---: | ---: | ---: |
| Original JSON | 1,133,916 | 7,702,588 | 8.50 |
| Original MessagePack f64 | 679,688 | 11,335,700 | 11.56 |
| MessagePack f64, only policy values restored to f32 grid | 679,688 | 6,834,997 | 7.23 |
| MessagePack f32 for every float (width probe only) | 644,525 | 6,485,700 | 6.86 |

These are smaller-sample payload projections, not the filesystem-allocation
figures in the main table. The policy-only change reduces native-game compressed
bytes by 39.7%, despite identical uncompressed lengths. Among 2,129,187 nonzero
native policy float values, 2,127,297 changed under f32 rounding; the largest
absolute change was about 2.98e-8. None of the 520,276 imported nonzero policy
floats changed. Imported MessagePack already compressed better than JSON, and its
policy-normalized version is byte-for-byte unchanged before compression.

This strongly identifies the decimal-to-binary64 conversion as the dominant cause
of the initial reversal, not a general weakness of binary formats. JSON keys,
punctuation and repeated decimals also compress well, so raw size alone cannot
predict compressed size.

Precision caveat: neither rounding probe is an exact round-trip of every JSON
number. The all-f32 probe also rounds true f64 fields such as duration, so it is
not a proposed production format. A future typed format should retain declared
f32 policy semantics, keep f64 where appropriate, and validate historical records
and Rust/Python compatibility. MessagePack supports both widths in its
[float format family](https://github.com/msgpack/msgpack/blob/master/spec.md#float-format-family).

Reproduce with:

```sh
uv run --offline --no-project --python 3.14 --with msgpack==1.2.2 python \
  scripts/benchmark_replay_floats.py
```

Detailed measurements and per-file results are in `float-diagnostics.json`.
