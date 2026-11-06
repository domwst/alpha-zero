# Game replay storage

New rich game archives use `<game-id>.msgpack.zst`: one MessagePack object inside
one Zstandard level-6 frame with a checksum and declared decompressed size.
The logical schema remains version 1. Structs are named maps, unit enums are
strings, board states and vectors are arrays, and optional fields may be null.
Native policy/value fields are encoded as **f32**, while durations remain **f64**,
seeds remain **u64**, and visits remain integers. Python converters preserve
unknown fields and their original types/precision.

This replaces rich replay storage only: checkpoint `replay.bin.zst` training
buffers, manifests, statistics, comparison reports and the browser's JSON HTTP
API are unchanged. Both Rust recovery and Python/dashboard readers still accept
old `<game-id>.json` games. Mixed-format directories list each ID only once and
prefer compressed files. Corrupt compressed files produce an error, not a silent
fallback to an older JSON copy. Readers cap decompressed records at 128 MiB.

JSON policies contain short decimal representations of native f32 values.
Conversion restores those f32 values before packing; it does not preserve the
original JSON bytes or Python's binary64 approximation to each decimal. This is
lossless relative to the original Rust f32 fields. It does not quantize duration
or arbitrary metadata to f32. Imported archives may retain null states/actions
and provenance; those remain dashboard-readable, not native resumable games.

## Optional conversion of existing games

Install dependencies with `uv sync --frozen` (or install
`deploy/dashboard/requirements.txt` in the dashboard environment). Deploy the new
readers first and stop training/import/archive writers. Read-only browsing can
remain active. Keep an external backup when removing originals.

```sh
# Inventory only; does not change games.
uv run python -m scripts.job_service.compress_replays --root ROOT/jobs

# First publish and verify a small pilot; omit --remove-json to retain originals.
uv run python -m scripts.job_service.compress_replays --root ROOT/jobs \
  --journal ROOT/replay-conversion.jsonl --apply --remove-json --limit 100

# Continue (also the command to resume after an interruption).
uv run python -m scripts.job_service.compress_replays --root ROOT/jobs \
  --journal ROOT/replay-conversion.jsonl --apply --remove-json --workers 4
```

Only hexadecimal game filenames directly inside eight-digit archive epochs are
selected. Manifests and checkpoints are untouched. Directory symlinks are not
followed; hard-linked games remain shared. Compressed output is published
atomically after round-trip verification, reread and compared, then original and
replacement SHA-256 hashes are journaled and fsynced **before** JSON removal.
Source identity is checked again before deletion. The journal is append-only;
an interrupted run can contain repeated entries for the same original path.
Do not run multiple conversions or archive writers concurrently.

Old migration/backup manifests remain historical evidence for original bytes;
the conversion journal maps their original JSON hashes to replacement hashes.
Byte-for-byte JSON restoration requires the external backup. A converted game
can otherwise be loaded with `scripts.job_service.replays.load(path)` and
exported as JSON for inspection.

## Verification

```sh
./run.sh cargo build --example replay_codec --bin alz
ALZ_REPLAY_CODEC=target/debug/examples/replay_codec \
  ./run.sh python -m unittest scripts.test_replay_storage
ALZ_TEST_BINARY=target/debug/alz OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./run.sh python -m unittest scripts.test_job_service_integration
```

These cover both codec directions with real Rust types, old JSON, native pause/
resume with mixed-format games, corrupt frames, scalar precision, duplicate IDs, hard links and migration
restart behavior. HTTP tests also cover inherited mixed-format archives.
