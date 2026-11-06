# MessagePack + Zstandard migration — 2026-09-22

## Result

Completed **198,414 games**, with zero remaining game JSON files in the selected
local archive. Every compressed replacement passed the final SHA-256 audit.

| Measurement | JSON before | MessagePack + Zstd after |
| --- | ---: | ---: |
| File payload | 50.489 GiB | 7.149 GiB |
| Allocated file blocks | 50.855 GiB | 7.543 GiB |

Replay files freed **43.313 GiB (85.17%)**. The 129.64 MiB migration journal leaves
**43.186 GiB net savings**, excluding negligible report overhead. This is an
actual full-archive measurement, not an estimate from the earlier samples.

The final audit found all 55 archive symlinks intact, 6,590 other retained files
unchanged by device/inode/size, and no unexpected changes outside converted
replays. Database counts remain 106 jobs, 54 experiments, 247,568 events and
548 artifacts. The post-conversion HTTP validation passed all 106 jobs and all
80 continuation epochs, including replays sampled from each of 58 jobs with games.

## Implementation

New native and imported rich game records use `.msgpack.zst`, while both Rust and
Python readers continue to support old `.json` archives. The logical schema is
unchanged (version 1), with named MessagePack maps and independent Zstd level-6
frames with checksums. Mixed-format listings deduplicate game IDs and prefer the
compressed representation. The browser still receives JSON over HTTP.

Policies and values retain their declared native f32 semantics. Durations and
unknown metadata are not rounded to f32; seeds retain all 64 bits. For JSON
migration, short decimal policy values are restored to the f32 values they
represent before binary encoding. Thus original JSON bytes are not preserved,
but native numerical values are. The earlier binary64 benchmark is not the
encoding used here; see the float investigation in `README.md`.

Checkpoint training buffers, manifests, statistics, model files and comparison
reports are unchanged. Zero-copy continuation inheritance is unchanged.

## Migration safeguards

The selected local archive contains 198,414 physical game files. The converter
does not follow shared directory aliases and preserves hard links. Each output
is atomically published, reread, decompressed and compared with the expected
record before an append-only SHA-256 journal is flushed to disk. Only then is
the old JSON file removed. Source identities are rechecked before removal.

The converter was first exercised on 100 games and checked through the live
dashboard. The production run was safely stopped and resumed twice: once to
replace expensive scalar float conversions with a tested bulk conversion, and
once to increase low-priority workers from four to eight. The optimized converter
matched the original scalar reference on 698 stratified games. Incomplete batches
retain JSON; already published compressed files are validated and reused on resume.

The external full `runs/`/`deployments/` backup remains the source for byte-exact
JSON restoration. Local receipts live in
`runs/dashboard-local/replay-migration-20260922/`. Old inventory/backup manifests
remain historical records; the new journal maps original hashes to replacements.

## Validation

- Rust library: 70 passed, four GPU tests ignored.
- Rust command layer: 29 passed, two opt-in integration tests ignored.
- CPU history reconstruction/resume/continuation test: explicitly run and passed.
- Python suite: 191 tests, 190 passed, native workflow opt-in skipped in this run.
  Rust/Python interoperability was enabled and passed in both directions.
- Native CPU workflow: separately passed, including self-play pause/resume,
  replay training, comparison recovery from mixed JSON/compressed games, and
  rejection of corrupt saved games.
- Codec tests cover f32/f64/u64 precision, unknown metadata, imported nulls,
  bad checksums, truncated frames, trailing data, size/schema limits, conflicting
  destinations, interrupted migration, hard links and changed-source refusal.
- Live dashboard check: all 106 jobs, replays sampled from all 58 jobs with games,
  all 80 final-continuation epochs, and 514,914 visible entries including shared
  and inherited archives. Read-only mutation rejection passed.
- All 55 retained archive symlinks were intact during conversion.

The dashboard image now installs MessagePack and Zstandard bindings and runs the
dual-format reader. The separate interactive demo was not restarted.

The final audit is in [migration-verification.json](migration-verification.json).
The local full journal and reproducible one-off verifier remain under
`runs/dashboard-local/replay-migration-20260922/`. The audit rehashed every output
against the journal after conversion; semantic round-trip verification happened
before each original JSON file was removed. Exact original JSON bytes remain in
the user's external backup. No database, model, or game-content migration is
needed when restoring an old JSON archive: the new readers still accept it.
