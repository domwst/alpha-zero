# RunPod decommissioned after verified local backup

Pod `rpii1ijfvmno97` (`noisy_azure_sturgeon`, L40) was permanently terminated
on 2026-09-08 after download, extraction, checksum validation and local model
load checks. The subsequent RunPod API inventory returned an empty pod list.
No independent network volume was attached.

Backup directory: `runs/pod-backup-rpii1ijfvmno97-20260908/`.

- Archive: 1,308,017,050 bytes; SHA-256
  `21b5cd4d3df5bfeed17381fe6a5c0ba4d9d902103f04f7c04e914dc170a61ac3`.
- 2,572 files verified against the remote manifest, representing
  20,565,361,326 logical bytes and 1,963,023,668 unique content bytes.
- 285 checkpoint directories preserved, including all 120 checkpoints from
  the six new recipe runs; optimizer states, replays, metrics, results,
  logs, profiling artifacts and deployed source versions retained.
- All six selected recipe models loaded and completed CPU policy games using
  the local executable. These are usability checks, not strength comparisons.
- The archive dashboard serves all 28 jobs and their saved log views on
  `127.0.0.1:8765`, independently of SSH. Frontend build and 13 dashboard tests
  passed; live archive endpoints were verified before and after termination.

The backup README explains layout and original-path mapping. Audit receipts
are `transfer-receipt.json`, `verification.json`, `model-load-verification.json`
and `decommissioned.json` in that directory. `restored/MANIFEST.json` records
every file and explicitly excluded rebuildable caches/toolchains. Duplicate
files share hard links in the restored tree; use copies for future edits.

The temporary authenticated HTTP transfer test endpoint was closed and its
local token deleted. The completed backup used the SSH gateway, with chunk
checksums and a final whole-archive SHA-256 check.
