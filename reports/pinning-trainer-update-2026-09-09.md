# Parameterless pinning API rollout

Reviewed source changes relative to the pod: `Cargo.toml`, `Cargo.lock`, and
`src/training_batches.rs` only. The trainer pins the same three CPU tensors and
checks pinned state for nonblocking CUDA transfer, now omitting the deprecated
device argument through tch revision `07f5604711b5bd9acd3ad501dbdf76fa758720a4`.
The CUDA-only guard and training algorithm are unchanged.

Validation: three local CPU batching tests and all four pod CPU/CUDA batching
tests passed. The new binary restored the saved model, optimizer and replay state.
A real-replay smoke check completed 20 measured B256 standard-Adam training steps
with CPU caching and prefetch=2; final loss was 4.8375076. No pinning deprecation
warnings appeared in CUDA batching, checkpoint restore or replay training. The source update is built in `target-pinning-update`,
leaving the production executable intact until validation succeeds.

The stop watcher waits for the next `epoch complete` log entry and corresponding
readable stats and checkpoint metadata. It then stops the worker before deploying.
The primary resumed at S3000/P500/B256, with the same 100-epoch target and
replay/LR schedule. Top-p 1.0 remains explicitly held; concurrency is one.

Deployment receipts and validation logs are stored remotely under
`runs/selfplay-nucleus/pinning-update/` and copied locally under
`runs/pinning-update-20260909/`.

Epoch 6 completed at 19:32:50 UTC; the boundary stop finished at 19:32:56 UTC.
Deployment completed at 19:37:13 UTC, resuming epoch 7 from snapshot `00000005`.
The running native process was verified against SHA-256
`b702391487e36a53c625730c36d689b05afc1ff682972ee0d9d2792d97f62c94`.
Fresh heartbeats and the read-only dashboard confirmed primary P500 and control
paused. The old executable and preflight remain archived on the pod.
