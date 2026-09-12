# Training-recipe comparisons — deployment, 2026-09-08

The first comparison started at 21:36 UTC on 2026-09-08; three successors are queued.

Four paired-seed comparisons: constant LR versus cosine decay and
constant LR versus BatchNorm gamma=1, separately for seeds 20260908 and 20260909.
Each uses the minimum combined validation-loss checkpoint from its 20-pass run.
Selected passes are constant/cosine/gamma = 11/15/11 for seed 1 and 10/16/19 for
seed 2. Training data and split are unchanged.

Match protocol: 1,000 seat-balanced games, 4,000 simulations, temperature 0.7,
game concurrency 300, inference batch 64, timeout 1 ms. The next comparison
starts at 60% completion, with at most two matches active.

## Pod selection

The first choice was Community Cloud RTX 4090 at $0.34/hour. Two distinct hosts
(107 GB and 125 GB RAM) failed before any experiment: `/dev/nvidia-uvm` returned
EIO and bare `cuInit(0)` returned 999. The second host failed identically with
RunPod's standard PyTorch image, excluding the minimal image as the explanation.
Both used driver 580.178.04. They were terminated; neither produced match data.
The underlying host/driver cause was not established.

The replacement is Secure Cloud RTX 3090, 125 GB RAM, 32 vCPUs, $0.50/hour plus
storage. Pod ID: `1amocr9mqhg10y`; driver 580.126.20. Bare CUDA initialization
and device access passed before setup. This is the usable fallback, not a
measured claim that it minimizes dollars per game across all current offers.
An early B64 inference benchmark measured 24,448 evaluations/s (200 measured
iterations after 30 warmups), versus 28,226 evaluations/s in the archived L40
preflight (10 iterations after 3 warmups, same architecture and batch). This
suggests about 20% more isolated inference per GPU dollar at $0.50 versus
$0.69/hour, but the different benchmark lengths and full-match tail mean this
is only an indicative comparison.

## Reproducibility and monitoring

Only six selected model weights and metadata, the source and runtime lockfiles,
a stripped copy of the previous validated executable, and archived dashboard
history were restored. Original replay/training archives remain unchanged locally.
All 135 restored files were checked against the transferred bundle.

The queue runs under `/workspace/alpha-zero-battles/runs/recipe-battles`.
Receipts, pricing, connection details and the selection plan are under
`runs/recipe-battles-deploy-20260908` locally. The bundle SHA-256 is
`9825748b831a373dc2e533abc6e9e9911e829b137e5de38801408319ee6eab34`.
The queue runner received a subsequent check binding its preflight receipt to
its exact binary; the deployed script hash is frozen in `worker-config.json`.

The existing dashboard at `http://127.0.0.1:8765/experiments` combines the 28
archived jobs with four live recipe comparisons. Both HTTP and SSH interfaces
remain read-only. Network labels distinguish recipe, training seed and selected
pass. The local systemd service uses its `30-recipe-battles.conf` override;
removing that override restores the previous archive-only configuration.

## Verification

- 22 Python queue/dashboard tests passed, including HTTP mutation rejection.
- Eight recipe/overlap tests passed; frontend build and six frontend test files passed.
- Native smoke: 300 games, 16 simulations, concurrency 300, batch 64, 6.2078 seconds.
  The report was validated for model identity, game count, seat balance and settings.
  This shallow smoke is not evidence about the recipes' playing strength.
- CPU/CUDA model-head agreement and the inference benchmark are required before
  starting the four production comparisons; their receipts are saved with the queue.

All 36 CPU/CUDA head checks passed; the maximum absolute difference was
0.00017643. The final isolated inference benchmark measured 29,983 evaluations/s,
compared with 24,448 during the earlier concurrent build. Treat this observed
range as preflight throughput, not a full-match rate. The production match is
confirmed running on CUDA and the dashboard receives live progress and telemetry.

The first production heartbeat reported 1,246,878 evaluations in 60.0 seconds
(20,780 evaluations/s). No games had completed at that early point; the match
was actively searching. Final API inventory contains only the healthy RTX 3090.
