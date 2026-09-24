#!/usr/bin/env bash
# =============================================================================
# d12 hybrid SWA + Muon, 10.0B tokens, on ONE GB10 (DGX Spark), attention: softplus_rmsnorm.
#
#   DRY_RUN=1 bash trains/exp_d12_10b_gb10_softplus_rmsnorm.sh               # print the config, do not train
#   SMOKE=1 bash trains/exp_d12_10b_gb10_softplus_rmsnorm.sh                 # 20-step pipeline check, gives tok/s
#   bash trains/exp_d12_10b_gb10_softplus_rmsnorm.sh                         # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d12_10b_gb10_softplus_rmsnorm.sh   # baseline arm
#   RESUME=<run_id> bash trains/exp_d12_10b_gb10_softplus_rmsnorm.sh         # continue after a crash / restart
#
# Local counterpart of the B200 exp_d12_*.sh arms, re-sized for GB10: device batch 32
# (the engine's measured fastest on 1x GB10; 64 is OOM-killed) and BF16, because FP8 is
# ~5% slower than BF16 on GB10. The optimizer step is unchanged (grad accumulation), so
# the loss curve is comparable to the B200 arms; the wall clock is not.
#
#   10.0B tokens = ratio 91 x 110.1M scaling params. At ~48k tok/s on an idle GB10 that
#   is ~57 h; the GPU is shared, so plan for more, and keep RESUME in mind.
#
# Compare exp_d12_10b_gb10_softmax.sh and exp_d12_10b_gb10_softplus_rmsnorm.sh with the
# same MUON_VARIANT; they differ only in ATTN_KIND.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
source trains/_exp_common.sh                        # clears everything below this line's control

# The variable under test.
export ATTN_KIND=softplus_rmsnorm

export TARGET_PARAM_DATA_RATIO=91               # 10.0B tokens
export DEVICE_BATCH_SIZE=32                         # throughput only; grad accum keeps the step fixed
export FP8=0
export SAVE_EVERY=1000                              # ~every 3 h on GB10; RESUME picks up from here
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
EXP_TOKENS=10.0B

run_experiment 12 "$@"
