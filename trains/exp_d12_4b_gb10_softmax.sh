#!/usr/bin/env bash
# =============================================================================
# d12 hybrid SWA + Muon, 4.0B tokens, on ONE GB10 (DGX Spark), attention: softmax.
#
#   DRY_RUN=1 bash trains/exp_d12_4b_gb10_softmax.sh               # print the config, do not train
#   SMOKE=1 bash trains/exp_d12_4b_gb10_softmax.sh                 # 20-step pipeline check, gives tok/s
#   bash trains/exp_d12_4b_gb10_softmax.sh                         # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d12_4b_gb10_softmax.sh   # baseline arm
#   RESUME=<run_id> bash trains/exp_d12_4b_gb10_softmax.sh         # continue after a crash / restart
#
# Local counterpart of the B200 exp_d12_*.sh arms, re-sized for GB10: device batch 32
# (the engine's measured fastest on 1x GB10; 64 is OOM-killed). FP8 (tensorwise) as on
# B200, so the numerics match those arms, although on GB10 FP8 is ~5% slower than BF16
# and takes ~6 GiB more. The optimizer step is unchanged (grad accumulation), so the loss
# curve is comparable to the B200 arms; the wall clock is not.
#
#   4.0B tokens = ratio 36 x 110.1M scaling params. At ~48k tok/s on an idle GB10 that
#   is ~23 h; the GPU is shared, so plan for more, and keep RESUME in mind.
#
# Compare exp_d12_4b_gb10_softmax.sh and exp_d12_4b_gb10_softplus_rmsnorm.sh with the
# same MUON_VARIANT; they differ only in ATTN_KIND.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
source trains/_exp_common.sh                        # clears everything below this line's control

# The variable under test.
export ATTN_KIND=softmax

export TARGET_PARAM_DATA_RATIO=36               # 3.96B tokens (36 per scaling param, ~20 per param at 200M)
export DEVICE_BATCH_SIZE=32                         # throughput only; grad accum keeps the step fixed
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=1000                              # ~every 3 h on GB10; RESUME picks up from here
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
EXP_TOKENS=4.0B

run_experiment 12 "$@"
