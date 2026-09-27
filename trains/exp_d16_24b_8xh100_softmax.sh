#!/usr/bin/env bash
# =============================================================================
# d16 hybrid SWA + Moonlight Muon, 23.7B tokens, 8x H100 DDP, FP8, attention: softmax.
#
#   DRY_RUN=1 bash trains/exp_d16_24b_8xh100_softmax.sh               # print the config, do not train
#   SMOKE=1 bash trains/exp_d16_24b_8xh100_softmax.sh                 # 20-step pipeline check, gives tok/s
#   bash trains/exp_d16_24b_8xh100_softmax.sh                         # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d16_24b_8xh100_softmax.sh   # baseline arm
#   RESUME=<run_id> bash trains/exp_d16_24b_8xh100_softmax.sh         # continue after a crash / restart
#
# Middle point of the fixed-ratio ladder d12 12.0B / d16 23.7B / d20 39.6B, all at
# ~44 tokens per parameter -- the 340M/15B ratio of GLA, DeltaNet and Gated Slot
# Attention. Holding the ratio fixed across sizes is what lets the softmax-vs-softplus
# gap be read as a function of scale; the older exp_d16_ctrl.sh (12B, 22/param) does not.
#
#   d16: dim 1024, 8 heads x 128, 536,871,738 params (234,881,792 scaling params)
#   23.7B tokens = ratio 101 x scaling params = 44.2 per parameter, 45,248 steps.
#
# The auto batch at d16 is 524,288 tokens per step (Power Lines, base_train.py), which
# must be a multiple of 8 ranks x 2048 x DEVICE_BATCH_SIZE: 32 is the largest that
# divides (one micro-step). Drop to 16 only if it OOMs -- that adds accumulation only.
#
# Estimated <= 3.3 h on 8x H100 (1.59 GFLOP/token at the ~3.2 PFLOP/s d12 measured
# here; the wider matrices should do a little better). Memory not measured at d16:
# run SMOKE first.
#
# A softplus arm should copy this file and change only ATTN_KIND.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUON_VARIANT="${MUON_VARIANT:-moonlight}"   # the variable under test
source trains/_exp_common.sh                        # clears everything below this line's control
export ATTN_KIND=softmax                            # baseline
export TARGET_PARAM_DATA_RATIO=101              # 23.7B tokens
export DEVICE_BATCH_SIZE=32                         # see above: 8 x 2048 x 32 = 524,288 = one step
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=5000                              # ~every 22 min
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
export RUN_NAME="d16_24b_8xh100_softmax_${MUON_VARIANT}"  # must start with d16_ for resume_latest.sh
EXP_TOKENS=23.7B
run_experiment 16 "$@"
