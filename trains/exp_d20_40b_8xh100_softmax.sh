#!/usr/bin/env bash
# =============================================================================
# d20 hybrid SWA + Moonlight Muon, 39.6B tokens, 8x H100 DDP, FP8, attention: softmax.
#
#   DRY_RUN=1 bash trains/exp_d20_40b_8xh100_softmax.sh               # print the config, do not train
#   SMOKE=1 bash trains/exp_d20_40b_8xh100_softmax.sh                 # 20-step pipeline check, gives tok/s
#   bash trains/exp_d20_40b_8xh100_softmax.sh                         # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d20_40b_8xh100_softmax.sh   # baseline arm
#   RESUME=<run_id> bash trains/exp_d20_40b_8xh100_softmax.sh         # continue after a crash / restart
#
# Top of the fixed-ratio ladder d12 12.0B / d16 23.7B / d20 39.6B, all at ~44 tokens
# per parameter (the 340M/15B ratio of GLA, DeltaNet and Gated Slot Attention). Not
# the literature's 1.3B/100B tier (77/param), so do not set it beside those tables;
# exp_d20_60b.sh is the older, differently-sized headline run.
#
#   d20: dim 1280, 10 heads x 128, 896,533,746 params (435,160,240 scaling params)
#   39.6B tokens = ratio 91 x scaling params = 44.2 per parameter, 37,765 steps.
#
# The auto batch at d20 is 1,048,576 tokens per step. 64 per device would make that one
# micro-step, but at d20 the activations plus the fp32 logits (bs x 2048 x 32768 x 4 B,
# 17 GiB at 64) are unlikely to fit in 80 GB, so 32 with 2 accumulation steps. Same
# optimizer step either way. If 32 OOMs: LOSS_CHUNK_TOKENS below, then 16.
#
# Measured on 8x H100 80GB SXM: ~676 ms/step, 1.55M tok/s, ~56% bf16 MFU -> ~7.2 h.
# bs 32 fits for training; the diagnostics forward used to OOM here at step 1500 by
# materialising full fp32 logits, fixed in nanochat/diagnostics.py. nanochat never prunes
# checkpoints, so every SAVE_EVERY leaves one behind; resume_latest.sh --keep N trims.
#
# A softplus arm should copy this file and change only ATTN_KIND.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUON_VARIANT="${MUON_VARIANT:-moonlight}"   # the variable under test
source trains/_exp_common.sh                        # clears everything below this line's control
export ATTN_KIND=softmax                            # baseline
export TARGET_PARAM_DATA_RATIO=91               # 39.6B tokens
export DEVICE_BATCH_SIZE=32                         # 8 x 2048 x 32 x 2 accum = 1,048,576 = one step
# export LOSS_CHUNK_TOKENS=32768                    # only if bs 32 OOMs: chunk lm_head + CE
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=2000                              # ~every 30 min
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=500
export RUN_NAME="d20_40b_8xh100_softmax_${MUON_VARIANT}"  # must start with d20_ for resume_latest.sh
EXP_TOKENS=39.6B
run_experiment 20 "$@"
