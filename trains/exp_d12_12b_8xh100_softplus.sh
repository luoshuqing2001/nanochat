#!/usr/bin/env bash
# =============================================================================
# d12 hybrid SWA + Moonlight Muon, 12.0B tokens, 8x H100 DDP, FP8, attention: softplus_rmsnorm.
#
#   DRY_RUN=1 bash trains/exp_d12_12b_8xh100_softplus.sh               # print the config, do not train
#   SMOKE=1 bash trains/exp_d12_12b_8xh100_softplus.sh                 # 20-step pipeline check, gives tok/s
#   bash trains/exp_d12_12b_8xh100_softplus.sh                         # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d12_12b_8xh100_softplus.sh   # baseline arm
#   RESUME=<run_id> bash trains/exp_d12_12b_8xh100_softplus.sh         # continue after a crash / restart
#
# 8x H100 counterpart of exp_d12_ctrl.sh (1x B200). Same budget, same FP8 recipe, same
# optimizer step; only the device batch differs, so the loss curves are comparable and
# the wall clock is not. The engine launches torchrun with one rank per visible GPU.
#
#   d12: dim 768, 6 heads x 128, 286,261,730 params (110,100,912 scaling params)
#   12.0B tokens = ratio 109 x scaling params = 42 per parameter, ~22,890 steps.
#   GLA / DeltaNet / Gated Slot Attention train 340M on 15B, i.e. 44 per parameter.
#
# Device batch is not free here. At d12 the auto batch is B_REF = 524,288 tokens per
# step (base_train.py), and it must be a multiple of 8 ranks x 2048 x DEVICE_BATCH_SIZE:
# 32 is the largest that divides (one micro-step, no grad accumulation); 64 fails
# base_train's assert. Drop to 16 only if it OOMs -- that adds accumulation, nothing else.
#
# Measured on 8x H100 80GB SXM (FA3, FP8 on 73/80 linears): ~125 ms/step, 4.1-4.2M
# tok/s, ~40% bf16 MFU, 34 GB peak per GPU -> ~48 min of training per arm.
#
# The softplus_rmsnorm arm of the softmax baseline in the same size (exp_*_softmax.sh): only
# ATTN_KIND and RUN_NAME differ. Exact softplus, the function rexp_rmsnorm stands in for.
#
# softplus_rmsnorm: f(x) = softplus(x) = log(1 + e^x), then the same per-head output RMSNorm
# (gain folded into c_proj). On H100 it runs the FA3 build of the flash-attention repo, branch
# lsq/softplus-fa3 (the CuTe kernels of nanochat/softplus_rmsnorm_attention.py are SM80/SM120
# only), log1p(2^-|x|) as an fp32 cubic (max abs error ~1e-4, below bf16's resolution), built with
#   python hopper_softplus/build.py bwd_sp_poly3        (in the flash-attention repo)
# and found via NANOCHAT_FA3_FN_DIR (default: ../flash-attention/hopper_softplus/build).
# Kernel-level: forward + backward ~0.90x FA3 softmax -- expect a few % slower steps than softmax.
# Everything else is identical to the softmax baseline in the same size, so compare the two.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUON_VARIANT="${MUON_VARIANT:-moonlight}"   # the variable under test
source trains/_exp_common.sh                        # clears everything below this line's control
export ATTN_KIND=softplus_rmsnorm                   # the variable under test (see header)
export TARGET_PARAM_DATA_RATIO=109              # 12.0B tokens
export DEVICE_BATCH_SIZE=32                         # see above: 8 x 2048 x 32 = 524,288 = one step
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=5000
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
export RUN_NAME="d12_12b_8xh100_softplus_${MUON_VARIANT}"  # must start with d12_ for resume_latest.sh
EXP_TOKENS=12.0B
run_experiment 12 "$@"
