#!/usr/bin/env bash
# =============================================================================
# d12 hybrid SWA + Moonlight Muon, 12.0B tokens, 4x H100, FP8, attention: softmax, at a HIGH matrix LR.
# A stability probe: the same high LR for softmax (exp_d12_12b_4xh100_hlr_softmax.sh, GPUs 0-3) and
# rexp_rmsnorm (exp_d12_12b_4xh100_hlr_rexp.sh, GPUs 4-7), run side by side.
#
#   bash trains/exp_d12_12b_4xh100_hlr_softmax.sh                # LR_MULT=8: MATRIX_LR = 8 x 0.004
#   LR_MULT=16 bash trains/exp_d12_12b_4xh100_hlr_softmax.sh     # another point of the ladder
#   GPUS=0,1,2,3 bash ...                                     # override the default GPU set (0,1,2,3)
#   DRY_RUN=1 bash ...                                        # print the config, do not train
#
# Only MATRIX_LR (Muon: every attention and MLP matrix) is scaled; embedding / unembedding / scalar
# LRs stay at the baseline's, so a divergence is about the transformer blocks. Everything else --
# data, 12B tokens, 524,288 tokens per step (4 GPUs x 32 x 2048 x 2 accumulation steps: the same
# optimizer steps as the 8-GPU runs, so the curves compare step for step), warmup, schedule, FP8 --
# is the 8x H100 baseline's (exp_d12_12b_8xh100_{softmax,rexp}.sh).
# Note: nanochat has QK-norm and a logit softcap, which remove softmax's classic high-LR failure
# (attention logit growth); a softmax divergence here, if any, may be a generic one.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
LR_MULT="${LR_MULT:-8}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
source trains/_exp_common.sh                        # clears everything below this line's control
export NPROC_PER_NODE=4                             # the engine counts nvidia-smi GPUs otherwise
export ATTN_KIND=softmax

export MATRIX_LR=$(python3 -c "print(round(0.004 * $LR_MULT, 6))")
export TARGET_PARAM_DATA_RATIO=109                  # 12.0B tokens
export DEVICE_BATCH_SIZE=32                         # 4 x 2048 x 32 = 262,144: 2 accumulation steps per step
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=5000
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
export RUN_NAME="d12_12b_4xh100_lr${MATRIX_LR}_softmax_${MUON_VARIANT}"  # must start with d12_ for resume_latest.sh
EXP_TOKENS=12.0B
run_experiment 12 "$@"
