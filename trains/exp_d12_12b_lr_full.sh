#!/usr/bin/env bash
# =============================================================================
# d12 hybrid SWA + Moonlight Muon, 12.0B tokens, FP8, a full run at MATRIX_LR = LR_MULT x 0.004 on the
# GPUs in GPUS (any count; 524,288 tokens / step always, the accumulation makes up the rest, so every
# run compares step for step with the 8x H100 baselines).
#
#   ARM=rexp    LR_MULT=1 GAMMA_MULT=4 GPUS=0 bash trains/exp_d12_12b_lr_full.sh
#   ARM=softmax LR_MULT=2              GPUS=1 bash trains/exp_d12_12b_lr_full.sh
#   ARM=softmax_rmsnorm ...            (softmax + the per-head output RMSNorm + gain)
#
# rexp / softmax_rmsnorm: the output RMSNorm gain's lr is GAMMA_MULT x its base, tied to MATRIX_LR
# (ATTN_GAMMA_LR_TIE=1: x LR_MULT on top); rexp runs NANOCHAT_FA3_DGAIN=2.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
ARM="${ARM:?set ARM=softmax|rexp|softmax_rmsnorm}"
LR_MULT="${LR_MULT:-1}"
GAMMA_MULT="${GAMMA_MULT:-1}"
export CUDA_VISIBLE_DEVICES="${GPUS:?set GPUS, e.g. GPUS=0}"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
source trains/_exp_common.sh                        # clears everything below this line's control
export NPROC_PER_NODE="$NGPU"
case "$ARM" in
    softmax)         export ATTN_KIND=softmax; TAG=softmax ;;
    rexp)            export ATTN_KIND=rexp_rmsnorm; TAG="rexp_g${GAMMA_MULT}" ;;
    softmax_rmsnorm) export ATTN_KIND=softmax_rmsnorm; TAG="softmax_rmsnorm_g${GAMMA_MULT}" ;;
    *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac
export NANOCHAT_FA3_DGAIN=2
export ATTN_GAMMA_LR_MULT="$GAMMA_MULT"
export ATTN_GAMMA_LR_TIE=1
export MATRIX_LR=$(python3 -c "print(round(0.004 * $LR_MULT, 6))")
export TARGET_PARAM_DATA_RATIO=109                  # 12.0B tokens
export DEVICE_BATCH_SIZE=32
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=5000
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
export RUN_NAME="d12_12b_full_lr${MATRIX_LR}_${TAG}_${MUON_VARIANT}"  # must start with d12_ for resume_latest.sh
EXP_TOKENS=12.0B
run_experiment 12 "$@"
