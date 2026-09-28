#!/usr/bin/env bash
# =============================================================================
# d12 hybrid SWA + Moonlight Muon, 12.0B tokens, 4x H100, FP8, attention: rexp_rmsnorm, at the baseline
# matrix LR (0.004), with the output RMSNorm gain's AdamW lr multiplied by GAMMA_MULT.
#
#   GAMMA_MULT=4 GPUS=0,1,2,3 bash trains/exp_d12_12b_4xh100_rexp_gamma.sh
#   GAMMA_MULT=8 GPUS=4,5,6,7 bash trains/exp_d12_12b_4xh100_rexp_gamma.sh
#
# The question: does the gain learning too slowly cost rexp quality even at the normal LR? At 8x the
# matrix LR it made rexp diverge (the gain is, with c_proj, the only knob on the attention sublayer's
# output scale); here the reference is the 8x H100 rexp run at GAMMA_MULT=1 (val bpb 0.7752, softmax
# 0.7745). 524,288 tokens / step as there (4 GPUs x 32 x 2048 x 2 accumulation steps): same steps.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
GAMMA_MULT="${GAMMA_MULT:?set GAMMA_MULT, e.g. 4}"
export CUDA_VISIBLE_DEVICES="${GPUS:?set GPUS, e.g. GPUS=0,1,2,3}"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
source trains/_exp_common.sh                        # clears everything below this line's control
export NPROC_PER_NODE="$NGPU"
export ATTN_KIND=rexp_rmsnorm
export NANOCHAT_FA3_DGAIN=2
export ATTN_GAMMA_LR_MULT="$GAMMA_MULT"
export ATTN_GAMMA_LR_TIE=1                          # a no-op at the default MATRIX_LR (factor 1)
export TARGET_PARAM_DATA_RATIO=109                  # 12.0B tokens
export DEVICE_BATCH_SIZE=32
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=5000
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
export RUN_NAME="d12_12b_${NGPU}xh100_rexp_gmult${GAMMA_MULT}_${MUON_VARIANT}"  # must start with d12_ for resume_latest.sh
EXP_TOKENS=12.0B
run_experiment 12 "$@"
