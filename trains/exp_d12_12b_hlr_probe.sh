#!/usr/bin/env bash
# =============================================================================
# d12 high-LR stability probes (MATRIX_LR = LR_MULT x 0.004, default 8x = 0.032, where rexp_rmsnorm
# diverged at ~1400 steps and softmax did not). One arm per call, on the GPUs in GPUS (2 each, so 4
# arms fit on 8 GPUs); diagnostics every 50 steps, now with the attention output path's gain / c_proj
# norms (nanochat/diagnostics.py weight_report).
#
#   ARM=rexp            GPUS=0,1 bash trains/exp_d12_12b_hlr_probe.sh   # gain lr tied to MATRIX_LR (the default)
#   ARM=rexp_untied     GPUS=2,3 bash ...  # gain lr fixed at its base (diverged at 8x: ~1.4k steps)
#   ARM=rexp_gamma8     GPUS=... bash ...  # untied, gain lr x8 by hand (= tied at LR_MULT=8)
#   ARM=rexp_nofp8      GPUS=... bash ...  # untied + FP8 off
#   ARM=rexp_nogain     GPUS=... bash ...  # output RMSNorm without its learned gain (--attn-gain=0)
#   ARM=softmax_rmsnorm GPUS=... bash ...  # softmax + the same per-head output RMSNorm + gain (tied)
#   ARM=softmax         GPUS=... bash ...  # plain softmax
#   STOP_AT_STEP=3000 (default) leaves the loop there, schedule unchanged; FINAL_EVAL off.
#
# Everything else is the baseline's (12B tokens, 524,288 tokens / step: 2 GPUs x 32 x 2048 x 4
# accumulation steps, warmup, schedule, FP8 unless nofp8), so all arms and the 8-GPU runs compare step
# for step. The divergence shows by ~1.5k steps: stop an arm once it has told you what it will.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
ARM="${ARM:?set ARM=rexp|rexp_untied|rexp_gamma8|rexp_nofp8|rexp_nogain|softmax_rmsnorm|softmax}"
STOP="${STOP_AT_STEP:-3000}"
: "${GPUS:?set GPUS, e.g. GPUS=0,1}"
LR_MULT="${LR_MULT:-8}"
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
source trains/_exp_common.sh                        # clears everything below this line's control
export NPROC_PER_NODE="$NGPU"
export MATRIX_LR=$(python3 -c "print(round(0.004 * $LR_MULT, 6))")
export FP8=1
export FP8_RECIPE=tensorwise
export STOP_AT_STEP="$STOP"
export FINAL_EVAL=none
export ATTN_GAMMA_LR_TIE=1
case "$ARM" in
    rexp)            export ATTN_KIND=rexp_rmsnorm ;;
    rexp_untied)     export ATTN_KIND=rexp_rmsnorm; export ATTN_GAMMA_LR_TIE=0 ;;
    rexp_gamma8)     export ATTN_KIND=rexp_rmsnorm; export ATTN_GAMMA_LR_TIE=0; export ATTN_GAMMA_LR_MULT=8 ;;
    rexp_nofp8)      export ATTN_KIND=rexp_rmsnorm; export ATTN_GAMMA_LR_TIE=0; export FP8=0 ;;
    rexp_nogain)     export ATTN_KIND=rexp_rmsnorm; set -- --attn-gain=0 "$@" ;;
    softmax_rmsnorm) export ATTN_KIND=softmax_rmsnorm ;;
    softmax)         export ATTN_KIND=softmax ;;
    *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac
export NANOCHAT_FA3_DGAIN=2                         # rexp arms: dgamma fused into the attention preprocess
export TARGET_PARAM_DATA_RATIO=109                  # 12.0B tokens
export DEVICE_BATCH_SIZE=32
export SAVE_EVERY=5000
export EVAL_EVERY=500
export DIAGNOSTICS_EVERY=50
export RUN_NAME="d12_12b_probe_lr${MATRIX_LR}_${ARM}_${MUON_VARIANT}"  # must start with d12_ for resume_latest.sh
EXP_TOKENS=12.0B
run_experiment 12 "$@"
