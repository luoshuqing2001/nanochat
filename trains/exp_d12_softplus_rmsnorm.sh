#!/usr/bin/env bash
# =============================================================================
# Ablation: d12 hybrid SWA + Muon, 12.0B tokens, FP8, SOFTPLUS + OUTPUT RMSNORM attention
# ("Softplus Attention with Normalization").
#
#   bash trains/exp_d12_softplus_rmsnorm.sh                        # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d12_softplus_rmsnorm.sh   # baseline arm
#   RESUME=<run_id> bash trains/exp_d12_softplus_rmsnorm.sh         # continue after a container restart
#   DRY_RUN=1 bash trains/exp_d12_softplus_rmsnorm.sh               # print the config, do not train
#
# Hermetic: _exp_common.sh clears every knob the engine reads before these are set,
# so nothing left in the shell can retarget the run. Only MUON_VARIANT (which arm),
# RESUME, DRY_RUN, SMOKE and the machine paths get through. To change the experiment,
# edit this file -- then the change is in git next to the results.
#
#   d12: dim 768, 6 heads x 128, 286,270,946 params (110,110,128 scaling params):
#   exp_d12_softplus.sh plus one 768-wide RMSNorm gain per layer
#   12.0B tokens = 42 per parameter, ~22,890 steps
#
# Compare against exp_d12_ctrl.sh (softmax) and exp_d12_softplus.sh (softplus scaled by
# n_i^-alpha) with the same MUON_VARIANT. Here the softplus output is not length-scaled but
# RMS-normalised per head, with a learned per-head gain (AdamW, lr scalar_lr*0.01, no weight
# decay) folded into attn.c_proj; under FP8 the gain scales c_proj's input instead, so the
# projection stays FP8 as in the other arms (see nanochat/gpt.py).
#
# Kernel: FA4 CuTe from the flash-attention repo (flash_attn.cute, editable install), SM80 /
# SM120 only -- see nanochat/softplus_rmsnorm_attention.py. NANOCHAT_SOFTPLUS_IMPL and
# SOFTPLUS_ALPHA do not apply to this score map.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
source trains/_exp_common.sh                        # clears everything below this line's control

# The variable under test. Everything else is copied from exp_d12_ctrl.sh so the arms
# are the same run apart from the score map; keep them in step if either changes.
export ATTN_KIND=softplus_rmsnorm

export TARGET_PARAM_DATA_RATIO=109              # 12.0B tokens
export DEVICE_BATCH_SIZE=128                        # throughput only; grad accum keeps the step fixed
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=5000
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
EXP_TOKENS=12.0B

run_experiment 12 "$@"
