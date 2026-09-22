#!/usr/bin/env bash
# =============================================================================
# Control run: d12 (286,261,730 params), hybrid SWA, FP8, Muon.
#
#   bash trains/exp_d12_ctrl.sh                       # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d12_ctrl.sh  # baseline arm
#   RESUME=<run_id> bash trains/exp_d12_ctrl.sh        # continue after a restart
#   DRY_RUN=1 bash trains/exp_d12_ctrl.sh              # print the config, do not train
#
# The token budget is an absolute 12.0B, not a ratio carried over from the d20 run.
# That follows what the architecture literature actually does: a cheap fixed config for
# the ablation sweep and a separate, larger one for the headline result -- 340M/15B and
# 1.3B/100B in GLA, Gated Slot Attention and the DeltaNet line, 600M/15B and 2B/100B
# elsewhere. Those two tiers do not share a tokens-per-parameter ratio (44 vs 77), and
# no one fits a scaling law across them; each tier is compared within itself.
#
# So: 12.0B here (42 tokens/param, ~22,890 steps, about 2.7 h on one B200 at the
# 937 TFLOPS measured there) against 60B for d20. What must hold is that both arms at
# *this* size see the same budget, which is why it is pinned here rather than exported
# by hand. What this tier cannot support is a precise claim about how an effect scales
# between sizes, since duration varies too; "the gain is present at both scales" is the
# statement this design buys.
#
#   d12: dim 768, 6 heads x 128, 286,261,730 params (110,100,912 scaling params)
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

unset NUM_ITERATIONS RUN_NAME TARGET_FLOPS 2>/dev/null || true

export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
export FP8="${FP8:-1}"
export FP8_RECIPE="${FP8_RECIPE:-tensorwise}"
export TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-109}"   # 12.0B tokens
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
export SAVE_EVERY="${SAVE_EVERY:-5000}"
export EVAL_EVERY="${EVAL_EVERY:-1000}"
export DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-250}"

if [ -n "${RESUME:-}" ]; then
    exec bash trains/resume_latest.sh "$RESUME" --keep 2 "$@"
fi
exec bash trains/train_d12_hybrid_swa_muon.sh "$@"
