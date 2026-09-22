#!/usr/bin/env bash
# =============================================================================
# Control run: d12 (286,261,730 params), hybrid SWA, FP8, Muon.
#
#   bash trains/exp_d12_ctrl.sh                       # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d12_ctrl.sh  # baseline arm
#   RESUME=<run_id> bash trains/exp_d12_ctrl.sh        # continue after a restart
#   DRY_RUN=1 bash trains/exp_d12_ctrl.sh              # print the config, do not train
#
# The token budget defaults to the *same* tokens-per-scaling-param as the d20/60B run
# (TARGET_PARAM_DATA_RATIO=138), not to a cheaper compute-optimal point. A control
# trained for a different duration than the run it controls for cannot support any
# statement about how an effect moves with scale -- size and training length would be
# confounded. At 138 this size is 15.2B tokens, ~28,980 steps, about 3.4 h on one
# B200 at the 937 TFLOPS measured there, which is cheap next to the d20 run's ~51 h.
#
# Cheaper points, for screening only (no cross-size claim):
#   TARGET_PARAM_DATA_RATIO=12   compute-optimal, 1.3B / 0.3 h
#   TARGET_PARAM_DATA_RATIO=52   Chinchilla 20x total params, 5.7B / 1.3 h
#
#   d12: dim 768, 6 heads x 128, 286,261,730 params (110,100,912 scaling params)
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

unset NUM_ITERATIONS RUN_NAME TARGET_FLOPS 2>/dev/null || true

export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
export FP8="${FP8:-1}"
export FP8_RECIPE="${FP8_RECIPE:-tensorwise}"
export TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-138}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
export SAVE_EVERY="${SAVE_EVERY:-5000}"
export EVAL_EVERY="${EVAL_EVERY:-1000}"
export DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-250}"

if [ -n "${RESUME:-}" ]; then
    exec bash trains/resume_latest.sh "$RESUME" --keep 2 "$@"
fi
exec bash trains/train_d12_hybrid_swa_muon.sh "$@"
