#!/usr/bin/env bash
# =============================================================================
# Control run: d16 (536,871,738 params), hybrid SWA, FP8, Muon.
#
#   bash trains/exp_d16_ctrl.sh                       # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d16_ctrl.sh  # baseline arm
#   RESUME=<run_id> bash trains/exp_d16_ctrl.sh        # continue after a restart
#   DRY_RUN=1 bash trains/exp_d16_ctrl.sh              # print the config, do not train
#
# The token budget defaults to the *same* tokens-per-scaling-param as the d20/60B run
# (TARGET_PARAM_DATA_RATIO=138), not to a cheaper compute-optimal point. A control
# trained for a different duration than the run it controls for cannot support any
# statement about how an effect moves with scale -- size and training length would be
# confounded. At 138 this size is 32.4B tokens, ~61,824 steps, about 15.2 h on one
# B200 at the 937 TFLOPS measured there, which is cheap next to the d20 run's ~51 h.
#
# Cheaper points, for screening only (no cross-size claim):
#   TARGET_PARAM_DATA_RATIO=12   compute-optimal, 2.8B / 1.3 h
#   TARGET_PARAM_DATA_RATIO=46   Chinchilla 20x total params, 10.8B / 5.1 h
#
#   d16: dim 1024, 8 heads x 128, 536,871,738 params (234,881,792 scaling params)
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

unset NUM_ITERATIONS RUN_NAME TARGET_FLOPS 2>/dev/null || true

export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
export FP8="${FP8:-1}"
export FP8_RECIPE="${FP8_RECIPE:-tensorwise}"
export TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-138}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export EVAL_EVERY="${EVAL_EVERY:-2000}"
export DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-500}"

if [ -n "${RESUME:-}" ]; then
    exec bash trains/resume_latest.sh "$RESUME" --keep 2 "$@"
fi
exec bash trains/train_d16_hybrid_swa_muon.sh "$@"
