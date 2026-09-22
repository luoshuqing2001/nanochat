#!/usr/bin/env bash
# =============================================================================
# Experiment: d20 (0.90B params) hybrid SWA, 60B tokens, FP8, Muon.
#
#   bash trains/exp_d20_60b.sh                      # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d20_60b.sh # baseline arm
#   RESUME=<run_id> bash trains/exp_d20_60b.sh       # continue after a container restart
#   DRY_RUN=1 bash trains/exp_d20_60b.sh             # print the config, do not train
#
# Why a file rather than a pile of exports: the knobs below have to be identical
# across the two arms and across every resume, and an environment variable left over
# in a shell (NUM_ITERATIONS, most easily) silently changes the experiment. This
# pins them in one place, and `unset` below clears the ones that must not leak in.
#
#   d20: dim 1280, 10 heads x 128, 896,533,746 params (435M scaling params)
#   60B tokens = ratio 138, ~57,270 steps at 1,048,576 tokens/step
#   67 tokens/param, i.e. the same over-training depth the 1.3B/100B architecture
#   papers use, scaled to this size
#   single epoch: needs ~1,340 shards of the 2,479 on disk
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# must not leak in from the shell -- these override the horizon and the run name
unset NUM_ITERATIONS RUN_NAME TARGET_FLOPS 2>/dev/null || true

export MUON_VARIANT="${MUON_VARIANT:-moonlight}"   # the variable under test
export FP8="${FP8:-1}"
export FP8_RECIPE="${FP8_RECIPE:-tensorwise}"
export TARGET_PARAM_DATA_RATIO=138                 # 138 x 435,160,240 = 60.05B tokens
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
export SAVE_EVERY="${SAVE_EVERY:-3000}"            # ~every 4 h; keeps a crash cheap
export EVAL_EVERY="${EVAL_EVERY:-2000}"
export DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-500}"

if [ -n "${RESUME:-}" ]; then
    exec bash trains/resume_latest.sh "$RESUME" --keep 2 "$@"
fi
exec bash trains/train_d20_hybrid_swa_muon.sh "$@"
