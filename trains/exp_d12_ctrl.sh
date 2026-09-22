#!/usr/bin/env bash
# =============================================================================
# Ablation: d12 hybrid SWA + Muon, 12.0B tokens, FP8.
#
#   bash trains/exp_d12_ctrl.sh                        # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d12_ctrl.sh   # baseline arm
#   RESUME=<run_id> bash trains/exp_d12_ctrl.sh         # continue after a container restart
#   DRY_RUN=1 bash trains/exp_d12_ctrl.sh               # print the config, do not train
#
# Hermetic: _exp_common.sh clears every knob the engine reads before these are set,
# so nothing left in the shell can retarget the run. Only MUON_VARIANT (which arm),
# RESUME, DRY_RUN, SMOKE and the machine paths get through. To change the experiment,
# edit this file -- then the change is in git next to the results.
#
#   d12: dim 768, 6 heads x 128, 286,261,730 params (110,100,912 scaling params)
#   12.0B tokens = 42 per parameter, ~22,890 steps, about 2.7 h on one B200
#   at the 937 TFLOPS measured there
#
# The ablation tier (d12, d16) and the headline run (d20) deliberately use different
# budgets, as the architecture literature does: 340M/15B with 1.3B/100B in GLA, Gated
# Slot Attention and the DeltaNet line. Those tiers differ in tokens per parameter by
# design; what must match is the two arms *within* a size, which is what this file
# guarantees. The claim such a series supports is "the effect holds at both scales",
# not a scaling slope.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MUON_VARIANT="${MUON_VARIANT:-moonlight}"   # the variable under test
source trains/_exp_common.sh                        # clears everything below this line's control

export TARGET_PARAM_DATA_RATIO=109              # 12.0B tokens
export DEVICE_BATCH_SIZE=128                        # throughput only; grad accum keeps the step fixed
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=5000
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
EXP_TOKENS=12.0B

run_experiment 12 "$@"
