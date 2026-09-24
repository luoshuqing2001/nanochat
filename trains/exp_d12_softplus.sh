#!/usr/bin/env bash
# =============================================================================
# Ablation: d12 hybrid SWA + Muon, 12.0B tokens, FP8, SOFTPLUS attention.
#
#   bash trains/exp_d12_softplus.sh                        # moonlight arm
#   MUON_VARIANT=nanochat bash trains/exp_d12_softplus.sh   # baseline arm
#   RESUME=<run_id> bash trains/exp_d12_softplus.sh         # continue after a container restart
#   DRY_RUN=1 bash trains/exp_d12_softplus.sh               # print the config, do not train
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
# The arm to compare against is exp_d12_ctrl.sh with the same MUON_VARIANT. Softplus
# replaces the softmax over each row with an elementwise softplus and no normalisation,
# output scaled by n_i^-alpha (see nanochat/softplus_attention.py). Expect this to be a
# question about quality -- length extrapolation, attention entropy -- not throughput:
# the kernel is 1.06x on attention, which is ~0.6% of a step.
#
# NANOCHAT_SOFTPLUS_IMPL picks the kernel (hybrid by default; fixed / fixed_triton
# force fixed KV/query tasks). This and the KV_CHUNK/Q_CHUNK/SPLITS overrides are
# cleared by _exp_common.sh; set them after sourcing it for a controlled experiment.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MUON_VARIANT="${MUON_VARIANT:-moonlight}"
source trains/_exp_common.sh                        # clears everything below this line's control

# The variable under test. Everything else is copied from exp_d12_ctrl.sh so the two
# are the same run apart from the score map; keep them in step if either changes.
export ATTN_KIND=softplus
export SOFTPLUS_ALPHA=1.0

export TARGET_PARAM_DATA_RATIO=109              # 12.0B tokens
export DEVICE_BATCH_SIZE=128                        # throughput only; grad accum keeps the step fixed
export FP8=1
export FP8_RECIPE=tensorwise
export SAVE_EVERY=5000
export EVAL_EVERY=1000
export DIAGNOSTICS_EVERY=250
EXP_TOKENS=12.0B

run_experiment 12 "$@"
