#!/usr/bin/env bash
# =============================================================================
# d48 hybrid-SWA + Muon. Thin wrapper over trains/_engine_hybrid_swa_muon.sh,
# which holds all the machinery; this file carries only what is specific to this
# size. Everything here is still overridable from the environment.
#
#   bash trains/train_d48_hybrid_swa_muon.sh
#   DRY_RUN=1 bash trains/train_d48_hybrid_swa_muon.sh     # print the config, do not train
#   SMOKE=1   bash trains/train_d48_hybrid_swa_muon.sh     # 20-step pipeline check
#   setsid nohup bash trains/train_d48_hybrid_swa_muon.sh > /dev/null 2>&1 &
#
# Shape and cost (seq_len 2048, data:param ratio 12, window_pattern SSSL):
#   model     dim 3072, 24 heads x 128, 8.05B parameters (5.54B "scaling" params)
#   horizon   66.4B tokens -> ~31,680 steps at 2,097,152 tokens/step
#   on GB10   not runnable on a GB10
#   storage   ~53.0 GB per checkpoint; SAVE_EVERY=8000 keeps about 3
#
# Does not fit a GB10: fp32 master weights + gradients + Muon momentum are about
# 78 GB of its 121 GB unified memory, before activations, and that memory is shared
# with the page cache. This needs a 180-288 GB datacenter card (B200/B300) or sharding.
# =============================================================================
set -euo pipefail

export DEPTH=48
export WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"

# Micro-batch. Throughput and memory only -- gradient accumulation keeps the tokens
# per optimizer step fixed, so the optimization is identical at any value.
# Not measured: this size is impractical on the GB10 this repo was tuned on. The value
# below targets a 180-288 GB datacenter card; confirm it with SMOKE=1 before a real run,
# and watch peak memory in the log.
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"

# -1 lets base_train apply the scaling law (Power Lines, B ~ D^0.383), giving
# 2,097,152 tokens/step here. Pinning it to chase speed changes the optimization:
# base_train derives the LR and weight-decay scaling from it.
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:--1}"

export SAVE_EVERY="${SAVE_EVERY:-8000}"
export EVAL_EVERY="${EVAL_EVERY:-250}"
export DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-50}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/_engine_hybrid_swa_muon.sh" "$@"
