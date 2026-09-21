#!/usr/bin/env bash
# =============================================================================
# d24 hybrid-SWA + Muon. Thin wrapper over trains/_engine_hybrid_swa_muon.sh,
# which holds all the machinery; this file carries only what is specific to this
# size. Everything here is still overridable from the environment.
#
#   bash trains/train_d24_hybrid_swa_muon.sh
#   DRY_RUN=1 bash trains/train_d24_hybrid_swa_muon.sh     # print the config, do not train
#   SMOKE=1   bash trains/train_d24_hybrid_swa_muon.sh     # 20-step pipeline check
#   setsid nohup bash trains/train_d24_hybrid_swa_muon.sh > /dev/null 2>&1 &
#
# Shape and cost (seq_len 2048, data:param ratio 12, window_pattern SSSL):
#   model     dim 1536, 12 heads x 128, 1.38B parameters (730M "scaling" params)
#   horizon   8.76B tokens -> ~8,352 steps at 1,048,576 tokens/step
#   on GB10   ~22 days
#   storage   ~9.1 GB per checkpoint; SAVE_EVERY=2000 keeps about 4
#
# nanochat's speedrun size; upstream trains it on 8xH100 in about 1.5 h.
# =============================================================================
set -euo pipefail

export DEPTH=24
export WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"

# Micro-batch. Throughput and memory only -- gradient accumulation keeps the tokens
# per optimizer step fixed, so the optimization is identical at any value.
# Measured on 1x GB10: bs 4 gives 4,681 tok/s at 22.3 GiB peak. Bigger is worse
# here -- GB10's memory is unified, and past ~25-30 GiB torch fights the page cache
# that streaming the parquet shards fills (bs 8 measured 2,876 tok/s, 1.6x slower).
# On a 180-288 GB datacenter card (B200/B300) start from 32 instead.
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"

# -1 lets base_train apply the scaling law (Power Lines, B ~ D^0.383), giving
# 1,048,576 tokens/step here. Pinning it to chase speed changes the optimization:
# base_train derives the LR and weight-decay scaling from it.
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:--1}"

export SAVE_EVERY="${SAVE_EVERY:-2000}"
export EVAL_EVERY="${EVAL_EVERY:-250}"
export DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-50}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/_engine_hybrid_swa_muon.sh" "$@"
