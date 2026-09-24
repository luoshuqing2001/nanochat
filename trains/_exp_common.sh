# Sourced by the exp_*.sh experiment definitions. Makes a run hermetic.
#
# Every knob the engine reads is cleared here, so nothing an earlier command left in
# the shell can retarget the run. This has gone wrong three times: NUM_ITERATIONS=300
# turning a 100B run into 300 steps (twice), and TARGET_PARAM_DATA_RATIO=137 from a d24
# session turning a d12 control into 15.08B tokens instead of 12B. An experiment
# definition that a stray export can change is not a definition.
#
# What survives, because it describes the machine rather than the experiment:
#   NANOCHAT_BASE_DIR, NANOCHAT_DATA_DIR, PYTHON_BIN, NPROC_PER_NODE, OMP_NUM_THREADS
# What survives, because it selects which run to make:
#   MUON_VARIANT (which arm), RESUME (continue a run), DRY_RUN, SMOKE (quick checks)

unset DEPTH WINDOW_PATTERN MAX_SEQ_LEN HEAD_DIM ASPECT_RATIO \
      DEVICE_BATCH_SIZE TOTAL_BATCH_SIZE \
      MATRIX_LR WEIGHT_DECAY EMBEDDING_LR UNEMBEDDING_LR SCALAR_LR \
      WARMUP_STEPS WARMDOWN_RATIO \
      NUM_ITERATIONS TARGET_PARAM_DATA_RATIO TARGET_FLOPS \
      EVAL_EVERY EVAL_TOKENS SAMPLE_EVERY SAVE_EVERY CORE_METRIC_EVERY DIAGNOSTICS_EVERY \
      FINAL_EVAL FINAL_EVAL_SPLIT_TOKENS FINAL_EVAL_MAX_PER_TASK \
      LOSS_CHUNK_TOKENS FP8 FP8_RECIPE RUN_NAME \
      ATTN_KIND SOFTPLUS_ALPHA NANOCHAT_SOFTPLUS_IMPL NANOCHAT_SOFTPLUS_SPLITS \
      NANOCHAT_SOFTPLUS_KV_CHUNK NANOCHAT_SOFTPLUS_Q_CHUNK 2>/dev/null || true

run_experiment() {
    local depth="$1"; shift
    echo "experiment: d${depth} | arm '${MUON_VARIANT}' | ${EXP_TOKENS} tokens"\
         "(ratio ${TARGET_PARAM_DATA_RATIO}) | bs ${DEVICE_BATCH_SIZE} | FP8 ${FP8}"\
         "| attn ${ATTN_KIND:-softmax}"
    if [ -n "${RESUME:-}" ]; then
        exec bash trains/resume_latest.sh "$RESUME" --keep 2 "$@"
    fi
    exec bash "trains/train_d${depth}_hybrid_swa_muon.sh" "$@"
}
