#!/usr/bin/env bash
# =============================================================================
# nanochat: pretrain a 12-layer hybrid-SWA transformer with the Muon optimizer
# on the local ClimbMix shards, writing every artifact of the experiment into
# its own folder under nanochat/logs/<RUN_NAME>_<timestamp>/.
#
# Self-contained: this file is the whole experiment launcher (config, preflight,
# provenance, training, evaluation, log parsing, summary).
#
#   bash trains/train_d12_hybrid_swa_muon.sh                  # the real run (~13-14h)
#   DRY_RUN=1 bash trains/train_d12_hybrid_swa_muon.sh        # write config/env, print cmd, don't train
#   SMOKE=1 bash trains/train_d12_hybrid_swa_muon.sh          # 20-step pipeline check (~1 min)
#   NUM_ITERATIONS=1000 bash trains/train_d12_hybrid_swa_muon.sh
#   WANDB_RUN=d12_hybrid_swa bash trains/train_d12_hybrid_swa_muon.sh
#   bash trains/train_d12_hybrid_swa_muon.sh --matrix-lr=0.03 # extra args -> base_train.py
#   bash trains/train_d12_hybrid_swa_muon.sh --summarize logs/<run_id>   # re-parse a run
#
# For a long run, keep it alive in screen:
#   screen -L -Logfile trains/d12_swa.screenlog -S d12swa \
#       bash trains/train_d12_hybrid_swa_muon.sh
#
# What the flags mean:
#   * 12 layers          --depth 12 => model_dim 12*64=768, 6 heads x 128
#   * hybrid SWA         --window-pattern SSSL, tiled across layers: S = sliding
#     window of ceil(seq_len/4) rounded to 128 (512 tokens at seq_len 2048),
#     L = full context. gpt.py always forces the final layer to L.
#   * Muon               needs no flag: GPT.setup_optimizer() builds a MuonAdamW
#     that runs Muon on every transformer matrix and AdamW on embeddings/scalars.
#     MATRIX_LR / WEIGHT_DECAY are the Muon-side hyperparameters (weight decay is
#     cosine-decayed to 0 and Muon momentum is scheduled 0.85 -> 0.97 -> 0.90).
#
# Data: read straight from $REPO_DIR/base_data_climbmix via NANOCHAT_DATA_DIR,
# which nanochat/dataset.py honors. NANOCHAT_BASE_DIR (default ~/.cache/nanochat)
# stays the home of the tokenizer, the checkpoints and the CORE eval bundle.
#
# Measured on 1x NVIDIA GB10 (torch 2.14+cu130, bf16, SDPA fallback since FA3 has
# no sm121 kernels): ~28.9k tok/s at DEVICE_BATCH_SIZE=16, ~15 GiB peak. bs=32 is
# slower (~24.5k). The default compute-optimal horizon is 2,682 steps x 524,288
# tokens = 1.41B tokens, i.e. roughly 13-14 hours.
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

# -----------------------------------------------------------------------------
# summarize(): parse a run's stdout into metrics.jsonl / summary.json / summary.md
# -----------------------------------------------------------------------------
summarize() {
    "$PYTHON_BIN" - "$1" <<'PYEOF'
import json, os, re, sys

run_dir = sys.argv[1]
train_log = os.path.join(run_dir, "train.log")
if not os.path.exists(train_log):
    print(f"no train.log in {run_dir}, nothing to summarize")
    raise SystemExit(0)

def _f(s):
    return float(s.replace(",", ""))

STEP_RE = re.compile(
    r"step (\d+)/(\d+) \(([\d.]+)%\) \| loss: ([\d.]+) \| lrm: ([\d.]+) \| "
    r"dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+) \| "
    r"epoch: (\d+) pq: (\d+) rg: (\d+) \| total time: ([\d.]+)m")
VAL_RE = re.compile(r"Step (\d+) \| Validation bpb: ([\d.]+)")
CORE_RE = re.compile(r"Step (\d+) \| CORE metric: ([\d.]+)")
PARAM_RE = re.compile(r"^(\w+)\s*:\s*([\d,]+)$")
SCALARS = [
    ("gpu", re.compile(r"^GPU: (.+?) \| Peak FLOPS"), str),
    ("compute_dtype", re.compile(r"^COMPUTE_DTYPE: (\S+)"), str),
    ("data_dir", re.compile(r"DATA_DIR -> (\S+)"), str),
    ("vocab_size", re.compile(r"^Vocab size: ([\d,]+)"), _f),
    ("flops_per_token", re.compile(r"^Estimated FLOPs per token: (\S+)"), float),
    ("total_batch_size", re.compile(r"^Auto-computed optimal batch size: ([\d,]+)"), _f),
    ("num_iterations", re.compile(r"number of iterations.*?: ([\d,]+)"), _f),
    ("total_tokens", re.compile(r"^Total number of training tokens: ([\d,]+)"), _f),
    ("tokens_per_scaling_param", re.compile(r"^Tokens : Scaling params ratio: ([\d.]+)"), float),
    ("total_train_flops", re.compile(r"^Total training FLOPs estimate: (\S+)"), float),
    ("grad_accum_steps", re.compile(r"gradient accumulation steps: (\d+)"), _f),
    ("peak_memory_mib", re.compile(r"^Peak memory usage: ([\d.]+)MiB"), float),
    ("train_time_min", re.compile(r"^Total training time: ([\d.]+)m"), float),
    ("min_val_bpb", re.compile(r"^Minimum validation bpb: ([\d.]+)"), float),
]

records, scalars, params = [], {}, {}
cfg_lines, in_cfg, in_params = [], False, False
with open(train_log, errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("Model config:"):
            in_cfg = True
            continue
        if in_cfg:
            cfg_lines.append(line)
            if line.startswith("}"):
                in_cfg = False
            continue
        if line.startswith("Parameter counts:"):
            in_params = True
            continue
        if in_params:
            m = PARAM_RE.match(line.strip())
            if m:
                params[m.group(1)] = int(m.group(2).replace(",", ""))
                continue
            in_params = False
        m = STEP_RE.search(line)
        if m:
            records.append({"type": "train", "step": int(m.group(1)), "num_iterations": int(m.group(2)),
                            "pct_done": float(m.group(3)), "train_loss_ema": float(m.group(4)),
                            "lr_mult": float(m.group(5)), "dt_ms": float(m.group(6)),
                            "tok_per_sec": int(m.group(7).replace(",", "")), "mfu_pct": float(m.group(8)),
                            "epoch": int(m.group(9)), "parquet_idx": int(m.group(10)),
                            "row_group_idx": int(m.group(11)), "elapsed_min": float(m.group(12))})
            continue
        m = VAL_RE.search(line)
        if m:
            records.append({"type": "val", "step": int(m.group(1)), "val_bpb": float(m.group(2))})
            continue
        m = CORE_RE.search(line)
        if m:
            records.append({"type": "core", "step": int(m.group(1)), "core_metric": float(m.group(2))})
            continue
        for key, rx, cast in SCALARS:
            if key not in scalars:
                m = rx.search(line)
                if m:
                    scalars[key] = cast(m.group(1))

if cfg_lines:
    try:
        scalars["model_config"] = json.loads("\n".join(cfg_lines))
    except json.JSONDecodeError:
        pass
if params:
    scalars["param_counts"] = params

eval_log = os.path.join(run_dir, "eval.log")
if os.path.exists(eval_log):
    with open(eval_log, errors="replace") as f:
        for line in f:
            m = re.search(r"^(train|val) bpb: ([\d.]+)", line.strip())
            if m:
                scalars[f"eval_{m.group(1)}_bpb"] = float(m.group(2))
            m = re.search(r"^CORE metric: ([\d.]+)", line.strip())
            if m:
                scalars["eval_core_metric"] = float(m.group(1))

with open(os.path.join(run_dir, "metrics.jsonl"), "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

train_recs = [r for r in records if r["type"] == "train"]
val_recs = [r for r in records if r["type"] == "val"]
core_recs = [r for r in records if r["type"] == "core"]
summary = dict(scalars)
summary["steps_logged"] = len(train_recs)
if train_recs:
    summary["last_step"] = train_recs[-1]["step"]
    summary["final_train_loss_ema"] = train_recs[-1]["train_loss_ema"]
    steady = [r for r in train_recs if r["step"] > 10] or train_recs  # skip compile/warmup
    summary["median_dt_ms"] = sorted(r["dt_ms"] for r in steady)[len(steady) // 2]
    summary["median_tok_per_sec"] = sorted(r["tok_per_sec"] for r in steady)[len(steady) // 2]
if val_recs:
    summary["final_val_bpb"] = val_recs[-1]["val_bpb"]
    summary["best_val_bpb"] = min(r["val_bpb"] for r in val_recs)
if core_recs:
    summary["final_core_metric"] = core_recs[-1]["core_metric"]
with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

def fmt(v):
    if isinstance(v, float) and v.is_integer():
        return f"{int(v):,}"
    if isinstance(v, float):
        return f"{v:,.6g}"
    return str(v)

L = [f"# Run summary: {os.path.basename(os.path.abspath(run_dir))}", ""]
cfg = summary.get("model_config", {})
if cfg:
    L += ["## Model", "",
          f"- layers (n_layer): {cfg.get('n_layer')}",
          f"- model dim (n_embd): {cfg.get('n_embd')}, heads: {cfg.get('n_head')} (kv: {cfg.get('n_kv_head')})",
          f"- sequence_len: {cfg.get('sequence_len')}, window_pattern: `{cfg.get('window_pattern')}` (hybrid SWA)",
          f"- vocab_size: {cfg.get('vocab_size')}", ""]
if params:
    L += ["## Parameters", ""] + [f"- {k}: {v:,}" for k, v in params.items()] + [""]
L += ["## Training horizon", ""]
for k, label in [("total_batch_size", "total batch size (tokens)"), ("grad_accum_steps", "grad accum steps"),
                 ("num_iterations", "iterations"), ("total_tokens", "total training tokens"),
                 ("tokens_per_scaling_param", "tokens : scaling params"),
                 ("total_train_flops", "estimated training FLOPs")]:
    if k in summary:
        L.append(f"- {label}: {fmt(summary[k])}")
L += ["", "## Results", ""]
for k, label in [("final_train_loss_ema", "final train loss (EMA)"), ("final_val_bpb", "final val bpb"),
                 ("best_val_bpb", "best val bpb"), ("min_val_bpb", "min val bpb (trainer)"),
                 ("final_core_metric", "CORE metric (during training)"), ("eval_val_bpb", "val bpb (base_eval)"),
                 ("eval_train_bpb", "train bpb (base_eval)"), ("eval_core_metric", "CORE metric (base_eval)")]:
    if k in summary:
        L.append(f"- {label}: {fmt(summary[k])}")
L += ["", "## Throughput", ""]
for k, label in [("gpu", "GPU"), ("compute_dtype", "compute dtype"), ("data_dir", "data dir"),
                 ("median_dt_ms", "median step time (ms)"), ("median_tok_per_sec", "median tokens/sec"),
                 ("peak_memory_mib", "peak memory (MiB)"), ("train_time_min", "total training time (min)")]:
    if k in summary:
        L.append(f"- {label}: {fmt(summary[k])}")
L += ["", "Raw per-step metrics: `metrics.jsonl`. Full stdout: `train.log`.", ""]
with open(os.path.join(run_dir, "summary.md"), "w") as f:
    f.write("\n".join(L))
print(f"wrote {run_dir}/metrics.jsonl, summary.json, summary.md ({len(records)} records)")
PYEOF
}

# re-parse an existing run folder and exit
if [ "${1:-}" = "--summarize" ]; then
    [ -n "${2:-}" ] || { echo "usage: $0 --summarize <run_dir>" >&2; exit 2; }
    summarize "$2"
    exit 0
fi

# -----------------------------------------------------------------------------
# Configuration (every knob is an env var override)
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# stdout is a pipe here (we tee into train.log), so python would block-buffer it in 8KB
# chunks and the log would lag ~50 step lines behind. Keep it line-buffered instead.
export PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"   # tokenizer, checkpoints, eval bundle
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$REPO_DIR/base_data_climbmix}"  # ClimbMix shards (dataset.py honors this)
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# model
DEPTH="${DEPTH:-12}"                             # 12 layers
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"         # hybrid sliding-window attention
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
HEAD_DIM="${HEAD_DIM:-128}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"               # model_dim = depth * aspect_ratio

# optimization (Muon on matrices, AdamW on the rest)
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"     # measured best on 1x GB10; lower to 8/4 if you OOM
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"   # tokens per optimizer step (-1 = auto)
MATRIX_LR="${MATRIX_LR:-0.02}"                   # Muon lr
WEIGHT_DECAY="${WEIGHT_DECAY:-0.28}"             # Muon cautious weight decay
EMBEDDING_LR="${EMBEDDING_LR:-0.3}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.008}"
SCALAR_LR="${SCALAR_LR:-0.5}"
WARMUP_STEPS="${WARMUP_STEPS:-40}"
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.65}"

# horizon: NUM_ITERATIONS wins when > 0, otherwise the data:param ratio decides
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-12}"

# evaluation / checkpointing during training
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_TOKENS="${EVAL_TOKENS:-2097152}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"     # needs the eval_bundle download, off by default

# post-training evaluation ("none" to skip); add ",core" to also run CORE
FINAL_EVAL="${FINAL_EVAL:-bpb,sample}"
FINAL_EVAL_SPLIT_TOKENS="${FINAL_EVAL_SPLIT_TOKENS:-2097152}"
FINAL_EVAL_MAX_PER_TASK="${FINAL_EVAL_MAX_PER_TASK:-200}"

WANDB_RUN="${WANDB_RUN:-dummy}"                  # "dummy" = no wandb

if [ "${SMOKE:-0}" = "1" ]; then                 # quick pipeline check
    NUM_ITERATIONS="${SMOKE_ITERS:-20}"
    TOTAL_BATCH_SIZE=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC_PER_NODE))
    EVAL_EVERY=10
    EVAL_TOKENS=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC_PER_NODE * 4))
    SAMPLE_EVERY=-1
    SAVE_EVERY=-1
    FINAL_EVAL="none"
    RUN_NAME="${RUN_NAME:-d${DEPTH}_hybridswa_muon_smoke}"
fi

RUN_NAME="${RUN_NAME:-d${DEPTH}_hybridswa_muon}"
RUN_ID="${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_DIR/logs/$RUN_ID"
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$RUN_ID"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
NUM_SHARDS="$(ls "$NANOCHAT_DATA_DIR"/*.parquet 2>/dev/null | wc -l)"
if [ "$NUM_SHARDS" -eq 0 ]; then
    cat >&2 <<MSG
ERROR: no *.parquet shards in the pretraining data dir:
  $NANOCHAT_DATA_DIR
Override it with: NANOCHAT_DATA_DIR=/path/to/shards bash trains/$(basename "$0")
MSG
    exit 1
fi

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
    echo "ERROR: no tokenizer at $NANOCHAT_BASE_DIR/tokenizer (train one with: python -m scripts.tok_train)" >&2
    exit 1
fi

for mod in torch pyarrow tiktoken rustbpe; do
    "$PYTHON_BIN" -c "import $mod" >/dev/null 2>&1 || { echo "ERROR: python module '$mod' is missing (pip install $mod)" >&2; exit 1; }
done

# base_train.py imports wandb unconditionally; fall back to the local stub if it is
# absent and wandb logging is disabled anyway.
if ! "$PYTHON_BIN" -c "import wandb" >/dev/null 2>&1; then
    if [ "$WANDB_RUN" != "dummy" ]; then
        echo "ERROR: WANDB_RUN=$WANDB_RUN but wandb is not installed (pip install wandb)" >&2
        exit 1
    fi
    export PYTHONPATH="$REPO_DIR/trains/_stubs${PYTHONPATH:+:$PYTHONPATH}"
    log "wandb not installed -> using trains/_stubs/wandb.py (logging disabled)"
fi

mkdir -p "$RUN_DIR"

# -----------------------------------------------------------------------------
# Provenance
# -----------------------------------------------------------------------------
{
    echo "run_id: $RUN_ID"
    echo "date: $(date -Iseconds)"
    echo "host: $(hostname)"
    echo "repo: $REPO_DIR"
    echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
    echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
    echo "git_dirty_files:"
    git status --short 2>/dev/null | sed 's/^/  /'
    echo "python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
    echo "torch: $("$PYTHON_BIN" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
    echo "nanochat_base_dir: $NANOCHAT_BASE_DIR"
    echo "data_dir: $NANOCHAT_DATA_DIR ($NUM_SHARDS parquet shards)"
    echo "checkpoint_dir: $CKPT_DIR"
    echo "nproc_per_node: $NPROC_PER_NODE"
    echo "gpus:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv 2>/dev/null | sed 's/^/  /'
} > "$RUN_DIR/env.txt"

cat > "$RUN_DIR/config.json" <<JSON
{
  "run_id": "$RUN_ID",
  "script": "trains/$(basename "$0")",
  "depth": $DEPTH,
  "window_pattern": "$WINDOW_PATTERN",
  "max_seq_len": $MAX_SEQ_LEN,
  "head_dim": $HEAD_DIM,
  "aspect_ratio": $ASPECT_RATIO,
  "optimizer": "MuonAdamW (Muon on matrices, AdamW on embeddings/scalars)",
  "device_batch_size": $DEVICE_BATCH_SIZE,
  "total_batch_size": $TOTAL_BATCH_SIZE,
  "matrix_lr": $MATRIX_LR,
  "weight_decay": $WEIGHT_DECAY,
  "embedding_lr": $EMBEDDING_LR,
  "unembedding_lr": $UNEMBEDDING_LR,
  "scalar_lr": $SCALAR_LR,
  "warmup_steps": $WARMUP_STEPS,
  "warmdown_ratio": $WARMDOWN_RATIO,
  "num_iterations": $NUM_ITERATIONS,
  "target_param_data_ratio": $TARGET_PARAM_DATA_RATIO,
  "eval_every": $EVAL_EVERY,
  "eval_tokens": $EVAL_TOKENS,
  "sample_every": $SAMPLE_EVERY,
  "save_every": $SAVE_EVERY,
  "core_metric_every": $CORE_METRIC_EVERY,
  "final_eval": "$FINAL_EVAL",
  "nproc_per_node": $NPROC_PER_NODE,
  "wandb_run": "$WANDB_RUN",
  "nanochat_base_dir": "$NANOCHAT_BASE_DIR",
  "data_dir": "$NANOCHAT_DATA_DIR",
  "num_shards": $NUM_SHARDS,
  "checkpoint_dir": "$CKPT_DIR"
}
JSON

# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
TRAIN_ARGS=(
    --depth="$DEPTH"
    --aspect-ratio="$ASPECT_RATIO"
    --head-dim="$HEAD_DIM"
    --max-seq-len="$MAX_SEQ_LEN"
    --window-pattern="$WINDOW_PATTERN"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --total-batch-size="$TOTAL_BATCH_SIZE"
    --matrix-lr="$MATRIX_LR"
    --weight-decay="$WEIGHT_DECAY"
    --embedding-lr="$EMBEDDING_LR"
    --unembedding-lr="$UNEMBEDDING_LR"
    --scalar-lr="$SCALAR_LR"
    --warmup-steps="$WARMUP_STEPS"
    --warmdown-ratio="$WARMDOWN_RATIO"
    --eval-every="$EVAL_EVERY"
    --eval-tokens="$EVAL_TOKENS"
    --sample-every="$SAMPLE_EVERY"
    --save-every="$SAVE_EVERY"
    --core-metric-every="$CORE_METRIC_EVERY"
    --model-tag="$RUN_ID"
    --run="$WANDB_RUN"
)
if [ "$NUM_ITERATIONS" -gt 0 ]; then
    TRAIN_ARGS+=(--num-iterations="$NUM_ITERATIONS")
else
    TRAIN_ARGS+=(--target-param-data-ratio="$TARGET_PARAM_DATA_RATIO")
fi
TRAIN_ARGS+=("$@")   # pass-through overrides

if [ "$NPROC_PER_NODE" -gt 1 ]; then
    LAUNCH=("torchrun" "--standalone" "--nproc_per_node=$NPROC_PER_NODE" "-m" "scripts.base_train" "--")
else
    LAUNCH=("$PYTHON_BIN" "-m" "scripts.base_train")
fi

log "run dir:  $RUN_DIR"
log "data dir: $NANOCHAT_DATA_DIR ($NUM_SHARDS shards)"
log "ckpt dir: $CKPT_DIR"
log "command:  ${LAUNCH[*]} ${TRAIN_ARGS[*]}"
printf '%s\n' "${LAUNCH[*]} ${TRAIN_ARGS[*]}" > "$RUN_DIR/command.txt"

if [ "${DRY_RUN:-0}" = "1" ]; then
    log "DRY_RUN=1 -> config.json / env.txt / command.txt written, not training."
    exit 0
fi

START_TIME=$(date +%s)
set +e
"${LAUNCH[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
TRAIN_SECONDS=$(( $(date +%s) - START_TIME ))
log "training exited with code $TRAIN_STATUS after $((TRAIN_SECONDS / 3600))h $(((TRAIN_SECONDS % 3600) / 60))m"

# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------
if [ "$TRAIN_STATUS" -eq 0 ] && [ -n "$FINAL_EVAL" ] && [ "$FINAL_EVAL" != "none" ]; then
    EVAL_ARGS=(
        --model-tag="$RUN_ID"
        --eval="$FINAL_EVAL"
        --device-batch-size="$DEVICE_BATCH_SIZE"
        --split-tokens="$FINAL_EVAL_SPLIT_TOKENS"
        --max-per-task="$FINAL_EVAL_MAX_PER_TASK"
    )
    if [ "$NPROC_PER_NODE" -gt 1 ]; then
        EVAL_LAUNCH=("torchrun" "--standalone" "--nproc_per_node=$NPROC_PER_NODE" "-m" "scripts.base_eval" "--")
    else
        EVAL_LAUNCH=("$PYTHON_BIN" "-m" "scripts.base_eval")
    fi
    log "evaluating: ${EVAL_LAUNCH[*]} ${EVAL_ARGS[*]}"
    set +e
    "${EVAL_LAUNCH[@]}" "${EVAL_ARGS[@]}" 2>&1 | tee "$RUN_DIR/eval.log"
    log "evaluation exited with code ${PIPESTATUS[0]}"
    set -e
fi

# -----------------------------------------------------------------------------
# Collect results
# -----------------------------------------------------------------------------
if [ -d "$CKPT_DIR" ]; then
    mkdir -p "$RUN_DIR/checkpoint_meta"
    cp "$CKPT_DIR"/meta_*.json "$RUN_DIR/checkpoint_meta/" 2>/dev/null || true
    du -sh "$CKPT_DIR" > "$RUN_DIR/checkpoint_meta/checkpoint_dir_size.txt" 2>/dev/null || true
    echo "$CKPT_DIR" > "$RUN_DIR/checkpoint_path.txt"
fi

summarize "$RUN_DIR" || true

log "all artifacts in: $RUN_DIR"
ls -la "$RUN_DIR"
exit "$TRAIN_STATUS"
