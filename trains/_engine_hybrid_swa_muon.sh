#!/usr/bin/env bash
# =============================================================================
# nanochat: pretrain a hybrid-SWA transformer with the Muon optimizer on the local
# ClimbMix shards, writing every artifact of the experiment into its own folder
# under nanochat/logs/<RUN_NAME>_<timestamp>/.
#
# This is the shared engine (config, preflight, provenance, training, evaluation,
# log parsing, summary). Do not run it directly -- use a per-size wrapper, which
# sets DEPTH and the knobs measured for that size:
#
#     trains/train_d12_hybrid_swa_muon.sh ... trains/train_d48_hybrid_swa_muon.sh
#
# Every knob below is an environment variable, so the wrappers are a few exports
# and an exec, and anything they set can still be overridden on the command line.
#
#   bash trains/train_d<N>_hybrid_swa_muon.sh                  # the real run (~7h)
#   DRY_RUN=1 bash trains/train_d<N>_hybrid_swa_muon.sh        # write config/env, print cmd, don't train
#   SMOKE=1 bash trains/train_d<N>_hybrid_swa_muon.sh          # 20-step pipeline check (~1 min)
#   NUM_ITERATIONS=1000 bash trains/train_d<N>_hybrid_swa_muon.sh
#   WANDB_RUN=d12_hybrid_swa bash trains/train_d<N>_hybrid_swa_muon.sh
#   bash trains/train_d<N>_hybrid_swa_muon.sh --matrix-lr=0.03 # extra args -> base_train.py
#   bash trains/train_d<N>_hybrid_swa_muon.sh --summarize logs/<run_id>   # re-parse a run
#
# For a long run, keep it alive in screen:
#   screen -L -Logfile trains/d12_swa.screenlog -S d12swa \
#       bash trains/train_d<N>_hybrid_swa_muon.sh
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
# Measured on 1x NVIDIA GB10 (torch 2.14+cu130, bf16, FA4 via torch.library custom
# ops). Throughput vs DEVICE_BATCH_SIZE, at seq_len 2048:
#     16 -> ~45.5k tok/s, 15.1 GiB peak
#     32 -> ~49.1k tok/s, 28.0 GiB peak   (the default)
#     64 -> killed by the OS: GB10 memory is unified, so torch competes with the
#           page cache that streaming the parquet shards fills
# DEVICE_BATCH_SIZE only trades memory for speed -- grad accumulation keeps
# TOTAL_BATCH_SIZE fixed, so the optimization is identical. It must divide
# TOTAL_BATCH_SIZE / MAX_SEQ_LEN (= 256 sequences per step by default), i.e. a power
# of two up to 256, or base_train asserts.
# A compute-optimal run (2,520 steps x 524,288 tokens = 1.32B) took 7.3 h at bs 16;
# expect ~6.8 h at 32.
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
    ("attention", re.compile(r"^Attention\s+: (.+?)\s*$"), str),
    ("optimizer", re.compile(r"^Optimizer\s+: (.+?)\s*$"), str),
    ("attn_layout", re.compile(r"^Attn layout\s+: (.+?)\s*$"), str),
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
        if line.startswith("diag_json "):
            try:
                records.append(json.loads(line[len("diag_json "):]))
            except json.JSONDecodeError:
                pass
            continue
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
L += ["## Setup", ""]
for k, label in [("attention", "attention"), ("optimizer", "optimizer"),
                 ("attn_layout", "attention layout")]:
    if k in summary:
        L.append(f"- {label}: {summary[k]}")
L += ["", "## Training horizon", ""]
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
diag_recs = [r for r in records if r.get("type") == "diag"]
if diag_recs:
    last = diag_recs[-1]
    L += ["", f"## Diagnostics (last sample, step {last.get('step')}; {len(diag_recs)} samples in metrics.jsonl)", ""]
    for k in ("grad_norm/global", "grad_norm/muon", "grad_norm/adamw",
              "update_ratio/muon", "update_ratio/adamw",
              "attn_logit/max/max", "attn_lse/max/max",
              "act_rms/block/min", "act_rms/block/max",
              "scalars/resid_lambdas/min", "scalars/resid_lambdas/max",
              "scalars/x0_lambdas/min", "scalars/x0_lambdas/max"):
        if k in last:
            L.append(f"- {k}: {last[k]:.6g}")

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

# -----------------------------------------------------------------------------
# write_run_header(): what this run actually resolved to -- attention backend,
# optimizer composition, per-layer window sizes -- none of which base_train.py
# prints. Goes to the top of train.log so every run is self-describing.
# -----------------------------------------------------------------------------
write_run_header() {
    RUN_ID="$RUN_ID" DEPTH="$DEPTH" ASPECT_RATIO="$ASPECT_RATIO" HEAD_DIM="$HEAD_DIM" \
    MAX_SEQ_LEN="$MAX_SEQ_LEN" WINDOW_PATTERN="$WINDOW_PATTERN" \
    DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" TOTAL_BATCH_SIZE="$TOTAL_BATCH_SIZE" \
    MATRIX_LR="$MATRIX_LR" WEIGHT_DECAY="$WEIGHT_DECAY" EMBEDDING_LR="$EMBEDDING_LR" \
    UNEMBEDDING_LR="$UNEMBEDDING_LR" SCALAR_LR="$SCALAR_LR" \
    NUM_ITERATIONS="$NUM_ITERATIONS" TARGET_PARAM_DATA_RATIO="$TARGET_PARAM_DATA_RATIO" \
    NPROC_PER_NODE="$NPROC_PER_NODE" NUM_SHARDS="$NUM_SHARDS" \
    DIAGNOSTICS_EVERY="$DIAGNOSTICS_EVERY" MUON_VARIANT="$MUON_VARIANT" \
    GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
    GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l)" \
    "$PYTHON_BIN" - <<'RUN_HEADER_PY'
import os
import sys

W = 88

def line(label, text):
    print(f"{label:<14}: {text}" if label else f"{'':<14}  {text}")

print("=" * W)
print(f"nanochat experiment: {os.environ['RUN_ID']}")
print("=" * W)

try:
    import contextlib
    import io

    import torch
    import nanochat.flash_attention as fa
    from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer

    depth = int(os.environ["DEPTH"])
    head_dim = int(os.environ["HEAD_DIM"])
    seq_len = int(os.environ["MAX_SEQ_LEN"])
    pattern = os.environ["WINDOW_PATTERN"]
    base_dim = depth * int(os.environ["ASPECT_RATIO"])
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = model_dim // head_dim
    vocab = get_tokenizer().get_vocab_size()

    # attention backend actually in force
    impl = fa.impl_name()
    if impl == "fa4":
        import flash_attn_4.fa3_compat as fa4
        desc = "FA4 (vendored flash_attn_4, CuTe SM120 kernels)"
        desc += (" via torch.library custom ops" if fa4.has_custom_op()
                 else " called directly -- WARNING: breaks torch.compile graphs")
    elif impl == "fa3":
        desc = "FA3 (kernels hub)"
    else:
        desc = "PyTorch SDPA (sliding-window layers build an explicit mask)"
    line("Attention", desc)
    line("", f"HAS_FA3={fa.HAS_FA3} HAS_FA4={fa.HAS_FA4} "
             f"NANOCHAT_ATTN={os.environ.get('NANOCHAT_ATTN', 'auto')}")
    line("", f"compute dtype {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    line("", "inference / KV-cache path always uses SDPA (FA4 has no kvcache entry point)")

    # model + per-layer windows, built on meta so this costs no memory
    cfg = GPTConfig(sequence_len=seq_len, vocab_size=vocab, n_layer=depth,
                    n_head=n_head, n_kv_head=n_head, n_embd=model_dim,
                    window_pattern=pattern)
    with torch.device("meta"):
        model = GPT(cfg)
    windows = [w for w, _ in model.window_sizes]
    n_short = sum(1 for w in windows if w < seq_len)
    short_w = min(windows) if n_short else seq_len
    line("Model", f"depth {depth}, model_dim {model_dim}, {n_head} heads x {head_dim}, "
                  f"seq_len {seq_len}, vocab {vocab:,}")
    line("Attn layout", f"window_pattern {pattern} -> {n_short} sliding-window layers "
                        f"({short_w} tokens) + {len(windows) - n_short} full-context layers")
    line("", " ".join(f"L{i:02d}:{'S' if w < seq_len else 'L'}{w}" for i, w in enumerate(windows)))

    # optimizer composition
    muon_variant = os.environ.get("MUON_VARIANT", "nanochat")
    with contextlib.redirect_stdout(io.StringIO()):
        opt = model.setup_optimizer(
            unembedding_lr=float(os.environ["UNEMBEDDING_LR"]),
            embedding_lr=float(os.environ["EMBEDDING_LR"]),
            scalar_lr=float(os.environ["SCALAR_LR"]),
            matrix_lr=float(os.environ["MATRIX_LR"]),
            weight_decay=float(os.environ["WEIGHT_DECAY"]),
            muon_variant=muon_variant,
        )
    kinds = {}
    for g in opt.param_groups:
        n = sum(p.numel() for p in g["params"])
        k = kinds.setdefault(g["kind"], {"tensors": 0, "params": 0, "lrs": []})
        k["tensors"] += len(g["params"])
        k["params"] += n
        k["lrs"].append(g["lr"])
    variant_desc = {
        "nanochat": "MuonEq row equilibration + Polar Express + Muon+ renorm + NorMuon "
                    "variance reduction + cautious weight decay",
        "moonlight": "Polar Express + update RMS matched to AdamW (0.2*sqrt(max(m,n))) "
                     "+ decoupled weight decay (arxiv 2502.16982)",
    }.get(muon_variant, "unknown variant")
    line("Optimizer", f"{type(opt).__name__} -- Muon on the transformer matrices, "
                      "AdamW on embeddings and scalars")
    line("", f"muon variant '{muon_variant}': {variant_desc}")
    for kind, k in kinds.items():
        lrs = ", ".join(str(x) for x in sorted({round(x, 6) for x in k["lrs"]}))
        line("", f"{kind:5s} {k['tensors']:3d} tensors, {k['params']:>12,} params, lr {lrs}")
    line("", f"muon weight_decay {os.environ['WEIGHT_DECAY']} (cosine-decayed to 0), "
             "momentum 0.85 -> 0.97 -> 0.90")
    line("", "these LRs are pre-scaling; base_train rescales by sqrt(B/B_ref) below")
    line("Parameters", ", ".join(f"{k} {v:,}" for k, v in model.num_scaling_params().items()))
except Exception as e:  # a broken header must never block a run
    print(f"(run header incomplete: {type(e).__name__}: {e})")

line("Diagnostics", f"model internals every {os.environ.get('DIAGNOSTICS_EVERY', '-1')} steps "
                    "(grad norms, update:param, attention logits/LSE, activation RMS, residual scalars)")
line("Data", f"{os.environ.get('NANOCHAT_DATA_DIR')} -- {os.environ['NUM_SHARDS']} shards "
             "(all but the last are train, the last is val)")
dbs, tbs = int(os.environ["DEVICE_BATCH_SIZE"]), int(os.environ["TOTAL_BATCH_SIZE"])
msl, nproc = int(os.environ["MAX_SEQ_LEN"]), int(os.environ["NPROC_PER_NODE"])
micro = dbs * msl * nproc
if tbs > 0:
    line("Batch", f"{dbs} x {msl} x {nproc} rank(s) = {micro:,} tok/micro-batch, "
                  f"{tbs:,} tok/step -> grad accum {tbs // micro if micro else '?'}")
else:
    line("Batch", f"{dbs} x {msl} x {nproc} rank(s) = {micro:,} tok/micro-batch; tokens per step "
                  "chosen by base_train's scaling law below (TOTAL_BATCH_SIZE=-1)")
ni = int(os.environ["NUM_ITERATIONS"])
line("Horizon", f"num_iterations {ni}" if ni > 0 else
                f"target_param_data_ratio {os.environ['TARGET_PARAM_DATA_RATIO']} "
                "(iteration count computed by base_train below)")
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    line("Environment", f"python {sys.version.split()[0]}, torch {torch.__version__}, {gpu}, "
                        f"git {os.environ['GIT_SHA']} ({os.environ['GIT_DIRTY']} dirty files)")
except Exception:
    pass
print("=" * W)
print()
RUN_HEADER_PY
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
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"    # fastest on 1x GB10 with FA4 (64 is OOM-killed); lower to 16/8 if you OOM
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"   # tokens per optimizer step (-1 = auto)
# Muon update rule: 'nanochat' (MuonEq + Muon+ + NorMuon + cautious decay) or
# 'moonlight' (RMS-matched update + decoupled decay, arxiv 2502.16982). The two need
# different MATRIX_LR -- at the same LR a Moonlight update is 0.2*sqrt(max(m,n)) larger.
MUON_VARIANT="${MUON_VARIANT:-nanochat}"
# Default LR follows the variant. 0.004 for moonlight was calibrated on this repo by
# matching the measured update:param ratio of the default variant at its own LR
# (3.1e-3 over the first 20 steps at d12); it is a starting point, not a tuned value.
if [ "$MUON_VARIANT" = "moonlight" ]; then
    MATRIX_LR="${MATRIX_LR:-0.004}"
else
    MATRIX_LR="${MATRIX_LR:-0.02}"
fi
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
# model internals: grad norms, update:param ratios, attention logits/LSE, activation RMS,
# residual scalars. Costs one extra no-grad forward on the uncompiled model, so at 50 it
# is well under 1% of the run. -1 disables.
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-50}"

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
    DIAGNOSTICS_EVERY="${SMOKE_DIAGNOSTICS_EVERY:-5}"
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
  "muon_variant": "$MUON_VARIANT",
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
  "diagnostics_every": $DIAGNOSTICS_EVERY,
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
    --muon-variant="$MUON_VARIANT"
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
    --diagnostics-every="$DIAGNOSTICS_EVERY"
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

write_run_header 2>&1 | tee "$RUN_DIR/train.log"

START_TIME=$(date +%s)
set +e
"${LAUNCH[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
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
