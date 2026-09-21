# trains/

Experiment launchers. Each launch writes **everything about one run** into its own
folder under `nanochat/logs/<RUN_NAME>_<YYYYmmdd_HHMMSS>/`.

Layout: `train_d12_hybrid_swa_muon.sh` is the entire launcher (config, preflight,
provenance, training, evaluation, log parsing, summary). `_stubs/wandb.py` is an
import-only fallback.

## d12 hybrid-SWA + Muon

```bash
cd /home/shuqing/nanochat
bash trains/train_d12_hybrid_swa_muon.sh
```

- **12 layers**, `--depth 12` (model_dim = 12 x 64 = 768, 6 heads x 128)
- **hybrid SWA**, `--window-pattern SSSL`: layers tiled S,S,S,L where `S` = sliding
  window of `ceil(seq_len/4)` rounded up to 128 (512 tokens at `seq_len=2048`) and
  `L` = full context. The last layer is always forced to full context by `gpt.py`.
- **Muon**, no flag needed: `GPT.setup_optimizer()` builds a `MuonAdamW` optimizer
  that runs Muon on every transformer matrix and AdamW on embeddings/scalars.
  `MATRIX_LR` (default 0.02) and `WEIGHT_DECAY` (0.28, cosine-decayed) are the Muon knobs.
- data: the ClimbMix parquet shards in `base_data_climbmix/`.

### Data path

`nanochat/dataset.py` now honors `NANOCHAT_DATA_DIR` (one-line change):

```python
DATA_DIR = os.environ.get("NANOCHAT_DATA_DIR") or os.path.join(base_dir, "base_data_climbmix")
```

The launcher exports it as `$REPO_DIR/base_data_climbmix`, so the shards are read in
place — no symlink, no copy. Override for another dataset:

```bash
NANOCHAT_DATA_DIR=/path/to/other/shards bash trains/train_d12_hybrid_swa_muon.sh
```

`NANOCHAT_BASE_DIR` (default `~/.cache/nanochat`) is unchanged and still holds the
**tokenizer**, the **checkpoints** and the CORE **eval bundle**. The override also applies
to `scripts/tok_train.py` and `scripts/tok_eval.py`, which read the same constant.

### Common invocations

```bash
DRY_RUN=1 bash trains/train_d12_hybrid_swa_muon.sh          # write config/env/command, don't train
SMOKE=1 bash trains/train_d12_hybrid_swa_muon.sh            # 20-step pipeline check (~1 min)
NUM_ITERATIONS=1000 bash trains/train_d12_hybrid_swa_muon.sh # fixed-length run
WANDB_RUN=d12_hybrid_swa bash trains/train_d12_hybrid_swa_muon.sh   # also log to wandb
DEVICE_BATCH_SIZE=8 bash trains/train_d12_hybrid_swa_muon.sh # if you OOM
bash trains/train_d12_hybrid_swa_muon.sh --matrix-lr=0.03    # extra args pass through to base_train.py
```

Long runs are best put in a screen session:

```bash
screen -L -Logfile trains/d12_swa.screenlog -S d12swa bash trains/train_d12_hybrid_swa_muon.sh
```

All knobs are environment variables with defaults at the top of the script:
`DEPTH, WINDOW_PATTERN, MAX_SEQ_LEN, HEAD_DIM, ASPECT_RATIO, DEVICE_BATCH_SIZE,
TOTAL_BATCH_SIZE, MATRIX_LR, WEIGHT_DECAY, EMBEDDING_LR, UNEMBEDDING_LR, SCALAR_LR,
WARMUP_STEPS, WARMDOWN_RATIO, NUM_ITERATIONS, TARGET_PARAM_DATA_RATIO, EVAL_EVERY,
EVAL_TOKENS, SAMPLE_EVERY, SAVE_EVERY, CORE_METRIC_EVERY, FINAL_EVAL, WANDB_RUN,
NPROC_PER_NODE, PYTHON_BIN, NANOCHAT_BASE_DIR`.

### Model internals

base_train.py logs loss and throughput only. `nanochat/diagnostics.py` adds the
internals, sampled every `DIAGNOSTICS_EVERY` steps (default 50, `-1` disables):

- gradient L2 norms: global, per optimizer kind (Muon / AdamW), and per layer
- update-to-parameter ratios `||dw||/||w||` per kind -- the standard Muon tuning
  signal, healthy around 1e-3
- attention log-sum-exp per layer (flash kernels never materialize the logits, so
  LSE is the cheap upper bound: `max_logit <= lse <= max_logit + log(n_keys)`), plus
  the exact max logit recomputed on a 128-query subsample
- per-layer activation RMS (residual stream, attention output, MLP output) and the
  learnable residual scalars `resid_lambdas` / `x0_lambdas`

Each sample prints a human-readable `diag <step> | ...` line plus a `diag_json {...}`
line that the log parser turns into `{"type": "diag", ...}` records in `metrics.jsonl`;
`summary.md` shows the last sample. Activations and attention logits come from one
extra no-grad forward on the *uncompiled* model, so the compiled training graph is
never instrumented; the gradient and update statistics are the only part inside the
timed region, which slightly inflates `dt` on diagnostic steps.

The same statistics can be computed offline, on any checkpoint (including runs that
predate this and other experiments' checkpoints):

```bash
python trains/probe_checkpoint.py --model-tag <run_id> --steps 500,1000 --out probe.json
```

That one does a real backward on a fixed validation batch, so it also reports per-layer
gradient norms. It needs the GPU -- do not run it next to a live training job.

### What lands in logs/<run_id>/

| file | contents |
|---|---|
| `config.json` | every resolved knob + checkpoint dir for this run |
| `env.txt` | host, GPUs, driver, python/torch versions, git branch/commit/dirty files, data dir + shard count |
| `command.txt` | the exact `base_train` command line |
| `train.log` | a run header (attention backend, optimizer composition, per-layer window sizes, batch/horizon, git commit) followed by the full training stdout, tee'd live |
| `eval.log` | `scripts/base_eval.py` output, if `FINAL_EVAL` is on |
| `metrics.jsonl` | one JSON record per logged step / val eval / CORE eval |
| `summary.json` | run-level numbers (setup, params, horizon, best val bpb, throughput, peak mem) |
| `summary.md` | the same, human-readable |
| `checkpoint_meta/` | copies of the checkpoint `meta_*.json` + checkpoint dir size |
| `checkpoint_path.txt` | where the weights actually are (under `NANOCHAT_BASE_DIR/base_checkpoints/`) |

Model weights stay in `$NANOCHAT_BASE_DIR/base_checkpoints/<run_id>/` so `logs/` stays small.

The log parser is built into the same script and can be re-run on any run folder (e.g.
after killing a run early):

```bash
bash trains/train_d12_hybrid_swa_muon.sh --summarize logs/<run_id>
```

### Notes for this machine (1x NVIDIA GB10)

- FA3 has no sm121 kernels. Attention runs on the vendored FA4 CuTe kernels
  (`flash_attn_4/`, see its `VENDOR.md`), wrapped in torch.library custom ops so the
  compiled model keeps a single graph. `NANOCHAT_ATTN=sdpa` falls back to PyTorch SDPA,
  where sliding-window layers build an explicit mask and cost ~6.5x more.
- Measured at `DEVICE_BATCH_SIZE=16`: ~44.9k tok/s with FA4 vs ~28.9k with SDPA
  (727 ms vs 1150 ms per micro-batch), ~15 GiB peak. bs=32 is slower, bs=8 slightly slower.
- The default compute-optimal horizon (`--target-param-data-ratio 12`) is 2,682 steps
  x 524,288 tokens = 1.41B tokens ≈ **8-9 h** with FA4 (13-14 h on SDPA). Use
  `NUM_ITERATIONS` for shorter runs.
- base_train.py still prints "Flash Attention 3 not available, using PyTorch SDPA
  fallback" and warns that sliding-window utilization will be terrible. That message
  only looks at FA3 and is stale when FA4 is active; check `train.log` for the real
  implementation via `python -c "import nanochat.flash_attention as f; print(f.impl_name())"`.
- MFU prints as 0.00 because `get_peak_flops()` in `nanochat/common.py` has no entry
  for "NVIDIA GB10".
- `CORE_METRIC_EVERY` / `FINAL_EVAL=...,core` download `eval_bundle.zip` on first use.
- `_stubs/wandb.py` is an import-only fallback used when wandb is missing and
  `WANDB_RUN=dummy`; real wandb is installed in the conda base env, so it is unused here.
