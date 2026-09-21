# trains/

Experiment launchers. Each launch writes **everything about one run** into its own
folder under `nanochat/logs/<RUN_NAME>_<YYYYmmdd_HHMMSS>/`.

Layout: `_engine_hybrid_swa_muon.sh` is the launcher itself (config, preflight,
provenance, training, evaluation, log parsing, summary). One wrapper per model size
sets `DEPTH` and the knobs measured for it, then execs the engine, so every size gets
the same machinery and there is one place to fix. `_stubs/wandb.py` is an import-only
fallback.

## Sizes

```bash
bash trains/train_d12_hybrid_swa_muon.sh     # ... d16, d20, d24, d32, d40, d48
```

All of them: hybrid SWA (`--window-pattern SSSL`), MuonAdamW, seq_len 2048, data:param
ratio 12, `TOTAL_BATCH_SIZE=-1` so base_train's scaling law picks the tokens per step.

| depth | dim | params | tokens | steps | micro-batch on GB10 | GB10 throughput | GB10 wall clock |
|---|---|---|---|---|---|---|---|
| 12 | 768 | 286M | 1.32B | 2,520 | 32 | 49.1k tok/s | **7.3 h (measured)** |
| 16 | 1024 | 537M | 2.82B | 5,376 | 16 | 23.0k tok/s | ~34 h |
| 20 | 1280 | 897M | 5.22B | 4,980 | 8 | 10.3k tok/s | ~5.8 days |
| 24 | 1536 | 1.38B | 8.76B | 8,352 | 4 | 4.7k tok/s | ~22 days |
| 32 | 2048 | 2.82B | 20.1B | 9,600 | — | — | ~6 months (extrapolated) |
| 40 | 2560 | 4.99B | 38.8B | 18,480 | — | — | ~2.5 years (extrapolated) |
| 48 | 3072 | 8.05B | 66.4B | 31,680 | — | — | does not fit |

Throughput and micro-batch for d12-d24 are measured on this box; d32-d48 are sized for
a datacenter GPU and their wrappers say so. **d48 does not fit a GB10 at all**: fp32
master weights, gradients and Muon momentum come to ~78 GB of its 121 GB unified
memory before any activations, and that memory is shared with the page cache.

The micro-batch numbers are not "as large as fits". GB10 has a throughput cliff once
torch's peak passes roughly 25-30 GiB, because its memory is unified and streaming the
parquet shards fills the page cache. Measured at d16: bs 16 -> 23.0k tok/s (25 GiB),
bs 32 -> 10.8k tok/s (47 GiB). Same shape at d20 (1.9x) and d24 (1.6x). On a
discrete-HBM datacenter card this cliff does not exist and the wrappers point at much
larger values.

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

### Another machine (1x B200 / B300, or any fresh box)

```bash
git clone <your fork> && cd nanochat && git checkout lsq/residuals
bash trains/setup_env.sh              # check everything, install nothing
INSTALL=1 bash trains/setup_env.sh    # also pip install the missing python packages
```

`setup_env.sh` checks the driver, that this torch build has kernels for the GPU's
compute-capability family, every python package nanochat and FA4 need, the tokenizer
and data shards, and which attention backend resolves. It deliberately does **not**
install torch -- pick the wheel for your CUDA first, e.g.
`pip install --index-url https://download.pytorch.org/whl/cu130 torch`.

Two assets do not come from git: the tokenizer (`$NANOCHAT_BASE_DIR/tokenizer`, two
files under 1 MB -- copy it to reproduce runs exactly, or retrain with
`python -m scripts.tok_train`) and the ClimbMix shards (copy them, or
`python -m nanochat.dataset -n 200`, then point `NANOCHAT_DATA_DIR` at them). The FA4
kernels *do* come from git, because they are vendored in `flash_attn_4/`.

What changes on Blackwell datacenter parts (B200 sm100, B300 sm103):

- FA4 dispatches on `arch // 10`, so every 10.x GPU gets the **native SM100 kernels**
  (`flash_fwd_sm100.py`), not the SM80-derived SM120 ones this box uses. There is no
  sm103 special case, and the SM100 path is the one upstream optimizes hardest.
- FP8 becomes available: FA4 asserts `arch // 10 == 10` for FP8 attention, and
  nanochat's own `--fp8` (Float8Linear for the matmuls) targets H100+. Both are worth
  an A/B; pass `--fp8` through the launcher.
- FA3 may resolve too (the kernels hub has more coverage for sm90/sm100). The resolver
  prefers FA3 when present, which may not be what you want -- compare all three with
  `NANOCHAT_ATTN=fa3|fa4|sdpa python trains/bench_attention.py`.
- Memory is discrete HBM, not unified, so the page-cache contention that kills
  `DEVICE_BATCH_SIZE=64` here does not apply. With 180-288 GB, start at 64 and try
  128; it must stay a power of two dividing `TOTAL_BATCH_SIZE / MAX_SEQ_LEN`.
- `get_peak_flops()` in `nanochat/common.py` has entries for b100/b200/gb200 but not
  b300, so MFU prints 0.00 until you add one with the official dense BF16 number.
- d12 is small for such a GPU. The launcher is generic: `DEPTH=20 bash trains/...`
  re-derives model dim, batch size, horizon and LR scaling.

Then the usual order: `DRY_RUN=1`, `SMOKE=1`, `python trains/bench_attention.py` to
pick the backend and micro-batch, and only then the real run.

### Notes for this machine (1x NVIDIA GB10)

- FA3 has no sm121 kernels. Attention runs on the vendored FA4 CuTe kernels
  (`flash_attn_4/`, see its `VENDOR.md`), wrapped in torch.library custom ops so the
  compiled model keeps a single graph. `NANOCHAT_ATTN=sdpa` falls back to PyTorch SDPA,
  where sliding-window layers build an explicit mask and cost ~6.5x more.
- Throughput vs `DEVICE_BATCH_SIZE` (FA4, measured): 16 -> ~45.5k tok/s / 15.1 GiB,
  **32 -> ~49.1k tok/s / 28.0 GiB (the default)**, 64 -> killed by the OS. GB10's memory
  is unified, so torch competes with the page cache that streaming parquet shards fill.
  This knob is throughput-only: grad accumulation keeps `TOTAL_BATCH_SIZE` fixed, so the
  optimization is unchanged. (Under the old SDPA path bs=32 was *slower* -- FA4 reversed it.)
- Do not raise `TOTAL_BATCH_SIZE` for speed: it changes the optimization (base_train
  rescales LRs by sqrt(B/B_ref) and the weight decay), and 524,288 is exactly what the
  repo's scaling law computes as optimal for d12 (`TOTAL_BATCH_SIZE=-1` reproduces it).
- A compute-optimal d12 run (2,520 steps x 524,288 tokens = 1.32B tokens) took **7.3 h**
  at bs 16 with FA4; expect ~6.8 h at bs 32. Use `NUM_ITERATIONS` for shorter runs.
- base_train.py still prints "Flash Attention 3 not available, using PyTorch SDPA
  fallback" and warns that sliding-window utilization will be terrible. That message
  only looks at FA3 and is stale when FA4 is active; check `train.log` for the real
  implementation via `python -c "import nanochat.flash_attention as f; print(f.impl_name())"`.
- MFU prints as 0.00 because `get_peak_flops()` in `nanochat/common.py` has no entry
  for "NVIDIA GB10".
- `CORE_METRIC_EVERY` / `FINAL_EVAL=...,core` download `eval_bundle.zip` on first use.
- `_stubs/wandb.py` is an import-only fallback used when wandb is missing and
  `WANDB_RUN=dummy`; real wandb is installed in the conda base env, so it is unused here.
