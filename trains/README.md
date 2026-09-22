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

### Getting only the data you need

```bash
python trains/fetch_data.py --depth 24 --dry-run   # what it would fetch, and why
python trains/fetch_data.py --depth 24             # fetch exactly that
```

It sizes the download from the model rather than from a raw shard count:
`tokens = 12 x scaling_params(depth)`, times a 1.25 margin, divided by **44.8M tokens
per shard**. That constant is measured, not assumed -- the finished d12 run consumed
2,520 x 524,288 = 1.321B tokens and stopped at parquet 29, row group 43 of 84, i.e.
29.51 shards. A shard is ~88 MB, so roughly 521M tokens per GB.

| target | tokens | shards | disk |
|---|---|---|---|
| d12 compute-optimal | 1.3B | 37 | 3 GB |
| d16 compute-optimal | 2.8B | 79 | 7 GB |
| d20 compute-optimal | 5.2B | 146 | 13 GB |
| d22 compute-optimal (~1.1B params) | 6.8B | 191 | 16 GB |
| d24 compute-optimal (~1.4B params) | 8.8B | 245 | 21 GB |
| **d24 Chinchilla (20 tok/param)** | 26B | 726 | 62 GB |
| **d24 ablation standard, 4 epochs** | 100B (25B unique) | 698 | 60 GB |
| **d24 ablation standard, 1 epoch** | 100B | 2791 | 240 GB |

### How many tokens for an architecture ablation?

`--depth` sizes a *compute-optimal* run, which is the wrong target for comparing
architectures. What the literature does at this scale:

- The de facto standard for efficient-attention architecture papers is **1.3B
  parameters on 100B tokens** -- [GLA](https://arxiv.org/pdf/2312.06635),
  [DeltaNet](https://proceedings.neurips.cc/paper_files/paper/2024/file/d13a3eae72366e61dfdc7eea82eeb685-Paper-Conference.pdf),
  [Gated Slot Attention](https://proceedings.neurips.cc/paper_files/paper/2024/file/d3f39e51f5f634fb16cc3e658f8512b9-Paper-Conference.pdf),
  [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791),
  [Physics of LMs 4.1](https://arxiv.org/pdf/2512.17351) all use it. That is ~77
  tokens/param, six times Chinchilla and six times nanochat's ratio-12 default.
- It is deliberately over-trained. Architecture rankings are **not stable across token
  budgets**: scaling curves cross, and an advantage visible at a small budget can
  vanish or reverse later. Compare only within a matched budget, and treat a cheap
  screening result as a hypothesis, not a finding.
- **Use single-epoch data for the headline comparison.** Those papers take 100B as a
  *subset* of a much larger corpus (SlimPajama is 627B, FineWeb-Edu is trillions), so
  their numbers are one pass over fresh tokens, and matching that is what makes your
  numbers comparable to theirs.
- [Muennighoff et al., "Scaling Data-Constrained Language Models"](https://arxiv.org/abs/2305.16264)
  (NeurIPS 2023) found up to ~4 epochs costs almost nothing versus fresh tokens, and
  `--epochs 4` does cut a 100B-token run to 25B unique tokens on disk. But that result
  is about being *data*-constrained; here the constraint is disk, which is cheap next
  to the GPU-days the run costs. More importantly, memorization under repetition
  depends on capacity, depth and width, so repeating data adds an
  architecture-dependent term to exactly the quantity an architecture ablation is
  trying to measure. Keep `--epochs > 1` for screening runs and for comparisons where
  both arms have identical capacity (e.g. optimizer variants).

A workable ladder: screen variants at d12 (1.3B tokens, 7.3 h on GB10), promote
survivors to d16, and run the claim at d24 with 26B tokens (Chinchilla) or 100B
(comparable to published numbers).

**Do not use `python -m nanochat.dataset -n N` to top up.** Besides sizing by raw
shard count, it always also downloads `shard_06542` (upstream's `MAX_SHARD`), which
then sorts last and silently becomes the validation set. nanochat validates on the
highest-numbered shard it finds (`dataset.py`: train is `paths[:-1]`, val is
`paths[-1:]`), so that one extra file invalidates every earlier bpb comparison.
`fetch_data.py` pins the validation shard instead (`--val-shard`, default 2499, which
is what the runs in `logs/` used) and refuses a budget that would run past it.

`--prune --yes` deletes shards outside the keep set; it never deletes the validation
shard, and without `--yes` it only reports. ClimbMix ships pre-shuffled
("climbmix-400b-shuffle"), so keeping a contiguous prefix is an unbiased sample.

### Runs longer than one container

Modal caps a function call at 24 h, so anything past a day has to survive a restart.
base_train restores model, optimizer and dataloader position from a checkpoint, and
`trains/resume_latest.sh` finds the newest one and relaunches:

```bash
bash trains/resume_latest.sh d24_hybridswa_muon_20260921_120000 --keep 2
```

It infers the size wrapper from the run id, checks that the step it picks has its
optimizer shard (a half-written checkpoint from a killed container is skipped rather
than loaded), and `--keep N` drops older checkpoints first -- necessary because
nanochat never prunes them and a d24 checkpoint is ~9 GB.

The environment is *not* stored in the checkpoint, so a resume must re-export what
the original run used (`MUON_VARIANT`, `TARGET_PARAM_DATA_RATIO`, `NANOCHAT_DATA_DIR`,
...). `logs/<run_id>/config.json` records them.

Set `SAVE_EVERY` from how much work a crash may cost, not from how many checkpoints
look tidy: at ~7.6 s/step for d24 on a B300, `SAVE_EVERY=2000` is a checkpoint every
~4 h, which bounds the loss from an unexpected restart.

### Muon variants

`MUON_VARIANT` picks the update rule for the matrix parameters:

```bash
MUON_VARIANT=moonlight bash trains/train_d12_hybrid_swa_muon.sh
```

| | `nanochat` (default) | `moonlight` |
|---|---|---|
| orthogonalization | Polar Express, 5 steps | Polar Express, 5 steps |
| pre-conditioning | MuonEq row equilibration | none |
| post-scaling | Muon+ Frobenius renorm, then NorMuon variance reduction | update RMS matched to AdamW: `0.2*sqrt(max(m,n))` |
| weight decay | cautious (only where update and weight agree in sign) | decoupled, all elements |
| default `MATRIX_LR` | 0.02 | 0.004 |

Moonlight is [arxiv 2502.16982](https://arxiv.org/abs/2502.16982). Its point is that a
semi-orthogonal m x n update has element RMS `1/sqrt(max(m,n))`, so its magnitude depends
on the shape and one learning rate cannot serve the whole model; rescaling to a fixed RMS
of 0.2 (AdamW's typical update RMS) removes that and is what lets the LR transfer across
shapes and scales. Verified here, one step at lr 0.01: the default variant produces update
RMS/lr of 0.036 / 0.018 / 0.036 for 768x768, 768x3072 and 3072x768 (exactly the predicted
shape dependence), while moonlight gives 0.201 / 0.199 / 0.199.

**The two variants need different learning rates.** At the same LR a moonlight update is
`0.2*sqrt(max(m,n))` larger -- about 11x for a 768x3072 matrix -- so reusing 0.02 would
blow up. The 0.004 default was calibrated by matching the measured `update_ratio/muon`
diagnostic of the default variant (3.1e-3 over the first 20 steps at d12); it is a
starting point, not a tuned value, and base_train warns if a moonlight run is given an
LR above 0.01. No quality claim is made either way: the short runs used for calibration
are all inside the LR warmup and say nothing about final loss. That comparison is the
experiment to run.

### Where the time goes

```bash
python trains/profile_step.py --depth 20 --device-batch-size 64 --fp8
```

Runs the same construction path as base_train (meta build, optional FP8 conversion,
torch.compile, MuonAdamW) on synthetic tokens, so the dataloader is excluded, and
reports step time, achieved TFLOPS, MFU, peak memory, a CUDA-time breakdown by kernel
category, and the top kernels. `--no-compile` isolates what inductor is buying.

Needs an idle GPU. Dataloader starvation does not show up here by construction --
watch `nvidia-smi` utilization during a real run for that.

MFU is only meaningful if the GPU is in `get_peak_flops()` (`nanochat/common.py`):
b200 is, b300 and GB10 are not.

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

The tokenizer now comes from git too: `trains/tokenizer/` holds the 545 KB it takes,
and `INSTALL=1 bash trains/setup_env.sh` copies it into `$NANOCHAT_BASE_DIR/tokenizer`.
Re-training it instead produces a *different* tokenizer, which silently makes bpb
numbers incomparable with the runs in `logs/` -- see `trains/tokenizer/README.md`.

Only the ClimbMix shards stay outside git: fetch them with
`python trains/fetch_data.py` (see "Getting only the data you need"). The FA4 kernels
are vendored in `flash_attn_4/`, so they come from git as well.

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
