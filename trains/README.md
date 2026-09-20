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

### What lands in logs/<run_id>/

| file | contents |
|---|---|
| `config.json` | every resolved knob + checkpoint dir for this run |
| `env.txt` | host, GPUs, driver, python/torch versions, git branch/commit/dirty files, data dir + shard count |
| `command.txt` | the exact `base_train` command line |
| `train.log` | full training stdout (tee'd live) |
| `eval.log` | `scripts/base_eval.py` output, if `FINAL_EVAL` is on |
| `metrics.jsonl` | one JSON record per logged step / val eval / CORE eval |
| `summary.json` | run-level numbers (params, horizon, best val bpb, throughput, peak mem) |
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

- FA3 has no sm121 kernels, so attention falls back to PyTorch SDPA. Full-context (`L`)
  layers still take the fast `is_causal` path; sliding-window (`S`) layers build an
  explicit mask, which works but is slower than a real SWA kernel.
- Measured: ~28.9k tok/s at `DEVICE_BATCH_SIZE=16`, ~15 GiB peak. bs=32 is *slower*
  (~24.5k tok/s), bs=8 is ~28.3k.
- The default compute-optimal horizon (`--target-param-data-ratio 12`) is 2,682 steps
  x 524,288 tokens = 1.41B tokens ≈ **13-14 h**. Use `NUM_ITERATIONS` for shorter runs.
- MFU prints as 0.00 because `get_peak_flops()` in `nanochat/common.py` has no entry
  for "NVIDIA GB10".
- `CORE_METRIC_EVERY` / `FINAL_EVAL=...,core` download `eval_bundle.zip` on first use.
- `_stubs/wandb.py` is an import-only fallback used when wandb is missing and
  `WANDB_RUN=dummy`; real wandb is installed in the conda base env, so it is unused here.
