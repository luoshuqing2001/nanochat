# Runbook: a 1B architecture experiment from an empty machine

Written against a single NVIDIA B200 (sm_100). Every command is meant to be pasted
as-is. Timings assume ~660 TFLOPS of achieved BF16 throughput; step 5 replaces that
assumption with a measurement, and everything downstream should be recomputed from it.

The experiment this targets: **d24 (1.38B params), hybrid SWA, Muon, 100B tokens**,
run twice -- once per Muon variant -- so the two arms can be compared.

---

## 1. Code

```bash
git clone -b lsq/residuals https://github.com/luoshuqing2001/nanochat.git
cd nanochat
git pull            # if the clone predates the latest fixes
```

Three things travel with the repo that usually do not: the FA4 CuTe kernels
(`flash_attn_4/`), the tokenizer (`trains/tokenizer/`, 545 KB) and every run's record
(`logs/`). One commit hash therefore pins the model code, the attention kernels and
the tokenization.

## 2. Environment

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
INSTALL=1 bash trains/setup_env.sh
```

`setup_env.sh` checks the driver, whether this torch build has kernels for the GPU's
compute-capability family, every python dependency, the tokenizer (installing it from
`trains/tokenizer/` when `INSTALL=1`), the data, and which attention backend resolves.
It never installs torch -- that choice depends on the CUDA version:

```bash
# only if torch is missing or too old for the GPU
pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch
```

Confirm the toolchain knows the architecture before trusting anything else:

```bash
P=$(python -c "import triton,os;print(os.path.join(os.path.dirname(triton.__file__),'backends/nvidia/bin/ptxas'))")
$P --version | tail -2
$P --help | grep -oE "sm_1[0-9]+a?" | sort -u
```

`sm_100a` must appear for a B200 (`sm_103a` for a B300, which needs CUDA >= 12.9).

## 3. Data

```bash
echo 'export NANOCHAT_DATA_DIR=/mnt/belta/nanochat/base_data_climbmix' >> ~/.bashrc
source ~/.bashrc

python trains/fetch_data.py --tokens 100e9 --margin 1.11 --dry-run
setsid nohup python trains/fetch_data.py --tokens 100e9 --margin 1.11 --workers 8 > fetch.log 2>&1 &
```

2,479 shards, 213 GB, one epoch. Idempotent and resumable: re-run the same command
after an interruption. When it finishes, confirm the plan actually holds:

```bash
python trains/fetch_data.py --tokens 100e9 --margin 1.11 --dry-run --verify
```

`--verify` runs the real dataloader over a few shards (~75 s, CPU) and reports what it
delivers, rather than trusting the 44.8M tokens/shard constant. Expect ~111B for a
100B budget, and `val = shard_02499.parquet` -- the validation shard must stay 2499 or
bits-per-byte stops comparing with earlier runs.

## 4. Preflight

```bash
bash trains/setup_env.sh                                        # 0 fail
nvidia-smi --query-compute-apps=pid,used_memory --format=csv    # nobody else on the GPU
SMOKE=1 MUON_VARIANT=moonlight DEVICE_BATCH_SIZE=32 bash trains/train_d24_hybrid_swa_muon.sh
```

The smoke run is 20 steps. Read the header it prints:

```
Attention     : FA4 (vendored flash_attn_4, CuTe SM100 kernels) via torch.library custom ops
Optimizer     : MuonAdamW -- Muon on the transformer matrices, AdamW on embeddings and scalars
                muon variant 'moonlight': ... update RMS matched to AdamW (0.2*sqrt(max(m,n)))
Attn layout   : window_pattern SSSL -> 18 sliding-window layers (512 tokens) + 6 full-context
```

base_train separately prints `WARNING: Flash Attention 3 not available ... utilization
will be terrible`. That warning only inspects FA3 and is wrong whenever the header
says FA4; the header is the authority.

Then pick the micro-batch, which is throughput-only (grad accumulation keeps the
tokens per step fixed, so the optimization is identical at any value):

```bash
python trains/bench_attention.py
for BS in 16 32 64; do
  SMOKE=1 DEVICE_BATCH_SIZE=$BS RUN_NAME=bs$BS bash trains/train_d24_hybrid_swa_muon.sh >/dev/null 2>&1
  echo -n "bs=$BS: "; grep -oE "tok/sec: [0-9,]+" logs/bs${BS}_*/train.log | tail -3 | tr '\n' ' '; echo
done
```

## 5. Calibration run (~40 min)

Do not commit 8 days to unmeasured assumptions.

```bash
export MUON_VARIANT=moonlight DEVICE_BATCH_SIZE=32
NUM_ITERATIONS=300 EVAL_EVERY=100 SAVE_EVERY=-1 DIAGNOSTICS_EVERY=50 \
  RUN_NAME=d24_calib bash trains/train_d24_hybrid_swa_muon.sh 2>&1 | tail -40
```

Read three things out of it:

| what | where | healthy |
|---|---|---|
| throughput | `tok/sec` on the step lines | recompute the real wall clock from it |
| Muon step size | `diag` lines, `upd/par muon` | ~1e-3; much larger means lower `MATRIX_LR` |
| attention scale | `diag` lines, `attn logit max` | stable, not climbing run-away |

`MATRIX_LR=0.004` for the moonlight variant was calibrated at d12, not d24. The RMS
matching makes it shape-independent by construction, so it should transfer, but the
diagnostic is what confirms it.

Real wall clock: `steps x tokens_per_step / measured_tok_per_sec`, with
95,367 steps x 1,048,576 tokens for the run below.

## 6. The run

```bash
export MUON_VARIANT=moonlight            # or: nanochat, for the other arm
export DEVICE_BATCH_SIZE=32              # from step 4
export TARGET_PARAM_DATA_RATIO=137       # 137 x 730M scaling params = 100B tokens
export SAVE_EVERY=2000 EVAL_EVERY=2000 DIAGNOSTICS_EVERY=500
unset NUM_ITERATIONS RUN_NAME

setsid nohup bash trains/train_d24_hybrid_swa_muon.sh > run.log 2>&1 &
tail -f logs/d24_hybridswa_muon_*/train.log
```

`TARGET_PARAM_DATA_RATIO` scales only the token count and the step count: the auto
batch size and the weight-decay scaling both cancel the ratio out, so the rest of the
recipe is untouched.

### Surviving container limits

Modal caps a function call at 24 h, so a multi-day run needs restarts. In each new
container, re-export the same variables and:

```bash
bash trains/resume_latest.sh <run_id> --keep 2
```

It resumes from the newest checkpoint (model, optimizer and dataloader position),
refuses a half-written one, and prunes older checkpoints -- nanochat never does, and
a d24 checkpoint is ~9 GB. The environment is not stored in the checkpoint;
`logs/<run_id>/config.json` records what the run started with.

## 7. Watching it

```bash
tail -f logs/<run_id>/train.log
grep "Validation bpb" logs/<run_id>/train.log | tail
grep "^diag" logs/<run_id>/train.log | tail -3
nvidia-smi
```

Per-run artifacts land in `logs/<run_id>/`: `config.json`, `env.txt` (host, GPUs,
versions, git commit), `command.txt`, `train.log`, `eval.log`, `metrics.jsonl` (one
record per logged step, validation and diagnostic sample), `summary.{json,md}`,
`checkpoint_meta/`. Weights stay in `$NANOCHAT_BASE_DIR/base_checkpoints/<run_id>/`.

Re-parse a log at any time (for instance after a killed run):

```bash
bash trains/train_d24_hybrid_swa_muon.sh --summarize logs/<run_id>
python trains/probe_checkpoint.py --model-tag <run_id> --steps 2000,10000 --out logs/<run_id>/probe.json
```

## 8. The comparison

One run answers nothing. Run both arms at the *same* budget:

```bash
MUON_VARIANT=moonlight  ... bash trains/train_d24_hybrid_swa_muon.sh
MUON_VARIANT=nanochat   ... bash trains/train_d24_hybrid_swa_muon.sh
```

and compare `best_val_bpb` in the two `summary.json`. Architecture and optimizer
rankings move with the token budget, so a result at 8.8B tokens does not license a
claim at 100B, and cross-budget losses are not comparable at all.

---

## Things that actually went wrong, and the fix

| symptom | cause | fix |
|---|---|---|
| `ptxas fatal: Value 'sm_103a' is not defined` | Triton's bundled ptxas predates CUDA 12.9, which introduced sm_103 (B300) | upgrade torch (triton 3.8.0 ships ptxas 12.9), or use a B200 (sm_100), or `CUTE_DSL_PTXAS_PATH=/usr/local/cuda/bin/ptxas` for FA4's own path |
| preflight reports 0 shards after a long download | `fetch_data` and the launcher resolved different data dirs | export `NANOCHAT_DATA_DIR` once, for both |
| `train.log` only has a few stderr lines | python block-buffers stdout into a pipe | fixed (`PYTHONUNBUFFERED=1`); if you see it again, check the launcher sets it |
| `WARNING: ... SDPA ... utilization will be terrible` while FA4 is active | base_train's warning only looks at FA3 | trust the run header, or `python -c "import nanochat.flash_attention as f; print(f.impl_name())"` |
| no `eval.log` / `summary.md` after training finished | the launcher shell was killed while python kept running | start with `setsid`; recover with `--summarize` |
| throughput far below a previous measurement | someone else is on the GPU | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` |
| bpb suddenly incomparable with earlier runs | the validation shard changed, or the tokenizer was re-trained | keep `--val-shard 2499`; never run `python -m nanochat.dataset -n N` (it also pulls shard_06542, which sorts last and becomes val) |
| MFU prints 0.00 | the GPU is not in `get_peak_flops()` in `nanochat/common.py` | b200 is in the table; b300 and GB10 are not |
