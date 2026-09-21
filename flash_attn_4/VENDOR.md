# Vendored: Flash Attention 4 (CuTe DSL)

Source: https://github.com/Dao-AILab/flash-attention, `flash_attn/cute/`
Upstream commit: `edb5c76` ("fix: compile with c++20 for pytorch 2.13+ (#2899)")
Upstream version: `4.0.0b32.dev5+gedb5c76` (distribution name `flash-attn-4`)
Vendored on: 2026-09-20, from `/home/shuqing/flash-attention/flash_attn/cute`
License: BSD 3-Clause (see `LICENSE`), (c) Tri Dao et al.

## Why vendored instead of installed

FA3 has no sm121 kernels, so nanochat falls back to PyTorch SDPA on this DGX Spark
(NVIDIA GB10, compute capability 12.1). FA4's CuTe DSL implementation *does* cover
this GPU: `flash_fwd_sm120.py` / `flash_bwd_sm120.py` are labelled "SM120 (Blackwell
GeForce / DGX Spark)" and subclass the SM80 kernels with SM120's 99 KB shared-memory
budget. Keeping the kernels in-tree means a run's `git rev-parse HEAD` pins the
attention implementation too.

## What was changed

1. Copied every `*.py` from `flash_attn/cute/`, minus the benchmark/tuning scripts
   (`benchmark.py`, `benchmark_flash_attention_fp8.py`, `bench_utils.py`,
   `sm90_config_search.py`) and the build metadata (`pyproject.toml`, egg-info).
2. Rewrote the package path so it imports as a top-level package:
   `flash_attn.cute` -> `flash_attn_4` (203 occurrences across 35 files).
3. Added `fa3_compat.py`, an FA3-API-shaped adapter for nanochat's call sites.

Nothing else was edited; the kernels are upstream code.

## Re-syncing with upstream

```bash
cd /home/shuqing/flash-attention && git pull
DST=/home/shuqing/nanochat/flash_attn_4
cp /home/shuqing/flash-attention/flash_attn/cute/*.py $DST/
rm -f $DST/benchmark.py $DST/benchmark_flash_attention_fp8.py $DST/bench_utils.py $DST/sm90_config_search.py
grep -rl 'flash_attn\.cute' $DST/*.py | xargs sed -i 's/flash_attn\.cute/flash_attn_4/g'
# then re-add fa3_compat.py if it was overwritten, and re-run the parity test
```

## Runtime dependencies

`nvidia-cutlass-dsl>=4.6.2`, `torch`, `einops`, `typing_extensions`,
`apache-tvm-ffi`, `torch-c-dlpack-ext`, `quack-kernels` — all present in the conda
base env on this machine (cutlass DSL 4.7.1).

## Measured on this machine (2026-09-20, NVIDIA GB10, torch 2.14+cu130, bf16)

Isolated attention, d12 shapes (B=16, T=2048, H=6, D=128), `trains/bench_attention.py`:

| layer type | SDPA fwd+bwd | FA4 fwd+bwd | speedup |
|---|---|---|---|
| L, full context (3 layers) | 9.74 ms | 10.63 ms | 0.92x |
| S, 512 sliding window (9 layers) | 54.70 ms | 8.33 ms | **6.56x** |
| whole d12 SSSL attention | 521 ms | 107 ms | 4.9x |

Numerics are equivalent: both land at max err 0.0093 / mean 1.1e-4 against an fp32
SDPA reference, and an 8-step training A/B matched to 4 decimal places on the loss
(10.397263 vs 10.397263 at step 0, 10.3364 vs 10.3359 at step 7).

End-to-end training step (same shapes, `SMOKE=1`):

| | SDPA | FA4 |
|---|---|---|
| eager (`TORCHDYNAMO_DISABLE=1`) | 2057 ms | **1632 ms** (-21%) |
| `torch.compile` (what base_train does) | **1150 ms** | 1216 ms (+6%) |

So FA4 wins by exactly the margin the microbenchmark predicts — until torch.compile
enters. Inductor optimizes the masked-SDPA path hard (2057 -> 1150 ms), while FA4 is
an opaque eager callback that breaks the graph, so compiled FA4 keeps only part of
its advantage and ends up behind. `torch._dynamo.allow_in_graph` does not help:
dynamo traces the function with FakeTensors and the CuTe kernel needs a real data
pointer ("Cannot access data pointer of Tensor").

### The fix: FA4 as a torch.library custom op

`fa3_compat.py` wraps FA4's raw `_flash_attn_fwd` / `_flash_attn_bwd` in
`flash_attn_4::attn_fwd` / `flash_attn_4::attn_bwd` custom ops, each with a
`register_fake` shape rule, tied together by `torch.library.register_autograd`.
Dynamo then sees one opaque node per attention call instead of tracing into FA4's
Python dispatch, and never runs the kernels on FakeTensors.

| | step time | graph breaks |
|---|---|---|
| compiled SDPA | 1150 ms | 0 |
| compiled FA4, called directly | 1216 ms | 52 |
| compiled FA4, via custom op | **727 ms** | **0** |

That is **1.58x** over the previous best, i.e. ~44,900 tok/s vs ~28,800, and about
**8.7 h instead of 13.7 h** for the compute-optimal d12 run.

Correctness: forward is bit-identical to FA4's own `flash_attn_func`, as are dk/dv;
dq differs by up to 3e-2 on a magnitude-274 tensor, and FA4's native path differs
from *itself* by exactly the same amount across reruns (non-deterministic atomic
accumulation in the dq backward), so this is FA4's own noise, not the wrapper's.
An 8-step training A/B against SDPA: loss 10.335936 vs 10.335933, val bpb 3.121077
vs 3.121080. Sampling and `base_eval` are unaffected — they use the KV-cache path,
which has no FA4 equivalent and stays on SDPA.

**FA4 is therefore the default** whenever the custom ops registered and FA3 is
absent; `NANOCHAT_ATTN=sdpa` switches back, and `FA4_CUSTOM_OP=0` calls FA4
directly (the slow, graph-breaking path) for comparison.

Note for future debugging: under a plain `FakeTensorMode` FA4 *almost* works — it
guards kernel launches with `if not fake_mode:` in ten places — but still reaches a
real `data_ptr()` at `interface.py:3750`, which is a warning there and a hard error
under dynamo. Fixing that one access would make `torch._dynamo.allow_in_graph`
viable as a lighter alternative to the custom ops.
