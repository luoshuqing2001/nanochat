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

---

## Local patch: softplus attention

Softplus attention (`softplus(s)` element-wise, no normalization, output scaled by
`n_i^-alpha`) is implemented inside this vendored tree rather than as a separate
Triton kernel. Three new files plus a small patch to two upstream ones:

| file | what |
|---|---|
| `softplus.py` | `Softplus`, a drop-in for `Softmax` at the three call sites in `flash_fwd.py` |
| `flash_fwd_softplus.py` | `SoftplusForwardMixin` + the SM80/SM120 concrete classes |
| `flash_bwd_softplus.py` | `SoftplusBackwardMixin` + its SM80/SM120 classes |
| `softplus_api.py` | `softplus_attn_fa4_func` (autograd), `softplus_attn_fa4`, `auto_num_splits` |
| `../trains/test_cute_softplus.py` | forward correctness vs a float32 PyTorch reference |
| `../trains/test_cute_softplus_bwd.py` | gradients, against float32 autograd and Triton |
| `../trains/bench_cute_softplus.py` | against the Triton kernel and FA4 softmax |

The patch to upstream is deliberately two lines, both in `flash_fwd.py`: the score
map became a class attribute (`score_map_cls = Softmax`) and the one construction
site inside `kernel()` now reads `self.score_map_cls.create(...)`. `interface.py`
gained an `attn_kind` / `softplus_alpha` argument that selects the subclass on the
SM80 and SM120 branches and asserts on SM90/SM100, which have no softplus kernel yet.
Everything else -- the mainloop, masking, pipelining, epilogue -- is upstream's.

Why the seam is this small: without normalization, nothing in the inner loop depends
on the other key tiles. `online_softmax` becomes a pure element-wise map,
`rescale_O` and `finalize` become no-ops, and the only surviving cross-tile quantity,
`n_i^-alpha`, depends on the row index alone. It is applied once in the epilogue, on
values already in registers, using the same row-index idiom `mask.py` uses for causal
masking -- so it costs one multiply per output element and no extra pass over O.

Three things that are not obvious and cost time to find:

- **Masked entries are `-inf`** (`mask.py`), so the `max(s, 0)` in the stable form
  must be a real `max`. The branchless `(s + |s|) / 2` gives `NaN` there. With a real
  max, `-inf` maps to 0, which is exactly the right contribution to an unnormalized
  sum, so softplus needs no separate masked-entry path at all.
- **`super()` inside a `@cute.jit` method recurses forever.** The epilogue override
  calls `FlashAttentionForwardBase.epilogue(self, ...)` by name instead.
- **`window_size_left` is compile-time here**, unlike the softmax kernel where it is
  a runtime argument, because the epilogue needs the per-row key count. It is
  therefore part of the compile key; leaving it out makes a second window silently
  reuse the first window's kernel. `cute.full()` and `cute.math.max(vec, scalar)`
  both fail on this DSL version, so the zero vector comes from a zeroed rmem fragment
  hoisted out of the row loop.

Correctness: 8 configurations (B 1-2, T 256-2048, H 1-6, D 64/128, full causal and
SWA windows 255/511, alpha 1.0 and 0.5) agree with a float32 PyTorch reference to
2e-3..4e-3 relative -- the same error as `nanochat/softplus_attention.py`'s Triton
kernel on the identical inputs, i.e. bf16 noise. Run `python trains/test_cute_softplus.py`.

The backward is in CuTe too, in `flash_bwd_softplus.py`. Its seam is even smaller than
the forward's: two methods factored out of `compute_one_m_block`. Softmax needs the LSE
to recompute P and D = rowsum(dO*O) to form dS = P*(dP - D); softplus needs neither,

    P_ij  = softplus(s_ij)                        (no LSE)
    dS_ij = n_i^-alpha * dP_ij * sigmoid(s_ij)    (element-wise, no row sum)

so the whole `flash_bwd_preprocess` pass over O and dO is skipped -- only its third job,
zeroing `dq_accum`, still has to happen. `sigmoid(s) = 1 - exp(-softplus(s))` reuses the
value just computed and is 0 at the `-inf` masked entries, so again no masked path. The
`n_i^-alpha` factor rides in through the LSE and dPsum buffers, which are already
per-query-row float32 tensors with a gmem -> smem -> rmem path that softplus has no other
use for -- cheaper than recomputing the row index in the kernel.

### Forward speed, 1x GB10, B8 T2048 H12 D128 (attention only)

| | S, window 512 | L, full causal |
|---|---|---|
| FA4 softmax | 1.063 ms | 1.978 ms |
| Triton softplus | 1.274 ms | 2.223 ms |
| **CuTe softplus** | **1.091 ms** | **1.703 ms** |

1.17x over the Triton kernel on windowed layers and 1.31x on full-causal ones. On the
full-causal layer it also beats FA4's own *softmax* by 1.16x, which is the point:
dropping the running max, the running sum and the per-tile rescale of the accumulator
is worth about that much.

### Does softplus actually make training faster? Barely.

Forward alone is the flattering half. With the backward included (softplus's is still
the Triton kernel), same shape:

| | fwd | bwd | fwd+bwd |
|---|---|---|---|
| **S, window 512** | | | |
| FA4 softmax | 1.108 | 5.160 | 6.268 ms |
| Triton softplus | 1.325 | 4.084 | 5.409 ms |
| CuTe softplus (+ Triton bwd) | 1.127 | 4.084 | **5.211 ms** |
| **L, full causal** | | | |
| FA4 softmax | 1.672 | 7.244 | **8.916 ms** |
| Triton softplus | 2.172 | 7.401 | 9.573 ms |
| CuTe softplus (+ Triton bwd) | 1.717 | 7.401 | 9.119 ms |

With the backward in CuTe as well (same shape, re-measured in one run):

| | fwd | bwd | fwd+bwd |
|---|---|---|---|
| **S, window 512** | | | |
| FA4 softmax | 1.120 | 4.789 | 5.909 ms |
| Triton softplus | 1.286 | 4.001 | 5.288 ms |
| CuTe softplus | 1.043 | 4.415 | 5.458 ms |
| **L, full causal** | | | |
| FA4 softmax | 1.670 | 6.883 | 8.553 ms |
| Triton softplus | 2.179 | 7.230 | 9.409 ms |
| CuTe softplus | 1.720 | 6.534 | **8.253 ms** |

Over an SSSL mix that is 26.28 -> 24.63 ms against FA4 softmax, **1.07x on attention**,
or 1.09x if each layer type takes its best backward (Triton for windowed, CuTe for full).
Attention is ~12% of a d12/bs32 step on GB10, so ~1% end to end -- which matches the
earlier end-to-end A/B, 45,179 tok/s for Triton softplus against 44,982 for FA4 softmax.

Moving the backward to CuTe was worth about what the skipped preprocess costs and no
more: 6.883 -> 6.534 ms on the full-causal layer. The inner loop did not get faster,
because softplus's P recompute (max, abs, exp2, log2) is *more* work than softmax's
single exp2, which cancels against the row-sum term it saves. A sweep of 32 backward
tile configurations changed nothing (4.54-4.58 ms windowed, 6.86-6.90 ms full), so the
kernel is not config-bound. On windowed layers the Triton backward still wins, because
FA4 accumulates dQ atomically into an fp32 `dq_accum` and converts it in a separate
pass -- a fixed cost that a short windowed backward feels more.

The conclusion to draw is that the softmax epilogue was never the bottleneck; the MMAs
and the memory traffic are. Softplus is worth running as an architecture experiment, not
as a throughput optimization.

### Tile splitting and atomic-add aggregation

`num_splits=S` cuts each query tile's key range into S programs that aggregate with
`atomic_add`. Softmax cannot do this without a combine pass that rescales each split by
its LSE; softplus can, because a partial sum is already the final quantity.

It does what it is supposed to do -- the longest program shrinks from `n_block_max`
tiles to `ceil(n_block_max / S)` -- and that is worth having only when there are too
few query-tile programs to fill the GPU:

| B1 H1 D128, full causal | S=1 | S=2 | S=4 | S=8 | S=16 |
|---|---|---|---|---|---|
| T 4096 | 0.181 ms | 0.125 | **0.117** | 0.130 | 0.182 |
| T 8192 | 0.452 ms | 0.343 | **0.320** | 0.348 | 0.428 |
| T 16384 | 1.336 ms | 1.131 | **1.075** | 1.088 | 1.224 |

1.24x to 1.54x, best at S=4, and the optimum turns over by S=16 -- the classic split-K
shape. At training shapes it loses badly, and the decomposition says why
(B8 T2048 H12 D128, full causal):

| | |
|---|---|
| kernel, S=1 | 1.702 ms |
| kernel, S=2 / S=4 / S=8 | 2.201 / 2.767 / 3.968 ms |
| zero the 96 MiB fp32 buffer | 0.537 ms |
| cast fp32 -> bf16 | 0.703 ms |

The kernel gets *slower* with splits before any of the buffer cost is counted: the
output writes are fp32 instead of bf16, the atomics serialize on contended rows, and
the empty splits still run a full mainloop. Then the fp32 accumulator adds a fixed
1.24 ms of zero-and-cast. There are already ~1500 query-tile programs at this shape
against 48 SMs, so there was no load to balance in the first place.

This is the same verdict the Triton implementation reached on both GB10 and B200, now
reproduced inside FA4's own machinery, and the CuTe version is the better one: at
B1 T8192 H1 the Triton kernel went 0.57 -> 0.47 ms with splits, where this one starts
at 0.452 and reaches 0.320.

`num_splits` therefore defaults to 1 and `softplus_attn_fa4_func` (the training path)
never splits.

### Where splitting does pay: long-context inference

`num_splits="auto"` calls `auto_num_splits`, which counts query-tile programs against
the SM count and splits only when the machine would otherwise sit idle. Forward only,
one sequence, GB10:

| | s=1 | s=2 | s=4 | s=8 | auto |
|---|---|---|---|---|---|
| prefill H1 T4096 | 0.193 | 0.124 | 0.116 | 0.130 | s=5, **1.70x** |
| prefill H1 T8192 | 0.447 | 0.351 | 0.322 | 0.338 | s=3, 1.40x |
| prefill H1 T16384 | 1.337 | 1.125 | 1.096 | 1.112 | s=2, 1.20x |
| prefill H12 T8192 | 3.153 | 3.673 | 3.934 | 4.282 | s=1, 1.00x |
| decode H6 Tk4096 | 0.181 | 0.088 | 0.049 | 0.030 | s=8, **6.00x** |
| decode H6 Tk16384 | 0.748 | 0.375 | 0.294 | 0.291 | s=8, 2.59x |
| decode H6 Tk65536 | 2.954 | 1.456 | 1.196 | 1.065 | s=8, 2.79x |
| decode H12 Tk65536 | 3.110 | 2.478 | 2.314 | 2.175 | s=8, 1.51x |

Decode is the case the idea was made for: one query token against a long cache is a
handful of programs on 48 SMs, and splitting the key range is the only parallelism
available. Multi-head prefill already fills the machine and `auto` correctly declines.
The heuristic lands within a few percent of the per-shape optimum everywhere; it is
tuned on one GPU, so re-check the constants in `softplus_api.py` on other hardware.

### Side effect: upstream's SWA forward was ignoring the window's lower bound

The split-aware bulk loop closes the `# TODO: local` that sat in `kernel()`. Upstream's
SM80/SM120 forward ran that loop down to block 0 regardless of `n_block_min`, so for
sliding-window attention it computed every block below the window and then let the mask
throw the results away. Windowed attention cost exactly as much as full causal:

| B8 T2048 H12 D128, **softmax** | upstream | patched | |
|---|---|---|---|
| window 512 | 1.656 ms | 1.070 ms | 1.55x |
| window 256 | 1.650 ms | 1.033 ms | 1.60x |
| window 128 | 1.723 ms | 0.950 ms | 1.81x |
| full causal | 1.657 ms | 1.658 ms | unchanged |

Full causal is untouched, which is the check that the patch is a no-op where
`n_block_min` is 0. The backward never had the bug -- `flash_bwd.py` already iterates
`cutlass.range(m_block_min, m_block_max)`.

This helps nanochat's hybrid-SWA runs on the ordinary softmax path, but only a little:
attention is ~12% of a d12/bs32 step on GB10 and the forward is a minority of that.
Measured with `trains/profile_step.py --depth 12 --device-batch-size 32`, the attention
category goes 662.0 -> 622.2 ms of CUDA time (-6%) and the step 1285.9 -> 1276.4 ms
(+0.7% throughput, which is close to run-to-run noise). It is worth more at larger
depth and sequence length, where attention's share grows.
