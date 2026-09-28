# Softplus attention + fused RMSNorm on FA3 (H100): results

2026-09-27, NVIDIA H100 80GB HBM3 (SXM, SM90), torch 2.9.1+cu128, nvcc/ptxas 12.8.93.
BF16 Q/K/V, head dim 128, MHA. Branch `lsq/softplus-fa3` of flash-attention.

```
U_i = sum_j softplus(q_i . k_j / sqrt(d)) v_j,     O_i = U_i / sqrt(mean_d(U_i^2) + 1e-6)
```

## Summary

| | best softplus variant / stock FA3 |
|---|---:|
| forward, training shapes (d12/d20, full and SWA 512) | 0.94 - 0.99x |
| forward, long causal 8k/16k, non-causal 2k/4k | 0.92 - 0.99x |
| backward | 0.88 - 0.91x |
| forward + backward | 0.90 - 0.91x |
| split-KV decode (atomic reduction vs FA3 split + combine) | 0.94 - 1.00x, 0.78x at the smallest |
| **quad** (softplus-shaped, not softplus): forward | **1.05 - 1.09x** |
| **quad**: forward + backward | 1.00 - 1.08x |
| **quad** with block skipping, 80-90% all-zero tiles: forward / backward | 1.15 - 1.28x / 1.07 - 1.23x |

Numerics match a float64 reference to BF16 precision in forward (rel. L2 2.4e-3, floor 1.7e-3)
and backward (dq/dk/dv 2.9e-3; stock FA3 softmax is 2.4e-3 against its own reference).

The attention *structure* without online softmax is faster than FA3: a diagnostic build with the
same kernel but a trivial elementwise function (A = 2^(s c)) runs **1.05-1.14x** stock forward.
Every percent softplus is below that is the cost of evaluating softplus itself: roughly 5% per
extra instruction per score, because what limits these kernels between the GEMMs is FP32 issue,
not MUFU throughput.

## Softplus-shaped replacements that beat FA3 (`fn_quad`, `fn_rexp`)

If the function only has to keep softplus's *shape* (0 far left, x far right, convex, smooth),
it can be chosen for cost. With x = s * scale:

- **quad** (`kFnQuad`): 0 for x <= -1, (x+1)^2/4 on (-1, 1), x for x >= 1. C1, no transcendental.
  u = sat(x/2 + 1/2) is one `FFMA.SAT` (inline PTX: `__saturatef(fmaf())` compiles to two
  instructions), A = u^2 + relu(x - 1): 3 FP32 + 1 INT per score. Its derivative is u itself, so
  the backward's dS = dP * u is one FMUL. Zero, not exponentially small, below x = -1.
- **rexp** (`kFnRexp`): relu(x) + e^-|x| / 2. softplus's asymptotes, C1 (both one-sided slopes
  1/2 at 0), convex, exponential tail. One MUFU; the backward derivative (1 - z/2 or z/2) reuses
  the same z = e^-|x|.

Both keep the scale in-kernel (no Q prescale), use the output RMSNorm unchanged, and are
checked forward and backward against float64 autograd through the same function
(`test_fn.py`: O 2.3e-3, rms 2e-4, dq/dk/dv 2.9e-3 - 3.4e-3).

| shape | fwd FA3 | fwd quad | fwd rexp | fwd+bwd FA3 | fwd+bwd quad |
|---|---:|---:|---:|---:|---:|
| d12 train (full) | 0.382 | 0.355 (**1.08x**) | 0.396 (0.96x) | 1.626 | 1.591 (1.02x) |
| d12 train (SWA 512) | 0.249 | 0.231 (**1.08x**) | 0.234 (1.06x) | 1.047 | 1.057 (0.99x) |
| d20 train (full) | 0.630 | 0.582 (**1.08x**) | 0.643 (0.98x) | 2.667 | 2.656 (1.00x) |
| d20 train (SWA 512) | 0.406 | 0.385 (**1.05x**) | 0.389 (1.04x) | 1.724 | 1.749 (0.99x) |
| causal 8k | 1.691 | 1.564 (**1.08x**) | 1.723 (0.98x) | 6.465 | 6.427 (1.01x) |
| causal 16k | 3.244 | 2.980 (**1.09x**) | 3.330 (0.97x) | | |
| non-causal 4k | 1.708 | 1.601 (**1.07x**) | 1.719 (0.99x) | 6.577 | 6.445 (1.02x) |
| non-causal 2k | 0.895 | 0.843 (**1.06x**) | 0.879 (1.02x) | 3.498 | 3.455 (1.01x) |

ms; `bench_fwd_h100_fn.md`, `bench_bwd_h100_fn2.md` (fwd+bwd quad = its best backward tiles: FA3's
for causal, the 64-row tile for non-causal too, now the `fn_quad` default). The forward sits at
the structural ceiling measured with `diag_exp`; forward+backward is at parity (0.99-1.02x, within
the ~3% noise) because the backward keeps its structural ~5% overhead (dU round trip).

### Round 3: what else was tried on quad / rexp

| # | change | result |
|---|---|---|
| 1 | rexp with the scale folded into Q (`fn_rexp_pre`: exp2 argument is a sign-bit OR, no FMUL) | forward 0.96-1.03x -> **1.03-1.08x** FA3 (`bench_fwd_h100_fn2.md`); now used by nanochat's `rexp_rmsnorm` (the multiply fuses into q's norm under torch.compile) |
| 2 | quad backward: recover u = sqrt(min(A,1)) after the dP wait instead of keeping it (frees 32 regs) | **2-6% slower** (MUFU.SQRT on the critical path); kept as `fn_quad_sqrtu`, default keeps u |
| 3 | backward tile search, 12 configs (V_in_regs, Stages_dS/dO = 1, SdP/dKV/dQ swapAB, M = 80/96/128, N = 96, 3 WGs) | none beats FA3's (`bench_bwd_h100_tiles.md`); M >= 96 exceed smem, N = 96 / 3 WGs fail static asserts |
| 4 | block skipping of all-zero tiles (quad's A is exactly 0 for x <= -1): forward skips PV, backward skips dV and dK | exact (bitwise = dense for causal); pays off above ~65% (fwd) / ~45% (bwd) all-zero tiles; at 80-90%: forward **1.15-1.28x** FA3, backward **1.07-1.23x** (`bench_skip_*_h100.md`) |

Dense quad after round 3 (`bench_bwd_h100_fn3.md`, `fn_quad` = keep-u, 64-row non-causal bwd tile):
forward 1.04-1.11x, backward 0.97-1.06x, forward+backward **1.00-1.08x** FA3 (causal ~1.00, non-causal 1.06-1.08).

Block skipping, details. The skip decision must be uniform over a warpgroup (WGMMA is collective):
the u bits are OR-ed while computing A (3-input LOP3, INT pipe, 1 per 2 scores) and reduced with one
`barrier.red.or.pred` on otherwise-unused named-barrier slots. The first attempt, in the
intra-warpgroup-overlap mainloop, was 2x *slower* even when dense: the two branches leave different
wgmma groups in flight (`wait<1>` vs `wait<0>`) and ptxas serializes every wgmma (C7514). The forward
skip therefore lives in the non-overlap mainloop (both branches end with `wait<0>`; that path is ~18%
slower dense, hence the ~65% break-even), and the backward skip ends both branches with `wait<0>`
after dK (~5% slower dense). Synthetic block-sparse inputs (`sparse_inputs.py`) supply the sparsity;
whether trained quad models are that sparse is measured with nanochat's `scripts/quad_sparsity.py`
(per layer, at the kernels' 64x128 fwd / 64x64 bwd decision granularity), and layers are opted in
with `NANOCHAT_QUAD_SKIP_FWD` / `NANOCHAT_QUAD_SKIP_BWD`.

nanochat wiring: `ATTN_KIND=quad_rmsnorm` / `rexp_rmsnorm` (`nanochat/fn_rmsnorm_attention.py`,
torch.library ops, 0 graph breaks under torch.compile; KV-cache inference on a torch reference),
experiment files `trains/exp_d{12,16,20}_*_8xh100_quad.sh`.

## What was built

All changes are behind `-DFLASHATTN_SOFTPLUS` (stock builds are unchanged):

- `hopper/softplus.h`: `flash::Softplus`, same interface as `flash::Softmax`
  (`max_get_scale`/`rescale_o`/`finalize` become no-ops, `online_softmax` becomes elementwise
  softplus) plus `normalize_o`, the RMSNorm, called once after the last PV GEMM. So the whole FA3
  forward machinery is kept: TMA + mbarrier pipeline, warp specialization, WGMMA with P from
  registers, pingpong, intra-warpgroup overlap, persistent LPT scheduler, causal/local skipping.
- Forward epilogue: RMSNorm in registers (one row of O = the 4 threads of a quad: 32 FMAs per
  thread-row, 2 shuffles, 1 rsqrt); the per-row rms goes where FA3 writes the LSE.
- `hopper/epilogue_fwd.hpp: store_softplus_split`: split-KV with in-kernel atomic reduction (below).
- Backward: `flash_bwd_preprocess_kernel.h` computes the RMSNorm backward
  dU = (dO - O mean(dO*O)) / rms and writes it; the main kernel reads dU where softmax reads dO,
  recomputes A = softplus(S), and uses dS = dA * sigmoid(S) (no dP_sum term).
- `hopper/tile_size.h`: softplus builds use 128 x 128 non-causal tiles (stock: 128 x 176).
- `hopper_softplus/`: `build.py` (every variant from the same sources with `hopper/setup.py`'s
  flags; hdim 128 / bf16 only; ~75 s fwd, ~110 s fwd+bwd), `sp_attn.py` (loading, calling,
  autograd wrapper, float64 reference), tests and benchmarks.

## Numerics

`test_correctness.py` (forward, causal / SWA 511 / non-causal / ragged T=777, scores at 1x and 4x):

| | rel. L2 of O | rel. L2 of rms |
|---|---:|---:|
| BF16 rounding of the exact O (floor for any BF16 kernel) | 1.66e-3 | |
| every variant with the scale in-kernel (exact MUFU, poly2-4, soft-exp, mix) | 2.31e-3 - 2.37e-3 | 1.7e-4 - 2.5e-4 |
| `pre_*` (scale folded into an already-BF16 Q: one extra rounding in the test) | 2.75e-3 | ~2e-4 |

The step from 1.66e-3 to 2.35e-3 is A rounded to BF16 before the PV GEMM, the same for every
variant: the polynomial approximations add nothing measurable. The `pre_*` excess is the test
rounding Q twice; folded into nanochat's q RMSNorm before its BF16 cast it would not exist.

`test_bwd.py`: dq, dk, dv vs float64 autograd through the reference: 2.9e-3 (`bwd_sp_poly3`),
3.4e-3 (`bwd_pre_poly3`, same double rounding). Stock FA3's softmax backward against its own
float64 reference, same inputs: 2.4e-3.

`test_split.py`: 2/3/8/32 splits on decode (Tq=1, Tk to 64k, SWA), chunked prefill, causal
prefill with empty splits, ragged shapes: 2.5e-3 vs float64 (as unsplit), ~1e-4 vs the unsplit
kernel, repeated and interleaved calls differ by at most 1-2 BF16 ulp (fp32 atomic order), no drift.

## Forward

`bench_fwd.py` (`bench_fwd_h100_final2.md`): ms (TFLOPS over unmasked pairs, 4*D flop each).

| shape | B | T | H | W | stock FA3 | diag_exp | sp_poly3x | pre_poly3 | pre_poly2 | best / stock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d12 train (full) | 32 | 2048 | 6 | -1 | 0.379 (544) | 0.362 (570) | 0.461 (448) | 0.415 (497) | 0.402 (513) | 0.943x |
| d12 train (SWA) | 32 | 2048 | 6 | 511 | 0.248 (363) | 0.218 (415) | 0.270 (334) | 0.263 (343) | 0.251 (359) | 0.989x |
| d20 train (full) | 32 | 2048 | 10 | -1 | 0.636 (541) | 0.582 (591) | 0.751 (457) | 0.678 (507) | 0.672 (511) | 0.946x |
| d20 train (SWA) | 32 | 2048 | 10 | 511 | 0.403 (373) | 0.357 (421) | 0.446 (337) | 0.435 (345) | 0.422 (357) | 0.955x |
| causal 8k | 4 | 8192 | 16 | -1 | 1.741 (632) | 1.574 (699) | 2.076 (530) | 1.836 (599) | 1.860 (591) | 0.948x |
| causal 16k | 2 | 16384 | 16 | -1 | 3.256 (675) | 2.951 (745) | 3.983 (552) | 3.546 (620) | 3.587 (613) | 0.918x |
| non-causal 4k | 8 | 4096 | 16 | -1 | 1.726 (637) | 1.569 (701) | 2.054 (535) | 1.873 (587) | 1.778 (618) | 0.971x |
| non-causal 2k | 32 | 2048 | 8 | -1 | 0.913 (602) | 0.818 (672) | 1.050 (524) | 0.970 (567) | 0.918 (599) | 0.994x |

- `diag_exp` is not softplus (A = 2^(s c)): it bounds what the structure alone gives. `diag_identity`
  (A = s) measured the same, so FFMA + MUFU per score is fully hidden behind the GEMMs.
- `pre_*` need `scale * log2e` folded into Q upstream (nanochat already RMS-normalizes q right
  before attention; the scale rides along for free). `pre_poly2` uses a degree-2 log polynomial
  (rel. err 2.8e-3, about BF16's half-ulp) -- its end-to-end error is indistinguishable from
  `pre_poly3` above, but it is the one to validate in training before relying on it.
- `sp_poly3x` is the best variant that keeps the scale in-kernel.
- The first build, before instruction-level work, was 0.83-0.92x (`bench_fwd_h100_run1.md`).

### Where the forward time went, and what fixed it

The first `sp_poly3` compiled, per score, to FMUL (scale), FADD + FADD for `-|y|` (MUFU.EX2 takes
no operand modifiers), FMNMX, 4 FFMA and 1/2 F2FP: ~8.5 FP32-pipe ops against softmax's ~4. The
evidence that FP32 issue rather than MUFU was the limit: `sp_softexp` (no MUFU, ~14 FP32 ops) was
the slowest variant and `sp_naive` (2 MUFU, ~3 FP32) tied `sp_poly3` (1 MUFU, ~8.5 FP32).

1. `max(y,0)` as a signed-int max of the float bits and `-|y|` as a sign-bit OR: both move to the
   INT pipe (VIMNMX, LOP3).
2. `sp_poly3x`: `-|s| * c` as one FMUL with operand modifiers feeding MUFU directly, and 1/c
   folded into the polynomial coefficients so `max(s,0)` needs no scale; the leftover global 1/c
   is cancelled by RMSNorm.
3. `pre_*`: the scale folded into Q upstream, removing the last per-score FMUL.
4. Tiles: FA3's 128 x 176 non-causal tile costs softplus 5-7%; 128 x 128 is used instead
   (`bench_fwd_h100_tiles2.md`: also tried N = 96/112/160, M = 192, no intra-WG overlap, SS PV;
   causal/SWA stay at FA3's 128 x 128; 128 x 176 with SS PV exceeds shared memory).

## Backward

`bench_bwd.py` (`bench_bwd_h100_final.md`). "bwd" is one backward op (preprocess + main +
postprocess); TFLOPS count 2.5x the forward flops, 3.5x for fwd+bwd.

| shape | B | T | H | W | bwd stock | bwd sp_poly3 | bwd pre_poly3 | fwd+bwd stock | fwd+bwd sp_poly3 | fwd+bwd pre_poly3 | best / stock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d12 train (full) | 32 | 2048 | 6 | -1 | 1.236 (417) | 1.370 (376) | 1.378 (374) | 1.594 (453) | 1.791 (403) | 1.758 (411) | 0.907x |
| d12 train (SWA) | 32 | 2048 | 6 | 511 | 0.819 (275) | 0.897 (251) | 0.885 (255) | 1.031 (306) | 1.146 (276) | 1.138 (277) | 0.906x |
| d20 train (full) | 32 | 2048 | 10 | -1 | 2.042 (421) | 2.294 (375) | 2.260 (380) | 2.643 (455) | 2.912 (413) | 2.917 (412) | 0.908x |
| d20 train (SWA) | 32 | 2048 | 10 | 511 | 1.332 (282) | 1.475 (255) | 1.476 (255) | 1.705 (309) | 1.880 (280) | 1.908 (276) | 0.907x |
| causal 8k | 4 | 8192 | 16 | -1 | 4.950 (555) | 5.534 (497) | 5.423 (507) | 6.552 (587) | 7.315 (526) | 7.211 (534) | 0.909x |
| non-causal 4k | 8 | 4096 | 16 | -1 | 4.875 (564) | 5.530 (497) | 5.317 (517) | 6.491 (593) | 7.331 (525) | 7.156 (538) | 0.907x |
| non-causal 2k | 32 | 2048 | 8 | -1 | 2.690 (511) | 3.027 (454) | 2.939 (468) | 3.480 (553) | 3.944 (488) | 3.858 (499) | 0.902x |

Design points of the backward:

- **dU in the preprocess.** RMSNorm's backward needs mean(dO * O) per row -- exactly the
  `rowsum(dO * O)` FA3's preprocess kernel already computes for softmax. It writes dU there (BF16,
  one extra B*T*H*D write), and the main kernel then differentiates unnormalized attention.
- **No dP_sum.** dS = dA * sigmoid(S): softplus has no row coupling, so the `- D` term is gone.
- **sigmoid from P, no extra tensor.** sigmoid(x) = 1 - e^-softplus(x) = 1 - 2^-sp2, and
  dS = dP - dP * 2^-sp2 is one FFMA. Computing 2^-sp2 before `warpgroup_wait<0>` so it overlaps
  the dP GEMM made no measurable difference.
- **Constants out of the loop.** The kernel's "P" is sp2 = softplus/ln2; the ln2 is applied once
  per dV element at the end instead of in every score (and dS needs no 1/ln2).

Why it is 0.90x and not closer: a diagnostic backward with the cheapest possible elementwise work
(`bwd_diag_exp`) was already ~0.95x -- the second MUFU for sigmoid and the dU round trip -- and
recomputing softplus adds ~3 FP32 ops per score on top. Recomputing A with `lg2(1 + ex2(y))`
instead (`bwd_pre_naive`: fewer FP32 ops, 3 MUFU per score) was slower, not faster.

## Split-KV with atomics

Softmax split partials carry their own max, so FA3 writes num_splits partial O + LSE and launches
a combine kernel that rescales them. Softplus partials are plain sums, U = sum_t U_t, so each split
reduces its fp32 fragment straight from registers into one workspace block, arrives on a counter,
and whoever arrives last reads the sum back, normalizes, and writes O and rms. Details:

- The unit of arrival is a **warp**, not a CTA: each warp owns 16 complete rows of O in the WGMMA
  accumulator layout and RMSNorm needs only the 4 threads of a row, so the last warp for its rows
  finishes them alone -- no cross-warp barrier, no smem flag. Empty KV ranges (causal) still arrive,
  without touching the smem barriers the producer also skipped.
- `red.global.add.v2.f32` (one fire-and-forget `REDG.E.ADD.F32x2`; `atomicAdd(float2*)` compiles to
  two returning scalar ATOMGs), release/acquire fences (`fence.acq_rel.gpu`, not `__threadfence()`
  = `MEMBAR.SC.GPU`).
- The last arrival zeroes its workspace slice and counter; the host caches one workspace per
  device, zeroed once. Single-stream assumption.
- The mainloop leaves U unnormalized when Split: RMSNorm(sum U_t) != sum RMSNorm(U_t).
- FA3 forces PackGQA whenever it splits on SM90; the softplus build drops that and instantiates a
  non-packed split kernel, whose epilogue already stores from registers.

`bench_split.py` (`bench_split_h100_run3.md`), best split count of a sweep on each side, FA3 time
includes its combine launch:

| shape | B | Tq | Tk | H | FA3 split+combine | split_diag_exp | split_pre_poly3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 4096 | 16 | 14.3 us | 17.3 us (0.83x) | 18.5 us (0.78x) |
| decode | 1 | 1 | 16384 | 16 | 53.4 us | 54.1 us (0.99x) | 56.8 us (0.94x) |
| decode | 1 | 1 | 65536 | 16 | 183.5 us | 181.1 us (1.01x) | 188.4 us (0.97x) |
| decode | 4 | 1 | 16384 | 16 | 184.3 us | 180.5 us (1.02x) | 187.7 us (0.98x) |
| decode (d20 heads) | 1 | 1 | 32768 | 10 | 64.8 us | 66.3 us (0.98x) | 65.4 us (0.99x) |
| chunked prefill | 1 | 512 | 32768 | 8 | 105.7 us | 113.8 us (0.93x) | 139.4 us (0.78x) |

**The atomic reduction does not beat FA3's combine on H100.** `split_diag_exp` isolates the
reduction and ties at memory-bound decode. FA3's combine is cheap here -- it reads
num_splits x B x H x Tq x D fp32, tens of KB at decode, and is launched with PDL so it overlaps the
main kernel's tail. The last-arrival pattern instead puts dependent L2 round trips (reds drain ->
fence -> counter -> fence -> read back) on every CTA's tail, which the smallest decode pays for.
Beyond that the loss is the elementwise cost again: MHA decode runs 128-row tiles with one valid
row, so it is compute-bound on padding and inherits the forward gap. A DSMEM variant (splits of an
m_block in one cluster, reduced through distributed shared memory into the leader, deterministic
and workspace-free) would avoid the global round trips, but cannot beat a combine that already
costs almost nothing.

## RMSNorm-specific optimizations, summarized

1. **Scale invariance.** RMSNorm(aU) = RMSNorm(U) up to eps, so any global factor on U can be
   dropped and eps rescaled by 1/a^2 instead -- exact. Used for the ln2 of the base-2 formulation,
   for the score scale in `sp_poly3x`, and in the backward to move ln2 out of the score loop. It
   also makes softplus attention's usual n_i^-alpha length normalization redundant.
2. **Normalize once per tile, in registers**, with the quad shuffle softmax uses for its row sum,
   instead of correcting a running statistic every KV block. Measured cost: zero.
3. **RMSNorm backward in FA3's own preprocess kernel**, which already computes the row dot product
   it needs; the main backward then has no row coupling at all.
4. **Linear partials** make split-KV / stream-K reductions LSE-free (implemented; see above for why
   it does not pay on H100).

## Measurement notes

- Clocks cannot be locked (no root). Identical kernels registered under different names, each in
  its own CUDA graph capture, differed by up to ~3% (0.415-0.433 ms for one kernel), and whole
  sessions drift by several percent. `bench()` therefore pools 3 captures per function over 21
  interleaved rounds; treat differences under ~3% as noise.
- `ncu` is not installed and `RmProfilingAdminOnly=1`: instruction mixes come from
  `cuobjdump -sass`, not hardware counters.

## Not done

- A training run: `pre_poly2`/`pre_poly3` and the backward are validated numerically, not for loss.
- nanochat integration: an H100 path for `ATTN_KIND=softplus_rmsnorm` would wrap
  `SoftplusAttn` (`sp_attn.py`) in torch.library custom ops and fold the scale into q's RMSNorm.
- GQA, varlen, head dims other than 128, FP8.

## dQ memset + postprocess folded into the backward (`*_dqf`, `FLASHATTN_SP_DQFUSE`): a negative result

Idea: stock FA3's backward runs a preprocess kernel that zeroes the fp32 dQaccum (~200 MB at d12) and a
postprocess kernel that reads it back and converts it to bf16 dQ; replace both with atomic arrival
counters per dQ row block, the conversion done where the row block completes, and a persistent dQaccum
that every call leaves all-zero. Correct in every variant (`test_dqfuse.py`: dK/dV bit-identical, dQ
within fp32 summation order, repeated calls and buffer regrowth), and slower in every variant:

| where the conversion ran | d12 causal bwd, ms (stock rexp_pre 1.25-1.28) |
|---|---:|
| last-arriving main CTA, arrivals counted per m_block in the store warp | 1.66 |
| last-arriving main CTA, arrivals counted at the CTA's end | 1.41 |
| same, conversion skipped (wrong dQ; lower bound of the approach) | 1.09 |
| arrivals in-loop, one iteration behind | 1.72 |
| converter CTAs appended to the grid, one tile at a time | 1.55 |
| converter CTAs, each warp its own tiles (final code, `bench_bwd_h100_dqf.md`) | 1.36-1.40 |

Why: the DRAM traffic does not change -- dQaccum still has to be zeroed once (now by the converter,
before by the memset) and read once (now by the converter, before by the postprocess), ~500 MB either
way at d12. The "skipped" row's 15% was that traffic, not overhead. What is left to win is reading a
row block while it is still in L2, which needs knowing the moment it completes: counting arrivals in
the store warp puts an atomic round trip (and a bulk-reduce completion wait) on the dQ pipeline's
critical path; counting at a CTA's end hands the conversions to the longest CTAs (in a causal
backward n = 0 covers every row block and finishes last), stretching the kernel's tail; separate
converter CTAs run only once every main CTA is dispatched, when the early tiles are long out of L2.
Kept behind the macro for reference; the nanochat switch NANOCHAT_FA3_DQFUSE stays off.

## Final: rexp_rmsnorm vs FA3 softmax, training / prefill / decode (H100, `bench_all.py`)

Kernel level (x = FA3 / rexp, > 1 = rexp faster; full tables in `bench_all_h100_{train,prefill,decode}.md`):

| workload | rexp vs FA3 |
|---|---|
| train fwd (B=32 x 2048, d12-d20, full / SWA 512; causal 8k) | 1.06-1.11x |
| train bwd | 0.96-1.00x |
| train fwd+bwd | 0.94-0.97x |
| prefill, B=1 2k / 8k / 32k prompts | 1.09 / 1.17 / 1.22x |
| prefill, 8 x 8k prompts; 32k SWA 512; 4 x 2k chunk vs 32k cache | 1.21x; 1.10x; 1.10x |
| prefill, 512-token chunk vs 32k cache (split territory) | 0.92-0.95x |
| decode B=1, 32k / 128k cache (split-KV, atomic reduction vs FA3 split + combine) | 1.02x (both at the KV bandwidth) |
| decode B=1, 2k / 8k cache | 0.83 / 0.93x (the atomic tail: 3 serialized L2 round trips) |
| decode B=8-64 | 1.01-1.05x |
| decode SWA 512, 8 x 32k | 1.07x unsplit (FA3's heuristic splits by cache length: 0.79x) |

The split-KV atomic reduction (`split_fn_rexp_pre`) is what makes small-batch long decode work at
all -- B=1 at 32k: 320 us unsplit -> 64 us with 8 splits, 128k: 1268 -> 224 us -- but FA3's combine
kernel is not the bottleneck there (reading the cache is), so it only reaches parity with FA3.

End to end (nanochat, one GPU, FP8 + torch.compile, hybrid SWA; `scripts/bench_attn_kinds.py`,
`scripts/bench_decode.py`): training step (fwd+bwd) rexp 1.1% slower at d12/d16/d20 (NANOCHAT_FA3_DGAIN
on); whole-model prefill with the KV cache 1.18-1.40x faster at 8k-32k; decode per token at parity
(eager decode is host-bound, ~14 ms/token at d20 whatever the kernel). 8 x H100 d12, 12B tokens (before
DGAIN): 130.4 vs 127.0 ms/step (2.7% slower), final val bpb 0.7752 vs 0.7745.
