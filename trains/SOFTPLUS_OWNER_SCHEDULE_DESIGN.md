# Softplus owner-preserving schedule design

Status: historical design. Forward and backward owner pairing, head-local LPT,
and an opt-in hybrid selector are now implemented; see `SOFTPLUS_OWNER_RESULTS.md`
for measurements and limitations. Based on GB10 experiments in
`SOFTPLUS_KV_CAP_RESULTS.md`, `SOFTPLUS_COMPLETION_RESULTS.md`, and
`SOFTPLUS_LONG_RESULTS.md`. Keep current production dispatch until validation.

## Evidence and objective

The objective is minimum full attention latency, including workspace initialization
and output reduction, rather than equal KV-token counts per CTA.

- Fixed cap512 helped B1/H1/T4096 prefill, but regressed many higher-parallelism cases.
- For 64K/H6/D128, cap1024 prefill was 134.237 ms; KV-major reduced it to
  107.037 ms, cap8192 KV-major to 84.732 ms. Unsplit default was still 79.973 ms;
  the faster Softmax FA4 control was 77.585 ms.
- At 64K/H1 a full M64 query grid has 1024 tasks, compared with 33,280 cap1024
  tasks. Smaller tasks duplicate Q loads and emit more FP32 output updates.
- Whole-query global LPT helps small batches but can severely regress larger ones.
  Therefore head interleaving and cache locality cannot be ignored by the model.
- Last-completer reduction did not provide a consistent win. D64 backward N128
  grouping achieved only about 0.697x over nine prior cases despite fewer dQ atomics.
- D64 long-context training already beats the measured FA4 baseline without the
  latest forward changes. D128 backward remains a separate optimization problem.

## 1. Large-workload forward: pair complete output owners

Retain current QK/Softplus/PV tiles and their pipeline: on SM120, M64/N128 for D64,
M64/N64 for D128. Do not retain two output accumulators at once.

Let M = ceil(Tq / BM), delta = Tk - Tq, and, for full causal attention,

    L(m) = ceil(min(Tk, delta + (m+1)*BM) / BN).

L(m) is the physical KV-iteration count, not an exact runtime prediction. For each
batch/head and pair p in [0, ceil(M/2)), process query tiles:

    high = M-1-p
    low  = p

The same CTA executes high first, then low if different. Each invocation of the
existing mainloop initializes fresh registers, consumes the whole visible KV
range, applies count scaling, and directly writes its own output tile. Drain
outstanding async copies and synchronize shared-memory reuse between tiles.
No partial workspace, zeroing kernel, output atomic, counter, or finish kernel.

For square causal sequences without a partial last tile, L(p)+L(M-1-p) is constant
or differs only by KV rounding. With BM=BN and aligned T this sum is M+1. Thus
large and small triangular jobs are paired *inside* the CTA instead of relying on
CUDA's block placement to pair them. An odd middle tile executes once. Unequal
Q/K lengths and partial tiles require the exact formula and coverage tests.

Use arithmetic indexing: initial prototype grid `(ceil(M/2), H, B)`, preserving
head-local neighboring block IDs. This is an ordering preference, not SM affinity
or a guarantee of physical launch order. Do not flatten head-first as the existing
global LPT variant does. The kernel holds only one live Q/O tile, so pair execution
need not double register/shared-memory footprint; verify compiled resources.
Long-first also lets the current descending-KV mainloop finish near low KV indices
before starting the short prefix; any cache benefit must be measured.

This differs from earlier persistent Stream-K: no concatenated KV stream, no
boundary splits, no CPU/GPU task table and no per-task global descriptor reads.
It differs from whole-query LPT: each CTA *owns a pair*, rather than only reversing
an otherwise one-query-per-CTA grid. It is a short sequential work list, not an
assumption that one persistent CTA can be permanently assigned to each SM.

Guard: pairing halves the number of independent CTAs. Let P = SM_count * R, where
R is a resource-derived candidate resident-CTA count, checked against actual
latency. Require enough paired CTAs for at least roughly two waves as an initial
candidate rule; calibrate rather than treating two as a proven optimum. Preserve
ordinary unsplit scheduling when pairing reduces parallelism too far. Windowed
attention is already close to uniform-cost: leave it unpaired initially. Nearly
uniform rectangular ranges also have less reason to pair.

Pairing is applicable to Softmax as well. Give Softmax the same implementation
when comparing, or the experiment cannot isolate a Softplus-specific advantage.

## 2. Underfilled forward: bounded extra split tasks

For small total output-owner counts, retain Softplus's ability to sum independent
KV intervals. Choose the amount of extra parallelism first, not a universal cap.
Let Q = B*H*M and use a candidate target of P to 2P total independent tasks.
Start from complete query owners, and split the longest predicted jobs only if
it improves estimated makespan after including communication costs.

Bound the total extra tasks by O(P), e.g. Bextra=max(P, 2P-Q) as a candidate budget.
Then total task count is at most Q+Bextra, rather than O(B*H*T^2/(BM*C)). The budget
is across batch and heads, not repeated independently for each head. R, the target
wave count and minimum useful grain are calibration parameters, not assertions.
The grain should initially be tested at 1024/2048/4096 KV tokens for long cases;
allow the existing smaller-grain small-context path where it has measured gains.
Split intervals along KV tile boundaries, with approximately equal iteration
counts to avoid tiny tails. A hard user-specified cap remains an explicit override.

For Q well above the saturation threshold, default to complete owners. Do not
split solely because T is large. A future tail split must demonstrate saved idle
waves exceeding its additional communication cost; pairing is the first attempt
to remove that tail without splitting.

Split owners use one zeroed FP32 output slot each, local accumulation over their
entire interval, and one atomic update per output element. Complete owners directly
write final output. A separate finish kernel processes only split owners. This
reuses the tested protocol without spinning or inter-CTA synchronization.

Keep private-partial and last-completer variants as controls, not prerequisites.
Do not make last-completer part of the first implementation: measured results do
not justify its fences, counters, metadata and extra code paths as the default.

## 3. Locality and reduction cost model

Never infer output reduction to be cheap just because it is algebraically legal.
Measure the cost of a full tile and split epilogue at the real BM/D/dtype/layout.
For extra split count E, extra output updates alone contain

    E * BM * Dv * sizeof(FP32)

bytes of logical atomic payload. This is not total DRAM traffic: read-modify-write,
cache-line effects, contention and finish reads add hardware-dependent costs.
Also account for repeated Q loads, CTA pipeline startup, workspace zeroing and
finish launches. KV traversal order changes cache behavior, so these costs cannot
be modeled only as a fixed two-KV-iteration charge per CTA/split.

A useful decision is:

    predicted idle-tail time saved > added split/merge time + uncertainty margin.

Calibrate separately for D64/D128, phase, and head-local vs interleaved layouts.
Use a small set of measured regimes rather than a per-test-shape winner table.
Require a material margin (initial experiment: 5%) before selecting an extra
workspace path; validate margins on held-out lengths and batches.

Prefer head-local order. KV-major can be an opt-in split-only alternative, useful
in the observed 64K/D128 regime but not globally. Grouped KV traversal within a
small query cohort is a later locality sweep, after the no-split paired baseline.
No claims about L2 hit rate or bandwidth bottlenecks without counters/profiling.

## 4. Backward: preserve KV ownership and test reverse-triangle pairing

Current backward assigns KV owners that accumulate dK/dV while visiting contributing
query tiles; dQ is accumulated with the existing atomic protocol. For full causal
attention, low-index KV owners have long Q ranges and high-index owners short ones.
Pair complementary KV owners inside one CTA, sequentially, with fresh dK/dV
accumulators and shared-memory pipeline reset between them.

Initially retain the validated `(BM,BN)=(64,64)` backward configuration and current
math/storage choices. Do not repeat the N128 grouping experiment or hold two owners'
dK/dV accumulators simultaneously. Pairing does not reduce the existing dQ atomic
count, nor does it add an output partial buffer for dK/dV. It addresses scheduling
only; shared atomic/cache interactions can still make it slower.

Guard against too few paired KV tasks, and preserve unsplit windows. Test backward
on its own using fixed Q/K/V/dO and a consistent input/output layout, then measure
forward+backward together. Do not attribute cache/allocation changes in combined
measurements to a faster backward instruction schedule.

## 5. Implementation and evaluation order

1. Add an arithmetic full-owner paired forward branch in `flash_fwd.py`, bypassing
   stream metadata and using the normal epilogue. Expose as an experimental schedule
   in the interface/API and custom-op schema. Implement the identical Softmax control.
2. Test exact tile ownership, odd counts, unequal Q/K, edge masks, BF16/FP16, strided
   tensors, compiled gradients, concurrent streams and repeated graphs. Verify no
   accidental partial allocations or increased live accumulator resources.
3. Benchmark native unsplit, head-local whole LPT, current global LPT, paired full
   owners, and both matched Softmax orders. Keep the existing cap1024/KV-major and
   cap8192 variants as reduction-cost controls.
4. Only after paired performance is known, add the bounded-extra-task hybrid for
   underfilled shapes. Its own ablation must separate task count, ordering and
   reduction mechanism. Compare against existing small-context winners.
5. Prototype paired KV-owner backward independently. Keep all default policies
   unchanged until both regressions and cross-shape timing support promotion.

Primary matrix: existing 4K/H1 gains; 2K/8K at B1/B8 and H6; 32K/64K at B1 and
H1/H6, D64/D128. Add holdout 12K/48K and odd lengths, windows and strided inputs.
Measure full attention prefill and forward+backward, plus separate backward to
explain training results. Use two processes with opposite orders, one live graph,
all launches included, and resource/workspace accounting. Whole-model throughput
requires an additional model run and must not be inferred from these timings.

Success is a stable improvement over the best available Softplus schedule without
large regressions, and an honest comparison against equally scheduled Softmax.
No current experiment proves that scheduling alone can make every Softplus case
faster than FA4; pairing itself remains an unmeasured hypothesis.
