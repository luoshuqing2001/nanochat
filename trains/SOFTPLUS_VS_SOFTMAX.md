# Current Softplus versus Softmax on GB10

Measured 2026-09-22. BF16, head dimension 128, equal Q/K/V head counts, causal attention, Softplus alpha=1. Inputs are random and contiguous in (B,T,H,D) layout. The two attention functions have different semantics; this is a speed comparison, not a quality-equivalence claim.

Timing is the median of three CUDA Graph measurements (60 ms target per trial), with alternating implementation order after JIT warmup. Includes output allocation kernels, accumulator zeroing and casts, and SDPA sliding-window mask construction. Excludes CPU dispatch/JIT, cache append, projections, the rest of the model, and optimizer. Train measures forward plus all three input gradients via autograd. All measured outputs/gradients were checked for finiteness.

Softplus uses the current auto prefill/decode and hybrid training dispatch. Softmax uses FA4 for training. Both FA4 and the actual nanochat SDPA inference helper are shown for forward. On this checkout SM120 FA4 requires num_splits=1; it is a weak decode baseline. Softmax can support split-KV algorithms in other implementations. Neither backend was exhaustively tuned here.

All times below are milliseconds; W is the left window, so W=511 means at most 512 visible keys, and -1 means full causal.

## prefill

| B | H | Tq | Tk | W | Softplus | Softmax FA4 | Softmax SDPA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4096 | 4096 | -1 | 0.08797 | 0.17241 | 0.11221 |
| 1 | 1 | 8192 | 8192 | -1 | 0.26894 | 0.42810 | 0.40494 |
| 1 | 1 | 16384 | 16384 | -1 | 0.98563 | 1.27303 | 1.20096 |
| 1 | 6 | 2048 | 2048 | -1 | 0.15752 | 0.15290 | 0.14580 |
| 1 | 6 | 8192 | 8192 | -1 | 1.57541 | 1.61962 | 1.52413 |
| 8 | 12 | 2048 | 2048 | -1 | 1.57103 | 1.61907 | 1.54951 |
| 8 | 12 | 2048 | 2048 | 512 | 1.03660 | 1.06523 | 9.47278 |

## decode

| B | H | Tq | Tk | W | Softplus | Softmax FA4 | Softmax SDPA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 4096 | -1 | 0.01172 | 0.17096 | 0.02277 |
| 1 | 6 | 1 | 16384 | -1 | 0.20076 | 0.68152 | 0.20854 |
| 1 | 6 | 1 | 65536 | -1 | 0.85684 | 2.72758 | 0.87294 |
| 1 | 12 | 1 | 65536 | -1 | 1.72114 | 2.73709 | 1.73089 |
| 1 | 6 | 1 | 65536 | 512 | 0.00527 | 0.02635 | 0.00927 |
| 8 | 6 | 1 | 4096 | -1 | 0.41284 | 0.42694 | 0.41789 |

## train

| B | H | Tq | Tk | W | Softplus | Softmax FA4 | Softmax SDPA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4096 | 4096 | -1 | 0.34630 | 0.48366 | — |
| 1 | 1 | 8192 | 8192 | -1 | 1.11092 | 1.34571 | — |
| 1 | 1 | 16384 | 16384 | -1 | 4.04028 | 4.64535 | — |
| 1 | 6 | 2048 | 2048 | -1 | 0.68709 | 0.60959 | — |
| 1 | 6 | 8192 | 8192 | -1 | 7.03753 | 6.93085 | — |
| 8 | 12 | 2048 | 2048 | -1 | 8.30172 | 8.56734 | — |
| 8 | 12 | 2048 | 2048 | 512 | 4.87824 | 5.69580 | — |
| 8 | 6 | 2048 | 2048 | -1 | 4.20018 | 4.32883 | — |
| 8 | 6 | 2048 | 2048 | 511 | 2.61571 | 2.90273 | — |
| 32 | 6 | 2048 | 2048 | -1 | 16.51658 | 17.26491 | — |
| 32 | 6 | 2048 | 2048 | 511 | 10.04372 | 11.55708 | — |

## Interpretation

- Prefill: at B=H=1 fixed-KV scheduling wins against both tested Softmax backends. At ordinary multihead full-causal shapes the difference is small, sometimes favoring SDPA. The 9.1x windowed win over SDPA reflects its explicit-mask fallback: against native local FA4 the same case is only 1.03x.
- Decode: B1/H6/T4096 is 1.94x faster than SDPA; a 513-key window is 1.76x. At 64K and at B8/H6/T4096 the difference is about 0–2%. This is consistent with KV traffic dominating once enough parallelism is available; no bandwidth-counter profiling was performed.
- Train: d12-like H6/T2048/B8–32 full-causal attention takes 3–4% less time, and a 512-key window takes 10–13% less. B1/H6/T2048 full-causal is 13% slower. The default retains the previously faster backward; these gains are not proof that forcing the new atomic backward everywhere helps.
- Whole-model throughput and convergence were not measured. Attention speedups cannot be used directly as training-step or generation-token speedups.

## Follow-up: why B1/H6/T2048 training loses

A controlled repeat (`--small-train-diagnosis`, raw results in
`softplus_small_train_diagnosis_gb10.json`) kept the training forward unchanged
and disabled only backward query splitting:

| Path | ms |
|---|---:|
| Softplus training forward, unsplit | 0.14719 |
| Softmax forward | 0.15161 |
| Softplus forward + backward, default auto | 0.68159 |
| Softplus forward + backward, backward unsplit | 0.61735 |
| Softmax forward + backward | 0.60946 |

The backward heuristic estimated 96 base KV tasks, below its 3 × 48 SM target,
and enabled splitting. A subsequent check found this estimate used 128-token
tiles, while SM120 actually uses 64-token tiles and has 192 tasks. Splitting
requires FP32 dK/dV accumulators, zeroing, atomic
updates and conversion. Disabling that split saves 0.06424 ms and removes about
89% of the default gap to Softmax; the residual is about 1.3%. This identifies
an overaggressive backward dispatch decision for this shape, not an inherent
batch-one limitation of Softplus. Production dispatch was not changed in that
initial diagnostic comparison. The unsplit training forward differs from the auto
prefill scheduler, which explains its different forward timing above.

## Implemented batch-one tuning

The default backward heuristic now uses the actual 64-token KV tile on SM120,
while retaining measured GB10 single-head 4K/8K/16K chunk choices. In particular,
16K still benefits from balancing the causal tail even with sufficient base tasks.
For GB10 BF16 B1/H6/T2048/D128 full causal, the automatic forward now uses
1024-token fixed KV chunks; backward retains the faster unsplit CuTe kernel.
No score-map or gradient formula changes were needed.

Two whole-call measurements after integration gave 0.593–0.597 ms versus
0.608–0.616 ms for FA4 Softmax. This is a modest roughly 2–3% lead, versus the
previous 0.677–0.682 ms Softplus. The values in the main tables above describe
the implementation before this follow-up tuning. Raw tuning and regression
results are in `softplus_small_tuning_gb10.json`,
`softplus_small_train_optimized_gb10.json`, and `softplus_small_regression_gb10.json`.

Final regression after retaining the long single-head choices:

| B / H / T | Softplus before (ms) | Softplus after (ms) | Softmax FA4 (ms) |
|---|---:|---:|---:|
| 1 / 6 / 2048 | 0.67372 | 0.59395 | 0.60504 |
| 1 / 1 / 4096 | 0.34449 | 0.34662 | 0.48540 |
| 1 / 1 / 8192 | 1.10937 | 1.10901 | 1.34517 |
| 1 / 1 / 16384 | 4.02807 | 4.05092 | 4.64936 |
| 8 / 12 / 2048 | 8.31831 | 8.23183 | 8.54095 |

The target's median is about 12% below the previous Softplus and 1.8% below
Softmax in this final run. Controls retain their former kernels/chunks, with
roughly 1% measurement variation. The two optimization test suites passed all
10 tests, including FP32 output/gradient references on packed QKV, fullgraph
compilation of the new auto path, and cache/decode regressions.

## Reproduce

```sh
python trains/bench_softplus_vs_softmax.py --output trains/softplus_vs_softmax_gb10.json
python trains/bench_softplus_vs_softmax.py --d12-train --output trains/softplus_vs_softmax_d12_gb10.json
python trains/bench_softplus_vs_softmax.py --small-regression --output trains/softplus_small_regression_gb10.json
```
