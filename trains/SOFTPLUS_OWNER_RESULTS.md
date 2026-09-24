# Complete-owner scheduling implementation and results

GB10 / SM120, BF16. Default dispatch is unchanged. The design is in
`SOFTPLUS_OWNER_SCHEDULE_DESIGN.md`; this document describes the implemented state.

## Implemented paths

```python
# Explicit experimental selector; production defaults are unchanged.
from flash_attn_4.softplus_api import softplus_owner_schedule_options
opts = softplus_owner_schedule_options(q, k)
out = softplus_attn_fa4_func(q, k, v, **opts)

```

```python
# Complete query tiles: longest first within each batch/head, no work table.
out = softplus_attn_fa4_func(q, k, v, head_lpt=True)

# One CTA sequentially computes a long query tile and its short complement.
out = softplus_attn_fa4_func(q, k, v, owner_pair=True)

# Experimental independent backward override, no additional KV-owner splitting.
out = softplus_attn_fa4_func(q, k, v, owner_pair=True,
                            bwd_schedule="paired_shared")  # D64 shared-P/dS control
# bwd_schedule="paired" retains separate P/dS storage (D128 auto normally does so).
```

Both forward options use the normal final-output epilogue. No output atomic,
partial workspace, zeroing, completion counter, or finish kernel is introduced.
Pairing drains async copies and synchronizes between owners before shared-memory
reuse. Odd middle tiles execute once. Current scope is dense full causal SM120,
equal Q/K/V head counts; explicit pairing/head-LPT reject local windows.

Backward pairing preserves the existing dQ atomic protocol and processes two
complete KV owners sequentially. It does not reduce the dQ update count. Both
shared and separate P/dS variants are explicit experimental overrides. No tile
widening is involved. Refactoring the old backward into `compute_owner_work`
allows the original path to call it once and the paired path to call it twice.

Softmax has matching `sm120_owner_pair` forward/backward controls and
`sm120_head_lpt` forward control. Results below use the fastest measured Softmax
control per shape, not only its original scheduling order.

The bounded split experiment reuses the existing `cute_stream_waves=2,
stream_tiles=True, stream_atomic=True` implementation. Its per-head budget is
ceil(total KV iterations / requested workers). For Q query owners, the additional
segment count is bounded by the requested workers per head (rounding aside), rather
than growing quadratically with sequence length at a fixed KV cap.

The benchmark's `hybrid_candidate` is a provisional regime rule, not default
dispatch: under 2*SM total owners use bounded splitting; otherwise use pairing
when per-head query count is below 2*SM and the paired grid still has at least
2*SM tasks; otherwise use head-local LPT. The thresholds are candidates and do
not constitute a calibrated hardware occupancy model.

## Measurement and validation

Primary data: `softplus_owner_stable.json`, 19 shapes x prefill/forward+backward.
Includes B1/B8, H1/H3/H6, D64/D128, 2K/4K/8K/12K/32K/48K/64K. The 12K/48K cases
are holdouts. Each measurement uses one live CUDA graph, three alternating-order
trials, >=20 ms warmup and a target 60 ms timed sample (up to 10000 replays).
All workspace/zeroing/finish kernels are included. These are attention timings,
not whole-model throughput. The earlier quick/full files used shorter windows
and are exploratory, not the primary basis for small percentage claims.

Softplus candidate outputs and gradients are checked against the original path;
selected query positions also use exact FP32 Softplus over the full KV sequence.
Softmax paired outputs/gradients have a same-math baseline check. Dedicated tests
cover odd tile counts, unequal sequence lengths, masks, strided BF16/FP16 inputs,
compiled gradients, streams and graph replay.

Quick compiled-resource data (`softplus_owner_pair_quick.json.metadata.json`)
showed zero local-memory allocation for forward variants. D64 Softplus registers
increased from 166 to 212 with pairing; D128 remained at 206. Sequential ownership
does not guarantee identical compiler register allocation. Shared storage layout
is unchanged; no achieved-occupancy or hardware-counter claims are made.

## Results

Speedup is original default divided by candidate; values above one are faster.
The training default can use a different automatic forward dispatch from explicit
native options; `native_unsplit` and `bwd_pair_control` are included as controls.

| Schedule | Prefill geomean vs default | Train geomean vs default |
|---|---:|---:|
| owner_pair | 1.085x | 1.017x |
| head_lpt | 1.079x | 1.015x |
| hybrid_candidate | 1.114x | 1.026x |

Milliseconds; the full raw file contains all variants and individual samples.

| B/H/T/D | Phase | Default | Pair forward | Head-local LPT | Hybrid | Best FA4 |
|---|---|---:|---:|---:|---:|---:|
| 1/1/4096/64 | prefill | 0.0778 | 0.0565 | 0.0617 | 0.0494 | 0.0521 |
| 1/1/4096/64 | train | 0.1901 | 0.1696 | 0.1748 | 0.1620 | 0.1812 |
| 1/1/4096/128 | prefill | 0.1384 | 0.0946 | 0.1082 | 0.0953 | 0.0912 |
| 1/1/4096/128 | train | 0.3673 | 0.3655 | 0.3760 | 0.3677 | 0.3366 |
| 8/6/2048/128 | prefill | 0.7446 | 0.7395 | 0.7237 | 0.7469 | 0.7082 |
| 8/6/2048/128 | train | 3.7428 | 3.7224 | 3.8416 | 3.7095 | 3.6342 |
| 1/6/32768/64 | prefill | 11.1457 | 10.9757 | 10.9802 | 10.9829 | 9.9018 |
| 1/6/32768/64 | train | 39.8437 | 39.5679 | 39.6287 | 39.5402 | 42.1559 |
| 1/6/32768/128 | prefill | 20.1505 | 20.0115 | 19.8840 | 19.5392 | 19.0978 |
| 1/6/32768/128 | train | 90.8040 | 89.5290 | 89.1246 | 89.5916 | 78.9297 |
| 1/1/65536/128 | prefill | 13.9913 | 13.6549 | 12.9587 | 12.9574 | 12.9578 |
| 1/1/65536/128 | train | 60.9748 | 60.8511 | 60.1788 | 60.4901 | 52.2653 |
| 1/1/32768/64 | prefill | 2.0228 | 2.0850 | 1.8422 | 1.8376 | 1.6629 |
| 1/1/32768/64 | train | 6.7603 | 6.8156 | 6.5822 | 6.5738 | 7.0008 |
| 1/1/32768/128 | prefill | 3.6646 | 3.7980 | 3.3253 | 3.3263 | 3.2478 |
| 1/1/32768/128 | train | 15.3108 | 15.4447 | 14.9069 | 14.9280 | 13.2749 |
| 1/1/65536/64 | prefill | 7.7170 | 7.6292 | 7.2575 | 7.2834 | 6.5590 |
| 1/1/65536/64 | train | 26.3416 | 26.2150 | 25.8835 | 25.9725 | 27.9193 |
| 1/6/65536/64 | prefill | 43.9597 | 43.4278 | 43.4331 | 43.4164 | 39.2355 |
| 1/6/65536/64 | train | 157.0718 | 158.8304 | 155.6897 | 157.0967 | 166.6434 |
| 1/6/65536/128 | prefill | 81.1507 | 78.8212 | 77.7818 | 77.6876 | 76.0479 |
| 1/6/65536/128 | train | 355.3892 | 355.7644 | 354.8846 | 352.3212 | 313.1512 |
| 1/6/2048/64 | prefill | 0.0744 | 0.0542 | 0.0724 | 0.0555 | 0.0495 |
| 1/6/2048/64 | train | 0.2420 | 0.2233 | 0.2417 | 0.2252 | 0.1994 |
| 8/6/2048/64 | prefill | 0.4232 | 0.4176 | 0.4117 | 0.4078 | 0.3805 |
| 8/6/2048/64 | train | 1.7253 | 1.7312 | 1.7060 | 1.7090 | 1.8556 |
| 1/6/8192/64 | prefill | 0.7958 | 0.7302 | 0.7408 | 0.7497 | 0.6536 |
| 1/6/8192/64 | train | 2.8054 | 2.7478 | 2.7835 | 2.7379 | 2.8253 |
| 1/6/8192/128 | prefill | 1.4357 | 1.3242 | 1.3231 | 1.3301 | 1.2791 |
| 1/6/8192/128 | train | 6.0626 | 5.9755 | 6.0396 | 6.0011 | 5.4134 |
| 1/3/12288/64 | prefill | 0.8893 | 0.8023 | 0.7913 | 0.7946 | 0.7213 |
| 1/3/12288/64 | train | 3.0375 | 2.9622 | 2.9587 | 2.9562 | 3.0872 |
| 1/3/12288/128 | prefill | 1.5990 | 1.4545 | 1.5029 | 1.4675 | 1.4021 |
| 1/3/12288/128 | train | 6.6619 | 6.5377 | 6.4993 | 6.5247 | 5.8574 |
| 1/1/49152/64 | prefill | 4.3810 | 4.1412 | 4.1005 | 4.1170 | 3.7151 |
| 1/1/49152/64 | train | 15.0491 | 14.7542 | 14.6363 | 14.7922 | 15.6477 |
| 1/1/49152/128 | prefill | 7.9362 | 7.5252 | 7.4229 | 7.4158 | 7.2045 |
| 1/1/49152/128 | train | 34.3892 | 33.7237 | 33.6567 | 33.3203 | 29.7513 |

## Interpretation

Equal per-CTA work is insufficient to predict minimum end-to-end latency. Pairing
halves independent task count, increases instruction/lifetime bookkeeping and can
increase register allocation. Head-local long-first scheduling leaves CUDA more
freedom to overlap independent owners and preserves head locality better than
the prior globally interleaved LPT schedule.

Backward has a separate triangular ownership pattern and atomic interactions;
a forward improvement does not establish that the backward pair will help.
Use the separate backward-only control before attributing a training speedup to
backward scheduling. Neither approximate fairness nor total CTA count alone
justifies enabling the hybrid rule by default.

Reproduce:

```sh
python -m unittest discover -s tests -p 'test_softplus_*.py'
python trains/bench_softplus_owner_pair_stable.py --output trains/softplus_owner_stable.json
python trains/bench_softplus_owner_pair_stable.py --quick --reverse --output trains/softplus_owner_stable_reverse.json
python trains/bench_softplus_owner_bwd.py --output trains/softplus_owner_bwd.json
python trains/summarize_softplus_owner.py
```

## Reverse-order confirmation

Six shapes were repeated in a separate process with reversed initial variant order.
prefill: hybrid/default speedup geomean 1.155x, range 0.983–1.575x.
train: hybrid/default speedup geomean 1.025x, range 0.982–1.176x.
The sample set differs from the full 19-shape set; do not compare these means as an improvement between runs. About 2% regressions appeared at B8/H6/T2048/D128 and small D128 training; the policy is not universally faster.

The full 48-test run had one failure in the new head-local scheduler factory. After correcting the inherited factory to return the head-local scheduler, all six owner-schedule tests passed (49.831 s); the other 47 full-suite tests had already passed. No remaining known test failure.

## Backward-only ablation

Same Q/K/V/dO, output/LSE, tile, P/dS configuration and unsplit owners; includes preprocessing and gradient conversion.

| T/H/D | Original backward ms | Paired backward ms | Speedup |
|---|---:|---:|---:|
| 4096/1/64 | 0.1635 | 0.1501 | 1.089x |
| 4096/1/128 | 0.2808 | 0.2845 | 0.987x |
| 32768/1/64 | 4.7686 | 5.4038 | 0.882x |
| 32768/1/128 | 11.5251 | 13.0652 | 0.882x |
| 32768/6/64 | 28.5161 | 28.9466 | 0.985x |
| 32768/6/128 | 68.6122 | 69.6061 | 0.986x |
| 65536/6/64 | 112.0445 | 112.5844 | 0.995x |
| 65536/6/128 | 273.0022 | 271.2154 | 1.007x |
Geomean 0.975x. The automatic backward can use query-range splitting on small shapes, so these unsplit controls are different from the production default. Backward pairing remains an explicit experimental override.
