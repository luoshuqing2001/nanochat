# Softplus long-context scheduling: 32K / 64K

NVIDIA GB10, BF16, B1, H1/H6, D64/D128, full causal. No default dispatch changes.

## Implementation

```python
out = softplus_attn_fa4_func(q, k, v, kv_split_size=1024,
                            stream_atomic=True, kv_major=True)
```

The new `kv_major` option reorders the same bounded KV tasks by KV start position,
then descending query tile. Each head visits different output owners sharing a KV
interval before moving to the next interval. This aims to reuse K/V and separate
updates to the same atomic output slot. It changes neither Softplus math nor tile
sizes, output workspace, or the number of CTAs. Short tiles retain direct writes.
The original query-major order remains selectable and remains the default.
Private partial and last-completer reductions also accept the new order.

Cache reuse and contention reduction are hypotheses motivated by task order, not
hardware-counter measurements. Reordering has no universal speedup guarantee.

## Method

Two processes with opposite initial variant orders, each with three alternating
trials. Aggregate timings are medians of the six samples. Only one CUDA graph is
alive at a time; every timing includes zeroing, reduction and output conversion.
Adaptive 1–10 replays target at least 30 ms per sample. Each candidate is warmed
before graph capture. Train means attention forward plus backward, not optimizer,
full-model training or data loading. No decode changes were made.

All candidates are checked against original Softplus outputs and gradients at
full sequence length. Eleven selected query positions are additionally checked
against PyTorch FP32 exact Softplus over the entire KV sequence. This avoids
allocating a dense T-by-T reference. Small regression cases cover compiled
gradients, unequal lengths, BF16/FP16, windows, strided inputs and graph replay.

FA4 is Softmax FA4 on this GPU, with the tuned SM120 tiles and a second control
using whole-query LPT scheduling. Comparisons below use the faster of these two
controls for each shape. Softplus and Softmax are different mathematical operators;
we compare latency, not numerical parity between them.

## Measurements

Milliseconds; lower is better. `1024 KV` is the new order. `8192 KV` changes both
segment size and order. `Whole LPT` is the previously implemented unsplit schedule.

| T | H | D | Phase | Default | 1024 original | 1024 KV | 8192 KV | Whole LPT | FA4 best |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 32K | 1 | 64 | prefill | 2.032 | 2.074 | 2.128 | 1.899 | 1.828 | 1.689 |
| 32K | 1 | 64 | train | 6.824 | 6.871 | 6.917 | 6.767 | 6.616 | 7.094 |
| 32K | 1 | 128 | prefill | 3.653 | 4.091 | 4.240 | 3.676 | 3.333 | 3.306 |
| 32K | 1 | 128 | train | 15.426 | 15.707 | 15.996 | 15.343 | 14.984 | 13.464 |
| 32K | 6 | 64 | prefill | 11.172 | 12.955 | 13.181 | 11.550 | 11.146 | 10.139 |
| 32K | 6 | 64 | train | 39.623 | 41.645 | 41.916 | 40.151 | 39.652 | 42.379 |
| 32K | 6 | 128 | prefill | 20.123 | 24.813 | 25.336 | 21.759 | 20.746 | 19.656 |
| 32K | 6 | 128 | train | 90.171 | 94.590 | 94.781 | 92.042 | 90.000 | 80.428 |
| 64K | 1 | 64 | prefill | 7.652 | 8.305 | 8.485 | 7.480 | 7.283 | 6.704 |
| 64K | 1 | 64 | train | 26.535 | 26.871 | 27.138 | 26.326 | 25.957 | 27.961 |
| 64K | 1 | 128 | prefill | 14.132 | 20.366 | 17.458 | 14.188 | 13.160 | 13.050 |
| 64K | 1 | 128 | train | 61.103 | 67.403 | 64.030 | 60.737 | 59.690 | 52.718 |
| 64K | 6 | 64 | prefill | 43.945 | 50.356 | 51.024 | 45.031 | 43.547 | 39.771 |
| 64K | 6 | 64 | train | 157.596 | 163.534 | 163.464 | 157.643 | 156.801 | 168.122 |
| 64K | 6 | 128 | prefill | 79.973 | 134.237 | 107.037 | 84.732 | 80.705 | 77.585 |
| 64K | 6 | 128 | train | 355.606 | 406.926 | 381.942 | 360.368 | 357.818 | 316.069 |

Geometric mean speedup against the faster FA4 control, eight shapes per phase:

| Schedule | Prefill | Train |
|---|---:|---:|
| default | 0.911x | 0.964x |
| whole_lpt | 0.944x | 0.977x |
| cap1024 | 0.746x | 0.916x |
| cap1024_kv | 0.770x | 0.925x |
| cap4096_kv | 0.874x | 0.958x |
| cap8192_kv | 0.898x | 0.962x |

## Interpretation and limits

At 64K/H6/D128, KV-major changes the 1024-cap prefill from 134.237 ms to
107.037 ms (1.254x), and forward+backward from 406.926 ms to 381.942 ms (1.065x).
However, default unsplit Softplus is still faster than both fixed-cap variants.
At 32K, KV-major does not produce a consistent improvement. The new option remains
experimental and is not automatically enabled. No tested Softplus prefill option
beats the faster FA4 control in these eight shapes. D64 training beats FA4 in all
four D64 cases, including with the preexisting default; D128 training does not.

Both KV-cap regression tests passed (21.282 s), followed by all seven existing
completion/scheduling regression tests (47.946 s). Across the first long-context
run, maximum normalized error versus the old Softplus outputs/gradients was
0.000178; sampled FP32 exact Softplus reference error was at most 0.00435.
Each independent reverse-order run also checks correctness before timing.

Longer sequences create more independent query CTAs even at B1/H1. On SM120 an
M64 tile gives 512 query tiles at 32K and 1024 at 64K, before multiplying by heads.
A fixed 1024 cap also makes total segment count grow approximately quadratically
with T. At 64K, H1, D128, there are 33,280 segment CTAs instead of 1024 whole-query
CTAs. Output atomic updates and Q reloads grow with the number of segments.
Increasing the cap reduces those costs; KV-major ordering cannot remove them.

Backward is unchanged, so a training win cannot be attributed entirely to this
forward scheduling change. Whole-query ordering also benefits Softmax. These
experiments do not establish universal Softplus superiority or a cross-GPU rule.

Reproduce:

```sh
python trains/bench_softplus_long.py --output trains/softplus_long_kv_major.json
python trains/bench_softplus_long.py --reverse --output trains/softplus_long_kv_major_reverse.json
python trains/summarize_softplus_long.py
```
