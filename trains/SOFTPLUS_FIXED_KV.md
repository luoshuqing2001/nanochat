# Fixed-size KV tasks for softplus attention

The subsequent prefill/training implementation and results are documented in
[SOFTPLUS_PREFILL_TRAIN.md](SOFTPLUS_PREFILL_TRAIN.md).

Measured on NVIDIA GB10, 2026-09-22, PyTorch 2.14.0+cu130, BF16.
Raw measurements, including the chunk sweep, are in
[`softplus_fixed_kv_gb10.json`](softplus_fixed_kv_gb10.json).

## Implementation

`nanochat/softplus_decode.py` assigns each single-query task a fixed number of KV
positions. Tasks compute QK and softplus in FP32, reduce their local weighted V
sum, and atomically add one output vector into an FP32 accumulator. The last task
is masked to the cache/window boundary. There are no host work tables or
inter-block barriers. CUDA still executes each Triton program as a thread block;
the scheduling unit is a KV chunk rather than an entire query's KV scan.

This avoids running a 128-row MMA tile for one real query. Default chunks are
64 positions for up to 1024 visible keys, and 256 for longer contexts. Very short
contexts of at most 64 keys need only one task and write directly in output dtype,
without zeroing or a conversion pass. Small tasks are important for short windows:
a single 512-key task had better eager launch overhead but worse graph replay
latency than 64-key tasks, so it is not the default.

`softplus_attn_fa4(..., num_splits="auto")` selects this path for Tq=1, FP16/BF16,
Dqk/Dv in {64, 128}, and matching Q/K/V head counts. The KV-cache inference entry
already uses this option, so no environment changes are needed. Explicit
`balanced_chunk` or integer `num_splits` continues to select the existing CuTe
path. Prefill and the training forward continue to use their existing kernels.

The hybrid training dispatch now normalizes windows before selecting backward.
The model represents full context as `(sequence_len, 0)`; previously this selected
Triton backward even though the normalized window was unlimited. Full-context
layers now select CuTe backward, and genuinely windowed layers retain Triton.

## Results

All decode rows below have B1/H6/D128. Times include accumulator initialization
and output conversion where applicable. The baseline is the previous CuTe
balanced path, not unsplit attention or softmax.

| Cache / left window | Eager baseline → new (µs) | CUDA Graph baseline → new (µs) |
|---|---:|---:|
| 4096 / unlimited | 31.71 → 24.06 | 26.92 → 12.00 |
| 16384 / unlimited | 237.16 → 232.34 | 234.70 → 225.18 |
| 65536 / unlimited | 916.04 → 892.77 | 898.69 → 880.18 |
| 65536 / 511 (512 visible keys) | 30.54 → 25.09 | 7.53 → 5.06 |
| 65536 / 512 (513 visible keys) | 30.88 → 25.13 | 10.64 → 5.37 |

Additional graph measurements: B1/H12/D128/Tk65536, 1809.93 → 1766.07 µs;
B8/H6/D128/Tk4096, 453.99 → 441.34 µs;
B1/H6/D64/Tk4096, 16.60 → 8.55 µs.
Small differences in the long-context cases are close to measurement variability.

For one call per newly encountered cache length (4097 through 4160), including
Python dispatch and table construction but excluding JIT compilation, mean time
per call in the median of three trials was 48.70 → 26.39 µs without a window and
45.79 → 27.32 µs with left window 511. Multiple model layers can share an existing
work table, so these are not whole-model per-token speedups.

Training attention-only forward + backward, B8/T2048/H12/D128:

| Layer | Eager before → after (ms) | CUDA Graph before → after (ms) |
|---|---:|---:|
| Full context, `(2048, 0)` | 9.013 → 8.367 | 8.837 → 8.258 |
| Windowed, `(511, 0)` | 5.132 → 5.037 | 5.008 → 5.016 |

The windowed algorithm is unchanged; its differences are measurement noise.
The full-context improvement is the backend-selection fix, not a new training
split-K algorithm. Whole-model training throughput has not been measured here.

## Reproduction and correctness

```sh
python trains/bench_softplus_decode.py --training --dynamic --output /tmp/softplus_decode.json
python trains/bench_softplus_decode.py --head-dim 64 --quick --output /tmp/softplus_decode_d64.json
python -m unittest discover -s tests -p test_softplus_optimization.py -v
```

The regression suite covers FP16/BF16, QK/V dimensions 64 and 128, strided feature
dimensions and overallocated cache views, arbitrary scale, alpha 0/0.5/1,
window and chunk boundaries, cache append, explicit CuTe comparison, and
full-context forward/dQ/dK/dV against an FP32 reference. The forward matrix has
96 combinations (24 shapes/dtypes × four chunk choices). Decode tolerances are
0.5% (BF16) and 0.1% (FP16) of reference max magnitude; full-context output and
gradient tolerances are 2%. FP32 atomic accumulation is not bitwise deterministic.
