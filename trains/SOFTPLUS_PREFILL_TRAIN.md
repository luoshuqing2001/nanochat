# Fixed KV tasks in prefill and training

GB10 measurements, 2026-09-22. This extends the earlier
[single-query decode optimization](SOFTPLUS_FIXED_KV.md).
The implementation is in `nanochat/softplus_fixed_kv.py`; raw final results are in
[`softplus_prefill_train_gb10.json`](softplus_prefill_train_gb10.json).

## Task layout and memory

Forward tasks own `(query tile, fixed absolute KV interval)`. The KV interval is
independent of the query's visible length. Only tasks intersecting the causal or
sliding-window band are scheduled; the inner loop skips fully masked tiles.
Tasks are ordered by diagonals to avoid issuing all updates to one output row
together. The shape-dependent task table is cached.

The key change to aggregation is ownership: a query tile with one contributor
writes directly to the output. Only query tiles with multiple contributors occupy
the compact FP32 workspace and use atomic-add. A final kernel converts those rows.
No query tile mixes a direct store with atomic writes to the same destination.

Backward uses a fixed KV tile and a fixed query interval. It recomputes each score
tile once for dQ, dK and dV, reuses `exp(-abs(s))` for softplus and sigmoid, and
accumulates dK/dV in registers across the query interval. dQ contributions use
FP32 atomics. dK/dV use compact FP32 accumulators only when multiple tasks own the
same KV tile; otherwise they write directly. The simpler query-resident all-atomic
backward remains available for benchmarking but is not the selected Triton path.

The softplus map uses the existing CuTe log1p polynomial (absolute approximation
error below 1.4e-5), rather than adding a second transcendental per score. The
sigmoid uses a stable piecewise expression instead of `1-exp(-softplus(s))`.
Accumulation stays FP32; atomic update order is not bitwise deterministic.

Both implementations register forward/backward custom ops with fake tensor rules
and autograd, so surrounding operations can stay in the compiled model graph.

## Selection and explicit controls

The default remains `NANOCHAT_SOFTPLUS_IMPL=hybrid`.
`NANOCHAT_SOFTPLUS_SPLITS` now defaults to `auto`. On measured GB10 shapes
(48 SMs, BF16, B×H=1, D128, equal Q/K lengths of 4096/8192/16384, full causal),
training and prefill select the new Triton forward with KV intervals of
512/1024/2048 tokens. A follow-up also enables a 1024-token interval for
B1/H6/T2048/D128 BF16 full causal on GB10. The backward heuristic now counts
SM120's actual 64-token KV tiles, avoiding unnecessary splitting on that shape.
See [Softmax comparison and batch-one follow-up](SOFTPLUS_VS_SOFTMAX.md).
Training retains the existing backward kernels, which
beat the new backward in those measurements. Other training shapes keep their
original forward, and prefill keeps the existing CuTe policy. Integer SPLITS
values retain the old fixed-number-of-splits experiment; SPLITS=1 disables the
new automatic training forward.

To force the fixed-interval algorithm for both forward and backward:

```sh
NANOCHAT_SOFTPLUS_IMPL=fixed_triton \
NANOCHAT_SOFTPLUS_KV_CHUNK=1024 \
NANOCHAT_SOFTPLUS_Q_CHUNK=1024 \
python trains/profile_step.py --depth 12 --device-batch-size 8 --seq-len 2048 --attn-kind softplus
```

`fixed_triton` uses the new compact-accumulator kernels. `fixed` uses CuTe's fixed
interval scheduler in both directions. KV_CHUNK and Q_CHUNK are token counts,
not split counts. The latter controls query intervals in backward. Both modes
also reach the KV-cache prefill entry point. The CuTe KV chunk must be a multiple
of its tile width (128 for D64 on SM120, otherwise 64); query chunks and Triton KV
chunks must be multiples of 64.

Experiment scripts sourcing `_exp_common.sh` clear these overrides. Set them
after sourcing that file in an experiment definition; an environment prefix on
`exp_d12_softplus.sh` will intentionally not override its configuration.

## Results

CUDA Graph timings include zeroing and conversion. BF16, D128, B1/H1, full causal:

| T | Prefill before → after (ms) | Training attention fwd+bwd before → after (ms) |
|---|---:|---:|
| 4096 | 0.09476 → 0.08811 | 0.42179 → 0.34824 |
| 8192 | 0.30114 → 0.26961 | 1.25180 → 1.11283 |
| 16384 | 1.17266 → 0.98892 | 4.26019 → 4.02622 |

Prefill is compared with the previous **already balanced CuTe** path. Training
is compared with the previous unsplit forward and the same backward; the training
gain is therefore not evidence that the new atomic backward is faster.

Controls retained their original kernels: B1/H6/T2048 training 0.6795 → 0.6803 ms;
B1/H6/T8192 7.014 → 7.090 ms; B8/H12/T2048 full context 8.250 → 8.283 ms;
the same shape with left window 512 4.911 → 4.870 ms. These differences are
measurement variation, not structural speedups. Testing a broad occupancy-only
training switch made H6/T2048 slower, so automatic selection is restricted to the
measured winning shapes. Forcing splitting on large training shapes was slower
because of additional accumulator traffic, and is left as an explicit experiment.
Whole-model training throughput or convergence has not been measured here.

## Validation and reproduction

```sh
python -m unittest discover -s tests -p test_softplus_fixed_kv.py -v
python trains/bench_softplus_fixed_kv.py --integrated --output /tmp/softplus_train.json
python trains/bench_softplus_fixed_kv.py --long --output /tmp/softplus_prefill.json
```

Tests cover FP16/BF16, ragged and rectangular Q/K lengths, alpha 0/0.5/1, sliding
windows including zero width, packed strided QKV inputs, output and dQ/dK/dV
against FP32 autograd, exact task coverage, real cache append, and fullgraph
`torch.compile` forward/backward for both explicit backends and automatic dispatch.
Max and RMS errors are checked; no training-data download is needed.
