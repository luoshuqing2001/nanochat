# Fixed KV token cap

2026-09-23, NVIDIA GB10. This is an opt-in schedule; default dispatch is unchanged.

```python
from flash_attn_4.softplus_api import softplus_attn_fa4_func
out = softplus_attn_fa4_func(q, k, v, kv_split_size=512, stream_atomic=True)
```

`softplus_attn_fa4` exposes the same options for inference. The cap is measured in
KV tokens, must be a positive multiple of the internal KV tile size, and is
independent of batch size, head count, SM count, and the worker-wave heuristic.
On SM120 the existing forward tile remains M64/N128 for D64 and M64/N64 for D128.
The planner conservatively caps the physical KV tile span, including masked edges.

For each batch/head/query tile, the planner obtains its causal/windowed KV range.
A range no longer than the cap runs in one CTA and writes output directly.
Longer ranges are partitioned into contiguous cap-sized intervals and a possible
short final interval. Each CTA runs the existing QK -> Softplus -> PV mainloop,
accumulating its entire interval in FP32 registers. It then atomically adds each
output accumulator element once to a zeroed FP32 slot shared by that query tile.
An independent finish kernel applies the existing count scaling and output cast.
There is no inter-CTA wait. Zeroing and finish are included in benchmark timings.
"One atomic submission" means one update per output element, not one instruction
for the entire output tile. Short query tiles avoid both atomic and finish work.

`stream_global=True` additionally interleaves batch/head tasks; it does not change
the hard cap. `stream_atomic=False` (default) selects private partials plus finish
for an independent reduction control. `stream_complete=True` selects private
partials with last-completer reduction. The cap affects the forward portion of
training; backward remains the existing implementation.

Compared with the older schedules:

- Default native forward assigns one query tile to each CTA, with no KV cap.
  Automatic dispatch can choose existing fixed-KV kernels in measured regimes.
- `stream_tiles=True, cute_stream_waves=...` derives a work budget from total KV
  iterations and requested worker count; it is not a fixed token cap.
- `stream_global=True` without a cap uses the experimental LPT cost model to
  selectively split long tasks, at most four segments per query tile.
- The preexisting `balanced_chunk` is in KV tile units and uses atomic output
  even for query tiles which need no split.

## Validation and benchmark

`tests/test_softplus_kv_cap.py` checks exact KV coverage, hard bounds, direct-write
bypass and slot ownership. GPU checks cover BF16/FP16, D64/D128, unequal Q/K lengths,
strided inputs, windows, compiled gradients, private/atomic/last-completer variants,
and ten CUDA graph replays. All 42 Softplus regression tests passed in 269.310 s.

```sh
python -m unittest discover -s tests -p 'test_softplus_*.py'
python trains/bench_softplus_completion.py --kv-caps --output trains/softplus_kv_caps_full.json
```

The benchmark covers 18 shapes and both prefill and attention forward+backward.
It includes caps 256/512/1024/2048 with both query-major and head-interleaved order,
a private-partial cap512 control, original default, native unsplit, whole-query LPT,
and tuned Softmax FA4 with/without the same whole-query LPT order. It uses the median
of three alternating CUDA graph trials. These are attention-operator timings,
not full GPT training steps or time to process a prompt through the whole model.

## Measurements

Speedup = original default / candidate, higher is better. 36 paired cases.

| Schedule | Prefill geomean | Train geomean |
|---|---:|---:|
| cap256_atomic | 0.537x | 0.815x |
| cap512_atomic | 0.703x | 0.897x |
| cap1024_atomic | 0.964x | 0.990x |
| cap2048_atomic | 1.011x | 0.998x |
| cap256_global_atomic | 0.495x | 0.789x |
| cap512_global_atomic | 0.635x | 0.858x |
| cap1024_global_atomic | 0.879x | 0.954x |
| cap2048_global_atomic | 0.972x | 0.987x |
| cap512_private | 0.659x | 0.875x |
| whole_lpt | 1.019x | 1.000x |

Selected B1/H6/T2048 full-causal cases, milliseconds:

| D | Phase | Default | Cap256 | Cap512 | Cap1024 | Whole LPT | Softmax tuned | Softmax LPT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | prefill | 0.0722 | 0.1048 | 0.0766 | 0.0718 | 0.0544 | 0.0663 | 0.0475 |
| 64 | train | 0.2429 | 0.2769 | 0.2474 | 0.2422 | 0.2233 | 0.2385 | 0.2249 |
| 128 | prefill | 0.1179 | 0.1908 | 0.1408 | 0.1301 | 0.0951 | 0.1174 | 0.0939 |
| 128 | train | 0.5316 | 0.5958 | 0.5461 | 0.5311 | 0.4886 | 0.4864 | 0.4543 |

The fixed cap guarantees a bound on KV iterations, not equal runtime for every CTA.
Short final segments, causal/window masking and atomic contention remain. A sliding
window of 512 visible keys per row can span more than 512 physical KV positions for
an M64 query tile, so a cap of 512 can still split that tile. Small caps increase
CTA startup work, Q reloads, and output traffic; they do not remove QK/PV work or
Softplus elementwise work. These are structural costs, not profiler measurements.
No default dispatch changes are made based on these experiments.

## Long-context follow-up

The optional `kv_major=True` traversal and 32K/64K measurements are documented in
`SOFTPLUS_LONG_RESULTS.md`. It preserves the same token cap and changes task order
only; previous measurements in this file used the original order.
