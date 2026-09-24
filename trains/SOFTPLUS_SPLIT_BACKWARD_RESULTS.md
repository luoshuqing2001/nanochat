# Separate dV and dQ/dK: implementation and rejection

Implemented two CuTe backward specializations on the same SM120 KV-owner
scheduler as the existing fused kernel. Neither is selected by default.

- `SoftplusBackwardQK`: QK, sigmoid, dP, dQ and dK; skips dV. The unused
  Softplus result from the common math helper is eliminated by the compiler.
- `SoftplusBackwardV`: QK, Softplus and dV only; does not load V or compute
  sigmoid/dP/dQ/dK. It overlaps the next Q load with the current dV GEMM.
- `softplus_split_backward(...)` combines their three useful gradient outputs.
- `_flash_attn_bwd` has an explicit `softplus_bwd_component="qk"/"v"` option,
  validated for dense SM120 MHA with unsplit owners. Component type is part of
  the compilation key. Other paths retain the previous fused implementation.

The implementation deliberately retains the common driver's preparation,
conversion, unused zero-gradient outputs and shared-storage structures. All
component timings below include those costs; split timings call both complete
components, rather than reporting only the faster subkernel.

## Backward results

GB10, B1/H6/D128, BF16, full causal. Baseline is the existing **8-warp** fused
kernel, not the older 4-warp baseline. Tiles are 64x64. Split candidates cover
128/256 threads and one/two Q stages for dQ/dK; dV uses one Q/dO stage.

|T|Fused ms|Best measured split ms|Change|
|---|---:|---:|---:|
|8K|4.047|5.694|+40.7%|
|32K|58.964|82.580|+40.1%|
|64K|227.852|325.294|+42.8%|

The best measured split uses 256 threads for both components, one Q/dO stage.
At 64K, complete dQ/dK and dV component calls separately cost 187.400 and
136.477 ms. Component medians need not sum exactly to the combined median.
Softmax's matched 8-warp backward control is 214.191 ms.

The initial 2K run shows a large timing anomaly (fused 0.763 ms versus previous
measurements around 0.34 ms, with large variations in other candidates).
It is not evidence that split wins at 2K. Raw samples are retained.
The initial D64 row used the unshared fused control; it is not used to claim
performance relative to the default D64 implementation. The benchmark now
selects shared-P/dS for the D64 fused control.

## Why the proposal did not win

Register use really decreased:

|D128 configuration|Registers/thread|Local bytes/thread|
|---|---:|---:|
|Existing fused, 256 threads|226|0|
|dQ/dK, 256 threads|197|0|
|dV, 256 threads|128|0|
|dQ/dK, 128 threads, Q stage 1|255|24|
|dQ/dK, 128 threads, Q stage 2|246|0|
|dV, 128 threads|166|0|

However, the existing 8-warp baseline already has no register spill. Reducing
register pressure does not itself guarantee more resident CTAs: the current
64x64 D128 tiles still consume substantial shared memory.

The arithmetic tradeoff is unfavorable in this implementation:

- Fused computes QK, dP, dV, dQ, dK: five matrix products per valid tile pair.
- Split computes QK twice: six products, approximately 20% more GEMM arithmetic
  when Q/K/V dimensions match.
- Softplus and sigmoid can no longer share the same exponential evaluation.
- Q/K/dO are visited in separate passes, with additional setup and output work.

These are code-derived costs, not a hardware-counter attribution of every
millisecond. The result rejects this tested split implementation, not every
possible specialized tile or producer/consumer design.

## Full attention forward + backward and assembly

`bench_softplus_split_train.py` compares the same head-local forward followed
by either the existing fused backward or the new complete split wrapper.
Results are in `softplus_split_train.json`; this is attention-operator timing,
not whole-model training. `softplus_split_sass.json` records retained cubin
hashes and static opcode counts. Static instruction counts are not runtime
cycles and must not be interpreted as proportional speedups.

|T|Fused forward+backward ms|Split forward+backward ms|Change|
|---|---:|---:|---:|
|32K|78.790|101.179|+28.4%|
|64K|310.778|419.254|+34.9%|

Three alternating trials show some timing outliers (raw samples retained), but
the split regression is much larger than the observed within-configuration
variation. Forward outputs match exactly; maximum gradient difference divided
by the reference gradient's peak magnitude is below 7.34e-5 in these rows.

Retained D128/256-thread binaries confirm that the intended computation was
actually removed from dQ/dK:

|Static opcode|Fused|dQ/dK|dV|Split sum|
|---|---:|---:|---:|---:|
|HMMA.16816|160|128|64|192|
|MUFU.EX2|16|16|16|32|
|FFMA|96|0|96|96|
|MUFU.RCP|16|16|0|16|

The log polynomial's FFMA chain is absent from dQ/dK. Splitting relocates that
work to dV rather than eliminating it from the complete backward.

## Validation and reproduction

- `test_softplus_split_backward.py`: independent FP32 gradient reference for
  BF16/FP16, strided inputs, rectangular/tail sequences, local/causal masks,
  alpha values, 128/256-thread components, extreme scores and CUDA graph replay.
- The existing `test_softplus_warp8.py` suite is rerun to check the fused path.
- Both passed: 2 new test methods and 5 fused-path regression methods.
- `bench_softplus_split_backward.py`: component and combined backward timings,
  relative gradient errors, compiler register/local-memory resources.
- Raw files: `softplus_split_backward_quick.json`,
  `softplus_split_backward_long.json`, `softplus_split_train.json`.
- Final hashes/test evidence: `softplus_split_summary.json`.

Production default dispatch, prefill and decode kernels are unchanged. Further
work should preserve shared QK/exponential computation when exploring separate
producer/consumer work inside a fused kernel; the expected benefit still needs
measurement rather than assuming that a split is inherently faster.
