(kernels-hub FA3 unavailable: ValueError: A kernel version or revision must be specified. Use `version=<major>` for a stable kernel API version or `revision=<branch/tag/commit>` for an explicit Hub revision. See: https://huggingface.co/docs/kernels/migration)
# NVIDIA H100 80GB HBM3, torch 2.9.1+cu128, bf16, D=128, forward only, ms (TFLOPS)
| shape | B | T | H | W | stock FA3 | sp_mufu2 | sp_naive | sp_poly3 | sp_poly4 | sp_softexp | sp_mix | best softplus / stock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d12 train (full) | 32 | 2048 | 6 | -1 | 0.373 (553) | 0.499 (414) | 0.421 (490) | 0.421 (490) | 0.456 (452) | 0.569 (362) | 0.465 (444) | sp_naive 0.885x |
| d12 train (SWA 512) | 32 | 2048 | 6 | 511 | 0.238 (379) | 0.287 (315) | 0.260 (348) | 0.267 (338) | 0.277 (326) | 0.341 (265) | 0.293 (308) | sp_naive 0.917x |
| d20 train (full) | 32 | 2048 | 10 | -1 | 0.604 (569) | 0.820 (419) | 0.703 (489) | 0.717 (480) | 0.767 (448) | 0.953 (361) | 0.769 (447) | sp_naive 0.859x |
| d20 train (SWA 512) | 32 | 2048 | 10 | 511 | 0.393 (383) | 0.455 (330) | 0.425 (354) | 0.445 (338) | 0.462 (325) | 0.565 (266) | 0.473 (318) | sp_naive 0.925x |
| long causal 8k | 4 | 8192 | 16 | -1 | 1.609 (683) | 2.140 (514) | 1.942 (566) | 1.968 (559) | 2.091 (526) | 2.678 (411) | 2.117 (519) | sp_naive 0.829x |
| long causal 16k | 2 | 16384 | 16 | -1 | 3.154 (697) | 4.388 (501) | 3.736 (589) | 3.715 (592) | 4.004 (549) | 5.238 (420) | 4.192 (525) | sp_poly3 0.849x |
| non-causal 4k | 8 | 4096 | 16 | -1 | 1.652 (666) | 2.407 (457) | 1.965 (560) | 1.983 (554) | 2.127 (517) | 2.950 (373) | 2.217 (496) | sp_naive 0.840x |
