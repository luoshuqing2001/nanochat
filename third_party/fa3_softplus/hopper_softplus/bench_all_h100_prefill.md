# prefill (forward only), NVIDIA H100 80GB HBM3, bf16, D=128, causal (bottom-right aligned), us
| shape | B | Tq | Tk | H | W | FA3 auto | FA3 best (splits) | rexp auto | rexp best (splits) | rexp 1 split | x auto | x best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prompt 2k | 1 | 2048 | 2048 | 10 | -1 | 28.8 | 29.0 (1) | 27.4 | 26.6 (1, non-split build) | 26.6 | **1.05** | **1.09** |
| prompt 8k | 1 | 8192 | 8192 | 10 | -1 | 305.1 | 304.8 (1) | 287.6 | 260.2 (1, non-split build) | 260.2 | **1.06** | **1.17** |
| prompt 32k | 1 | 32768 | 32768 | 10 | -1 | 4227.1 | 4131.4 (1) | 3902.3 | 3382.1 (1, non-split build) | 3382.1 | **1.08** | **1.22** |
| batched prompts 8x8k | 8 | 8192 | 8192 | 10 | -1 | 2183.1 | 2152.9 (1) | 2054.0 | 1773.9 (1, non-split build) | 1773.9 | **1.06** | **1.21** |
| prompt 32k, SWA 512 | 1 | 32768 | 32768 | 10 | 512 | 223.6 | 216.8 (1) | 216.1 | 197.4 (1, non-split build) | 197.4 | **1.03** | **1.10** |
| chunked prefill 512 vs 32k cache | 1 | 512 | 32768 | 10 | -1 | 142.2 | 161.8 (8) | 150.5 | 176.4 (8) | 322.8 | **0.95** | **0.92** |
| chunked prefill 2k vs 32k cache | 4 | 2048 | 32768 | 10 | -1 | 2081.8 | 2078.7 (1) | 1973.8 | 1897.8 (1, non-split build) | 1897.8 | **1.05** | **1.10** |
