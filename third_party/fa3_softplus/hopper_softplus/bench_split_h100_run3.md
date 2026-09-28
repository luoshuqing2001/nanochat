
## split_diag_exp
# NVIDIA H100 80GB HBM3: split-KV, bf16, D=128, ms. 'auto' = FA3 get_num_splits heuristic; best = min over (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
| shape | B | Tq | Tk | H | W | FA3 auto | FA3 best (splits) | softplus auto | softplus best (splits) | softplus 1 split | best/best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 4096 | 16 | -1 | 15.6 us | 14.3 us (8) | 17.8 us | 17.3 us (8) | 42.1 us | 0.83x |
| decode | 1 | 1 | 16384 | 16 | -1 | 53.6 us | 53.4 us (8) | 56.8 us | 54.1 us (4) | 148.3 us | 0.99x |
| decode | 1 | 1 | 65536 | 16 | -1 | 184.4 us | 183.5 us (4) | 187.0 us | 181.1 us (4) | 582.9 us | 1.01x |
| decode | 4 | 1 | 16384 | 16 | -1 | 185.1 us | 184.3 us (4) | 185.3 us | 180.5 us (1) | 180.5 us | 1.02x |
| decode | 16 | 1 | 8192 | 16 | -1 | 352.0 us | 352.1 us (1) | 351.6 us | 351.8 us (1) | 351.8 us | 1.00x |
| decode nanochat d20 | 1 | 1 | 32768 | 10 | -1 | 66.4 us | 64.8 us (8) | 68.3 us | 66.3 us (6) | 293.2 us | 0.98x |
| decode SWA 512 | 8 | 1 | 32768 | 10 | 511 | 18.8 us | 14.4 us (1) | 22.2 us | 13.5 us (1) | 13.5 us | 1.07x |
| chunked prefill | 1 | 512 | 32768 | 8 | -1 | 106.5 us | 105.7 us (4) | 119.0 us | 113.8 us (4) | 321.2 us | 0.93x |
| long prefill | 1 | 8192 | 8192 | 4 | -1 | 109.9 us | 105.5 us (1) | 99.1 us | 96.3 us (1) | 96.3 us | 1.10x |

## split_pre_poly3
# NVIDIA H100 80GB HBM3: split-KV, bf16, D=128, ms. 'auto' = FA3 get_num_splits heuristic; best = min over (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
| shape | B | Tq | Tk | H | W | FA3 auto | FA3 best (splits) | softplus auto | softplus best (splits) | softplus 1 split | best/best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 4096 | 16 | -1 | 15.7 us | 14.4 us (8) | 19.5 us | 18.5 us (8) | 53.1 us | 0.78x |
| decode | 1 | 1 | 16384 | 16 | -1 | 53.4 us | 53.4 us (8) | 56.8 us | 56.8 us (8) | 194.1 us | 0.94x |
| decode | 1 | 1 | 65536 | 16 | -1 | 184.3 us | 183.3 us (4) | 188.3 us | 188.4 us (8) | 764.5 us | 0.97x |
| decode | 4 | 1 | 16384 | 16 | -1 | 184.3 us | 184.5 us (4) | 187.4 us | 187.7 us (2) | 203.1 us | 0.98x |
| decode | 16 | 1 | 8192 | 16 | -1 | 352.5 us | 352.0 us (1) | 353.7 us | 353.1 us (1) | 353.1 us | 1.00x |
| decode nanochat d20 | 1 | 1 | 32768 | 10 | -1 | 66.4 us | 64.7 us (8) | 71.0 us | 65.4 us (8) | 384.1 us | 0.99x |
| decode SWA 512 | 8 | 1 | 32768 | 10 | 511 | 18.8 us | 14.6 us (1) | 23.1 us | 15.0 us (1) | 15.0 us | 0.97x |
| chunked prefill | 1 | 512 | 32768 | 8 | -1 | 107.3 us | 108.1 us (4) | 138.9 us | 139.4 us (4) | 397.6 us | 0.78x |
| long prefill | 1 | 8192 | 8192 | 4 | -1 | 107.7 us | 104.2 us (1) | 117.9 us | 116.4 us (1) | 116.4 us | 0.90x |
