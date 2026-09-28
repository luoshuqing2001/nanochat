
## split_diag_exp
# NVIDIA H100 80GB HBM3: split-KV, bf16, D=128, ms. 'auto' = FA3 get_num_splits heuristic; best = min over (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
| shape | B | Tq | Tk | H | W | FA3 auto | FA3 best (splits) | softplus auto | softplus best (splits) | softplus 1 split | best/best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 4096 | 16 | -1 | 15.6 us | 14.3 us (8) | 17.9 us | 17.3 us (8) | 41.9 us | 0.83x |
| decode | 1 | 1 | 16384 | 16 | -1 | 53.8 us | 53.4 us (8) | 56.7 us | 54.1 us (4) | 148.2 us | 0.99x |
| decode | 1 | 1 | 65536 | 16 | -1 | 184.4 us | 183.3 us (4) | 186.9 us | 180.9 us (4) | 582.9 us | 1.01x |
| decode | 4 | 1 | 16384 | 16 | -1 | 183.9 us | 184.0 us (2) | 184.7 us | 183.2 us (1) | 183.2 us | 1.00x |
| decode | 16 | 1 | 8192 | 16 | -1 | 351.5 us | 351.7 us (1) | 351.1 us | 351.1 us (1) | 351.1 us | 1.00x |
| decode nanochat d20 | 1 | 1 | 32768 | 10 | -1 | 66.4 us | 64.8 us (8) | 68.3 us | 66.2 us (6) | 293.3 us | 0.98x |
| decode SWA 512 | 8 | 1 | 32768 | 10 | 511 | 18.8 us | 14.5 us (1) | 22.1 us | 13.4 us (1) | 13.4 us | 1.08x |
| chunked prefill | 1 | 512 | 32768 | 8 | -1 | 107.0 us | 107.0 us (4) | 115.6 us | 115.8 us (4) | 313.2 us | 0.92x |
| long prefill | 1 | 8192 | 8192 | 4 | -1 | 109.6 us | 104.9 us (1) | 99.0 us | 94.7 us (1) | 94.7 us | 1.11x |

## split_pre_poly3
# NVIDIA H100 80GB HBM3: split-KV, bf16, D=128, ms. 'auto' = FA3 get_num_splits heuristic; best = min over (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
| shape | B | Tq | Tk | H | W | FA3 auto | FA3 best (splits) | softplus auto | softplus best (splits) | softplus 1 split | best/best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 4096 | 16 | -1 | 15.7 us | 14.3 us (8) | 20.0 us | 19.5 us (8) | 53.7 us | 0.74x |
| decode | 1 | 1 | 16384 | 16 | -1 | 53.3 us | 53.3 us (8) | 57.0 us | 57.0 us (8) | 193.6 us | 0.94x |
| decode | 1 | 1 | 65536 | 16 | -1 | 184.2 us | 183.2 us (4) | 187.9 us | 188.4 us (8) | 763.2 us | 0.97x |
| decode | 4 | 1 | 16384 | 16 | -1 | 184.9 us | 184.5 us (4) | 186.8 us | 187.1 us (2) | 200.2 us | 0.99x |
| decode | 16 | 1 | 8192 | 16 | -1 | 352.5 us | 352.4 us (1) | 353.7 us | 354.0 us (1) | 354.0 us | 1.00x |
| decode nanochat d20 | 1 | 1 | 32768 | 10 | -1 | 66.8 us | 64.8 us (8) | 70.9 us | 65.5 us (8) | 382.3 us | 0.99x |
| decode SWA 512 | 8 | 1 | 32768 | 10 | 511 | 18.8 us | 14.4 us (1) | 23.1 us | 15.2 us (1) | 15.2 us | 0.94x |
| chunked prefill | 1 | 512 | 32768 | 8 | -1 | 106.6 us | 108.7 us (4) | 140.2 us | 139.2 us (4) | 395.2 us | 0.78x |
| long prefill | 1 | 8192 | 8192 | 4 | -1 | 108.3 us | 104.1 us (1) | 119.4 us | 115.7 us (1) | 115.7 us | 0.90x |
