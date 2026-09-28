# NVIDIA H100 80GB HBM3: split-KV, bf16, D=128, ms. 'auto' = FA3 get_num_splits heuristic; best = min over (1, 2, 4, 8, 16, 32, 64)
| shape | B | Tq | Tk | H | W | FA3 auto | FA3 best (splits) | softplus auto | softplus best (splits) | softplus 1 split | best/best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 4096 | 16 | -1 | 15.6 us | 14.3 us (8) | 19.7 us | 19.0 us (8) | 53.1 us | 0.75x |
| decode | 1 | 1 | 16384 | 16 | -1 | 53.1 us | 52.9 us (8) | 57.5 us | 57.5 us (8) | 192.4 us | 0.92x |
| decode | 1 | 1 | 65536 | 16 | -1 | 186.0 us | 182.5 us (4) | 190.9 us | 190.6 us (8) | 758.3 us | 0.96x |
| decode | 4 | 1 | 16384 | 16 | -1 | 183.4 us | 183.6 us (2) | 186.7 us | 186.7 us (2) | 201.2 us | 0.98x |
| decode | 16 | 1 | 8192 | 16 | -1 | 351.7 us | 351.6 us (1) | 352.9 us | 352.9 us (1) | 352.9 us | 1.00x |
| decode nanochat d20 | 1 | 1 | 32768 | 10 | -1 | 66.7 us | 64.5 us (8) | 68.3 us | 65.2 us (8) | 382.7 us | 0.99x |
| decode SWA 512 | 8 | 1 | 32768 | 10 | 511 | 18.8 us | 14.9 us (1) | 23.0 us | 15.0 us (1) | 15.0 us | 0.99x |
| chunked prefill | 1 | 512 | 32768 | 8 | -1 | 107.8 us | 105.3 us (4) | 132.9 us | 143.1 us (4) | 394.0 us | 0.74x |
| long prefill | 1 | 8192 | 8192 | 4 | -1 | 107.3 us | 103.9 us (1) | 119.9 us | 115.3 us (1) | 115.3 us | 0.90x |
