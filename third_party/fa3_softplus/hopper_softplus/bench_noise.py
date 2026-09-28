"""How much do *identical* kernels differ under bench()? Same call registered under several names."""
import torch

from bench_fwd import bench
from sp_attn import fwd, prescale_q

torch.manual_seed(0)
B, T, H, D = 32, 2048, 6, 128
q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
qp = prescale_q(q)
for trial in range(3):
    fns = {}
    for i in range(3):
        fns[f"stock#{i}"] = lambda: fwd("stock", q, k, v, causal=True)
        fns[f"pre_poly3#{i}"] = lambda: fwd("pre_poly3", qp, k, v, causal=True)
        fns[f"tile_nc128#{i}"] = lambda: fwd("tile_nc128", qp, k, v, causal=True)
    ms = bench(fns, rounds=15, target_ms=60.0)
    print(f"trial {trial}: " + "  ".join(f"{n}={t:.3f}" for n, t in ms.items()), flush=True)
