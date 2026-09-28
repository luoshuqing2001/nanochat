"""
Backward and forward+backward timing: softplus+RMSNorm (bwd_* builds) vs stock FA3 softmax, same
build flags. "bwd" is one call of the backward op (preprocess + main + postprocess kernels, plus the
dU buffer for softplus); "fwd+bwd" adds one forward. TFLOPS for bwd count 2.5x the forward flops.

    python hopper_softplus/bench_bwd.py
"""
import math

import torch

from bench_fwd import bench, pairs
from sp_attn import bwd, fwd, prescale_q

LN2 = math.log(2.0)
SHAPES = [  # name, B, T, H, causal, window_left
    ("d12 train (full)", 32, 2048, 6, True, -1),
    ("d12 train (SWA 512)", 32, 2048, 6, True, 511),
    ("d20 train (full)", 32, 2048, 10, True, -1),
    ("d20 train (SWA 512)", 32, 2048, 10, True, 511),
    ("long causal 8k", 4, 8192, 16, True, -1),
    ("non-causal 4k", 8, 4096, 16, False, -1),
    ("non-causal 2k", 32, 2048, 8, False, -1),
]
VARIANTS = {"stock": ("bwd_stock", None, False),
            "fn_rexp": ("fn_rexp", None, False), "fn_rexp_pre": ("fn_rexp_pre", LN2, True)}


def main():
    print(f"# {torch.cuda.get_device_name()}, bf16, D=128, ms (TFLOPS)")
    cols = list(VARIANTS)
    print("| shape | B | T | H | W | " + " | ".join(f"bwd {c}" for c in cols) + " | "
          + " | ".join(f"fwd+bwd {c}" for c in cols) + " | best fwd+bwd / stock |")
    print("|---|---:|---:|---:|---:|" + "---:|" * (2 * len(cols)) + "---:|")
    for name, B, T, H, causal, wl in SHAPES:
        D = 128
        q, k, v, do = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        qp = prescale_q(q)
        fns = {}
        for c, (build, scale, pre) in VARIANTS.items():
            qq = qp if pre else q
            out, lse = fwd(build, qq, k, v, causal=causal, window_left=wl, softmax_scale=scale)
            fns[("bwd", c)] = (lambda b=build, qq=qq, out=out, lse=lse, s=scale:
                               bwd(b, do, qq, k, v, out, lse, causal=causal, window_left=wl, softmax_scale=s))

            def both(b=build, qq=qq, s=scale):
                o, l = fwd(b, qq, k, v, causal=causal, window_left=wl, softmax_scale=s)
                bwd(b, do, qq, k, v, o, l, causal=causal, window_left=wl, softmax_scale=s)
            fns[("all", c)] = both
        ms = bench(fns, rounds=11, target_ms=60.0)
        f_fwd = 4 * B * H * D * pairs(T, causal, wl)
        cells = [f"{ms[('bwd', c)]:.3f} ({2.5 * f_fwd / ms[('bwd', c)] / 1e9:.0f})" for c in cols]
        cells += [f"{ms[('all', c)]:.3f} ({3.5 * f_fwd / ms[('all', c)] / 1e9:.0f})" for c in cols]
        best = min([c for c in cols[1:] if not c.startswith("diag")], key=lambda c: ms[("all", c)])
        print(f"| {name} | {B} | {T} | {H} | {wl} | " + " | ".join(cells)
              + f" | {best} {ms[('all', 'stock')] / ms[('all', best)]:.3f}x |", flush=True)
        del q, k, v, do, qp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
