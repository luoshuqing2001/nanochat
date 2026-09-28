"""
Split-KV: softplus with the atomic in-kernel reduction vs stock FA3 split + combine kernel.

Both sides get FA3's own num_splits heuristic (num_splits=0 -> get_num_splits, same code in both
builds) and, separately, the best of an explicit sweep. FA3 times include its combine launch;
softplus has none. Decode (Tq=1) and low-parallelism long prefill.

    python hopper_softplus/bench_split.py [softplus_variant]
"""
import sys

import torch

from bench_fwd import bench
from sp_attn import fwd, prescale_q

SHAPES = [  # name, B, Tq, Tk, H, window_left
    ("decode", 1, 1, 4096, 16, -1),
    ("decode", 1, 1, 16384, 16, -1),
    ("decode", 1, 1, 65536, 16, -1),
    ("decode", 4, 1, 16384, 16, -1),
    ("decode", 16, 1, 8192, 16, -1),
    ("decode nanochat d20", 1, 1, 32768, 10, -1),
    ("decode SWA 512", 8, 1, 32768, 10, 511),
    ("chunked prefill", 1, 512, 32768, 8, -1),
    ("long prefill", 1, 8192, 8192, 4, -1),
]
SWEEP = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)


def main(var):
    dev = torch.cuda.get_device_name()
    print(f"# {dev}: split-KV, bf16, D=128, ms. 'auto' = FA3 get_num_splits heuristic; best = min over {SWEEP}")
    print("| shape | B | Tq | Tk | H | W | FA3 auto | FA3 best (splits) | softplus auto | softplus best (splits) | softplus 1 split | best/best |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, B, Tq, Tk, H, wl in SHAPES:
        D = 128
        q = torch.randn(B, Tq, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, Tk, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, Tk, H, D, device="cuda", dtype=torch.bfloat16)
        qs = prescale_q(q) if "pre_" in var else q
        fns = {("fa3", 0): lambda: fwd("stock", q, k, v, causal=True, window_left=wl, num_splits=0),
               ("sp", 0): lambda: fwd(var, qs, k, v, causal=True, window_left=wl, num_splits=0)}
        for ns in SWEEP:
            fns[("fa3", ns)] = (lambda ns=ns: fwd("stock", q, k, v, causal=True, window_left=wl, num_splits=ns))
            fns[("sp", ns)] = (lambda ns=ns: fwd(var, qs, k, v, causal=True, window_left=wl, num_splits=ns))
        ms = bench(fns, rounds=9, target_ms=30.0)
        fb = min(SWEEP, key=lambda n: ms[("fa3", n)])
        sb = min(SWEEP, key=lambda n: ms[("sp", n)])
        print(f"| {name} | {B} | {Tq} | {Tk} | {H} | {wl} | {ms[('fa3', 0)]*1e3:.1f} us "
              f"| {ms[('fa3', fb)]*1e3:.1f} us ({fb}) | {ms[('sp', 0)]*1e3:.1f} us "
              f"| {ms[('sp', sb)]*1e3:.1f} us ({sb}) | {ms[('sp', 1)]*1e3:.1f} us "
              f"| {ms[('fa3', fb)] / ms[('sp', sb)]:.2f}x |", flush=True)
        del q, k, v, qs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for var in (sys.argv[1:] or ["split_pre_poly3"]):
        print(f"\n## {var}")
        main(var)
